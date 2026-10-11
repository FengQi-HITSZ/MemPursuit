"""LongMemEval-S and BEAM ingestion, retrieval and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from .adapter import MemPursuitAdapter, _token_count
from .beam_judge import _evaluate_beam_with_parse_retries
from .beam_official import answer_prompt as beam_answer_prompt
from .datasets import load_beam, load_longmemeval
from .evaluation import exact_match, summarize, token_f1
from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .models import RetrievalResult
from .openai_gateway import OpenAIGateway
from .prompts import PromptSnapshot, extract_answer, parse_judgment
from .security import safe_component

ROOT = Path(__file__).resolve().parents[2]


def scope_routes(scopes, endpoints):
    """Assign memory scopes to endpoints round robin in dataset order."""
    if not endpoints or len(endpoints) != len(set(endpoints)):
        raise ValueError("Memory endpoints must be nonempty and unique")
    return {scope.spec.scope_id: endpoints[index % len(endpoints)] for index, scope in enumerate(scopes)}


def prefix(result: RetrievalResult, cutoff: int) -> RetrievalResult:
    value = RetrievalResult.from_dict(result.to_dict())
    value.requested_k = cutoff
    value.main_hits = value.main_hits[:cutoff]
    value.context_tokens = _token_count([hit.text for hit in value.all_context()])
    value.backend_metadata["retrieval_reuse"] = {
        "source_cutoff": 200,
        "derived_cutoff": cutoff,
        "policy": "truncate-ranked-main-hits",
    }
    return value


class CheckpointAdapter(MemPursuitAdapter):
    """Persist successful ADD responses without changing pair boundaries or order."""

    def __init__(self, endpoint, directory):
        super().__init__(endpoint)
        self.directory = directory
        self.pair_index = 0

    def _request(self, method, path, payload=None, **kwargs):
        if method != "POST" or path != "/memories":
            return super()._request(method, path, payload, **kwargs)
        index = self.pair_index
        self.pair_index += 1
        checkpoint = self.directory / f"pair_{index:06d}.json"
        fingerprint = sha256_json(payload)
        old = read_json(checkpoint)
        if old is not None:
            if old["payload_sha256"] != fingerprint:
                raise RuntimeError("Ingestion input changed at an existing checkpoint")
            return old["response"]
        response = super()._request(method, path, payload, **kwargs)
        if not isinstance(response, dict) or not isinstance(response.get("results"), list):
            raise RuntimeError("ADD returned an invalid response; checkpoint not advanced")
        atomic_write_json(checkpoint, {"payload_sha256": fingerprint, "response": response})
        return response


def ingest_scope(scope, endpoint, run_dir):
    directory = run_dir / "ingestion" / safe_component(scope.spec.scope_id)
    events = [event for session in scope.sessions for event in session]
    expected = sha256_json([event.to_dict() for event in events])
    done_path = directory / "complete.json"
    done = read_json(done_path)
    if done is not None:
        if done["events_sha256"] != expected:
            raise RuntimeError("Completed memory scope does not match the dataset")
        return
    adapter = CheckpointAdapter(endpoint, directory)
    result = adapter._build(scope.spec, events)
    atomic_write_json(done_path, {"events_sha256": expected, **result})


def evaluate_one(question, retrieval, cutoff, benchmark, prompts, answerer, judge):
    """Generate an answer and score it with the benchmark-specific evaluator."""
    track = "official-v1" if benchmark == "beam" else "current"
    reader_context = None
    if benchmark == "beam":
        prompt, reader_context = beam_answer_prompt(question, retrieval)
    else:
        prompt = prompts.answer_prompt(question, retrieval)
    started = time.monotonic()
    raw_answer, answer_response = answerer.text(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
    )
    answer_seconds = time.monotonic() - started
    answer = extract_answer(raw_answer, benchmark)
    started = time.monotonic()
    beam_evaluation = None
    if benchmark == "beam":
        beam_evaluation = _evaluate_beam_with_parse_retries(judge, question, raw_answer)
        score = float(beam_evaluation["primary_score"])
        raw_judgment = json.dumps(beam_evaluation, ensure_ascii=False)
        judge_usage = beam_evaluation["usage"]
    else:
        system, prompt = prompts.judge_prompt(question, answer)
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        raw_judgment, judge_response = judge.text(messages, temperature=0.0, max_tokens=1024)
        score = float(parse_judgment(raw_judgment, benchmark, track))
        judge_usage = judge_response.get("usage", {})
    return {
        "benchmark": benchmark,
        "question_id": question.question_id,
        "scope_id": question.scope_id,
        "category": question.category,
        "question": question.question,
        "answer": question.answer,
        "question_date": question.question_date,
        "prompt_track": track,
        "cutoff": cutoff,
        "generated_answer": answer,
        "raw_generated_answer": raw_answer,
        "raw_judgment": raw_judgment,
        "correct": score,
        "score": score,
        "exact_match": exact_match(answer, question.answer),
        "token_f1": token_f1(answer, question.answer),
        "retrieval": retrieval.to_dict(),
        "reader_context": reader_context,
        "beam_official_evaluation": beam_evaluation,
        "timing": {
            "retrieval_seconds": retrieval.elapsed_seconds,
            "answer_seconds": answer_seconds,
            "judge_seconds": time.monotonic() - started,
        },
        "api_usage": {"answer": answer_response.get("usage", {}), "judge": judge_usage},
    }


def build_parser(benchmark):
    parser = argparse.ArgumentParser(description=f"MemPursuit {benchmark} evaluation pipeline")
    parser.add_argument("--project-name", "--run-id", dest="run_id", required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--beam-size", choices=["500K", "1M"], default="500K")
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--top-k-cutoffs", default="10,20,50,200")
    parser.add_argument("--stages", default="ingest,retrieve,evaluate")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--reuse-retrieval-dir",
        type=Path,
        help="Completed run directory; reuse its frozen retrieval for another answerer",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mem0-host", default=os.getenv("MEM0_HOST", "http://localhost:8889"))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / benchmark)
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel answer/judge workers (default: 16)")
    parser.add_argument(
        "--ingest-workers",
        type=int,
        default=12 if benchmark == "longmemeval" else 4,
        help="Parallel isolated memory scopes",
    )
    parser.add_argument("--retrieval-workers", type=int, default=12 if benchmark == "longmemeval" else 4)
    parser.add_argument("--mem0-hosts", help="Comma-separated memory endpoints for distributing memory scopes")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--answerer-model", default=os.getenv("ANSWER_MODEL", "gpt-4o-mini"))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--answerer-base-url",
        default=os.getenv("ANSWER_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    return parser


def run(args, benchmark):
    cutoffs = sorted({int(x) for x in args.top_k_cutoffs.split(",")})
    if not cutoffs or min(cutoffs) < 1 or max(cutoffs) > 200:
        raise ValueError("Context cutoffs must be between 1 and 200")
    if args.max_workers < 1 or args.ingest_workers < 1 or args.retrieval_workers < 1:
        raise ValueError("Worker counts must be positive")
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("max-questions must be positive")
    if args.predict_only and args.evaluate_only:
        raise ValueError("predict-only and evaluate-only are mutually exclusive")
    stages = args.stages.split(",")
    if args.predict_only:
        stages = ["ingest", "retrieve"]
    if args.evaluate_only or args.reuse_retrieval_dir:
        stages = ["evaluate"]
    if not stages or any(s not in {"ingest", "retrieve", "evaluate"} for s in stages):
        raise ValueError("stages must contain ingest, retrieve and/or evaluate")
    source = args.dataset_path or (
        Path("datasets/longmemeval/longmemeval_s_cleaned.json")
        if benchmark == "longmemeval"
        else Path(f"datasets/beam/beam-{args.beam_size}.arrow")
    )
    if benchmark == "longmemeval":
        scopes = load_longmemeval(source, args.run_id, args.mode, allow_download=args.download_dataset)
    else:
        scopes = load_beam(source, args.run_id, args.mode, chat_size=args.beam_size)
    endpoints = [item.strip().rstrip("/") for item in (args.mem0_hosts or args.mem0_host).split(",") if item.strip()]
    routes = scope_routes(scopes, endpoints)
    questions = [(scope, question) for scope in scopes for question in scope.questions]
    if args.max_questions:
        questions = questions[: args.max_questions]
    selected = {scope.spec.scope_id for scope, _ in questions}
    scopes = [scope for scope in scopes if scope.spec.scope_id in selected]
    run_dir = args.output_dir / safe_component(args.run_id)
    manifest_path = run_dir / "manifest.json"
    prompts = PromptSnapshot(ROOT, run_dir, benchmark)
    contract = {
        "benchmark": benchmark,
        "mode": args.mode,
        "dataset_sha256": sha256_file(source),
        "questions": [q.question_id for _, q in questions],
        "cutoffs": cutoffs,
        "answer_model": args.answerer_model,
        "judge_model": args.judge_model,
        "answer_base_url": args.answerer_base_url,
        "judge_base_url": args.judge_base_url,
        "prompt_sha256": prompts.sha256,
        "memory_run_id": args.run_id,
        "memory_endpoints": endpoints,
        "memory_routes": routes,
        "retrieval_source": str(args.reuse_retrieval_dir.resolve()) if args.reuse_retrieval_dir else None,
    }
    previous = read_json(manifest_path)
    if previous is not None:
        if not args.resume:
            raise RuntimeError("Run already exists; use --resume or a new project name")
        if previous["contract"] != contract:
            raise RuntimeError(
                "Dataset, question selection, models, prompts or cutoffs changed; use a new project name"
            )
    manifest = previous or {"contract": contract, "completed_stages": []}
    retrieval_dir = args.reuse_retrieval_dir or run_dir
    if args.reuse_retrieval_dir:
        other = read_json(retrieval_dir / "manifest.json")
        if not other or "retrieve" not in other.get("completed_stages", []):
            raise RuntimeError("Source run has not completed retrieval")
        for key in ["benchmark", "dataset_sha256", "questions"]:
            if other["contract"][key] != contract[key]:
                raise RuntimeError(f"Source retrieval has different {key}")
        manifest["retrieval_source"] = str(retrieval_dir.resolve())
    if any(s in stages for s in ["ingest", "retrieve"]):
        algorithms = [
            MemPursuitAdapter._algorithm_manifest(MemPursuitAdapter(endpoint)._validated_health())
            for endpoint in endpoints
        ]
        algorithm = algorithms[0]
        if any(item != algorithm for item in algorithms):
            raise RuntimeError("Memory shards have different algorithm settings")
        if "algorithm" in manifest and manifest["algorithm"] != algorithm:
            raise RuntimeError("Memory server configuration changed during the run")
        manifest["algorithm"] = algorithm
    manifest["execution"] = {
        "evaluation_workers": args.max_workers,
        "ingest_workers": args.ingest_workers,
        "retrieval_workers": args.retrieval_workers,
        "endpoint_pool_size": len(endpoints),
    }
    atomic_write_json(manifest_path, manifest)
    if "ingest" in stages:
        with ThreadPoolExecutor(max_workers=args.ingest_workers) as pool:
            list(pool.map(lambda scope: ingest_scope(scope, routes[scope.spec.scope_id], run_dir), scopes))
        manifest["completed_stages"] = sorted(set(manifest["completed_stages"]) | {"ingest"})
        atomic_write_json(manifest_path, manifest)
    if "retrieve" in stages:
        if "ingest" not in manifest["completed_stages"]:
            raise RuntimeError("Complete ingestion before retrieval")

        def retrieve_question(item):
            scope, question = item
            target = run_dir / "retrieval" / f"{safe_component(question.question_id)}.json"
            if not target.exists():
                adapter = MemPursuitAdapter(routes[scope.spec.scope_id])
                result = adapter._retrieve(scope.spec, question.question, 200)
                atomic_write_json(target, result.to_dict())

        with ThreadPoolExecutor(max_workers=args.retrieval_workers) as pool:
            for index, _ in enumerate(pool.map(retrieve_question, questions), 1):
                print(f"Retrieved {index}/{len(questions)}", flush=True)
        manifest["completed_stages"] = sorted(set(manifest["completed_stages"]) | {"retrieve"})
        atomic_write_json(manifest_path, manifest)
    if "evaluate" not in stages:
        return {"run_directory": str(run_dir), "completed_stages": manifest["completed_stages"]}
    if not args.reuse_retrieval_dir and "retrieve" not in manifest["completed_stages"]:
        raise RuntimeError("Complete retrieval before evaluation")
    answerer = OpenAIGateway(
        args.answerer_base_url, os.getenv("ANSWER_API_KEY") or os.environ["OPENAI_API_KEY"], model=args.answerer_model
    )
    judge = OpenAIGateway(
        args.judge_base_url, os.getenv("JUDGE_API_KEY") or os.environ["OPENAI_API_KEY"], model=args.judge_model
    )

    def evaluate_question(item):
        _, question = item
        result = RetrievalResult.from_dict(
            read_json(retrieval_dir / "retrieval" / f"{safe_component(question.question_id)}.json")
        )
        if result.query != question.question or result.scope_id != question.scope_id:
            raise RuntimeError("Frozen retrieval does not match this question")
        records = []
        for cutoff in cutoffs:
            target = run_dir / "records" / f"{safe_component(question.question_id)}.k{cutoff}.json"
            record = read_json(target)
            if record is None:
                record = evaluate_one(question, prefix(result, cutoff), cutoff, benchmark, prompts, answerer, judge)
                atomic_write_json(target, record)
            records.append(record)
        return records

    records = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for rows in pool.map(evaluate_question, questions):
            records.extend(rows)
            print(f"Evaluated {len(records)}/{len(questions) * len(cutoffs)} contexts", flush=True)
    metrics = summarize(records, benchmark)
    metrics["score_scale"] = "0–1; multiply by 100 for a 0–100 score"
    atomic_write_json(run_dir / "metrics.json", metrics)
    manifest["completed_stages"] = sorted(set(manifest["completed_stages"]) | {"evaluate"})
    atomic_write_json(manifest_path, manifest)
    return metrics


def main(benchmark, argv=None):
    load_dotenv(ROOT / ".env")
    args = build_parser(benchmark).parse_args(argv)
    print(json.dumps(run(args, benchmark), ensure_ascii=False, indent=2))
