"""
LOCOMO Benchmark Runner
=======================

Combined ingest + search + answer + judge pipeline for the LOCOMO-10
benchmark (Snap Research, ACL 2024).

Flow:
    1. Download dataset (auto-download from GitHub if missing)
    2. For each conversation:
        a. Parse into sessions, ingest via Mem0
        b. For each question:
            - Search Mem0 -> retrieved memories
            - Generate answer (answerer model)
            - Judge answer vs ground truth (judge model)
            - Save checkpoint
    3. Compute metrics (by category, by cutoff)
    4. Write unified result JSON

Usage:
    python -m benchmarks.locomo.run --project-name test
    python -m benchmarks.locomo.run --project-name full --answerer-model gpt-4o --judge-model gpt-4o
    python -m benchmarks.locomo.run --project-name full --predict-only
"""

from __future__ import annotations
import argparse
import asyncio
import json
import math
import os
import re
import statistics
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


from tqdm import tqdm
from benchmarks.common.llm_client import LLMClient
from benchmarks.common.mem0_client import Mem0Client, RerankerSearchError, format_search_results
from benchmarks.common.utils import (
    GracefulShutdown,
    IngestionCheckpoint,
    cutoff_label,
    download_file,
    parse_cutoffs,
    save_result_json,
    setup_logging,
)
from .prompts import (
    CATEGORY_NAMES,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    get_judge_prompt_with_evidence,
    preprocess_answer,
)

if os.getenv("BENCHMARKS_SKIP_DOTENV") != "1":
    load_dotenv(override=False)
DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_DATASET_DIR = "datasets/locomo"
DEFAULT_DATASET_FILE = "locomo10.json"
CHUNK_SIZE = 1
ADAPTIVE_CONTEXT_LABEL = "adaptive_sparse_omp"
SPARSE_OMP_PERFORMANCE_FILE = "_sparse_omp_performance.json"
RERANKER_PERFORMANCE_FILE = "_reranker_performance.json"
SPARSE_OMP_POLICY_VERSION = "sparse_omp_v2"


def parse_search_threshold(value: str) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("threshold must be a float or one of: none, null, off") from exc
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("threshold must be <= 1.0; use a negative value or 'none' to disable")
    if parsed < 0.0:
        return None
    return parsed


def download_dataset(dataset_dir: str, logger: Any) -> str:
    """Download locomo10.json from GitHub if not present."""
    path = os.path.join(dataset_dir, DEFAULT_DATASET_FILE)
    if os.path.exists(path):
        logger.info("Dataset already exists: %s", path)
        return path
    os.makedirs(dataset_dir, exist_ok=True)
    logger.info("Downloading LOCOMO-10 dataset...")
    download_file(DATASET_URL, path, description="Downloading LOCOMO-10")
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 10:
        os.remove(path)
        raise RuntimeError(f"Invalid dataset: expected 10 conversations, got {len(data)}")
    logger.info("Downloaded: %s (%d conversations)", path, len(data))
    return path


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_locomo_date(date_str: str) -> datetime | None:
    """Parse LOCOMO date: '1:56 pm on 8 May, 2023'."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def locomo_date_to_epoch(date_str: str) -> int | None:
    parsed = parse_locomo_date(date_str)
    if parsed:
        return int(parsed.replace(tzinfo=timezone.utc).timestamp())
    return None


def get_sorted_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """Extract and sort sessions chronologically."""
    session_keys = [k for k in conversation if re.match("^session_\\d+$", k)]
    paired = []
    for key in session_keys:
        date_key = f"{key}_date_time"
        date_str = conversation.get(date_key, "")
        turns = conversation[key]
        paired.append((key, date_str, turns))

    def sort_key(item: tuple) -> tuple:
        parsed = parse_locomo_date(item[1])
        if parsed:
            return (0, parsed)
        num = int(re.search("\\d+", item[0]).group())
        return (1, datetime(2000, 1, num))

    paired.sort(key=sort_key)
    return paired


def session_to_chunks(turns: list[dict], speaker_a: str, speaker_b: str) -> list[list[dict]]:
    """Convert turns to message chunks for ingestion."""
    messages = []
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        blip = turn.get("blip_caption", "")
        query = turn.get("query", "")
        if query and blip:
            photo_tag = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo_tag = f"[Sharing image - query for: {query}]"
        elif blip:
            photo_tag = f"[Sharing image that shows: {blip}]"
        else:
            photo_tag = ""
        if photo_tag:
            text = f"{text} {photo_tag}" if text else photo_tag
        if not text:
            continue
        role = "user" if speaker == speaker_a else "assistant"
        messages.append({"role": role, "content": f"{speaker}: {text}"})
    chunks = []
    for i in range(0, len(messages), CHUNK_SIZE):
        chunk = messages[i : i + CHUNK_SIZE]
        if chunk:
            chunks.append(chunk)
    return chunks


def load_evidence_lookup(dataset_path: str) -> dict[tuple, str]:
    """Build lookup: (conv_idx, dia_id) -> formatted turn text."""
    with open(dataset_path) as f:
        data = json.load(f)
    lookup = {}
    for conv_idx, conv in enumerate(data):
        conversation = conv["conversation"]
        session_dates = {}
        for key in conversation:
            if key.endswith("_date_time") and key.startswith("session_"):
                session_num = key.replace("session_", "").replace("_date_time", "")
                session_dates[session_num] = conversation[key]
        for key in conversation:
            if key.startswith("session_") and (not key.endswith("date_time")):
                if not isinstance(conversation[key], list):
                    continue
                for turn in conversation[key]:
                    dia_id = turn.get("dia_id", "")
                    if dia_id:
                        speaker = turn.get("speaker", "")
                        text = turn.get("text", "")
                        dia_match = re.match("D(\\d+):", dia_id)
                        date_suffix = ""
                        if dia_match:
                            snum = dia_match.group(1)
                            sdate = session_dates.get(snum, "")
                            if sdate:
                                date_suffix = f", said on {sdate}"
                        lookup[conv_idx, dia_id] = f'[{dia_id}{date_suffix}] {speaker}: "{text}"'
    return lookup


async def ingest_conversation(
    conv_idx: int,
    entry: dict,
    mem0: Mem0Client,
    logger: Any,
    run_id: str,
    project_name: str,
    output_dir: str,
    shutdown: GracefulShutdown,
    debug: bool = True,
) -> tuple[bool, str, int]:
    """Ingest all sessions of a LOCOMO conversation into Mem0.

    Returns: (success, user_id, total_chunks_processed)
    """
    conversation = entry["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    user_id = f"locomo_{conv_idx}_{run_id}"
    checkpoint = IngestionCheckpoint(output_dir)
    key = str(conv_idx)
    (is_done, cp_data) = checkpoint.is_complete(key, CHUNK_SIZE)
    if is_done and cp_data:
        chunks_done = cp_data.get("total_chunks_processed", 0)
        user_id = cp_data.get("user_id", user_id)
        logger.info("Conversation %d already ingested (user_id=%s, %d chunks)", conv_idx, user_id, chunks_done)
        return (True, user_id, chunks_done)
    (chunks_already_done, resumed_uid) = checkpoint.load_progress(key, CHUNK_SIZE)
    if resumed_uid and chunks_already_done:
        user_id = resumed_uid
        logger.info("Resuming conversation %d from %d completed chunks", conv_idx, len(chunks_already_done))
    sorted_sessions = get_sorted_sessions(conversation)
    total_chunks = sum((len(session_to_chunks(s, speaker_a, speaker_b)) for (_, _, s) in sorted_sessions))
    logger.info(
        "Ingesting conversation %d: %s & %s, %d sessions, %d chunks",
        conv_idx,
        speaker_a,
        speaker_b,
        len(sorted_sessions),
        total_chunks,
    )
    debug_file = None
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"conv_{conv_idx}_ingestion.txt")
        debug_mode = "a" if chunks_already_done else "w"
        debug_file = open(debug_path, debug_mode, encoding="utf-8")
        if not chunks_already_done:
            debug_file.write(f"{'=' * 80}\n")
            debug_file.write(f"CONVERSATION {conv_idx}: {speaker_a} & {speaker_b}\n")
            debug_file.write(f"Sessions: {len(sorted_sessions)}, Chunks: {total_chunks}\n")
            debug_file.write(f"User ID: {user_id}\n")
            debug_file.write(f"{'=' * 80}\n\n")
    pbar = tqdm(total=total_chunks, desc=f"Ingest conv {conv_idx}", initial=len(chunks_already_done), leave=True)
    total_processed = len(chunks_already_done)
    total_failed = 0
    for session_key, date_str, turns in sorted_sessions:
        chunks = session_to_chunks(turns, speaker_a, speaker_b)
        if not chunks:
            continue
        session_epoch = locomo_date_to_epoch(date_str)
        if debug_file and f"{session_key}_header" not in chunks_already_done:
            debug_file.write(f"\n{'---' * 27}\n")
            debug_file.write(f"SESSION: {session_key}  |  Date: {date_str}  |  Chunks: {len(chunks)}\n")
            debug_file.write(f"{'---' * 27}\n\n")
        for chunk_idx, messages in enumerate(chunks):
            chunk_key = f"{session_key}_c{chunk_idx}"
            if chunk_key in chunks_already_done:
                continue
            if shutdown.requested:
                logger.info("Shutdown requested at conv %d, chunk %s", conv_idx, chunk_key)
                pbar.close()
                if debug_file:
                    debug_file.close()
                return (True, user_id, total_processed)
            if any((not msg.get("content", "").strip() for msg in messages)):
                chunks_already_done.add(chunk_key)
                total_processed += 1
                pbar.update(1)
                continue
            if debug_file:
                debug_file.write(f"--- Chunk {chunk_idx} ({len(messages)} messages) ---\n")
                for msg in messages:
                    debug_file.write(f"  {msg['role']}: {msg['content']}\n")
                debug_file.write("\n")
            metadata = {
                "benchmark": "locomo",
                "locomo_conversation_idx": conv_idx,
                "locomo_session_key": session_key,
                "locomo_session_date_time": date_str,
                "locomo_session_epoch": session_epoch,
                "locomo_chunk_idx": chunk_idx,
                "locomo_chunk_key": chunk_key,
            }
            response = await mem0.add(messages, user_id, timestamp=session_epoch, metadata=metadata)
            if response is not None:
                total_processed += 1
                if debug_file:
                    results = response.get("results", [])
                    if results:
                        debug_file.write(f"--- Chunk {chunk_idx} (extracted) ---\n")
                        for mem_item in results:
                            mem_text = mem_item.get("memory", "")
                            event_type = mem_item.get("event", "")
                            debug_file.write(f"  [{event_type}] {mem_text}\n")
                        debug_file.write("\n")
            else:
                total_failed += 1
                logger.warning("Ingestion failed: conv %d %s chunk %d", conv_idx, session_key, chunk_idx)
            chunks_already_done.add(chunk_key)
            checkpoint.save_progress(
                key,
                {
                    "conversation_idx": conv_idx,
                    "user_id": user_id,
                    "run_id": run_id,
                    "chunk_size": CHUNK_SIZE,
                    "completed_chunks": list(chunks_already_done),
                },
            )
            pbar.update(1)
    pbar.close()
    if debug_file:
        debug_file.write(f"\nSUMMARY: {total_processed}/{total_chunks} OK, {total_failed} failed\n")
        debug_file.close()
    checkpoint.save_complete(
        key,
        {
            "conversation_idx": conv_idx,
            "user_id": user_id,
            "run_id": run_id,
            "chunk_size": CHUNK_SIZE,
            "total_chunks_processed": total_processed,
            "total_chunks_failed": total_failed,
        },
    )
    return (total_failed == 0, user_id, total_processed)


def context_policy_labels(cutoffs: list[int], adaptive_context: bool) -> list[str]:
    if adaptive_context:
        return [ADAPTIVE_CONTEXT_LABEL]
    return [cutoff_label(cutoff) for cutoff in cutoffs]


def validate_context_cutoffs(cutoffs: list[int], top_k: int, adaptive_context: bool) -> None:
    """Validate fixed cutoffs only when fixed-cutoff contexts are requested."""
    if not adaptive_context and cutoffs and (max(cutoffs) > top_k):
        raise ValueError("The maximum --top-k-cutoffs value must not exceed --top-k")


def labels_to_evaluate(existing_results: dict[str, Any], requested_labels: list[str], rejudge: bool) -> list[str]:
    if rejudge:
        return list(requested_labels)
    return [label for label in requested_labels if label not in existing_results]


def _resume_result_complete(result: Any, requested_labels: list[str], *, predict_only: bool) -> bool:
    """Return whether a checkpoint is safe to skip during a resumed run."""
    if not isinstance(result, dict) or not isinstance(result.get("question_id"), str):
        return False
    retrieval = result.get("retrieval")
    if not isinstance(retrieval, dict) or not isinstance(retrieval.get("search_results"), list):
        return False
    if predict_only:
        return True
    cutoff_results = result.get("cutoff_results")
    if not isinstance(cutoff_results, dict):
        return False
    for label in requested_labels:
        cutoff = cutoff_results.get(label)
        if not isinstance(cutoff, dict):
            return False
        generated_answer = cutoff.get("generated_answer")
        if not isinstance(generated_answer, str) or not generated_answer.strip():
            return False
        judgment = cutoff.get("judgment")
        score = cutoff.get("score")
        if judgment not in {"CORRECT", "WRONG"} or score not in {0, 1, 0.0, 1.0}:
            return False
    return True


def _retryable_question_failure(exc: Exception) -> bool:
    """Identify transient search/answer/judge failures worth retrying as a whole question."""
    message = str(exc).lower()
    retryable_markers = (
        "answer generation failed",
        "judge failed",
        "connection error",
        "server disconnected",
        "timed out",
        "timeout",
    )
    return any((marker in message for marker in retryable_markers))


def compatible_sparse_omp_metadata(metadata: Any, expected_mode: str | None = None) -> bool:
    if not isinstance(metadata, dict) or metadata.get("provider") != "sparse_omp":
        return False
    if metadata.get("policy_version") != SPARSE_OMP_POLICY_VERSION or "fallback_used" in metadata:
        return False
    required = {
        "mode",
        "early_stopping",
        "effective_iteration_limit",
        "qp_solve_count",
        "retain_zero_coefficients",
        "support_count",
        "support_ids",
        "stop_reason",
        "converged",
        "budget_capped",
    }
    if not required.issubset(metadata) or not isinstance(metadata.get("support_ids"), list):
        return False
    mode = metadata.get("mode")
    if mode not in {"fixed", "adaptive"} or (expected_mode is not None and mode != expected_mode):
        return False
    if bool(metadata.get("early_stopping")) != (mode == "adaptive"):
        return False
    if bool(metadata.get("retain_zero_coefficients")) != (mode == "fixed"):
        return False
    try:
        return (
            int(metadata["support_count"]) == len(metadata["support_ids"])
            and int(metadata["effective_iteration_limit"]) >= 0
            and (int(metadata["qp_solve_count"]) >= 0)
        )
    except (TypeError, ValueError):
        return False


def resolve_locomo_context(
    formatted: list[dict[str, Any]], context_label: str, rerank_metadata: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve fixed or pure sparse-OMP context with identical online/offline semantics."""
    metadata = rerank_metadata or {}
    common = {
        "context_policy": context_label,
        "support_count": int(metadata.get("support_count", 0) or 0),
        "stop_reason": metadata.get("stop_reason"),
        "converged": bool(metadata.get("converged", False)),
        "budget_capped": bool(metadata.get("budget_capped", False)),
        "sparse_omp_policy_version": metadata.get("policy_version"),
        "sparse_omp_mode": metadata.get("mode"),
    }
    if context_label != ADAPTIVE_CONTEXT_LABEL:
        if not context_label.startswith("top_"):
            raise ValueError(f"Unsupported LOCOMO context policy: {context_label}")
        cutoff = int(context_label.removeprefix("top_"))
        return (formatted[:cutoff], common)
    if not compatible_sparse_omp_metadata(metadata, expected_mode="adaptive"):
        raise RuntimeError(
            "--adaptive-context requires sparse_omp rerank_metadata for every question; rerun prediction with --rerank using reranker.provider=sparse_omp"
        )
    support_ids = [str(value) for value in metadata.get("support_ids", [])]
    memories_by_id = {str(item.get("id")): item for item in formatted}
    missing_ids = [memory_id for memory_id in support_ids if memory_id not in memories_by_id]
    if missing_ids:
        raise RuntimeError(
            f"Adaptive sparse OMP context cannot be reproduced because support memories are missing from the saved search results: {missing_ids[:5]}. Increase --top-k and rerun prediction."
        )
    return ([memories_by_id[memory_id] for memory_id in support_ids], common)


async def evaluate_locomo_contexts(
    *,
    formatted: list[dict[str, Any]],
    rerank_metadata: dict[str, Any] | None,
    labels: list[str],
    question: str,
    category: int,
    processed_answer: str,
    reference_date_human: str | None,
    user_profile: dict | None,
    evidence_context: str,
    answerer: LLMClient,
    judge_llm: LLMClient,
    existing_results: dict[str, dict] | None = None,
    reuse_generated_answers: bool = False,
    on_label_complete: Any = None,
) -> dict[str, dict]:
    cutoff_results: dict[str, dict] = dict(existing_results or {})
    reused_answer_judgments: dict[str, dict[str, Any]] = {}
    for label in labels:
        (sliced, context_diagnostics) = resolve_locomo_context(formatted, label, rerank_metadata)
        if reuse_generated_answers:
            saved_label_result = cutoff_results.get(label)
            generated_answer = (
                saved_label_result.get("generated_answer") if isinstance(saved_label_result, dict) else None
            )
            if not isinstance(generated_answer, str) or not generated_answer.strip():
                raise RuntimeError(
                    f"Cannot reuse generated answer for {label}: saved cutoff result has no generated_answer"
                )
        else:
            gen_prompt = get_answer_generation_prompt(
                question, sliced, reference_date=reference_date_human, user_profile=user_profile
            )
            generated_answer = await answerer.generate(system="", user=gen_prompt)
            if not isinstance(generated_answer, str) or not generated_answer.strip():
                raise RuntimeError(f"Answer generation failed for {label}: empty response after retries")
            if "ANSWER:" in generated_answer:
                generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()
            if not generated_answer:
                raise RuntimeError(f"Answer generation failed for {label}: empty ANSWER section")
        if evidence_context:
            judge_prompt = get_judge_prompt_with_evidence(
                category, question, processed_answer, generated_answer, evidence_context
            )
        else:
            judge_prompt = get_judge_prompt(category, question, processed_answer, generated_answer)
        cached_raw = reused_answer_judgments.get(generated_answer) if reuse_generated_answers else None
        judge_response_reused = cached_raw is not None
        if cached_raw is not None:
            raw = cached_raw
        else:
            raw = await judge_llm.generate_structured(system=JUDGE_SYSTEM_PROMPT, user=judge_prompt)
            if reuse_generated_answers and isinstance(raw, dict):
                reused_answer_judgments[generated_answer] = raw
        if not isinstance(raw, dict):
            raise RuntimeError(f"Judge failed for {label}: no structured response after retries")
        judgment = str(raw.get("label", "")).upper()
        if judgment not in {"CORRECT", "WRONG"}:
            raise RuntimeError(f"Judge failed for {label}: invalid label {raw.get('label')!r}")
        correct = judgment == "CORRECT"
        cutoff_results[label] = {
            "judgment": "CORRECT" if correct else "WRONG",
            "score": 1.0 if correct else 0.0,
            "generated_answer": generated_answer,
            "memories_evaluated": len(sliced),
            "reason": raw.get("reasoning", "") if isinstance(raw, dict) else "",
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_response_reused": judge_response_reused,
            **context_diagnostics,
        }
        if on_label_complete is not None:
            on_label_complete(label, cutoff_results[label])
    return cutoff_results


async def process_question(
    qa: dict,
    qa_idx: int,
    conv_idx: int,
    user_id: str,
    mem0: Mem0Client,
    answerer: LLMClient,
    judge_llm: LLMClient,
    cutoffs: list[int],
    top_k: int,
    threshold: float | None,
    reference_date_human: str | None,
    user_profile: dict | None,
    evidence_lookup: dict | None,
    predict_only: bool,
    logger: Any,
    score_debug: bool = False,
    rerank: bool = False,
    sparse_omp_rerank: bool = False,
    adaptive_context: bool = False,
    result_path: str | None = None,
) -> dict[str, Any]:
    """Process a single question: search + answer + judge at multiple cutoffs.

    Returns a result dict suitable for serialization.
    """
    question_id = f"conv{conv_idx}_q{qa_idx}"
    question = qa["question"]
    category = qa["category"]
    answer = str(qa["answer"])
    rerank_request_latency: float | None = None
    start = time.monotonic()
    rerank_options = None
    if rerank and sparse_omp_rerank:
        rerank_options = {"early_stopping": adaptive_context}
        if not adaptive_context:
            rerank_options["min_iterations"] = max(cutoffs) if cutoffs else top_k
            rerank_options["short_pool_policy"] = "use_available"
    search_results = await mem0.search(
        question,
        user_id,
        top_k=top_k,
        threshold=threshold,
        rerank=rerank,
        score_debug=score_debug,
        rerank_options=rerank_options,
    )
    search_latency = (time.monotonic() - start) * 1000
    rerank_metadata = search_results.get("rerank_metadata") if isinstance(search_results, dict) else None
    (formatted, query_debug) = format_search_results(search_results)
    if adaptive_context and (not compatible_sparse_omp_metadata(rerank_metadata, expected_mode="adaptive")):
        raise RuntimeError(
            f"{question_id}: --adaptive-context requires a sparse_omp rerank response with compatible metadata"
        )
    result: dict[str, Any] = {
        "question_id": question_id,
        "conversation_idx": conv_idx,
        "category": category,
        "category_name": CATEGORY_NAMES.get(category, "unknown"),
        "question": question,
        "ground_truth_answer": answer,
        "evidence": qa.get("evidence", []),
        "user_id": user_id,
        "reference_date": reference_date_human,
        "retrieval": {
            "search_query": question,
            "search_results": formatted,
            "search_latency_ms": round(search_latency, 1) if search_latency is not None else None,
            "total_results": len(formatted),
            "threshold": threshold,
        },
    }
    if rerank_metadata:
        result["retrieval"]["rerank_metadata"] = rerank_metadata
    sparse_performance = query_debug.get("sparse_omp_performance") if isinstance(query_debug, dict) else None
    if sparse_performance:
        adaptive_count = None
        if adaptive_context:
            (adaptive_memories, _) = resolve_locomo_context(formatted, ADAPTIVE_CONTEXT_LABEL, rerank_metadata)
            adaptive_count = len(adaptive_memories)
        result["_sparse_omp_performance"] = {
            **sparse_performance,
            "search_total": search_latency,
            "outcome": "success",
            "error_code": None,
            "support_count": int((rerank_metadata or {}).get("support_count", 0) or 0),
            "adaptive_context_count": adaptive_count,
            "converged": bool((rerank_metadata or {}).get("converged", False)),
            "budget_capped": bool((rerank_metadata or {}).get("budget_capped", False)),
        }
    reranker_performance = query_debug.get("reranker_performance") if isinstance(query_debug, dict) else None
    if reranker_performance:
        performance_sample = {
            **reranker_performance,
            "outcome": "success",
            "error_code": None,
            "provider": (rerank_metadata or {}).get("provider"),
            "model": (rerank_metadata or {}).get("model"),
            "device": (rerank_metadata or {}).get("resolved_devices") or (rerank_metadata or {}).get("device"),
            "torch_dtype": (rerank_metadata or {}).get("torch_dtype"),
            "batch_size": (rerank_metadata or {}).get("batch_size"),
        }
        performance_sample["search_total"] = search_latency
        performance_sample["retrieval_reused"] = False
        result["_reranker_performance"] = performance_sample
    if user_profile is not None:
        result["user_profile"] = user_profile
    if predict_only:
        return result
    processed_answer = preprocess_answer(category, answer)
    ev_ctx = ""
    if evidence_lookup:
        for ref in qa.get("evidence", []):
            key = (conv_idx, ref)
            if key in evidence_lookup:
                ev_ctx += evidence_lookup[key] + "\n"
        ev_ctx = ev_ctx.strip()

    def save_completed_label(label: str, label_result: dict[str, Any]) -> None:
        result.setdefault("cutoff_results", {})[label] = label_result
        if result_path:
            saved_result = dict(result)
            saved_result.pop("_sparse_omp_performance", None)
            saved_result.pop("_reranker_performance", None)
            save_result_json(result_path, saved_result)

    result["cutoff_results"] = await evaluate_locomo_contexts(
        formatted=formatted,
        rerank_metadata=rerank_metadata,
        labels=context_policy_labels(cutoffs, adaptive_context),
        question=question,
        category=category,
        processed_answer=processed_answer,
        reference_date_human=reference_date_human,
        user_profile=user_profile,
        evidence_context=ev_ctx,
        answerer=answerer,
        judge_llm=judge_llm,
        on_label_complete=save_completed_label,
    )
    return result


async def apply_locomo_judge_to_saved_result(
    result: dict,
    qa: dict,
    conv_idx: int,
    answerer: LLMClient,
    judge_llm: LLMClient,
    cutoffs: list[int],
    evidence_lookup: dict | None,
    adaptive_context: bool = False,
    labels: list[str] | None = None,
    result_path: str | None = None,
    reuse_generated_answers: bool = False,
) -> None:
    """Fill ``cutoff_results`` using ``retrieval.search_results`` only (no Mem0)."""
    formatted = list(result["retrieval"]["search_results"])
    question = result["question"]
    category = qa["category"]
    answer = str(qa["answer"])
    reference_date_human = result.get("reference_date")
    user_profile = result.get("user_profile")
    rerank_metadata = result.get("retrieval", {}).get("rerank_metadata")
    processed_answer = preprocess_answer(category, answer)
    ev_ctx = ""
    if evidence_lookup:
        for ref in qa.get("evidence", []):
            key = (conv_idx, ref)
            if key in evidence_lookup:
                ev_ctx += evidence_lookup[key] + "\n"
        ev_ctx = ev_ctx.strip()
    requested_labels = labels if labels is not None else context_policy_labels(cutoffs, adaptive_context)
    existing_results = dict(result.get("cutoff_results") or {})

    def save_completed_label(label: str, label_result: dict[str, Any]) -> None:
        result.setdefault("cutoff_results", {})[label] = label_result
        if result_path:
            save_result_json(result_path, result)

    result["cutoff_results"] = await evaluate_locomo_contexts(
        formatted=formatted,
        rerank_metadata=rerank_metadata,
        labels=requested_labels,
        question=question,
        category=category,
        processed_answer=processed_answer,
        reference_date_human=reference_date_human,
        user_profile=user_profile,
        evidence_context=ev_ctx,
        answerer=answerer,
        judge_llm=judge_llm,
        existing_results=existing_results,
        reuse_generated_answers=reuse_generated_answers,
        on_label_complete=save_completed_label,
    )


def expected_locomo_question_items(
    dataset: list[dict], conv_indices: list[int], categories: list[int], max_questions: int | None
) -> list[tuple[str, int, int, dict]]:
    """(question_id, conv_idx, qa_idx, qa_dict) for every question in scope."""
    items: list[tuple[str, int, int, dict]] = []
    for conv_idx in conv_indices:
        if conv_idx >= len(dataset):
            continue
        entry = dataset[conv_idx]
        questions = entry.get("qa", entry.get("qa_pairs", []))
        conv_questions = [(qi, qa) for (qi, qa) in enumerate(questions) if qa.get("category") in categories]
        if max_questions is not None:
            conv_questions = conv_questions[:max_questions]
        for qi, qa in conv_questions:
            items.append((f"conv{conv_idx}_q{qi}", conv_idx, qi, qa))
    return items


def locomo_predict_outputs_complete(
    output_dir: str, expected_items: list[tuple[str, int, int, dict]]
) -> tuple[bool, list[str]]:
    """True if every expected question has JSON with retrieval.search_results."""
    missing: list[str] = []
    for qid, _, _, _ in expected_items:
        path = os.path.join(output_dir, f"{qid}.json")
        if not os.path.isfile(path):
            missing.append(qid)
            continue
        try:
            data = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            missing.append(f"{qid} (unreadable)")
            continue
        retr = data.get("retrieval") or {}
        if "search_results" not in retr:
            missing.append(f"{qid} (no search_results)")
    return (len(missing) == 0, missing)


def nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted((float(value) for value in values))
    rank = max(1, min(len(ordered), math.ceil(percentile * len(ordered))))
    return ordered[rank - 1]


def summarize_numeric_samples(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
        "p90": nearest_rank_percentile(numeric, 0.9),
        "p95": nearest_rank_percentile(numeric, 0.95),
        "max": max(numeric),
    }


def build_sparse_omp_performance_report(samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    timing_names = (
        "query_embedding",
        "hybrid_candidate_retrieval",
        "concurrency_wait",
        "dense_vector_fetch",
        "vector_prepare",
        "gram_build",
        "support_selection",
        "tail_fill",
        "rerank_total",
        "search_total",
    )
    stage_summary = {}
    for name in timing_names:
        values = [sample[name] for sample in samples.values() if isinstance(sample.get(name), (int, float))]
        stage_summary[name] = summarize_numeric_samples(values)

    def count_values(field: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for sample in samples.values():
            counts[str(sample.get(field))] += 1
        return dict(sorted(counts.items()))

    return {
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "timings_ms": stage_summary,
            "stop_reason": count_values("stop_reason"),
            "embedding_reused": count_values("embedding_reused"),
            "outcome": count_values("outcome"),
            "error_code": count_values("error_code"),
            "converged": count_values("converged"),
            "budget_capped": count_values("budget_capped"),
            "support_count": summarize_numeric_samples(
                [
                    sample["support_count"]
                    for sample in samples.values()
                    if isinstance(sample.get("support_count"), (int, float))
                ]
            ),
            "adaptive_context_count": summarize_numeric_samples(
                [
                    sample["adaptive_context_count"]
                    for sample in samples.values()
                    if isinstance(sample.get("adaptive_context_count"), (int, float))
                ]
            ),
            "selected_count": summarize_numeric_samples(
                [
                    sample["support_count"]
                    for sample in samples.values()
                    if isinstance(sample.get("support_count"), (int, float))
                ]
            ),
            "tail_count": summarize_numeric_samples(
                [
                    sample["tail_count"]
                    for sample in samples.values()
                    if isinstance(sample.get("tail_count"), (int, float))
                ]
            ),
            "vector_cache_hits": summarize_numeric_samples(
                [
                    sample["vector_cache_hits"]
                    for sample in samples.values()
                    if isinstance(sample.get("vector_cache_hits"), (int, float))
                ]
            ),
            "vector_cache_misses": summarize_numeric_samples(
                [
                    sample["vector_cache_misses"]
                    for sample in samples.values()
                    if isinstance(sample.get("vector_cache_misses"), (int, float))
                ]
            ),
            "vector_cache_hit_rate": sum(
                (float(sample.get("vector_cache_hits", 0) or 0) for sample in samples.values())
            )
            / max(
                sum(
                    (
                        float(sample.get("vector_cache_hits", 0) or 0)
                        + float(sample.get("vector_cache_misses", 0) or 0)
                        for sample in samples.values()
                    )
                ),
                1.0,
            ),
        },
    }


def update_sparse_omp_performance_report(path: str, question_id: str, sample: dict[str, Any]) -> None:
    report_path = Path(path)
    samples: dict[str, dict[str, Any]] = {}
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
            samples = dict(existing.get("samples") or {})
        except (json.JSONDecodeError, OSError, TypeError):
            samples = {}
    samples[question_id] = sample
    save_result_json(str(report_path), build_sparse_omp_performance_report(samples))


def build_reranker_performance_report(samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate server-internal reranker timings with the same percentile method as sparse OMP."""
    timing_names = (
        "query_embedding",
        "hybrid_candidate_retrieval",
        "queue_wait",
        "pair_prepare",
        "prompt_prepare",
        "tokenize",
        "inference",
        "score_postprocess",
        "decode",
        "sort",
        "rerank_total",
        "rerank_request_total",
        "search_total",
    )
    stage_summary = {}
    for name in timing_names:
        values = [sample[name] for sample in samples.values() if isinstance(sample.get(name), (int, float))]
        stage_summary[name] = summarize_numeric_samples(values)

    def numeric_summary(field: str) -> dict[str, float | int]:
        return summarize_numeric_samples(
            [sample[field] for sample in samples.values() if isinstance(sample.get(field), (int, float))]
        )

    def count_values(field: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for sample in samples.values():
            counts[str(sample.get(field))] += 1
        return dict(sorted(counts.items()))

    docs_per_second = []
    output_tokens_per_second = []
    for sample in samples.values():
        inference_ms = sample.get("inference")
        if not isinstance(inference_ms, (int, float)) or inference_ms <= 0:
            continue
        candidate_count = sample.get("candidate_count")
        if isinstance(candidate_count, (int, float)):
            docs_per_second.append(float(candidate_count) * 1000.0 / float(inference_ms))
        output_tokens = sample.get("output_tokens")
        if isinstance(output_tokens, (int, float)):
            output_tokens_per_second.append(float(output_tokens) * 1000.0 / float(inference_ms))
    return {
        "samples": samples,
        "summary": {
            "sample_count": len(samples),
            "provider": count_values("provider"),
            "model": count_values("model"),
            "device": count_values("device"),
            "torch_dtype": count_values("torch_dtype"),
            "batch_size": count_values("batch_size"),
            "outcome": count_values("outcome"),
            "error_code": count_values("error_code"),
            "retrieval_reused": count_values("retrieval_reused"),
            "timings_ms": stage_summary,
            "candidate_count": numeric_summary("candidate_count"),
            "batch_count": numeric_summary("batch_count"),
            "input_tokens": numeric_summary("input_tokens"),
            "output_tokens": numeric_summary("output_tokens"),
            "docs_per_second": summarize_numeric_samples(docs_per_second),
            "output_tokens_per_second": summarize_numeric_samples(output_tokens_per_second),
        },
    }


def update_reranker_performance_report(path: str, question_id: str, sample: dict[str, Any]) -> None:
    report_path = Path(path)
    samples: dict[str, dict[str, Any]] = {}
    if report_path.exists():
        try:
            existing = json.loads(report_path.read_text())
            samples = dict(existing.get("samples") or {})
        except (json.JSONDecodeError, OSError, TypeError):
            samples = {}
    samples[question_id] = sample
    save_result_json(str(report_path), build_reranker_performance_report(samples))


def compute_locomo_metrics(evaluations: list[dict], context_labels: list[str]) -> dict:
    """Compute per-category and overall metrics at each cutoff."""
    metrics_by_cutoff = {}
    for label in context_labels:
        total = len(evaluations)
        scores = [e.get("cutoff_results", {}).get(label, {}).get("score", 0.0) for e in evaluations]
        correct = sum((1 for s in scores if s >= 0.5))
        by_category: dict[str, list] = defaultdict(list)
        for e in evaluations:
            cat_name = e.get("category_name", "unknown")
            by_category[cat_name].append(e.get("cutoff_results", {}).get(label, {}).get("score", 0.0))
        cat_metrics = {}
        for cat_name in sorted(by_category):
            cat_scores = by_category[cat_name]
            cat_correct = sum((1 for s in cat_scores if s >= 0.5))
            cat_metrics[cat_name] = {
                "total": len(cat_scores),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_scores) * 100 if cat_scores else 0.0,
                "avg_score": statistics.mean(cat_scores) * 100 if cat_scores else 0.0,
            }
        metric: dict[str, Any] = {
            "overall": {
                "total": total,
                "correct": correct,
                "accuracy": correct / total * 100 if total else 0.0,
                "avg_score": statistics.mean(scores) * 100 if scores else 0.0,
            },
            "by_category": cat_metrics,
        }
        if label == ADAPTIVE_CONTEXT_LABEL:
            memory_counts = [
                e.get("cutoff_results", {}).get(label, {}).get("memories_evaluated", 0) for e in evaluations
            ]
            metric["context_size"] = {
                "mean": statistics.mean(memory_counts) if memory_counts else 0.0,
                "median": statistics.median(memory_counts) if memory_counts else 0.0,
                "p90": nearest_rank_percentile(memory_counts, 0.9),
            }
            by_stop_reason: dict[str, list[float]] = defaultdict(list)
            for evaluation in evaluations:
                cutoff_result = evaluation.get("cutoff_results", {}).get(label, {})
                by_stop_reason[str(cutoff_result.get("stop_reason"))].append(cutoff_result.get("score", 0.0))
            metric["by_stop_reason"] = {
                reason: {
                    "total": len(reason_scores),
                    "accuracy": sum((score >= 0.5 for score in reason_scores)) / len(reason_scores) * 100,
                }
                for (reason, reason_scores) in sorted(by_stop_reason.items())
            }
        metrics_by_cutoff[label] = metric
    return metrics_by_cutoff


def display_results(metrics_by_cutoff: dict, context_labels: list[str]) -> None:
    """Print metrics to console."""
    for label in context_labels:
        m = metrics_by_cutoff.get(label, {})
        overall = m.get("overall", {})
        print(f"\n--- {label} ---")
        print(
            f"  Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} ({overall.get('accuracy', 0):.1f}%) avg={overall.get('avg_score', 0):.1f}%"
        )
        for cat_name, cm in sorted(m.get("by_category", {}).items()):
            print(f"  {cat_name}: {cm['correct']}/{cm['total']} ({cm['accuracy']:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOCOMO-10 benchmark: ingest + search + answer + judge")
    parser.add_argument("--project-name", required=True, help="Name for this eval run")
    parser.add_argument(
        "--answerer-model", default=os.getenv("ANSWER_MODEL", "gpt-4o-mini"), help="Model for answer generation"
    )
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"), help="Model for judging")
    parser.add_argument("--provider", default="openai", help="OpenAI-compatible LLM provider")
    parser.add_argument("--judge-provider", default=None, help="Judge provider (defaults to --provider)")
    parser.add_argument(
        "--answer-base-url",
        default=os.getenv("ANSWER_BASE_URL") or None,
        help="optional OpenAI-compatible endpoint used only by the answerer",
    )
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("JUDGE_BASE_URL") or None,
        help="optional OpenAI-compatible endpoint used only by the judge",
    )
    parser.add_argument("--answer-timeout", type=float, default=900.0, help="answer request timeout in seconds")
    parser.add_argument("--judge-timeout", type=float, default=900.0, help="judge request timeout in seconds")
    parser.add_argument(
        "--answer-provider-only", default=None, help="comma-separated OpenRouter providers allowed for answer calls"
    )
    parser.add_argument(
        "--answer-provider-order",
        default=None,
        help="comma-separated OpenRouter providers to prefer in order; other providers remain eligible when fallbacks are enabled",
    )
    parser.add_argument(
        "--disable-answer-provider-fallbacks",
        action="store_true",
        help="set OpenRouter provider.allow_fallbacks=false for answer calls",
    )
    parser.add_argument(
        "--judge-provider-only", default=None, help="comma-separated OpenRouter providers allowed for judge calls"
    )
    parser.add_argument(
        "--disable-judge-provider-fallbacks",
        action="store_true",
        help="set OpenRouter provider.allow_fallbacks=false for judge calls",
    )
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated conversation indices")
    parser.add_argument("--top-k", type=int, default=200, help="Number of search results to retrieve")
    parser.add_argument(
        "--threshold",
        type=parse_search_threshold,
        default=0.1,
        help="Unified Mem0 search threshold for all semantic_score_mode values. Use 'none' or a negative value to disable.",
    )
    parser.add_argument("--top-k-cutoffs", default="10,20,50,200", help="Comma-separated cutoffs for evaluation")
    parser.add_argument("--max-workers", type=int, default=2, help="Max parallel workers")
    parser.add_argument("--output-dir", default="results/locomo", help="Output directory")
    parser.add_argument("--predict-only", action="store_true", help="Skip answer+judge, only ingest+search")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Judge only: requires all predict outputs on disk (use after --predict-only or full search). No Mem0.",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="With --evaluate-only: re-run answer+judge even if cutoff_results already exist",
    )
    parser.add_argument(
        "--reuse-generated-answers",
        action="store_true",
        help="With --evaluate-only --rejudge: keep each saved generated_answer and rerun only the judge, so the judge prompt is the sole experimental variable",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--score-debug", action="store_true", default=True, help="Include score breakdowns in output")
    parser.add_argument(
        "--rerank", action="store_true", default=True, help="Apply configured Mem0 reranker during search"
    )
    parser.add_argument(
        "--adaptive-context",
        action="store_true",
        help="Evaluate only the exact final adaptive sparse OMP support (mutually exclusive with fixed cutoffs)",
    )
    parser.add_argument("--dataset-path", default=None, help="Path to local locomo10.json")
    parser.add_argument("--run-id", default=None, help="Reuse a specific run_id for resume")
    parser.add_argument("--categories", default="1,2,3,4", help="Comma-separated categories")
    parser.add_argument("--with-evidence", action="store_true", help="Pass evidence to judge")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions to process (for quick testing)")
    parser.add_argument("--rpm", type=int, default=200, help="Requests per minute for LLM")
    parser.add_argument(
        "--question-max-attempts",
        type=int,
        default=3,
        help="Whole-question attempts after the answer/judge clients perform their own retries",
    )
    parser.add_argument(
        "--question-retry-delay",
        type=float,
        default=30.0,
        help="Base seconds between whole-question retries (multiplied by the failed attempt number)",
    )
    parser.add_argument("--backend", default="oss", choices=["oss"], help="Mem0 backend: self-hosted memory server")
    parser.add_argument("--mem0-host", default=None, help="Mem0 server URL (default: http://localhost:8889)")
    parser.add_argument("--mem0-api-key", default=None, help="Optional HTTP API key")
    parser.add_argument("--mem0-timeout", type=float, default=900.0)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    logger = setup_logging("locomo", debug=args.debug)
    if args.question_max_attempts < 1:
        raise ValueError("--question-max-attempts must be at least 1")
    if args.question_retry_delay < 0:
        raise ValueError("--question-retry-delay must be nonnegative")
    if args.answer_timeout <= 0 or args.judge_timeout <= 0:
        raise ValueError("--answer-timeout and --judge-timeout must be positive")
    if args.reuse_generated_answers and (not (args.evaluate_only and args.rejudge)):
        raise RuntimeError("--reuse-generated-answers requires --evaluate-only --rejudge")
    cutoffs = parse_cutoffs(args.top_k_cutoffs)
    validate_context_cutoffs(cutoffs, args.top_k, args.adaptive_context)
    labels = context_policy_labels(cutoffs, args.adaptive_context)
    categories = [int(c) for c in args.categories.split(",")]
    conv_indices = [int(c) for c in args.conversations.split(",")]
    run_id = args.run_id or uuid.uuid4().hex[:8]
    output_dir = os.path.join(args.output_dir, f"predicted_{args.project_name}")
    os.makedirs(output_dir, exist_ok=True)
    backend = os.getenv("MEM0_BACKEND", args.backend)
    print(f"LOCOMO Benchmark | project={args.project_name} run_id={run_id}")
    print(f"  Answerer: {args.answerer_model} ({args.provider})")
    print(f"  Judge: {args.judge_model} ({args.judge_provider or args.provider})")
    print(f"  Conversations: {args.conversations}")
    print(f"  Cutoffs: {args.top_k_cutoffs}")
    print(f"  Context policies: {', '.join(labels)}")
    print(f"  Search threshold: {(args.threshold if args.threshold is not None else 'disabled')}")
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = download_dataset(DEFAULT_DATASET_DIR, logger)
    dataset = load_dataset(dataset_path)
    evidence_lookup = None
    if args.with_evidence:
        evidence_lookup = load_evidence_lookup(dataset_path)
        print(f"  Evidence lookup: {len(evidence_lookup)} entries")
    answer_provider_only = (
        [value.strip() for value in args.answer_provider_only.split(",") if value.strip()]
        if args.answer_provider_only
        else []
    )
    answer_provider_order = (
        [value.strip() for value in args.answer_provider_order.split(",") if value.strip()]
        if args.answer_provider_order
        else []
    )
    if answer_provider_only and answer_provider_order:
        raise ValueError("--answer-provider-only and --answer-provider-order are mutually exclusive")
    answer_routing_key = "only" if answer_provider_only else "order"
    answer_routing_values = answer_provider_only or answer_provider_order
    answer_extra_body = (
        {
            "provider": {
                answer_routing_key: answer_routing_values,
                "allow_fallbacks": not args.disable_answer_provider_fallbacks,
            }
        }
        if answer_routing_values
        else None
    )
    answerer = LLMClient(
        api_key=os.getenv("ANSWER_API_KEY") or None,
        model=args.answerer_model,
        provider=args.provider,
        rpm=args.rpm,
        base_url=args.answer_base_url,
        timeout=args.answer_timeout,
        extra_body=answer_extra_body,
    )
    judge_provider = args.judge_provider or args.provider
    judge_provider_only = (
        [value.strip() for value in args.judge_provider_only.split(",") if value.strip()]
        if args.judge_provider_only
        else []
    )
    judge_extra_body = (
        {"provider": {"only": judge_provider_only, "allow_fallbacks": not args.disable_judge_provider_fallbacks}}
        if judge_provider_only
        else None
    )
    judge_llm = LLMClient(
        api_key=os.getenv("JUDGE_API_KEY") or None,
        model=args.judge_model,
        provider=judge_provider,
        rpm=args.rpm,
        base_url=args.judge_base_url,
        timeout=args.judge_timeout,
        extra_body=judge_extra_body,
    )
    if args.evaluate_only:
        expected_items = expected_locomo_question_items(dataset, conv_indices, categories, args.max_questions)
        if not expected_items:
            print("No questions in scope (check --conversations / --categories).")
            return
        (complete, missing) = locomo_predict_outputs_complete(output_dir, expected_items)
        if not complete:
            print(
                "Evaluate-only aborted: not all predict outputs are on disk. Finish ingest+search for every in-scope question first (full run without --predict-only, or --predict-only until complete)."
            )
            print(f"  Missing or invalid: {len(missing)} (showing up to 25): {missing[:25]}")
            return
        expected_mode = "adaptive" if args.adaptive_context else "fixed"
        incompatible = []
        for qid, _, _, _ in expected_items:
            saved = json.loads(Path(os.path.join(output_dir, f"{qid}.json")).read_text())
            metadata = saved.get("retrieval", {}).get("rerank_metadata")
            is_sparse_omp_prediction = isinstance(metadata, dict) and metadata.get("provider") == "sparse_omp"
            if (
                is_sparse_omp_prediction
                and (not compatible_sparse_omp_metadata(metadata, expected_mode=expected_mode))
                or (args.adaptive_context and (not is_sparse_omp_prediction))
            ):
                incompatible.append(qid)
        if incompatible:
            print(
                f"Evaluate-only aborted: saved predictions use missing, fallback-era, or incompatible sparse OMP metadata for {expected_mode} mode. Rerun prediction with the current sparse OMP policy."
            )
            print(f"  Missing or incompatible: {len(incompatible)} (showing up to 25): {incompatible[:25]}")
            return
        print(f"  Predict complete ({len(expected_items)} questions). Running judge phase (no Mem0)...")
        if args.reuse_generated_answers:
            print("  Reusing saved generated answers; only judge calls will be rerun")
        sem = asyncio.Semaphore(args.max_workers)

        async def judge_one(qid: str, conv_idx: int, qi: int, qa: dict) -> None:
            path = os.path.join(output_dir, f"{qid}.json")
            for question_attempt in range(1, args.question_max_attempts + 1):
                data = json.loads(Path(path).read_text())
                existing_cutoffs = data.get("cutoff_results") or {}
                labels_to_judge = labels_to_evaluate(existing_cutoffs, labels, args.rejudge)
                if not labels_to_judge:
                    return
                try:
                    async with sem:
                        await apply_locomo_judge_to_saved_result(
                            data,
                            qa,
                            conv_idx,
                            answerer,
                            judge_llm,
                            cutoffs,
                            evidence_lookup,
                            adaptive_context=args.adaptive_context,
                            labels=labels_to_judge,
                            result_path=path,
                            reuse_generated_answers=args.reuse_generated_answers,
                        )
                    return
                except Exception as exc:
                    if question_attempt >= args.question_max_attempts or not _retryable_question_failure(exc):
                        raise
                    delay = args.question_retry_delay * question_attempt
                    logger.warning(
                        "Evaluate-only question %s attempt %d/%d failed: %s; retrying in %.1fs",
                        qid,
                        question_attempt,
                        args.question_max_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        await asyncio.gather(*[judge_one(qid, conv_idx, qi, qa) for (qid, conv_idx, qi, qa) in expected_items])
        all_evaluations = [
            json.loads(Path(os.path.join(output_dir, f"{qid}.json")).read_text()) for (qid, _, _, _) in expected_items
        ]
        metrics = compute_locomo_metrics(all_evaluations, labels)
        display_results(metrics, labels)
        run_id_meta = args.run_id or run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_path = os.path.join(args.output_dir, f"locomo_results_{args.project_name}_{timestamp}.json")
        save_result_json(
            unified_path,
            {
                "metadata": {
                    "benchmark": "locomo",
                    "project_name": args.project_name,
                    "run_id": run_id_meta,
                    "timestamp": timestamp,
                    "answerer_model": args.answerer_model,
                    "judge_model": args.judge_model,
                    "provider": args.provider,
                    "answer_provider_routing": answer_extra_body.get("provider") if answer_extra_body else None,
                    "judge_provider_routing": judge_extra_body.get("provider") if judge_extra_body else None,
                    "top_k": args.top_k,
                    "top_k_cutoffs": labels,
                    "total_questions": len(all_evaluations),
                    "categories": categories,
                    "evaluate_only": True,
                    "reuse_generated_answers": args.reuse_generated_answers,
                    "judge_prompt_version": JUDGE_PROMPT_VERSION,
                },
                "metrics_by_cutoff": metrics,
                "evaluations": all_evaluations,
            },
        )
        print(f"\nResults saved to: {unified_path}")
        print(f"\nTotal questions evaluated: {len(all_evaluations)}")
        return
    if args.adaptive_context and (not args.rerank):
        raise RuntimeError("--adaptive-context requires --rerank with reranker.provider=sparse_omp")
    mem0 = Mem0Client(
        mode=backend, host=args.mem0_host, api_key=args.mem0_api_key, rpm=args.rpm, timeout=args.mem0_timeout
    )
    shutdown = GracefulShutdown()
    sparse_omp_rerank = False
    runtime_reranker_provider: str | None = None
    runtime_reranker_model: str | None = None
    all_evaluations: list[dict] = []
    if args.resume:
        incomplete_resume_ids: list[str] = []
        for p in sorted(Path(output_dir).glob("*.json")):
            if p.name.startswith("_"):
                continue
            try:
                data = json.loads(p.read_text())
                if data.get("category") in categories and _resume_result_complete(
                    data, labels, predict_only=args.predict_only
                ):
                    all_evaluations.append(data)
                elif data.get("category") in categories:
                    incomplete_resume_ids.append(data.get("question_id") or p.stem)
            except (json.JSONDecodeError, KeyError):
                incomplete_resume_ids.append(p.stem)
                continue
        print(f"  Loaded {len(all_evaluations)} existing results")
        if incomplete_resume_ids:
            print(
                f"  Reprocessing incomplete checkpoints: {len(incomplete_resume_ids)} (showing up to 25): {incomplete_resume_ids[:25]}"
            )
    existing_ids = {e["question_id"] for e in all_evaluations}
    results_lock = asyncio.Lock()
    conv_semaphore = asyncio.Semaphore(args.max_workers)

    async def process_conversation(conv_idx: int):
        """Process a single conversation: ingest → answer questions."""
        async with conv_semaphore:
            if shutdown.requested:
                return
            if conv_idx >= len(dataset):
                logger.warning("Conversation %d out of range (dataset has %d)", conv_idx, len(dataset))
                return
            entry = dataset[conv_idx]
            conversation = entry["conversation"]
            (success, user_id, chunks) = await ingest_conversation(
                conv_idx, entry, mem0, logger, run_id, args.project_name, output_dir, shutdown, debug=args.debug
            )
            if not success:
                logger.error("Ingestion failed for conversation %d", conv_idx)
            if shutdown.requested:
                return
            user_profile = None
            if args.user_profile and (not False):
                user_profile = await mem0.get_user_profile(user_id)
            sorted_sessions = get_sorted_sessions(conversation)
            ref_date_human = None
            if sorted_sessions:
                ref_date_human = sorted_sessions[-1][1]
            questions = entry.get("qa", entry.get("qa_pairs", []))
            conv_questions = [(qi, qa) for (qi, qa) in enumerate(questions) if qa.get("category") in categories]
            if args.max_questions is not None:
                conv_questions = conv_questions[: args.max_questions]
            search_pbar = tqdm(conv_questions, desc=f"Questions conv {conv_idx}", leave=True)
            for qi, qa in search_pbar:
                qid = f"conv{conv_idx}_q{qi}"
                if shutdown.requested:
                    break
                async with results_lock:
                    if qid in existing_ids:
                        continue
                result_path = os.path.join(output_dir, f"{qid}.json")
                for question_attempt in range(1, args.question_max_attempts + 1):
                    try:
                        result = await process_question(
                            qa=qa,
                            qa_idx=qi,
                            conv_idx=conv_idx,
                            user_id=user_id,
                            mem0=mem0,
                            answerer=answerer,
                            judge_llm=judge_llm,
                            cutoffs=cutoffs,
                            top_k=args.top_k,
                            threshold=args.threshold,
                            reference_date_human=ref_date_human,
                            user_profile=user_profile,
                            evidence_lookup=evidence_lookup,
                            predict_only=args.predict_only,
                            logger=logger,
                            score_debug=args.score_debug,
                            rerank=args.rerank,
                            sparse_omp_rerank=sparse_omp_rerank,
                            adaptive_context=args.adaptive_context,
                            result_path=result_path,
                        )
                        break
                    except RerankerSearchError as exc:
                        failure_performance = {
                            **exc.performance,
                            "provider": runtime_reranker_provider,
                            "model": runtime_reranker_model,
                        }
                        async with results_lock:
                            if exc.code == "sparse_omp_rerank_failed":
                                update_sparse_omp_performance_report(
                                    os.path.join(output_dir, SPARSE_OMP_PERFORMANCE_FILE), qid, failure_performance
                                )
                            else:
                                update_reranker_performance_report(
                                    os.path.join(output_dir, RERANKER_PERFORMANCE_FILE), qid, failure_performance
                                )
                        raise
                    except Exception as exc:
                        if question_attempt >= args.question_max_attempts or not _retryable_question_failure(exc):
                            raise
                        delay = args.question_retry_delay * question_attempt
                        logger.warning(
                            "Question %s whole-question attempt %d/%d failed: %s; retrying in %.1fs",
                            qid,
                            question_attempt,
                            args.question_max_attempts,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                sparse_performance = result.pop("_sparse_omp_performance", None)
                reranker_performance = result.pop("_reranker_performance", None)
                async with results_lock:
                    save_result_json(result_path, result)
                    if sparse_performance:
                        update_sparse_omp_performance_report(
                            os.path.join(output_dir, SPARSE_OMP_PERFORMANCE_FILE), qid, sparse_performance
                        )
                    if reranker_performance:
                        update_reranker_performance_report(
                            os.path.join(output_dir, RERANKER_PERFORMANCE_FILE), qid, reranker_performance
                        )
                    all_evaluations.append(result)
                    existing_ids.add(qid)

    async with mem0:
        if args.rerank and backend == "oss":
            health = await mem0.health()
            reranker_provider = health.get("reranker")
            if not isinstance(reranker_provider, str):
                raise RuntimeError("OSS /health did not report the configured reranker provider")
            sparse_omp_rerank = reranker_provider == "sparse_omp"
            runtime_reranker_provider = reranker_provider
            reranker_model = health.get("reranker_model")
            runtime_reranker_model = reranker_model if isinstance(reranker_model, str) else None
            model_suffix = f" ({runtime_reranker_model})" if runtime_reranker_model else ""
            print(f"  Configured reranker: {reranker_provider}{model_suffix}")
        if args.adaptive_context and (not sparse_omp_rerank):
            raise RuntimeError(
                "--adaptive-context requires the OSS server to be configured with reranker.provider=sparse_omp"
            )
        with shutdown:
            tasks = [process_conversation(idx) for idx in conv_indices]
            await asyncio.gather(*tasks)
    if not args.predict_only and all_evaluations:
        has_cutoffs = any(("cutoff_results" in e for e in all_evaluations))
        if has_cutoffs:
            metrics = compute_locomo_metrics(all_evaluations, labels)
            display_results(metrics, labels)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unified_path = os.path.join(args.output_dir, f"locomo_results_{args.project_name}_{timestamp}.json")
            save_result_json(
                unified_path,
                {
                    "metadata": {
                        "benchmark": "locomo",
                        "project_name": args.project_name,
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "answerer_model": args.answerer_model,
                        "judge_model": args.judge_model,
                        "provider": args.provider,
                        "top_k": args.top_k,
                        "top_k_cutoffs": labels,
                        "total_questions": len(all_evaluations),
                        "categories": categories,
                        "judge_prompt_version": JUDGE_PROMPT_VERSION,
                    },
                    "metrics_by_cutoff": metrics,
                    "evaluations": all_evaluations,
                },
            )
            print(f"\nResults saved to: {unified_path}")
    print(f"\nTotal questions processed: {len(all_evaluations)}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
