from __future__ import annotations

import importlib.util

import json

import re

import shutil

from pathlib import Path

from typing import Any

from .io import sha256_file

from .models import BenchmarkQuestion, MemoryHit, RetrievalResult

EXPECTED_PROMPT_HASHES = {
    "locomo": "c091566e935bdf11f5a3075f88b5b4a04fd4d5101cba42f48106504d875d283b",
    "longmemeval": "ba8cf60d26f1390ecbef0f07b3e950556fe3bc5a37ba4b5343f28217f18c144f",
    "beam": "67f5197b37123a8497ae4d808163aea4b9ea76532b3ab637a18a9c44e7f40d6a",
}


def _memory_dict(hit: MemoryHit) -> dict[str, Any]:
    return {"memory": hit.text, "score": hit.score or 0.0, "created_at": hit.timestamp or ""}


def _profile(result: RetrievalResult) -> dict[str, str] | None:
    values = [hit.text for hit in result.resident_context if hit.kind == "profile"]
    return {"memory_system_profile": "\n".join(values)} if values else None


class PromptSnapshot:
    """Freeze prompt source per run and load resumed runs from the frozen copy."""

    def __init__(self, repository_root: str | Path, run_dir: str | Path, benchmark: str):
        self.repository_root = Path(repository_root)
        self.run_dir = Path(run_dir)
        self.benchmark = benchmark
        source = (
            self.repository_root / "benchmarks" / "pipeline" / "beam_official.py"
            if benchmark == "beam"
            else self.repository_root / "benchmarks" / benchmark / "prompts.py"
        )
        snapshot_dir = self.run_dir / "prompt_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.path = snapshot_dir / f"{benchmark}_prompts.py"
        if not self.path.exists():
            actual_hash = sha256_file(source)
            expected_hash = EXPECTED_PROMPT_HASHES[benchmark]
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{benchmark} prompt drift: expected {expected_hash}, found {actual_hash}; "
                    "refuse to begin a run with an unreviewed prompt"
                )
            shutil.copy2(source, self.path)
        self.sha256 = sha256_file(self.path)
        if benchmark == "beam":
            if sha256_file(source) != self.sha256:
                raise RuntimeError("BEAM prompt implementation changed since this run was created")
            # BEAM's official module is imported normally by the runner.  The
            # frozen copy exists solely as a byte-identical audit artifact;
            # loading it outside its package would break its relative imports.
            self.module = None
            return
        module_spec = importlib.util.spec_from_file_location(
            f"mempursuit_frozen_{benchmark}_{self.sha256[:12]}", self.path
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"cannot load frozen prompt module {self.path}")
        self.module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.module)

    def answer_prompt(self, question: BenchmarkQuestion, result: RetrievalResult) -> str:
        memories = [_memory_dict(hit) for hit in result.all_context() if hit.kind != "profile"]
        if self.benchmark == "locomo":
            return self.module.get_answer_generation_prompt(
                question.question,
                memories,
                reference_date=question.question_date,
                user_profile=_profile(result),
            )
        return self.module.get_answer_generation_prompt(
            question.question,
            memories,
            question.question_date,
            user_profile=_profile(result),
        )

    def judge_prompt(self, question: BenchmarkQuestion, answer: str) -> tuple[str, str]:
        if self.benchmark == "locomo":
            prompt = self.module.get_judge_prompt(int(question.category), question.question, question.answer, answer)
            return str(getattr(self.module, "JUDGE_SYSTEM_PROMPT", "")), prompt
        prompt = self.module.get_judge_prompt(
            question.category,
            question.question_id,
            question.question,
            question.answer,
            answer,
            question.question_date,
        )
        return "", prompt


def extract_answer(raw: str, benchmark: str) -> str:
    value = raw.strip()
    value = re.sub(r"<mem_thinking>.*?</mem_thinking>", "", value, flags=re.I | re.S).strip()
    if benchmark == "locomo" and re.search(r"\bANSWER\s*:", value, flags=re.I):
        value = re.split(r"\bANSWER\s*:", value, flags=re.I)[-1].strip()
    return value


def parse_judgment(raw: str, benchmark: str, track: str) -> bool:
    value = raw.strip()
    if benchmark == "locomo":
        try:
            parsed = json.loads(value)
            label = str(parsed.get("label", ""))
        except json.JSONDecodeError:
            label = value
        labels = re.findall(r"\b(CORRECT|WRONG)\b", label.upper())
        if not labels:
            raise ValueError(f"cannot parse LoCoMo judgment: {raw!r}")
        return labels[-1] == "CORRECT"
    cleaned = re.sub(r"<judge_thinking>.*?</judge_thinking>", "", value, flags=re.I | re.S)
    labels = re.findall(r"\b(yes|no)\b", cleaned.lower())
    if labels:
        return labels[-1] == "yes"
    # Some OpenAI-compatible gateways place the explicit verdict inside the
    # reasoning tag instead of after it. Accept only the unambiguous final
    # verdict marker in that case; arbitrary yes/no words in the reasoning are
    # deliberately ignored.
    verdicts = re.findall(r"\bfinal\s+verdict\s*:\s*(yes|no)\b", value, flags=re.I)
    if verdicts:
        return verdicts[-1].lower() == "yes"
    # A few OpenAI-compatible providers occasionally keep the requested
    # newline verdict immediately inside the closing reasoning tag.  Accept an
    # exact standalone yes/no line there, while still rejecting incidental or
    # conflicting yes/no words in the reasoning prose.
    reasoning_blocks = re.findall(r"<judge_thinking>(.*?)</judge_thinking>", value, flags=re.I | re.S)
    for block in reversed(reasoning_blocks):
        line_verdicts = re.findall(r"(?im)^\s*(yes|no)\s*$", block)
        if line_verdicts:
            return line_verdicts[-1].lower() == "yes"
    raise ValueError(f"cannot parse LongMemEval judgment: {raw!r}")
