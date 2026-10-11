from __future__ import annotations

import json

import pytest

from benchmarks.pipeline.beam_official import (
    ANSWER_GENERATION_FOR_RAG,
    UNIFIED_LLM_JUDGE_BASE_PROMPT,
    answer_prompt,
    evaluate_answer,
)
from benchmarks.pipeline.models import BenchmarkQuestion, MemoryHit, RetrievalResult


class ScriptedGateway:
    def __init__(self, values: list[str]):
        self.values = iter(values)
        self.messages = []

    def text(self, messages, **kwargs):
        self.messages.append((messages, kwargs))
        value = next(self.values)
        return value, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def _question(category: str, rubric: tuple[str, ...]) -> BenchmarkQuestion:
    return BenchmarkQuestion("q", "s", "What happened?", "answer", category, rubric=rubric)


def test_official_answer_prompt_isolated_from_legacy_beam_prompt() -> None:
    result = RetrievalResult("s", "q", 5, main_hits=[MemoryHit("retrieved fact")])
    prompt, context = answer_prompt(_question("abstention", ("criterion",)), result)

    assert prompt == ANSWER_GENERATION_FOR_RAG.replace("<context>", "retrieved fact").replace(
        "<question>", "What happened?"
    )
    assert "9. Cite" not in prompt
    assert context["included_items"] == 1
    assert "RUBRIC CRITERION (what to check)" in UNIFIED_LLM_JUDGE_BASE_PROMPT
    assert "<question>" not in UNIFIED_LLM_JUDGE_BASE_PROMPT


def test_official_rubric_score_is_mean_of_all_criteria() -> None:
    gateway = ScriptedGateway(
        [
            json.dumps({"score": 1.0, "reason": "yes"}),
            json.dumps({"score": 0.5, "reason": "partial"}),
        ]
    )
    result = evaluate_answer(gateway, _question("summarization", ("one", "two")), "response")

    assert result["primary_score"] == pytest.approx(0.75)
    assert result["llm_judge_score"] == pytest.approx(0.75)
    assert len(result["rubric_judgments"]) == 2


def test_official_event_ordering_runs_alignment_then_rubric_and_reports_tau() -> None:
    # Two successful first-candidate alignments followed by two rubric judges.
    gateway = ScriptedGateway(
        [
            "YES",
            "YES",
            json.dumps({"score": 1.0, "reason": "present"}),
            json.dumps({"score": 1.0, "reason": "present"}),
        ]
    )
    result = evaluate_answer(
        gateway,
        _question("event_ordering", ("first event", "second event")),
        "first event\nsecond event",
    )

    assert result["event_ordering"]["tau_norm"] == pytest.approx(1.0)
    assert result["primary_score"] == pytest.approx(1.0)
    assert result["official_report_rule"] == "tau_norm"
    assert len(result["event_alignment_comparisons"]) == 2
