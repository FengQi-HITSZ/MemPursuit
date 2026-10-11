import json

import pytest

from benchmarks.locomo.prompts import JUDGE_SYSTEM_PROMPT, get_judge_prompt, get_judge_prompt_with_evidence
from benchmarks.locomo.run import apply_locomo_judge_to_saved_result


def test_mem0_accuracy_prompt_matches_reference_protocol():
    prompt = get_judge_prompt(
        1,
        "Which activities did Sam try?",
        "Kayaking and pottery",
        "Kayaking",
    )

    assert "as long as it touches on the same topic as the gold answer" in prompt
    assert "as long as it refers to the same date or time period" in prompt
    assert 'json format with the key as "label"' in prompt
    assert "Question: Which activities did Sam try?" in prompt
    assert "Gold answer: Kayaking and pottery" in prompt
    assert "Generated answer: Kayaking" in prompt
    assert JUDGE_SYSTEM_PROMPT == ""
    assert "AT LEAST ONE correct item" not in prompt
    assert "Dates within 14 days" not in prompt


def test_mem0_accuracy_prompt_can_include_evidence_without_changing_core_prompt():
    prompt = get_judge_prompt_with_evidence(
        2,
        "When did Sam visit?",
        "7 May 2023",
        "8 May 2023",
        "Sam visited on 7 May 2023.",
    )

    assert "Evidence (actual conversation messages containing the answer)" in prompt
    assert "Sam visited on 7 May 2023." in prompt
    assert "as long as it touches on the same topic as the gold answer" in prompt


@pytest.mark.asyncio
async def test_rejudge_can_reuse_saved_generated_answer_without_calling_answerer(tmp_path):
    class ExplodingAnswerer:
        async def generate(self, **kwargs):
            raise AssertionError("answerer must not be called when reusing saved answers")

    class RecordingJudge:
        def __init__(self):
            self.users = []

        async def generate_structured(self, **kwargs):
            self.users.append(kwargs["user"])
            return {"label": "WRONG", "reasoning": "The response omits pottery."}

    result_path = tmp_path / "conv0_q0.json"
    result = {
        "question": "Which activities did Sam try?",
        "reference_date": None,
        "retrieval": {
            "search_results": [
                {"id": "m1", "memory": "Sam tried kayaking."},
                {"id": "m2", "memory": "Sam tried pottery."},
            ]
        },
        "cutoff_results": {
            "top_1": {
                "judgment": "CORRECT",
                "score": 1.0,
                "generated_answer": "Kayaking",
                "memories_evaluated": 1,
            },
            "top_2": {
                "judgment": "CORRECT",
                "score": 1.0,
                "generated_answer": "Kayaking",
                "memories_evaluated": 2,
            }
        },
    }
    judge = RecordingJudge()

    await apply_locomo_judge_to_saved_result(
        result,
        {"category": 1, "answer": "Kayaking and pottery", "evidence": []},
        0,
        ExplodingAnswerer(),
        judge,
        [1, 2],
        None,
        labels=["top_1", "top_2"],
        result_path=str(result_path),
        reuse_generated_answers=True,
    )

    saved = json.loads(result_path.read_text())
    assert saved["cutoff_results"]["top_2"]["generated_answer"] == "Kayaking"
    assert saved["cutoff_results"]["top_2"]["judgment"] == "WRONG"
    assert saved["cutoff_results"]["top_2"]["score"] == 0.0
    assert saved["cutoff_results"]["top_2"]["judge_prompt_version"] == "locomo_mem0_accuracy_v1"
    assert saved["cutoff_results"]["top_1"]["judge_response_reused"] is False
    assert saved["cutoff_results"]["top_2"]["judge_response_reused"] is True
    assert len(judge.users) == 1
    assert "Kayaking" in judge.users[0]
