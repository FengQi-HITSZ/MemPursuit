from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.pipeline.datasets import (
    BEAM_1M_SMOKE_ROW_INDEX,
    BEAM_QUESTION_TYPES,
    BEAM_SMOKE_ROW_INDICES,
    QUESTION_TYPES,
    load_beam,
    load_locomo,
    load_longmemeval,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_locomo_preserves_names_time_order_and_image_caption(tmp_path: Path) -> None:
    qa = [
        {"question": f"question-{category}", "answer": f"answer-{category}", "category": category}
        for category in range(1, 5)
    ]
    payload = [
        {
            "sample_id": "conv/one",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_2": [{"speaker": "Bob", "text": "Later", "dia_id": "d3"}],
                "session_2_date_time": "10:30 AM on 02 May, 2024",
                "session_1": [
                    {"speaker": "Alice", "text": "Hello", "dia_id": "d1"},
                    {
                        "speaker": "Bob",
                        "text": "Here",
                        "blip_caption": "a blue bicycle",
                        "dia_id": "d2",
                    },
                ],
                "session_1_date_time": "09:15 PM on 01 May, 2024",
            },
            "qa": qa,
        }
    ]
    source = tmp_path / "locomo.json"
    _write_json(source, payload)

    scopes = load_locomo(source, "run-one", "smoke")

    assert len(scopes) == 1
    assert [event.session_id for session in scopes[0].sessions for event in session] == [
        "session_1",
        "session_1",
        "session_2",
    ]
    events = [event for session in scopes[0].sessions for event in session]
    assert [event.order for event in events] == [0, 1, 2]
    assert events[0].speaker == "Alice" and events[0].role == "user"
    assert events[1].speaker == "Bob" and events[1].role == "assistant"
    assert events[0].timestamp == "2024-05-01T21:15:00"
    assert events[1].timestamp_raw == "09:15 PM on 01 May, 2024"
    assert "a blue bicycle" in events[1].content
    assert events[1].provenance["blip_caption"] == "a blue bicycle"
    assert [question.category for question in scopes[0].questions] == ["1", "2", "3", "4"]


def test_longmemeval_fixed_smoke_selection_and_has_answer_removal(tmp_path: Path) -> None:
    payload = []
    for index in range(500):
        qtype = QUESTION_TYPES[index % len(QUESTION_TYPES)]
        payload.append(
            {
                "question_id": f"{qtype}-{index:04d}",
                "question_type": qtype,
                "question": "What happened?",
                "answer": "Something",
                "question_date": "2024/05/10",
                "haystack_session_ids": [f"session-{index}"],
                "haystack_dates": ["2024/05/01"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "U", "has_answer": True},
                        {"role": "assistant", "content": "A", "has_answer": False},
                        {"role": "tool", "content": "must be excluded", "has_answer": True},
                    ]
                ],
                "answer_session_ids": [f"session-{index}"],
            }
        )
    source = tmp_path / "long.json"
    payload[0]["haystack_session_ids"].append(payload[0]["haystack_session_ids"][0])
    payload[0]["haystack_dates"].append("2024/05/02")
    payload[0]["haystack_sessions"].append([{"role": "user", "content": "duplicate-id session", "has_answer": False}])
    _write_json(source, payload)

    first = load_longmemeval(source, "run-one", "smoke")
    second = load_longmemeval(source, "run-two", "smoke")

    assert len(first) == 6
    assert [scope.spec.scope_id for scope in first] == [scope.spec.scope_id for scope in second]
    assert [scope.questions[0].category for scope in first] == list(QUESTION_TYPES)
    assert [session[0].session_id for session in first[0].sessions] == [
        "session-0",
        "session-0__duplicate_1",
    ]
    assert first[0].sessions[1][0].provenance["original_session_id"] == "session-0"
    for scope in first:
        assert [event.role for event in scope.sessions[0]] == ["user", "assistant"]
        assert all("has_answer" not in event.to_dict() for event in scope.sessions[0])
        assert all("has_answer" not in event.provenance for event in scope.sessions[0])
        assert scope.sessions[0][0].timestamp == "2024/05/01"


def test_longmemeval_rejects_non_cleaned_cardinality(tmp_path: Path) -> None:
    source = tmp_path / "short.json"
    _write_json(source, [])
    with pytest.raises(ValueError, match="expected 500"):
        load_longmemeval(source, "run", "full")


def test_beam_1m_smoke_is_fixed_complete_conversation_without_question_leakage(tmp_path, monkeypatch) -> None:
    source = tmp_path / "beam-1M.arrow"
    monkeypatch.setattr("benchmarks.pipeline.datasets._load_beam_arrow", lambda path: _beam_rows())
    scopes = load_beam(source, "beam-test", "smoke")

    assert len(scopes) == 1
    scope = scopes[0]
    assert scope.spec.benchmark == "beam"
    assert scope.spec.metadata["conversation_row_index"] == BEAM_1M_SMOKE_ROW_INDEX
    assert len(scope.sessions) == 2
    assert sum(len(session) for session in scope.sessions) == 4
    assert [question.category for question in scope.questions] == [
        category for category in BEAM_QUESTION_TYPES for _ in range(2)
    ]
    assert all(question.rubric for question in scope.questions)
    question_texts = {question.question for question in scope.questions}
    assert all(event.content not in question_texts for session in scope.sessions for event in session)


def test_beam_500k_smoke_is_fixed_complete_conversation_without_question_leakage(tmp_path, monkeypatch) -> None:
    source = tmp_path / "beam-500K.arrow"
    monkeypatch.setattr("benchmarks.pipeline.datasets._load_beam_arrow", lambda path: _beam_rows())
    scopes = load_beam(source, "beam-500k-test", "smoke", chat_size="500K")

    assert len(scopes) == 1
    scope = scopes[0]
    assert scope.spec.scope_id == "beam-500k-conversation-26"
    assert scope.spec.metadata["conversation_row_index"] == BEAM_SMOKE_ROW_INDICES["500K"]
    assert scope.spec.metadata["chat_size"] == "500K"
    assert scope.spec.metadata["adaptation"] == "adapted-beam-500k"
    assert len(scope.sessions) == 2
    assert sum(len(session) for session in scope.sessions) == 4
    assert [question.category for question in scope.questions] == [
        category for category in BEAM_QUESTION_TYPES for _ in range(2)
    ]
    assert all(question.rubric for question in scope.questions)
    question_texts = {question.question for question in scope.questions}
    assert all(event.content not in question_texts for session in scope.sessions for event in session)


def test_beam_loader_rejects_a_split_label_file_mismatch() -> None:
    one_million = Path("beam-1M.arrow")
    with pytest.raises(ValueError, match="chat_size=500K requires beam-500K.arrow"):
        load_beam(one_million, "mislabeled", "smoke", chat_size="500K")


def _beam_rows():
    return [
        {
            "conversation_id": index + 1,
            "chat": [
                [
                    {"role": "user", "content": "history only", "time_anchor": "2024-05-01"},
                    {"role": "assistant", "content": "reply only"},
                ]
                for _ in range(2)
            ],
            "probing_questions": {
                category: [{"question": f"{category} query {j}", "rubric": ["gold criterion"]} for j in range(2)]
                for category in BEAM_QUESTION_TYPES
            },
        }
        for index in range(35)
    ]
