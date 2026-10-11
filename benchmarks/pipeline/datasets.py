from __future__ import annotations

import ast
import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .models import BenchmarkQuestion, BenchmarkScope, CanonicalEvent, ScopeSpec


LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
)
QUESTION_TYPES = (
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
)
BEAM_QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
BEAM_CHAT_SIZES = ("500K", "1M")
# Fixed median conversations by source-character count. Keeping one complete
# conversation makes smoke costs comparable across backends and includes two
# questions from every one of BEAM's ten ability categories (20 total).
BEAM_SMOKE_ROW_INDICES = {"500K": 25, "1M": 26}
# Backward-compatible public name used by the existing BEAM 1M tests/tools.
BEAM_1M_SMOKE_ROW_INDEX = BEAM_SMOKE_ROW_INDICES["1M"]


def _locomo_timestamp(raw: str) -> str:
    raw = raw.strip().strip("()")
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(raw, fmt).isoformat(timespec="seconds")
        except ValueError:
            pass
    return raw


def _ordered_session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers = []
    for key in conversation:
        match = re.fullmatch(r"session_(\d+)", key)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def load_locomo(path: str | Path, run_id: str, mode: str = "smoke") -> list[BenchmarkScope]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    selected = data[:1] if mode == "smoke" else data
    scopes: list[BenchmarkScope] = []
    for conv_index, item in enumerate(selected):
        conversation = item["conversation"]
        speaker_a = str(conversation["speaker_a"])
        speaker_b = str(conversation["speaker_b"])
        scope_id = str(item.get("sample_id") or f"conversation-{conv_index}")
        sessions: list[list[CanonicalEvent]] = []
        global_order = 0
        for session_num in _ordered_session_numbers(conversation):
            session_id = f"session_{session_num}"
            raw_time = str(conversation.get(f"{session_id}_date_time", ""))
            timestamp = _locomo_timestamp(raw_time)
            events: list[CanonicalEvent] = []
            for session_order, turn in enumerate(conversation[session_id]):
                content = str(turn.get("text", "")).strip()
                caption = str(turn.get("blip_caption") or "").strip()
                if caption:
                    content = f"{content} [Image description: {caption}]"
                speaker = str(turn.get("speaker", "Unknown"))
                events.append(
                    CanonicalEvent(
                        benchmark="locomo",
                        scope_id=scope_id,
                        session_id=session_id,
                        timestamp=timestamp,
                        timestamp_raw=raw_time,
                        speaker=speaker,
                        role="user" if speaker == speaker_a else "assistant",
                        content=content,
                        order=global_order,
                        session_order=session_order,
                        provenance={
                            "conversation_index": conv_index,
                            "dia_id": turn.get("dia_id"),
                            "speaker_a": speaker_a,
                            "speaker_b": speaker_b,
                            "blip_caption": caption or None,
                        },
                    )
                )
                global_order += 1
            sessions.append(events)

        questions: list[BenchmarkQuestion] = []
        for q_index, qa in enumerate(item.get("qa", [])):
            category = int(qa.get("category", 0))
            if category not in {1, 2, 3, 4}:
                continue
            answer = qa.get("answer")
            if answer in (None, ""):
                answer = qa.get("adversarial_answer", "")
            questions.append(
                BenchmarkQuestion(
                    question_id=f"{scope_id}:q{q_index:04d}",
                    scope_id=scope_id,
                    question=str(qa["question"]),
                    answer=str(answer),
                    category=str(category),
                    question_date=sessions[-1][0].timestamp if sessions and sessions[-1] else "",
                    evidence=tuple(str(x) for x in qa.get("evidence", [])),
                    metadata={"conversation_index": conv_index, "question_index": q_index},
                )
            )
        if mode == "smoke":
            fixed: list[BenchmarkQuestion] = []
            for category in ("1", "2", "3", "4"):
                match = next((q for q in questions if q.category == category), None)
                if match is None:
                    raise ValueError(f"LoCoMo smoke conversation lacks category {category}")
                fixed.append(match)
            questions = fixed
        scopes.append(
            BenchmarkScope(
                spec=ScopeSpec(
                    benchmark="locomo",
                    scope_id=scope_id,
                    run_id=run_id,
                    metadata={
                        "adaptation": "native-locomo",
                        "conversation_index": conv_index,
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                    },
                ),
                sessions=sessions,
                questions=questions,
            )
        )
    return scopes


def download_longmemeval(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(LONGMEMEVAL_URL, headers={"User-Agent": "mempursuit/1"})
    with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    with open(target, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list) or len(value) != 500:
        target.unlink(missing_ok=True)
        raise ValueError(f"expected cleaned LongMemEval-S with 500 questions, got {len(value)}")
    return target


def _clean_longmem_message(message: dict[str, Any]) -> dict[str, Any]:
    # has_answer is intentionally and explicitly excluded from canonical events.
    return {key: value for key, value in message.items() if key != "has_answer"}


def load_longmemeval(
    path: str | Path,
    run_id: str,
    mode: str = "smoke",
    *,
    allow_download: bool = False,
) -> list[BenchmarkScope]:
    source = Path(path)
    if not source.exists():
        if not allow_download:
            raise FileNotFoundError(f"{source} is missing; pass --download-dataset to fetch it")
        download_longmemeval(source)
    with open(source, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if len(data) != 500:
        raise ValueError(f"expected 500 cleaned LongMemEval-S questions, got {len(data)}")
    if mode == "smoke":
        selected = []
        for qtype in QUESTION_TYPES:
            candidates = sorted(
                (x for x in data if x.get("question_type") == qtype),
                key=lambda x: str(x.get("question_id", "")),
            )
            if not candidates:
                raise ValueError(f"LongMemEval dataset lacks question type {qtype}")
            selected.append(candidates[0])
    elif mode == "full":
        selected = data
    else:
        raise ValueError("mode must be smoke or full")

    scopes: list[BenchmarkScope] = []
    for item in selected:
        scope_id = str(item["question_id"])
        sessions: list[list[CanonicalEvent]] = []
        global_order = 0
        dates = item.get("haystack_dates", [])
        session_ids = item.get("haystack_session_ids", [])
        seen_session_ids: dict[str, int] = {}
        for session_index, raw_session in enumerate(item.get("haystack_sessions", [])):
            original_session_id = (
                str(session_ids[session_index]) if session_index < len(session_ids) else f"session_{session_index}"
            )
            occurrence = seen_session_ids.get(original_session_id, 0)
            seen_session_ids[original_session_id] = occurrence + 1
            session_id = original_session_id if occurrence == 0 else f"{original_session_id}__duplicate_{occurrence}"
            raw_time = str(dates[session_index]) if session_index < len(dates) else ""
            events: list[CanonicalEvent] = []
            for session_order, raw_message in enumerate(raw_session):
                message = _clean_longmem_message(dict(raw_message))
                role = str(message.get("role", "user"))
                if role not in {"user", "assistant"}:
                    continue
                events.append(
                    CanonicalEvent(
                        benchmark="longmemeval",
                        scope_id=scope_id,
                        session_id=session_id,
                        timestamp=raw_time,
                        timestamp_raw=raw_time,
                        speaker=role,
                        role=role,  # type: ignore[arg-type]
                        content=str(message.get("content", "")),
                        order=global_order,
                        session_order=session_order,
                        provenance={
                            "session_index": session_index,
                            "original_session_id": original_session_id,
                            "session_id_occurrence": occurrence,
                        },
                    )
                )
                global_order += 1
            sessions.append(events)
        question = BenchmarkQuestion(
            question_id=scope_id,
            scope_id=scope_id,
            question=str(item["question"]),
            answer=str(item.get("answer", "")),
            category=str(item["question_type"]),
            question_date=str(item.get("question_date", "")),
            evidence=tuple(str(x) for x in item.get("answer_session_ids", [])),
            metadata={"is_abstention": "_abs" in scope_id},
        )
        scopes.append(
            BenchmarkScope(
                spec=ScopeSpec(
                    benchmark="longmemeval",
                    scope_id=scope_id,
                    run_id=run_id,
                    metadata={
                        "adaptation": "adapted",
                        "question_type": item["question_type"],
                        "is_abstention": "_abs" in scope_id,
                    },
                ),
                sessions=sessions,
                questions=[question],
            )
        )
    return scopes


def _load_beam_arrow(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("loading the BEAM Arrow cache requires pyarrow") from exc
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_stream(source).read_all().to_pylist()


def _beam_probing_questions(value: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("BEAM probing_questions is not a valid Python literal") from exc
    if not isinstance(value, dict):
        raise ValueError("BEAM probing_questions must be a mapping")
    missing = [name for name in BEAM_QUESTION_TYPES if name not in value]
    if missing:
        raise ValueError(f"BEAM conversation lacks question types: {missing}")
    return {str(key): list(items) for key, items in value.items()}


def load_beam(
    path: str | Path,
    run_id: str,
    mode: str = "smoke",
    *,
    chat_size: str = "1M",
) -> list[BenchmarkScope]:
    """Load an official HuggingFace BEAM Arrow split.

    Every conversation is one isolated memory scope and each source batch is a
    chronological session.  Probing questions and rubrics are evaluation-only:
    neither is inserted into canonical history events.
    """
    source = Path(path)
    if chat_size not in BEAM_CHAT_SIZES:
        raise ValueError(f"chat_size must be one of {BEAM_CHAT_SIZES}, got {chat_size!r}")
    if source.suffix != ".arrow":
        raise ValueError("the official BEAM track requires a cached beam-500K.arrow or beam-1M.arrow split")
    expected_filename = f"beam-{chat_size}.arrow"
    if source.name != expected_filename:
        raise ValueError(f"BEAM chat_size={chat_size} requires {expected_filename}, got {source.name}")
    rows = _load_beam_arrow(source)
    if len(rows) != 35:
        raise ValueError(f"expected the BEAM {chat_size} split with 35 conversations, got {len(rows)}")
    if mode == "smoke":
        smoke_row_index = BEAM_SMOKE_ROW_INDICES[chat_size]
        selected = [(smoke_row_index, rows[smoke_row_index])]
    elif mode == "full":
        selected = list(enumerate(rows))
    else:
        raise ValueError("mode must be smoke or full")

    scopes: list[BenchmarkScope] = []
    size_slug = chat_size.lower()
    for row_index, item in selected:
        conversation_id = str(item["conversation_id"])
        scope_id = f"beam-{size_slug}-conversation-{conversation_id}"
        sessions: list[list[CanonicalEvent]] = []
        global_order = 0
        for batch_index, raw_batch in enumerate(item.get("chat") or []):
            session_id = f"batch_{batch_index + 1:03d}"
            raw_time = next(
                (str(message.get("time_anchor")) for message in raw_batch if message.get("time_anchor")),
                "",
            )
            events: list[CanonicalEvent] = []
            for session_order, message in enumerate(raw_batch):
                role = str(message.get("role", ""))
                if role not in {"user", "assistant"}:
                    continue
                events.append(
                    CanonicalEvent(
                        benchmark="beam",
                        scope_id=scope_id,
                        session_id=session_id,
                        timestamp=raw_time,
                        timestamp_raw=raw_time,
                        speaker=role,
                        role=role,  # type: ignore[arg-type]
                        content=str(message.get("content", "")),
                        order=global_order,
                        session_order=session_order,
                        provenance={
                            "conversation_id": conversation_id,
                            "conversation_row_index": row_index,
                            "batch_index": batch_index,
                            "message_id": message.get("id"),
                            "source_index": message.get("index"),
                            "source_question_type": message.get("question_type"),
                        },
                    )
                )
                global_order += 1
            sessions.append(events)

        questions: list[BenchmarkQuestion] = []
        probing = _beam_probing_questions(item.get("probing_questions"))
        for category in BEAM_QUESTION_TYPES:
            category_items = probing[category]
            if len(category_items) != 2:
                raise ValueError(
                    f"BEAM {chat_size} conversation {conversation_id} has {len(category_items)} "
                    f"questions for {category}, expected 2"
                )
            for question_index, raw_question in enumerate(category_items):
                ideal = next(
                    (
                        raw_question[key]
                        for key in ("ideal_response", "ideal_answer", "answer")
                        if raw_question.get(key) not in (None, "")
                    ),
                    "",
                )
                rubric = tuple(str(value) for value in raw_question.get("rubric", []))
                if not rubric:
                    raise ValueError(f"BEAM question {conversation_id}/{category}/{question_index} has no rubric")
                questions.append(
                    BenchmarkQuestion(
                        question_id=(f"{scope_id}:{category}:q{question_index + 1}"),
                        scope_id=scope_id,
                        question=str(raw_question["question"]),
                        answer=str(ideal),
                        category=category,
                        question_date=sessions[-1][0].timestamp if sessions and sessions[-1] else "",
                        rubric=rubric,
                        metadata={
                            "conversation_id": conversation_id,
                            "conversation_row_index": row_index,
                            "question_index_within_category": question_index,
                            "difficulty": raw_question.get("difficulty"),
                        },
                    )
                )
        seed = item.get("conversation_seed") or {}
        scopes.append(
            BenchmarkScope(
                spec=ScopeSpec(
                    benchmark="beam",
                    scope_id=scope_id,
                    run_id=run_id,
                    metadata={
                        "adaptation": f"adapted-beam-{size_slug}",
                        "conversation_id": conversation_id,
                        "conversation_row_index": row_index,
                        "title": seed.get("title"),
                        "chat_size": chat_size,
                        "smoke_selection": ("fixed-median-by-source-character-count" if mode == "smoke" else None),
                    },
                ),
                sessions=sessions,
                questions=questions,
            )
        )
    return scopes


def iter_questions(scopes: Iterable[BenchmarkScope]) -> Iterable[BenchmarkQuestion]:
    for scope in scopes:
        yield from scope.questions
