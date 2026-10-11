from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class CanonicalEvent:
    benchmark: Literal["locomo", "longmemeval", "beam"]
    scope_id: str
    session_id: str
    timestamp: str
    timestamp_raw: str
    speaker: str
    role: Literal["user", "assistant"]
    content: str
    order: int
    session_order: int
    provenance: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: JsonDict) -> "CanonicalEvent":
        return cls(**value)


@dataclass(frozen=True)
class ScopeSpec:
    benchmark: Literal["locomo", "longmemeval", "beam"]
    scope_id: str
    run_id: str
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class MemoryHit:
    text: str
    score: float | None = None
    timestamp: str | None = None
    provenance: JsonDict = field(default_factory=dict)
    kind: Literal["main", "neighbor", "profile", "knowledge", "recent"] = "main"
    rank: int | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: JsonDict) -> "MemoryHit":
        return cls(**value)


@dataclass
class RetrievalResult:
    scope_id: str
    query: str
    requested_k: int
    main_hits: list[MemoryHit] = field(default_factory=list)
    expanded_hits: list[MemoryHit] = field(default_factory=list)
    resident_context: list[MemoryHit] = field(default_factory=list)
    backend_metadata: JsonDict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    context_tokens: int = 0

    @property
    def actual_count(self) -> int:
        return len(self.main_hits) + len(self.expanded_hits) + len(self.resident_context)

    def all_context(self) -> list[MemoryHit]:
        return [*self.main_hits, *self.expanded_hits, *self.resident_context]

    def to_dict(self) -> JsonDict:
        value = asdict(self)
        value["actual_count"] = self.actual_count
        return value

    @classmethod
    def from_dict(cls, value: JsonDict) -> "RetrievalResult":
        clean = dict(value)
        clean.pop("actual_count", None)
        clean["main_hits"] = [MemoryHit.from_dict(x) for x in clean.get("main_hits", [])]
        clean["expanded_hits"] = [MemoryHit.from_dict(x) for x in clean.get("expanded_hits", [])]
        clean["resident_context"] = [MemoryHit.from_dict(x) for x in clean.get("resident_context", [])]
        return cls(**clean)


@dataclass(frozen=True)
class BenchmarkQuestion:
    question_id: str
    scope_id: str
    question: str
    answer: str
    category: str
    question_date: str = ""
    evidence: tuple[str, ...] = ()
    rubric: tuple[str, ...] = ()
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        value = asdict(self)
        value["evidence"] = list(self.evidence)
        value["rubric"] = list(self.rubric)
        return value


@dataclass
class BenchmarkScope:
    spec: ScopeSpec
    sessions: list[list[CanonicalEvent]]
    questions: list[BenchmarkQuestion]

    def to_dict(self) -> JsonDict:
        return {
            "spec": self.spec.to_dict(),
            "sessions": [[e.to_dict() for e in session] for session in self.sessions],
            "questions": [q.to_dict() for q in self.questions],
        }
