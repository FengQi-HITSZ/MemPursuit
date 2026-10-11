from __future__ import annotations
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Any
from .models import CanonicalEvent, ScopeSpec, MemoryHit, RetrievalResult
from .security import safe_component


def _token_count(texts: list[str]) -> int:
    combined = "\n".join(texts)
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(combined))
    except Exception:
        import re

        return len(re.findall(r"\w+|[^\w\s]", combined, flags=re.UNICODE))


def _retrieval_profile(spec):
    return "native"


class MemPursuitAdapter:
    """MemPursuit memory ingestion and retrieval adapter."""

    _search_limit = 200
    _rerank_options = {"early_stopping": False, "min_iterations": 200, "short_pool_policy": "use_available"}

    def __init__(self, endpoint="http://localhost:8889", timeout=900):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _rerank_enabled():
        return True

    @staticmethod
    def _user_id(spec: ScopeSpec) -> str:
        return f"longmemeval-omp::{safe_component(spec.run_id)}::{safe_component(spec.scope_id)}"

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=body,
            method=method,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "mempursuit/1",
            },
        )
        attempts = 5
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                    raise RuntimeError(f"Mem0 OMP {method} {path} failed ({exc.code}): {error_body}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == attempts - 1:
                    raise RuntimeError(f"Mem0 OMP {method} {path} failed: {exc}") from exc
            time.sleep(min(2**attempt, 15))
        raise AssertionError("unreachable")

    @staticmethod
    def _ordered_sessions(events: list[CanonicalEvent]) -> list[list[CanonicalEvent]]:
        sessions: dict[str, list[CanonicalEvent]] = {}
        for event in events:
            sessions.setdefault(event.session_id, []).append(event)
        ordered = list(sessions.values())
        for session in ordered:
            session.sort(key=lambda event: event.session_order)
        ordered.sort(
            key=lambda session: (
                session[0].timestamp if session else "",
                session[0].order if session else 0,
            )
        )
        return ordered

    def _build(self, spec: ScopeSpec, events: list[CanonicalEvent]) -> dict[str, Any]:
        user_id = self._user_id(spec)
        add_calls = 0
        extracted_events = 0
        for session in self._ordered_sessions(events):
            for offset in range(0, len(session), 2):
                pair = session[offset : offset + 2]
                messages = [{"role": event.role, "content": event.content} for event in pair if event.content.strip()]
                if not messages:
                    continue
                first = pair[0]
                response = self._request(
                    "POST",
                    "/memories",
                    {
                        "messages": messages,
                        "user_id": user_id,
                        "timestamp": first.timestamp,
                        "metadata": {
                            "benchmark": spec.benchmark,
                            "scope_id": spec.scope_id,
                            "session_id": first.session_id,
                            "event_timestamp": first.timestamp,
                            "timestamp_raw": first.timestamp_raw,
                            "pair_offset": offset,
                            "first_event_order": first.order,
                        },
                    },
                )
                if not isinstance(response, dict) or not isinstance(response.get("results"), list):
                    raise RuntimeError("Mem0 OMP add returned an invalid result shape")
                add_calls += 1
                extracted_events += len(response["results"])
        rows = self._request(
            "GET",
            f"/memories?{urllib.parse.urlencode({'user_id': user_id})}",
        )
        memories = rows.get("results", rows) if isinstance(rows, dict) else rows
        if not isinstance(memories, list):
            raise RuntimeError("Mem0 OMP get_all returned an invalid result shape")
        health = self._validated_health()
        return {
            "memory_count": len(memories),
            "native_add_calls": add_calls,
            "extraction_events": extracted_events,
            "batch_size": 2,
            "llm_model": health.get("llm_model"),
            "embedding_model": health.get("embedder_model"),
            "retrieval": self._algorithm_manifest(health),
        }

    @classmethod
    def _algorithm_manifest(cls, health: dict[str, Any]) -> dict[str, Any]:
        rerank_enabled = cls._rerank_enabled()
        return {
            "semantic_score_mode": health.get("semantic_score_mode"),
            "enable_bm25": health.get("enable_bm25"),
            "enable_rrf": health.get("enable_rrf"),
            "enable_entity_boost": health.get("enable_entity_boost"),
            "threshold": None,
            "rerank_enabled": rerank_enabled,
            "reranker": health.get("reranker") if rerank_enabled else "none",
            "configured_reranker": health.get("reranker"),
            "candidate_pool_size": (health.get("reranker_candidate_pool_size") if rerank_enabled else None),
            "max_support": (health.get("sparse_omp_max_support") if rerank_enabled else None),
            "early_stopping": (health.get("sparse_omp_early_stopping") if rerank_enabled else None),
            "search_limit": cls._search_limit,
            "cutoffs": [10, 20, 50, 200],
            "rerank_options": dict(cls._rerank_options) if rerank_enabled else None,
        }

    def _retrieve(self, spec: ScopeSpec, query: str, k: int) -> RetrievalResult:
        if k != self._search_limit:
            raise ValueError(f"Mem0 OMP native retrieval must be called once at k={self._search_limit}; got {k}")
        started = time.monotonic()
        rerank_enabled = self._rerank_enabled()
        payload: dict[str, Any] = {
            "query": query,
            "user_id": self._user_id(spec),
            "limit": self._search_limit,
            "threshold": None,
            "rerank": rerank_enabled,
            "score_debug": True,
        }
        if rerank_enabled:
            payload["rerank_options"] = dict(self._rerank_options)
        response = self._request(
            "POST",
            "/search",
            payload,
        )
        if not isinstance(response, dict) or not isinstance(response.get("results"), list):
            raise RuntimeError("Mem0 OMP search returned an invalid result shape")
        hits: list[MemoryHit] = []
        for rank, row in enumerate(response["results"], 1):
            metadata = dict(row.get("metadata") or {})
            score = row.get("rerank_score", row.get("score")) if rerank_enabled else row.get("score")
            hits.append(
                MemoryHit(
                    text=str(row.get("memory", row.get("data", ""))),
                    score=float(score) if score is not None else None,
                    timestamp=(metadata.get("event_timestamp") or metadata.get("created_at") or row.get("created_at")),
                    rank=rank,
                    provenance={
                        "memory_id": row.get("id"),
                        "retrieval_score": row.get("score"),
                        "rerank_score": row.get("rerank_score"),
                        "session_id": metadata.get("session_id"),
                        "first_event_order": metadata.get("first_event_order"),
                        "score_breakdown": row.get("score_breakdown") or row.get("score_debug"),
                    },
                )
            )
        health = self._validated_health()
        result = RetrievalResult(
            scope_id=spec.scope_id,
            query=query,
            requested_k=k,
            main_hits=hits,
            backend_metadata={
                "cutoff_semantics": "ranked-prefixes-from-single-top-200-search",
                "retrieval_profile": _retrieval_profile(spec),
                "algorithm": self._algorithm_manifest(health),
                "rerank_metadata": (
                    response.get("rerank_metadata", {}) if rerank_enabled else {"provider": "none", "applied": False}
                ),
                "query_debug": response.get("query_debug", {}),
            },
            elapsed_seconds=time.monotonic() - started,
        )
        result.context_tokens = _token_count([hit.text for hit in hits])
        return result

    def _validated_health(self) -> dict[str, Any]:
        value = self._request("GET", "/health", timeout=30)
        if not isinstance(value, dict):
            raise RuntimeError("Mem0 OMP health returned a non-object")
        expected = {
            "semantic_score_mode": "embedding",
            "enable_bm25": True,
            "enable_rrf": False,
            "enable_entity_boost": True,
            "reranker": "sparse_omp",
            "reranker_candidate_pool_size": 200,
            "sparse_omp_max_support": 200,
            "sparse_omp_early_stopping": False,
        }
        mismatches = {
            key: {"expected": expected_value, "actual": value.get(key)}
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        }
        if mismatches:
            raise RuntimeError(f"Mem0 OMP runtime configuration mismatch: {mismatches}")
        return value
