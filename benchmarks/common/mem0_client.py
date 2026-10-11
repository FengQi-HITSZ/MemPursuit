"""Async HTTP adapter for the MemPursuit memory service."""

from __future__ import annotations
import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

try:
    import aiohttp
except ImportError:
    aiohttp = None
try:
    from aiolimiter import AsyncLimiter
except ImportError:
    from benchmarks.common.llm_client import AsyncLimiter
logger = logging.getLogger(__name__)
_THRESHOLD_UNSET = object()
_OSS_SCORE_DEBUG_EXTRA_KEYS = (
    "semantic_score_source",
    "pre_sparse_omp_score",
    "pre_sparse_omp_rank",
    "sparse_omp_score_source",
    "sparse_omp_score_mode",
    "sparse_omp_selection_solver",
    "sparse_omp_constraint",
    "sparse_omp_step",
    "sparse_omp_step_score",
    "sparse_omp_coefficient",
    "sparse_omp_scan_count",
    "sparse_omp_candidate_count",
    "sparse_omp_selected_count",
    "sparse_omp_residual_norm",
    "sparse_omp_marginal_gain",
    "sparse_omp_skipped_missing_data",
    "sparse_omp_skipped_missing_id",
    "sparse_omp_skipped_missing_vector",
    "sparse_omp_skipped_invalid_vector",
    "sparse_omp_skipped_zero_vector",
    "sparse_omp_skipped_dimension_mismatch",
)


class RerankerSearchError(RuntimeError):
    """Non-retryable reranker response returned by an OSS Mem0 server."""

    def __init__(self, message: str, *, code: str, performance: dict[str, Any]):
        super().__init__(message)
        self.code = code
        self.retryable = False
        self.performance = performance


class SparseOmpSearchError(RerankerSearchError):
    """Backward-compatible typed error for sparse OMP failures."""


def _sort_search_results(results: list[dict[str, Any]]) -> None:
    """Sort reranked results by rerank_score, otherwise by retrieval score."""
    if any(("rerank_score" in result for result in results)):
        results.sort(key=lambda result: result.get("rerank_score") or 0, reverse=True)
    else:
        results.sort(key=lambda result: result.get("score") or 0, reverse=True)


def _threshold_payload_value(threshold: Any) -> tuple[bool, float | None]:
    if threshold is _THRESHOLD_UNSET:
        return (False, None)
    if threshold is None:
        return (True, None)
    numeric = float(threshold)
    if numeric < 0:
        return (True, None)
    return (True, numeric)


def _normalise_oss_search_results(data: Any) -> Any:
    results = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(results, list):
        results = []
    normalised = []
    for r in results:
        entry: dict[str, Any] = {
            "memory": r.get("memory", r.get("data", "")),
            "score": r.get("score", 0),
            "id": r.get("id", ""),
        }
        if r.get("created_at"):
            entry["created_at"] = r["created_at"]
        if r.get("updated_at"):
            entry["updated_at"] = r["updated_at"]
        score_debug = r.get("score_debug")
        breakdown = r.get("score_breakdown") or score_debug
        if score_debug and (
            "combined_score" in score_debug or "semantic_score" in score_debug or "bm25_score" in score_debug
        ):
            entry["score_debug"] = dict(score_debug)
            entry["score_debug"].setdefault("combined_score", r.get("score", 0))
        elif breakdown:
            entry["score_debug"] = {
                "combined_score": r.get("score", 0),
                "semantic_score": breakdown.get("semantic", 0),
                "bm25_score": breakdown.get("bm25", 0),
                "entity_boost": breakdown.get("entity_boost", 0),
            }
            for key in _OSS_SCORE_DEBUG_EXTRA_KEYS:
                if key in breakdown:
                    entry["score_debug"][key] = breakdown[key]
        if "rerank_score" in r:
            entry["rerank_score"] = r["rerank_score"]
        normalised.append(entry)
    _sort_search_results(normalised)
    if isinstance(data, dict) and ("rerank_metadata" in data or "query_debug" in data):
        wrapper: dict[str, Any] = {"results": normalised}
        if "rerank_metadata" in data:
            wrapper["rerank_metadata"] = data["rerank_metadata"]
        if "query_debug" in data:
            wrapper["query_debug"] = data["query_debug"]
        return wrapper
    return normalised


class Mem0Client:
    """Async HTTP client for the self-hosted MemPursuit backbone."""

    def __init__(
        self,
        mode: str = "oss",
        host: str | None = None,
        api_key: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        rpm: int = 60,
        timeout: float = 300.0,
    ):
        if aiohttp is None:
            raise RuntimeError(
                "aiohttp is required when constructing Mem0Client; install it for retrieval/ingestion runs"
            )
        if mode != "oss":
            raise ValueError("Only the self-hosted memory service is supported")
        self.mode = mode
        default_host = "http://localhost:8889"
        self.host = (host or os.getenv("MEM0_HOST", default_host)).rstrip("/")
        self.api_key = api_key or os.getenv("MEM0_API_KEY", "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.limiter = AsyncLimiter(100000, 60)
        self._session: aiohttp.ClientSession | None = None

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(headers=self._headers, timeout=self.timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session and (not self._session.closed):
            await self._session.close()

    async def __aenter__(self) -> Mem0Client:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def health(self) -> dict[str, Any]:
        """Return the effective runtime configuration exposed by the OSS server."""
        if self.mode != "oss":
            raise ValueError("Runtime health/config discovery is only supported by the OSS backend")
        session = await self._get_session()
        async with session.get(f"{self.host}/health") as resp:
            resp.raise_for_status()
            data = await resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("OSS /health returned a non-object response")
        return data

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Add memories from a conversation.

        Returns dict with "results" key listing extracted memories, or None on failure.
        """
        return await self._add_oss(messages, user_id, observation_date, timestamp, custom_instructions, metadata)

    async def _add_oss(
        self, messages, user_id, observation_date, timestamp, custom_instructions, metadata
    ) -> dict | None:
        """Add via OSS server — synchronous endpoint, no event polling."""
        session = await self._get_session()
        payload: dict[str, Any] = {"messages": messages, "user_id": user_id}
        if timestamp is not None:
            payload["timestamp"] = timestamp
        elif observation_date is not None:
            try:
                d = datetime.strptime(observation_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                payload["timestamp"] = int(d.timestamp())
            except ValueError:
                pass
        if custom_instructions:
            payload["custom_instructions"] = custom_instructions
        if metadata:
            payload["metadata"] = metadata
        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(f"{self.host}/memories", json=payload) as resp:
                        if resp.status >= 500:
                            raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp.status)
                        resp.raise_for_status()
                        data = await resp.json()
                if isinstance(data, dict) and "results" in data:
                    return data
                if isinstance(data, list):
                    return {"results": data}
                return {"results": []}
            except Exception as exc:
                logger.warning(
                    "ADD attempt %d/%d failed (user=%s): %s", attempt + 1, self.max_retries, user_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("ADD failed after %d attempts for user=%s", self.max_retries, user_id)
                    return None

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
        *,
        threshold: Any = _THRESHOLD_UNSET,
        rerank_options: dict[str, Any] | None = None,
    ) -> Any:
        """Search memories, preserving OSS diagnostic wrappers when present."""
        search_kwargs: dict[str, Any] = {"threshold": threshold}
        if rerank_options is not None:
            search_kwargs["rerank_options"] = rerank_options
        return await self._search_oss(query, user_id, top_k, rerank, score_debug, **search_kwargs)

    async def _search_oss(
        self,
        query,
        user_id,
        top_k,
        rerank,
        score_debug,
        *,
        threshold=_THRESHOLD_UNSET,
        rerank_options: dict[str, Any] | None = None,
    ) -> Any:
        """Search via OSS server."""
        search_started = time.monotonic()
        session = await self._get_session()
        payload: dict[str, Any] = {"query": query, "user_id": user_id, "limit": top_k}
        include_threshold, threshold_value = _threshold_payload_value(threshold)
        if include_threshold:
            payload["threshold"] = threshold_value
        if rerank:
            payload["rerank"] = True
            if rerank_options is not None:
                payload["rerank_options"] = rerank_options
        if score_debug:
            payload["score_debug"] = True
        for attempt in range(self.max_retries):
            try:
                async with self.limiter:
                    async with session.post(f"{self.host}/search", json=payload) as resp:
                        if resp.status >= 500:
                            error_body = await resp.json()
                            detail = error_body.get("detail") if isinstance(error_body, dict) else None
                            if isinstance(detail, dict) and detail.get("code") and (detail.get("retryable") is False):
                                performance = dict(detail.get("performance") or {})
                                performance.update(
                                    {
                                        "search_total": (time.monotonic() - search_started) * 1000.0,
                                        "outcome": "error",
                                        "error_code": detail["code"],
                                    }
                                )
                                error_class = (
                                    SparseOmpSearchError
                                    if detail["code"] == "sparse_omp_rerank_failed"
                                    else RerankerSearchError
                                )
                                raise error_class(
                                    detail.get("message") or "reranking failed",
                                    code=detail["code"],
                                    performance=performance,
                                )
                            raise aiohttp.ClientResponseError(resp.request_info, resp.history, status=resp.status)
                        resp.raise_for_status()
                        data = await resp.json()
                return _normalise_oss_search_results(data)
            except RerankerSearchError:
                logger.error("SEARCH failed with a non-retryable reranker error (user=%s)", user_id)
                raise
            except Exception as exc:
                logger.warning(
                    "SEARCH attempt %d/%d failed (user=%s): %s", attempt + 1, self.max_retries, user_id, str(exc)[:200]
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("SEARCH failed after %d attempts for user=%s", self.max_retries, user_id)
                    return []

    async def delete_user(self, user_id: str) -> bool:
        """Delete all memories for a user. Returns True on success."""
        return await self._delete_user_oss(user_id)

    async def _delete_user_oss(self, user_id: str) -> bool:
        session = await self._get_session()
        try:
            async with self.limiter:
                async with session.delete(f"{self.host}/memories", params={"user_id": user_id}) as resp:
                    resp.raise_for_status()
            logger.info("Deleted memories for user %s", user_id)
            return True
        except Exception as exc:
            logger.warning("Failed to delete user %s: %s", user_id, exc)
            return False


def format_search_results(search_results: Any) -> tuple[list[dict], dict | None]:
    """Normalize search results into a clean format for benchmark output.

    Returns:
        Tuple of (formatted results list, query_debug dict or None).
    """
    if not search_results:
        return ([], None)
    query_debug = None
    if isinstance(search_results, dict):
        query_debug = search_results.get("query_debug")
        search_results = search_results.get("results", [])
    sorted_results = list(search_results)
    _sort_search_results(sorted_results)
    formatted = []
    for r in sorted_results:
        entry: dict[str, Any] = {"memory": r.get("memory", ""), "score": r.get("score", 0), "id": r.get("id", "")}
        if r.get("created_at"):
            entry["created_at"] = r["created_at"]
        if r.get("updated_at"):
            entry["updated_at"] = r["updated_at"]
        if r.get("score_debug"):
            entry["score_debug"] = r["score_debug"]
        if "rerank_score" in r:
            entry["rerank_score"] = r["rerank_score"]
        formatted.append(entry)
    return (formatted, query_debug)
