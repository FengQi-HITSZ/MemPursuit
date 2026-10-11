import gc
import hashlib
import json
import logging
import math
import os
import time
import uuid
import warnings
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    generate_additive_extraction_prompt,
)
from mem0.exceptions import ValidationError as Mem0ValidationError
from mem0.memory.base import MemoryBase
from mem0.memory.setup import mem0_dir
from mem0.memory.storage import SQLiteManager
from mem0.memory.utils import (
    extract_json,
    parse_messages,
    remove_code_blocks,
)
from mem0.reranker.sparse_omp_reranker import SparseOmpRerankError, SparseOmpReranker
from mem0.utils.entity_extraction import extract_entities, extract_entities_batch
from mem0.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
)
from mem0.utils.lemmatization import lemmatize_for_bm25
from mem0.utils.scoring import (
    ENTITY_BOOST_WEIGHT,
    get_bm25_params,
    normalize_bm25,
    score_and_rank,
)

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)


# Fields that hold runtime auth/connection objects and must be preserved.
# These are non-serializable objects (e.g. AWSV4SignerAuth, RequestsHttpConnection)
# needed by clients like OpenSearch — not sensitive strings to redact.
_RUNTIME_FIELDS = frozenset(
    {
        "http_auth",
        "auth",
        "connection_class",
        "ssl_context",
    }
)

# Fields that are known to contain sensitive secrets and must be redacted.
_SENSITIVE_FIELDS_EXACT = frozenset(
    {
        "api_key",
        "secret_key",
        "private_key",
        "access_key",
        "password",
        "credentials",
        "credential",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "auth_token",
        "session_token",
        "client_secret",
        "auth_client_secret",
        "azure_client_secret",
        "service_account_json",
        "aws_session_token",
    }
)

# Suffixes that indicate a field likely holds a secret value.
_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_credential",
    "_credentials",
)

# Entity parameters that must be passed via filters, not top-level kwargs
ENTITY_PARAMS = frozenset({"user_id", "agent_id", "run_id"})


@dataclass
class SearchArtifacts:
    """Request-local retrieval outputs that may be reused by a reranker."""

    memories: list[Dict[str, Any]]
    query_embedding: Any = None
    query_embedding_ms: float = 0.0
    hybrid_candidate_retrieval_ms: float = 0.0


_RETRIEVAL_QUERY_COMPRESSION_ATTEMPTS = 2


def _prepare_retrieval_query_for_embedding(query: str, embedding_model: Any, llm: Any) -> str:
    """Fit an overlong retrieval query to the embedder without changing extraction input.

    Only embedders that expose ``max_input_tokens``, ``count_tokens`` and
    ``truncate_tokens`` opt into this behavior. LLM call failures deliberately
    propagate. An overlong LLM result is retried once; only then is it truncated.
    """
    token_limit = getattr(embedding_model, "max_input_tokens", None)
    count_tokens = getattr(embedding_model, "count_tokens", None)
    truncate_tokens = getattr(embedding_model, "truncate_tokens", None)
    if (
        isinstance(token_limit, bool)
        or not isinstance(token_limit, int)
        or token_limit < 1
        or not callable(count_tokens)
        or not callable(truncate_tokens)
    ):
        return query

    query_tokens = count_tokens(query)
    if query_tokens <= token_limit:
        return query

    logger.warning(
        "Retrieval query exceeds embedding token limit; compressing with LLM (tokens=%d, limit=%d)",
        query_tokens,
        token_limit,
    )

    compressed_query = query
    compressed_tokens = query_tokens
    for attempt in range(1, _RETRIEVAL_QUERY_COMPRESSION_ATTEMPTS + 1):
        response = llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Compress the supplied retrieval query only enough to fit within the requested token limit. "
                        "Preserve as much information and original wording as possible; do not aggressively summarize. "
                        "Remove redundancy and low-information formatting first. Preserve speaker roles, entities, "
                        "relationships, dates, chronology, numbers, locations, preferences, decisions, state changes, "
                        "negations, and conflicts. Do not invent information. Output only the compressed query."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The query currently contains approximately {compressed_tokens} tokens. "
                        f"Rewrite it to no more than {token_limit} tokens. Use the available token budget when needed "
                        "to retain detail.\n\nQUERY:\n" + compressed_query
                    ),
                },
            ],
            max_tokens=token_limit,
        )
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("LLM retrieval-query compression returned an empty or non-text response")

        compressed_query = response.strip()
        compressed_tokens = count_tokens(compressed_query)
        logger.info(
            "Retrieval-query compression attempt %d/%d produced %d tokens (limit=%d)",
            attempt,
            _RETRIEVAL_QUERY_COMPRESSION_ATTEMPTS,
            compressed_tokens,
            token_limit,
        )
        if compressed_tokens <= token_limit:
            return compressed_query

    truncated_query = truncate_tokens(compressed_query, token_limit)
    truncated_tokens = count_tokens(truncated_query)
    if truncated_tokens > token_limit:
        raise RuntimeError(
            "Embedding tokenizer failed to truncate retrieval query within its token limit "
            f"({truncated_tokens} > {token_limit})"
        )
    logger.warning(
        "LLM retrieval-query compression remained overlong after %d attempts; truncated from %d to %d tokens",
        _RETRIEVAL_QUERY_COMPRESSION_ATTEMPTS,
        compressed_tokens,
        truncated_tokens,
    )
    return truncated_query


def _reject_top_level_entity_params(kwargs: Dict[str, Any], method_name: str) -> None:
    """Reject top-level entity parameters - must use filters instead."""
    invalid_keys = ENTITY_PARAMS & set(kwargs.keys())
    if invalid_keys:
        raise ValueError(
            f"Top-level entity parameters {invalid_keys} are not supported in {method_name}(). "
            f"Use filters={{'user_id': '...'}} instead."
        )


def _validate_and_trim_entity_id(value: Optional[str], name: str) -> Optional[str]:
    """
    Validates and normalizes an entity ID.
    - Trims leading/trailing whitespace
    - Rejects empty or whitespace-only strings
    - Rejects strings containing internal whitespace

    Args:
        value: The entity ID value to validate
        name: The parameter name (for error messages)

    Returns:
        The trimmed entity ID, or None if input is None

    Raises:
        ValueError: If entity ID is invalid
    """
    if value is None:
        return None
    trimmed = value.strip()
    if trimmed == "":
        raise ValueError(f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier.")
    if any(c.isspace() for c in trimmed):
        raise ValueError(f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces.")
    return trimmed


def _validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    """
    Validates search parameters.

    Args:
        threshold: Similarity threshold. None or a negative value disables threshold filtering;
            otherwise it must be between 0 and 1.
        top_k: Number of results to return (must be non-negative integer)

    Raises:
        ValueError: If threshold or top_k are invalid
    """
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a valid number")
        if threshold > 1:
            raise ValueError(
                f"Invalid threshold: {threshold}. Must be less than or equal to 1, or negative to disable."
            )
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(f"Invalid top_k: {top_k}. Must be a non-negative integer.")


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field should be redacted in configuration copies.

    Uses a layered approach:
    1. Runtime fields (allowlist) — always preserved, highest priority.
    2. Exact deny list — known secret field names.
    3. Suffix deny list — catches patterns like db_password, auth_secret, etc.
    """
    name = field_name.lower().strip()
    if name in _RUNTIME_FIELDS:
        return False
    if name in _SENSITIVE_FIELDS_EXACT:
        return True
    return any(name.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)


def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to dict-based cloning for non-serializable objects."""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using dict-based cloning: {e}")

        config_class = type(config)

        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump()
            except Exception:
                clone_dict = dict(config.__dict__)
        else:
            clone_dict = dict(config.__dict__)

        # Restore runtime fields, redact sensitive ones
        for field_name in list(clone_dict.keys()):
            if field_name in _RUNTIME_FIELDS and hasattr(config, field_name):
                clone_dict[field_name] = getattr(config, field_name)
            elif _is_sensitive_field(field_name):
                clone_dict[field_name] = None

        try:
            return config_class(**clone_dict)
        except Exception:
            logger.debug("Config reconstruction failed, returning shallow dict clone")
            return type("Config", (), clone_dict)()


def _normalize_iso_timestamp_to_utc(timestamp: Optional[str]) -> Optional[str]:
    """Normalize timezone-aware ISO timestamps to UTC without rewriting naive values."""
    if not timestamp:
        return timestamp
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    if parsed.tzinfo is None:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat()


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- validate and add all provided session ids ----------
    session_ids_provided = []

    # Validate and trim entity IDs
    user_id = _validate_and_trim_entity_id(user_id, "user_id")
    agent_id = _validate_and_trim_entity_id(agent_id, "agent_id")
    run_id = _validate_and_trim_entity_id(run_id, "run_id")

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation.",
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


def _build_session_scope(filters):
    """Build deterministic session scope string from entity IDs."""
    parts = []
    for key in sorted(["user_id", "agent_id", "run_id"]):
        val = filters.get(key)
        if val:
            parts.append(f"{key}={val}")
    return "&".join(parts)


def _get_prompt_observation_timestamp(metadata: Optional[Dict[str, Any]]) -> Any:
    """Return the best available conversation-time anchor for extraction prompts."""
    if not metadata:
        return None

    for key in (
        "created_at",
        "observation_date",
        "locomo_session_date_time",
        "timestamp",
        "locomo_session_epoch",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


_SPARSE_OMP_EPS = 1e-12


def _uses_sparse_omp_reranker(config: MemoryConfig) -> bool:
    reranker_config = getattr(config, "reranker", None)
    return bool(reranker_config and getattr(reranker_config, "provider", None) == "sparse_omp")


def _resolve_sparse_omp_rerank_options(
    config: MemoryConfig,
    reranker: Any,
    *,
    rerank: bool,
    top_k: int,
    rerank_options: Any,
) -> Dict[str, Any] | None:
    """Validate request-level OMP controls only when reranking is actually enabled."""
    if not rerank:
        return None

    uses_sparse_omp = _uses_sparse_omp_reranker(config)
    if rerank_options is None:
        options: Dict[str, Any] = {}
    elif isinstance(rerank_options, dict):
        options = dict(rerank_options)
    else:
        raise ValueError("rerank_options must be an object")

    if not uses_sparse_omp:
        if rerank_options is not None:
            provider = getattr(getattr(config, "reranker", None), "provider", None) or "none"
            raise ValueError(
                "rerank_options are only supported for reranker.provider='sparse_omp'; "
                f"configured provider is {provider!r}"
            )
        return None
    if reranker is None:
        raise ValueError("reranker.provider='sparse_omp' is configured but no reranker instance is available")

    unknown = sorted(set(options) - {"early_stopping", "min_iterations", "short_pool_policy"})
    if unknown:
        raise ValueError(f"Unknown sparse OMP rerank_options fields: {', '.join(unknown)}")

    early_stopping = options.get("early_stopping", reranker.config.early_stopping)
    if not isinstance(early_stopping, bool):
        raise ValueError("rerank_options.early_stopping must be a boolean")

    min_iterations = options.get("min_iterations")
    if min_iterations is not None:
        if isinstance(min_iterations, bool) or not isinstance(min_iterations, int) or min_iterations < 1:
            raise ValueError("rerank_options.min_iterations must be a positive integer")
        if early_stopping:
            raise ValueError("rerank_options.min_iterations is only valid when early_stopping=false")

    short_pool_policy = options.get("short_pool_policy", "error")
    if short_pool_policy not in {"error", "use_available"}:
        raise ValueError("rerank_options.short_pool_policy must be 'error' or 'use_available'")

    candidate_pool_size = int(reranker.config.candidate_pool_size)
    if candidate_pool_size < int(top_k):
        raise ValueError(
            f"sparse OMP candidate_pool_size must be greater than or equal to top_k ({candidate_pool_size} < {top_k})"
        )

    max_support = reranker.config.max_support
    if early_stopping:
        effective_iteration_limit = int(max_support) if max_support is not None else candidate_pool_size
        if effective_iteration_limit > int(top_k):
            raise ValueError(
                "adaptive sparse OMP effective iteration limit must not exceed top_k "
                f"({effective_iteration_limit} > {top_k})"
            )
    else:
        if min_iterations is not None and min_iterations > int(top_k):
            raise ValueError(
                "fixed sparse OMP min_iterations (the maximum requested cutoff) must not exceed top_k "
                f"({min_iterations} > {top_k})"
            )
        effective_iteration_limit = max(int(max_support or 0), int(min_iterations or 0))
        if effective_iteration_limit < 1:
            raise ValueError("fixed sparse OMP requires max_support or rerank_options.min_iterations to be set")
        if candidate_pool_size < effective_iteration_limit:
            raise ValueError(
                "fixed sparse OMP candidate_pool_size must cover its effective iteration limit "
                f"({candidate_pool_size} < {effective_iteration_limit})"
            )

    return {
        "early_stopping": early_stopping,
        "min_iterations": min_iterations,
        "effective_iteration_limit": effective_iteration_limit,
        "short_pool_policy": short_pool_policy,
    }


def _rerank_sparse_omp_candidates(
    *,
    reranker: Any,
    embedding_model: Any,
    vector_store: Any,
    vector_store_provider: str,
    query: str,
    memories: list[Dict[str, Any]],
    limit: int,
    query_embedding: Any = None,
    query_embedding_ms: float = 0.0,
    hybrid_candidate_retrieval_ms: float = 0.0,
    score_debug: bool = False,
    rerank_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rerank_started = time.perf_counter()
    embedding_reused = query_embedding is not None
    performance: Dict[str, Any] = {
        "query_embedding": float(query_embedding_ms) if query_embedding is not None else None,
        "hybrid_candidate_retrieval": float(hybrid_candidate_retrieval_ms),
        "dense_vector_fetch": None,
        "concurrency_wait": 0.0,
        "vector_cache_hits": 0,
        "vector_cache_misses": 0,
        "vector_prepare": None,
        "gram_build": None,
        "support_selection": None,
        "tail_fill": None,
        "rerank_total": None,
        "embedding_reused": embedding_reused,
        "candidate_count": len(memories),
        "valid_vector_count": 0,
        "support_count": 0,
        "tail_count": 0,
        "qp_solve_count": 0,
        "stop_reason": None,
        "relative_residual": None,
        "final_fw_gap": None,
        "iterations": 0,
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
    }
    fetch_started: float | None = None
    embedding_started: float | None = None
    try:
        if not hasattr(reranker, "rerank_with_vectors_detailed"):
            raise NotImplementedError(
                "Configured sparse_omp reranker does not implement rerank_with_vectors_detailed()."
            )

        if not memories:
            detailed = reranker.rerank_with_vectors_detailed(
                query_embedding,
                [],
                {},
                limit,
                early_stopping=(rerank_options or {}).get("early_stopping"),
                min_iterations=(rerank_options or {}).get("min_iterations"),
                short_pool_policy=(rerank_options or {}).get("short_pool_policy", "error"),
            )
        else:
            if not hasattr(vector_store, "get_dense_vectors"):
                raise NotImplementedError(
                    "reranker.provider='sparse_omp' requires stored dense vector lookup. "
                    f"Current vector_store provider '{vector_store_provider}' does not implement get_dense_vectors()."
                )
            if query_embedding is None:
                embedding_started = time.perf_counter()
                query_embedding = embedding_model.embed(query, "search")
                query_embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
                performance["query_embedding"] = query_embedding_ms

            slot_context = reranker.execution_slot() if hasattr(reranker, "execution_slot") else nullcontext(0.0)
            with slot_context as concurrency_wait_ms:
                performance["concurrency_wait"] = float(concurrency_wait_ms or 0.0)
                memory_ids = [item.get("id") for item in memories if item.get("id") is not None]
                fetch_started = time.perf_counter()
                memory_vectors = vector_store.get_dense_vectors(memory_ids)
                performance["dense_vector_fetch"] = (time.perf_counter() - fetch_started) * 1000.0
                performance["vector_cache_hits"] = int(getattr(memory_vectors, "cache_hits", 0))
                performance["vector_cache_misses"] = int(getattr(memory_vectors, "cache_misses", len(memory_vectors)))
                detailed = reranker.rerank_with_vectors_detailed(
                    query_embedding,
                    memories,
                    memory_vectors,
                    limit,
                    early_stopping=(rerank_options or {}).get("early_stopping"),
                    min_iterations=(rerank_options or {}).get("min_iterations"),
                    short_pool_policy=(rerank_options or {}).get("short_pool_policy", "error"),
                )

        metadata = detailed.get("rerank_metadata") or {}
        performance.update(detailed.get("performance") or {})
        performance.update(
            {
                "query_embedding": float(query_embedding_ms) if query_embedding is not None else None,
                "hybrid_candidate_retrieval": float(hybrid_candidate_retrieval_ms),
                "dense_vector_fetch": performance.get("dense_vector_fetch"),
                "rerank_total": (time.perf_counter() - rerank_started) * 1000.0,
                "embedding_reused": embedding_reused,
                "candidate_count": len(memories),
                "stop_reason": metadata.get("stop_reason"),
                "relative_residual": metadata.get("relative_residual"),
                "final_fw_gap": metadata.get("final_fw_gap"),
                "iterations": metadata.get("iterations", 0),
                "qp_solve_count": metadata.get("qp_solve_count", performance.get("qp_solve_count", 0)),
            }
        )
        if score_debug:
            logger.info(json.dumps({"event": "sparse_omp_rerank_profile", **performance}, sort_keys=True))
        return {
            "results": detailed.get("results", []),
            "rerank_metadata": metadata,
            "performance": performance,
        }
    except SparseOmpRerankError as exc:
        performance.update(exc.performance)
        performance["query_embedding"] = (
            (time.perf_counter() - embedding_started) * 1000.0
            if embedding_started is not None and performance.get("query_embedding") is None
            else performance.get("query_embedding")
        )
        performance["dense_vector_fetch"] = (
            (time.perf_counter() - fetch_started) * 1000.0
            if fetch_started is not None and performance.get("dense_vector_fetch") is None
            else performance.get("dense_vector_fetch")
        )
        performance["rerank_total"] = (time.perf_counter() - rerank_started) * 1000.0
        exc.performance = performance
        raise
    except Exception as exc:
        if embedding_started is not None and performance["query_embedding"] is None:
            performance["query_embedding"] = (time.perf_counter() - embedding_started) * 1000.0
        if fetch_started is not None and performance["dense_vector_fetch"] is None:
            performance["dense_vector_fetch"] = (time.perf_counter() - fetch_started) * 1000.0
        performance["rerank_total"] = (time.perf_counter() - rerank_started) * 1000.0
        raise SparseOmpRerankError(
            f"sparse OMP reranking failed: {exc}",
            cause=exc,
            performance=performance,
        ) from exc


logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        self.custom_instructions = self.config.custom_instructions

        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(config.reranker.provider, config.reranker.config)

        # Entity store is initialized lazily on first use
        self._entity_store = None

    @property
    def entity_store(self):
        """Lazily initialize entity store on first use."""
        if self._entity_store is None:
            entity_config = _safe_deepcopy_config(self.config.vector_store.config)
            entity_collection = f"{self.collection_name}_entities"
            # Set collection name on the cloned config
            if hasattr(entity_config, "collection_name"):
                entity_config.collection_name = entity_collection
            elif isinstance(entity_config, dict):
                entity_config["collection_name"] = entity_collection
            # For Qdrant, share the existing client to avoid RocksDB lock contention
            # when using embedded mode (path=...). QdrantConfig.client takes precedence
            # over host/port/path.
            if self.config.vector_store.provider == "qdrant" and hasattr(self.vector_store, "client"):
                if hasattr(entity_config, "client"):
                    entity_config.client = self.vector_store.client
                elif isinstance(entity_config, dict):
                    entity_config["client"] = self.vector_store.client
            self._entity_store = VectorStoreFactory.create(self.config.vector_store.provider, entity_config)
        return self._entity_store

    def _upsert_entity(self, entity_text, entity_type, memory_id, filters):
        """Upsert an entity into the entity store, linking it to a memory."""
        try:
            entity_embedding = self.embedding_model.embed(entity_text, "add")
            search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}

            existing = self.entity_store.search(
                query=entity_text,
                vectors=entity_embedding,
                top_k=1,
                filters=search_filters,
            )

            if existing and existing[0].score >= 0.95:
                # Update existing entity's linked_memory_ids
                match = existing[0]
                payload = match.payload or {}
                linked_ids = payload.get("linked_memory_ids", [])
                if memory_id not in linked_ids:
                    linked_ids.append(memory_id)
                    payload["linked_memory_ids"] = linked_ids
                    self.entity_store.update(
                        vector_id=match.id,
                        vector=None,
                        payload=payload,
                    )
            else:
                # Create new entity
                entity_id = str(uuid.uuid4())
                entity_payload = {
                    "data": entity_text,
                    "entity_type": entity_type,
                    "linked_memory_ids": [memory_id],
                    **{k: v for k, v in search_filters.items()},
                }
                self.entity_store.insert(
                    vectors=[entity_embedding],
                    ids=[entity_id],
                    payloads=[entity_payload],
                )
        except Exception as e:
            logger.warning(f"Entity upsert failed for '{entity_text}': {e}")

    def _remove_memory_from_entity_store(self, memory_id, filters):
        """Strip `memory_id` from every entity record scoped to `filters`.

        For each entity whose `linked_memory_ids` contains `memory_id`:
          - remove the id; if the list becomes empty, delete the entity record.
          - otherwise re-embed the entity text and update the payload
            (the vector store's update() requires a vector).

        No-op if the entity store has never been initialized in this process.
        Errors on individual entities are swallowed at debug level; outer
        failures are swallowed at warning level so the primary delete/update
        path is never broken by entity cleanup.
        """
        if self._entity_store is None:
            return
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        try:
            listed = self.entity_store.list(filters=search_filters, top_k=10000)
            rows = listed[0] if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list) else listed
            for row in rows or []:
                try:
                    payload = getattr(row, "payload", None) or {}
                    linked = payload.get("linked_memory_ids", [])
                    if not isinstance(linked, list) or memory_id not in linked:
                        continue
                    remaining = [mid for mid in linked if mid != memory_id]
                    if not remaining:
                        try:
                            self.entity_store.delete(vector_id=row.id)
                        except Exception as e:
                            logger.debug(f"Entity delete failed for id={row.id}: {e}")
                    else:
                        entity_text = payload.get("data")
                        if not isinstance(entity_text, str) or not entity_text:
                            logger.debug(f"Entity id={row.id} missing 'data'; skipping update during cleanup")
                            continue
                        try:
                            vec = self.embedding_model.embed(entity_text, "update")
                        except Exception as e:
                            logger.debug(f"Entity re-embed failed for '{entity_text}': {e}")
                            continue
                        new_payload = {**payload, "linked_memory_ids": remaining}
                        try:
                            self.entity_store.update(
                                vector_id=row.id,
                                vector=vec,
                                payload=new_payload,
                            )
                        except Exception as e:
                            logger.debug(f"Entity update failed for id={row.id}: {e}")
                except Exception as e:
                    logger.debug(f"Entity cleanup error: {e}")
        except Exception as e:
            logger.warning(f"Entity store cleanup failed for memory_id={memory_id}: {e}")

    def _link_entities_for_memory(self, memory_id, text, filters):
        """Extract entities from `text` and link them to `memory_id` in the
        entity store, scoped to `filters`. Simpler single-memory variant of
        Phase 7 in add(): per-entity search-then-update-or-insert via the
        existing `_upsert_entity` helper. Non-fatal on any failure.
        """
        try:
            entities = extract_entities(text)
            if not entities:
                return
            seen = set()
            for entity_type, entity_text in entities:
                key = entity_text.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                try:
                    self._upsert_entity(entity_text, entity_type, memory_id, filters)
                except Exception as e:
                    logger.debug(f"Entity link failed for '{entity_text}': {e}")
        except Exception as e:
            logger.warning(f"Entity linking failed for memory_id={memory_id}: {e}")

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False

        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.

        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None

        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)

        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type: Reserved; leave unset.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None:
            raise ValueError("Only conversational memory is supported by this research backbone")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries.",
            )

        vector_store_result = self._add_to_vector_store(
            messages, processed_metadata, effective_filters, infer, prompt=prompt
        )
        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer, prompt=None):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, {msg_content: msg_embeddings}, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        # === V3 PHASED BATCH PIPELINE ===

        # Phase 0: Context gathering
        session_scope = _build_session_scope(filters)
        last_messages = self.db.get_last_messages(session_scope, limit=10)
        parsed_messages = parse_messages(messages)

        # Phase 1: Existing memory retrieval
        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        retrieval_query = _prepare_retrieval_query_for_embedding(parsed_messages, self.embedding_model, self.llm)
        query_embedding = self.embedding_model.embed(retrieval_query, "search")
        existing_results = self.vector_store.search(
            query=retrieval_query,
            vectors=query_embedding,
            top_k=10,
            filters=search_filters,
        )

        # Map UUIDs to integers (anti-hallucination)
        existing_memories = []
        uuid_mapping = {}
        for idx, mem in enumerate(existing_results):
            uuid_mapping[str(idx)] = mem.id
            existing_memories.append({"id": str(idx), "text": mem.payload.get("data", "")})

        # Phase 2: LLM extraction (single call)
        is_agent_scoped = bool(filters.get("agent_id")) and not filters.get("user_id")
        system_prompt = ADDITIVE_EXTRACTION_PROMPT
        if is_agent_scoped:
            system_prompt += AGENT_CONTEXT_SUFFIX

        custom_instr = prompt or self.custom_instructions

        user_prompt = generate_additive_extraction_prompt(
            existing_memories=existing_memories,
            new_messages=parsed_messages,
            last_k_messages=last_messages,
            timestamp=_get_prompt_observation_timestamp(metadata),
            custom_instructions=custom_instr,
        )

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise RuntimeError("LLM memory extraction failed") from e

        # Parse response
        try:
            response = remove_code_blocks(response)
            if not response or not response.strip():
                extracted_memories = []
            else:
                try:
                    extracted_memories = json.loads(response, strict=False).get("memory", [])
                except json.JSONDecodeError:
                    extracted_json = extract_json(response)
                    extracted_memories = json.loads(extracted_json, strict=False).get("memory", [])
        except Exception as e:
            raise RuntimeError("Could not parse the memory extraction response") from e

        if not extracted_memories:
            # Save messages even if nothing extracted
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 3: Batch embed all extracted memory texts
        mem_texts = [m.get("text", "") for m in extracted_memories if m.get("text")]
        try:
            mem_embeddings_list = self.embedding_model.embed_batch(mem_texts, "add")
            embed_map = dict(zip(mem_texts, mem_embeddings_list))
        except Exception:
            # Fallback: embed individually
            embed_map = {}
            for text in mem_texts:
                try:
                    embed_map[text] = self.embedding_model.embed(text, "add")
                except Exception as e:
                    logger.warning(f"Failed to embed memory text: {e}")

        # Phase 4: Per-memory CPU processing + Phase 5: Hash dedup
        # Build set of existing hashes for dedup
        existing_hashes = set()
        for mem in existing_results:
            h = mem.payload.get("hash") if hasattr(mem, "payload") and mem.payload else None
            if h:
                existing_hashes.add(h)

        records = []  # (memory_id, text, embedding, payload)
        seen_hashes = set()  # dedup within the current batch
        for mem in extracted_memories:
            text = mem.get("text")
            if not text or text not in embed_map:
                continue

            mem_hash = hashlib.md5(text.encode()).hexdigest()
            if mem_hash in existing_hashes or mem_hash in seen_hashes:
                logger.debug(f"Skipping duplicate memory (hash match): {text[:50]}")
                continue
            seen_hashes.add(mem_hash)

            text_lemmatized = lemmatize_for_bm25(text)

            memory_id = str(uuid.uuid4())
            mem_metadata = deepcopy(metadata)
            mem_metadata["data"] = text
            mem_metadata["text_lemmatized"] = text_lemmatized
            mem_metadata["hash"] = mem_hash
            if "created_at" not in mem_metadata:
                mem_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            mem_metadata["updated_at"] = mem_metadata["created_at"]
            if mem.get("attributed_to"):
                mem_metadata["attributed_to"] = mem["attributed_to"]

            records.append((memory_id, text, embed_map[text], mem_metadata))

        if not records:
            self.db.save_messages(messages, session_scope)
            return []

        # Phase 6: Batch persist
        all_vectors = [r[2] for r in records]
        all_ids = [r[0] for r in records]
        all_payloads = [r[3] for r in records]
        # Example payload includes benchmark/user metadata plus data, text_lemmatized, hash, created_at, and updated_at.

        try:
            self.vector_store.insert(
                vectors=all_vectors,
                ids=all_ids,
                payloads=all_payloads,
            )
        except Exception:
            # Fallback: insert one by one
            for mid, vec, pay in zip(all_ids, all_vectors, all_payloads):
                try:
                    self.vector_store.insert(vectors=[vec], ids=[mid], payloads=[pay])
                except Exception as e:
                    logger.error(f"Failed to insert memory {mid}: {e}")

        # Batch history
        history_records = [
            {
                "memory_id": r[0],
                "old_memory": None,
                "new_memory": r[1],
                "event": "ADD",
                "created_at": r[3].get("created_at"),
                "is_deleted": 0,
            }
            for r in records
        ]
        try:
            self.db.batch_add_history(history_records)
        except Exception:
            # Fallback: add one by one
            for hr in history_records:
                try:
                    self.db.add_history(hr["memory_id"], None, hr["new_memory"], "ADD", created_at=hr.get("created_at"))
                except Exception as e:
                    logger.error(f"Failed to add history for {hr['memory_id']}: {e}")

        # Phase 7: Batch entity linking
        try:
            all_texts = [r[1] for r in records]
            all_entities = extract_entities_batch(all_texts)

            # 7a: Global dedup — collect unique entities across all memories
            global_entities = {}  # normalized_key -> (entity_type, entity_text, set of memory_ids)
            for idx, (memory_id, text, embedding, payload) in enumerate(records):
                entities = all_entities[idx] if idx < len(all_entities) else []
                for entity_type, entity_text in entities:
                    key = entity_text.strip().lower()
                    if key in global_entities:
                        global_entities[key][2].add(memory_id)
                    else:
                        global_entities[key] = [entity_type, entity_text, {memory_id}]

            if global_entities:
                ordered_keys = list(global_entities.keys())
                entity_texts = [global_entities[k][1] for k in ordered_keys]

                # 7b: Single batch embed for all unique entities
                try:
                    entity_embeddings = self.embedding_model.embed_batch(entity_texts, "add")
                except Exception:
                    # Fallback: embed individually, use None for failures
                    entity_embeddings = []
                    for t in entity_texts:
                        try:
                            entity_embeddings.append(self.embedding_model.embed(t, "add"))
                        except Exception:
                            entity_embeddings.append(None)

                # Filter out entities with failed embeddings
                valid = [(i, k) for i, k in enumerate(ordered_keys) if entity_embeddings[i] is not None]
                if valid:
                    valid_indices, valid_keys = zip(*valid)
                    valid_vectors = [entity_embeddings[i] for i in valid_indices]

                    # 7c: Batch search for existing entities
                    valid_texts = [global_entities[k][1] for k in valid_keys]
                    existing_matches = self.entity_store.search_batch(
                        queries=valid_texts,
                        vectors_list=valid_vectors,
                        top_k=1,
                        filters=search_filters,
                    )

                    # 7d: Separate into inserts vs updates
                    to_insert_vectors, to_insert_ids, to_insert_payloads = [], [], []
                    for j, key in enumerate(valid_keys):
                        entity_type, entity_text, memory_ids = global_entities[key]
                        matches = existing_matches[j] if j < len(existing_matches) else []

                        if matches and matches[0].score >= 0.95:
                            # Update existing entity
                            match = matches[0]
                            payload = match.payload or {}
                            linked = set(payload.get("linked_memory_ids", []))
                            linked |= memory_ids
                            payload["linked_memory_ids"] = sorted(linked)
                            try:
                                self.entity_store.update(
                                    vector_id=match.id,
                                    vector=None,
                                    payload=payload,
                                )
                            except Exception as e:
                                logger.debug(f"Entity update failed for '{entity_text}': {e}")
                        else:
                            # New entity — collect for batch insert
                            to_insert_vectors.append(valid_vectors[j])
                            to_insert_ids.append(str(uuid.uuid4()))
                            to_insert_payloads.append(
                                {
                                    "data": entity_text,
                                    "entity_type": entity_type,
                                    "linked_memory_ids": sorted(memory_ids),
                                    **search_filters,
                                }
                            )

                    # 7e: Single batch insert for all new entities
                    if to_insert_vectors:
                        try:
                            self.entity_store.insert(
                                vectors=to_insert_vectors,
                                ids=to_insert_ids,
                                payloads=to_insert_payloads,
                            )
                        except Exception as e:
                            logger.warning(f"Batch entity insert failed: {e}")
        except Exception as e:
            logger.warning(f"Batch entity linking failed: {e}")

        # Phase 8: Save messages + return
        self.db.save_messages(messages, session_scope)

        returned_memories = [{"id": r[0], "memory": r[1], "event": "ADD"} for r in records]

        return returned_memories

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {
            "data",
            "hash",
            "created_at",
            "updated_at",
            "id",
            "text_lemmatized",
            "attributed_to",
            *promoted_payload_keys,
        }

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        **kwargs,
    ):
        """
        List all memories.

        Args:
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}
            top_k (int, optional): The maximum number of memories to return. Defaults to 20.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if top_k is invalid.
        """
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "get_all")

        # Validate top_k
        _validate_search_params(top_k=top_k)

        # Validate and trim entity IDs in filters
        effective_filters = dict(filters) if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(effective_filters["user_id"], "user_id")
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(effective_filters["agent_id"], "agent_id")
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(effective_filters["run_id"], "run_id")

        # Validate filters contains at least one entity ID
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. Example: filters={'user_id': 'u1'}"
            )

        limit = top_k

        all_memories_result = self._get_all_from_vector_store(effective_filters, limit)

        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, top_k=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {
            "data",
            "hash",
            "created_at",
            "updated_at",
            "id",
            "text_lemmatized",
            "attributed_to",
            *promoted_payload_keys,
        }

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        rerank_options: Optional[Dict[str, Any]] = None,
        score_debug: bool = False,
        **kwargs,
    ):
        """
        Searches for memories based on a query.

        Args:
            query (str): Query to search for.
            top_k (int, optional): Maximum number of results to return. Defaults to 20.
            filters (dict): Filter dict containing entity IDs and optional metadata filters.
                Must contain at least one of: user_id, agent_id, run_id.
                Example: filters={"user_id": "u1", "agent_id": "a1"}

                Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT
            threshold (float, optional): Minimum score for a memory to be included. Defaults to 0.1.
            rerank (bool, optional): Whether to rerank results. Defaults to False.
            rerank_options (dict, optional): Request-level sparse OMP controls. Ignored unless rerank is True.
            score_debug (bool, optional): Include semantic/BM25/entity score breakdowns.

        Returns:
            dict: A dictionary containing the search results under a "results" key.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`

        Raises:
            ValueError: If filters doesn't contain at least one of user_id, agent_id, run_id,
                or if threshold/top_k values are invalid.
        """
        search_started = time.perf_counter()
        # Reject top-level entity params - must use filters instead
        _reject_top_level_entity_params(kwargs, "search")

        # Validate search parameters (before applying defaults)
        _validate_search_params(threshold=threshold, top_k=top_k)
        if threshold is not None and threshold < 0:
            threshold = None

        # Validate and trim entity IDs in filters
        effective_filters = filters.copy() if filters else {}
        if "user_id" in effective_filters:
            effective_filters["user_id"] = _validate_and_trim_entity_id(effective_filters["user_id"], "user_id")
        if "agent_id" in effective_filters:
            effective_filters["agent_id"] = _validate_and_trim_entity_id(effective_filters["agent_id"], "agent_id")
        if "run_id" in effective_filters:
            effective_filters["run_id"] = _validate_and_trim_entity_id(effective_filters["run_id"], "run_id")
        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "filters must contain at least one of: user_id, agent_id, run_id. Example: filters={'user_id': 'u1'}"
            )

        limit = top_k
        sparse_request_options = _resolve_sparse_omp_rerank_options(
            self.config,
            self.reranker,
            rerank=rerank,
            top_k=limit,
            rerank_options=rerank_options,
        )
        retrieval_limit = limit
        if rerank and _uses_sparse_omp_reranker(self.config) and self.reranker is not None:
            retrieval_limit = int(self.reranker.config.candidate_pool_size)
        elif rerank and self.reranker is not None:
            candidate_pool_size = getattr(self.reranker.config, "candidate_pool_size", None)
            if candidate_pool_size is not None:
                retrieval_limit = int(candidate_pool_size)
                if retrieval_limit < limit:
                    raise ValueError(
                        "reranker candidate_pool_size must be greater than or equal to top_k "
                        f"({retrieval_limit} < {limit})"
                    )

        # Apply enhanced metadata filtering if advanced operators are detected
        if self._has_advanced_operators(effective_filters):
            processed_filters = self._process_metadata_filters(effective_filters)
            # Remove logical/operator keys that have been reprocessed
            for logical_key in ("AND", "OR", "NOT"):
                effective_filters.pop(logical_key, None)
            for fk in list(effective_filters.keys()):
                if fk not in ("AND", "OR", "NOT", "user_id", "agent_id", "run_id") and isinstance(
                    effective_filters.get(fk), dict
                ):
                    effective_filters.pop(fk, None)
            effective_filters.update(processed_filters)

        artifacts = self._search_vector_store_detailed(
            query,
            effective_filters,
            retrieval_limit,
            threshold,
            score_debug,
        )
        original_memories = artifacts.memories
        if sparse_request_options is not None:
            original_memories = original_memories[:retrieval_limit]
        response: Dict[str, Any] = {}

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker:
            if sparse_request_options is not None:
                sparse_result = _rerank_sparse_omp_candidates(
                    reranker=self.reranker,
                    embedding_model=self.embedding_model,
                    vector_store=self.vector_store,
                    vector_store_provider=self.config.vector_store.provider,
                    query=query,
                    memories=original_memories,
                    limit=limit,
                    query_embedding=artifacts.query_embedding,
                    query_embedding_ms=artifacts.query_embedding_ms,
                    hybrid_candidate_retrieval_ms=artifacts.hybrid_candidate_retrieval_ms,
                    score_debug=score_debug,
                    rerank_options=sparse_request_options,
                )
                original_memories = sparse_result["results"]
                response["rerank_metadata"] = sparse_result["rerank_metadata"]
                if score_debug:
                    response["query_debug"] = {"sparse_omp_performance": sparse_result["performance"]}
            else:
                try:
                    if hasattr(self.reranker, "rerank_detailed"):
                        detailed = self.reranker.rerank_detailed(query, original_memories, limit)
                        original_memories = detailed.get("results", [])
                        response["rerank_metadata"] = detailed.get("rerank_metadata", {})
                        if score_debug:
                            performance = dict(detailed.get("performance", {}))
                            performance.setdefault("query_embedding", artifacts.query_embedding_ms)
                            performance.setdefault(
                                "hybrid_candidate_retrieval",
                                artifacts.hybrid_candidate_retrieval_ms,
                            )
                            response["query_debug"] = {"reranker_performance": performance}
                    else:
                        original_memories = self.reranker.rerank(query, original_memories, limit)
                except Exception as e:
                    if getattr(e, "retryable", None) is False:
                        raise
                    logger.warning(f"Reranking failed, using original results: {e}")

        response["results"] = original_memories[:limit]
        if score_debug:
            query_debug = response.setdefault("query_debug", {})
            detailed_performance = (
                query_debug.get("sparse_omp_performance") or query_debug.get("reranker_performance") or {}
            )
            search_performance = dict(detailed_performance)
            search_performance.update(
                {
                    "server_search_total": (time.perf_counter() - search_started) * 1000.0,
                    "query_embedding": artifacts.query_embedding_ms,
                    "hybrid_candidate_retrieval": artifacts.hybrid_candidate_retrieval_ms,
                    "rerank_total": float(detailed_performance.get("rerank_total", 0.0)),
                }
            )
            query_debug["search_performance"] = search_performance
        return response

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq",
                    "ne": "ne",
                    "gt": "gt",
                    "gte": "gte",
                    "lt": "lt",
                    "lte": "lte",
                    "in": "in",
                    "nin": "nin",
                    "contains": "contains",
                    "icontains": "icontains",
                }

                if operator in operator_map:
                    result.setdefault(key, {})[operator_map[operator]] = value
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        def merge_filters(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            """Merge source into target, deep-merging nested operator dicts for the same key."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key].update(value)
                else:
                    target[key] = value

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        merge_filters(processed_filters, process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(or_condition, process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        merge_filters(not_condition, process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                merge_filters(processed_filters, process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold=0.1, score_debug=False):
        """Compatibility wrapper returning only formatted memories."""
        return self._search_vector_store_detailed(query, filters, limit, threshold, score_debug).memories

    def _search_vector_store_detailed(self, query, filters, limit, threshold=0.1, score_debug=False):
        retrieval_started = time.perf_counter()
        query_embedding = None
        query_embedding_ms = 0.0
        enable_bm25 = bool(getattr(self.config.retrieval, "enable_bm25", True))
        enable_entity_boost = bool(getattr(self.config.retrieval, "enable_entity_boost", True))

        # Step 1: Preprocess query only for enabled retrieval signals.
        query_lemmatized = lemmatize_for_bm25(query) if enable_bm25 else None
        query_entities = extract_entities(query) if enable_entity_boost else []

        # Step 2: Semantic search (over-fetch for scoring pool)
        internal_limit = max(limit * 4, 60)
        semantic_score_mode = self.config.retrieval.semantic_score_mode
        if semantic_score_mode == "embedding":
            embedding_started = time.perf_counter()
            embeddings = self.embedding_model.embed(query, "search")
            query_embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
            query_embedding = embeddings
            semantic_results = self.vector_store.search(
                query=query, vectors=embeddings, top_k=internal_limit, filters=filters
            )
            candidates = []
            for mem in semantic_results:
                mem_id = str(mem.id)
                candidates.append(
                    {
                        "id": mem_id,
                        "score": mem.score,
                        "payload": mem.payload if hasattr(mem, "payload") else {},
                        "score_breakdown": {"semantic_score_source": "embedding"},
                    }
                )
        else:
            raise ValueError(f"Unsupported semantic_score_mode: {semantic_score_mode}")

        # Step 3/4: Keyword search and BM25 scoring.
        bm25_scores = {}
        bm25_payloads = {}
        if enable_bm25:
            keyword_results = self.vector_store.keyword_search(
                query=query_lemmatized, top_k=internal_limit, filters=filters
            )
        else:
            keyword_results = None
        if keyword_results is not None:
            midpoint, steepness = get_bm25_params(query, lemmatized=query_lemmatized)
            for mem in keyword_results:
                mem_id = str(mem.id) if hasattr(mem, "id") else str(mem.get("id", ""))
                raw_score = mem.score if hasattr(mem, "score") else mem.get("score", 0)
                if raw_score and raw_score > 0:
                    bm25_scores[mem_id] = normalize_bm25(raw_score, midpoint, steepness)
                    bm25_payloads[mem_id] = mem.payload if hasattr(mem, "payload") else mem.get("payload", {})

        # Step 5: Compute entity boosts
        entity_boosts = {}
        if enable_entity_boost and query_entities:
            entity_boosts = self._compute_entity_boosts(query_entities, filters)

        ranking_weights = {
            "semantic": 1.0,
            "bm25": 1.0 if enable_bm25 else 0.0,
            "entity": 1.0 if enable_entity_boost else 0.0,
        }

        # Step 6: Score and rank
        scored_results = score_and_rank(
            semantic_results=candidates,
            bm25_scores=bm25_scores,
            entity_boosts=entity_boosts,
            threshold=threshold,
            top_k=limit,
            bm25_payloads=bm25_payloads,
            score_debug=score_debug,
            weights=ranking_weights,
        )

        # Step 7: Format results
        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {
            "data",
            "hash",
            "created_at",
            "updated_at",
            "id",
            "text_lemmatized",
            "attributed_to",
            *promoted_payload_keys,
        }

        original_memories = []
        for scored in scored_results:
            payload = scored.get("payload") or {}

            if not payload.get("data"):
                continue  # Skip candidates with no payload data

            memory_item_dict = MemoryItem(
                id=scored["id"],
                memory=payload.get("data", ""),
                hash=payload.get("hash"),
                created_at=payload.get("created_at"),
                updated_at=payload.get("updated_at"),
                score=scored["score"],
            ).model_dump()

            for key in promoted_payload_keys:
                if key in payload:
                    memory_item_dict[key] = payload[key]

            additional_metadata = {k: v for k, v in payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                if not memory_item_dict.get("metadata"):
                    memory_item_dict["metadata"] = {}
                memory_item_dict["metadata"].update(additional_metadata)
            if scored.get("score_breakdown"):
                memory_item_dict["score_breakdown"] = scored["score_breakdown"]

            original_memories.append(memory_item_dict)

        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        return SearchArtifacts(
            memories=original_memories,
            query_embedding=query_embedding,
            query_embedding_ms=query_embedding_ms,
            hybrid_candidate_retrieval_ms=max(retrieval_ms - query_embedding_ms, 0.0),
        )

    def _compute_entity_boosts(self, query_entities, filters):
        """Compute per-memory entity boosts from entity store search.

        For each extracted entity from the query:
        1. Embed the entity text
        2. Search the entity store (threshold >= 0.5)
        3. For each matched entity, boost its linked memories

        Returns:
            Dict mapping memory_id (str) -> max entity boost [0, 0.5].
        """
        # Deduplicate entities (max 8)
        seen = set()
        deduped = []
        for entity_type, entity_text in query_entities[:8]:
            key = entity_text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append((entity_type, entity_text))

        if not deduped:
            return {}

        search_filters = {k: v for k, v in filters.items() if k in ("user_id", "agent_id", "run_id") and v}
        memory_boosts = {}

        try:
            for _, entity_text in deduped:
                entity_embedding = self.embedding_model.embed(entity_text, "search")
                matches = self.entity_store.search(
                    query=entity_text,
                    vectors=entity_embedding,
                    top_k=500,
                    filters=search_filters,
                )

                for match in matches:
                    similarity = match.score if hasattr(match, "score") else 0.0
                    if similarity < 0.5:
                        continue

                    payload = match.payload if hasattr(match, "payload") else {}
                    linked_memory_ids = payload.get("linked_memory_ids", [])
                    if not isinstance(linked_memory_ids, list):
                        continue

                    # Spread-attenuated boost: entities linking to many memories get attenuated
                    num_linked = max(len(linked_memory_ids), 1)
                    memory_count_weight = 1.0 / (1.0 + 0.001 * ((num_linked - 1) ** 2))
                    boost = similarity * ENTITY_BOOST_WEIGHT * memory_count_weight

                    for memory_id in linked_memory_ids:
                        if memory_id:
                            memory_key = str(memory_id)
                            memory_boosts[memory_key] = max(memory_boosts.get(memory_key, 0.0), boost)

        except Exception as e:
            logger.warning(f"Entity boost computation failed: {e}")

        return memory_boosts

    def update(self, memory_id, data, metadata: Optional[Dict[str, Any]] = None):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.
            metadata (dict, optional): Metadata to update with the memory. Defaults to None.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings, metadata)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """

        existing_memory = self.vector_store.get(vector_id=memory_id)
        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found")

        self._delete_memory(memory_id, existing_memory)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        # delete all vector memories and reset the collections
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        new_metadata = deepcopy(metadata) if metadata is not None else {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        if "created_at" not in new_metadata:
            new_metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        new_metadata["updated_at"] = new_metadata["created_at"]
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[new_metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=new_metadata.get("created_at"),
            updated_at=new_metadata.get("updated_at"),
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        if existing_memory is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["text_lemmatized"] = lemmatize_for_bm25(data)
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )

        # Entity-store cleanup: strip this memory's id from old-text entities,
        # then re-extract entities from the new text and link them back.
        session_filters = {k: new_metadata[k] for k in ("user_id", "agent_id", "run_id") if new_metadata.get(k)}
        self._remove_memory_from_entity_store(memory_id, session_filters)
        self._link_entities_for_memory(memory_id, data, session_filters)

        return memory_id

    def _delete_memory(self, memory_id, existing_memory=None):
        logger.info(f"Deleting memory with {memory_id=}")
        if existing_memory is None:
            existing_memory = self.vector_store.get(vector_id=memory_id)
            if existing_memory is None:
                raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data", "")
        created_at = _normalize_iso_timestamp_to_utc(existing_memory.payload.get("created_at"))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = existing_memory.payload or {}
        session_filters = {k: payload[k] for k in ("user_id", "agent_id", "run_id") if payload.get(k)}
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            created_at=created_at,
            updated_at=updated_at,
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        # Entity-store cleanup: strip this memory's id from any entity records
        # that linked to it. Non-fatal — the helper swallows errors.
        self._remove_memory_from_entity_store(memory_id, session_filters)

        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.reset()
            self.db.connection.close()

        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        # Reset entity store if initialized
        if self._entity_store is not None:
            try:
                self._entity_store.reset()
            except Exception as e:
                logger.warning(f"Failed to reset entity store: {e}")
            self._entity_store = None

    def close(self):
        """Release resources held by this Memory instance (SQLite connections, etc.)."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None
