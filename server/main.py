"""Self-hosted HTTP API for MemPursuit memory."""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger("mem0-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Priority:
#   1. /app/config.yaml  (custom config file, mounted via docker-compose)
#   2. Environment variables (simple defaults for OpenAI-only setup)
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(os.getenv("MEM0_CONFIG_PATH", "configs/mempursuit.yaml"))


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} references in string values."""
    if isinstance(obj, str):
        import re

        def _replace(m: re.Match) -> str:
            return os.getenv(m.group(1), m.group(0))

        return re.sub(r"\$\{(\w+)\}", _replace, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


DESTRUCTIVE_ENDPOINTS_DISABLED = _is_truthy(os.getenv("DISABLE_DESTRUCTIVE_ENDPOINTS"))


def _parse_optional_bool_env(env_name: str) -> bool | None:
    value = os.getenv(env_name)
    if value in (None, ""):
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    logger.warning("Ignoring invalid %s=%r; expected one of 1,true,yes,on,0,false,no,off", env_name, value)
    return None


def _apply_retrieval_env_overrides(cfg: dict) -> dict:
    """Apply semantic recall overrides independently from final reranking."""
    retrieval_cfg = cfg.setdefault("retrieval", {})
    retrieval_cfg.setdefault("semantic_score_mode", "embedding")

    semantic_score_mode = os.getenv("SEMANTIC_SCORE_MODE")
    if semantic_score_mode not in (None, ""):
        retrieval_cfg["semantic_score_mode"] = semantic_score_mode

    for env_name, config_key in {
        "ENABLE_BM25": "enable_bm25",
        "ENABLE_ENTITY_BOOST": "enable_entity_boost",
    }.items():
        parsed = _parse_optional_bool_env(env_name)
        if parsed is not None:
            retrieval_cfg[config_key] = parsed

    return cfg


def _apply_sparse_omp_reranker_env_overrides(cfg: dict) -> dict:
    """Override only an explicitly configured sparse OMP reranker."""
    reranker_cfg = cfg.get("reranker", {}) or {}
    if reranker_cfg.get("provider") != "sparse_omp":
        return cfg
    provider_cfg = reranker_cfg.get("config")
    if not isinstance(provider_cfg, dict):
        provider_cfg = {}
        reranker_cfg["config"] = provider_cfg
    for env_name, config_key in {
        "SPARSE_OMP_MAX_SUPPORT": "max_support",
        "SPARSE_OMP_CANDIDATE_POOL_SIZE": "candidate_pool_size",
        "SPARSE_OMP_MAX_CONCURRENCY": "max_concurrency",
    }.items():
        raw = os.getenv(env_name)
        if raw in (None, ""):
            continue
        try:
            parsed = int(raw)
            if parsed <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid %s=%r; expected a positive integer", env_name, raw)
            continue
        provider_cfg[config_key] = parsed
    return cfg


def _apply_qdrant_env_overrides(cfg: dict) -> dict:
    """Apply dense-vector cache and strict BM25 settings to a Qdrant vector store."""
    vector_store_cfg = cfg.get("vector_store", {}) or {}
    if vector_store_cfg.get("provider") != "qdrant":
        return cfg
    provider_cfg = vector_store_cfg.setdefault("config", {})

    raw = os.getenv("QDRANT_DENSE_VECTOR_CACHE_SIZE")
    if raw not in (None, ""):
        try:
            parsed = int(raw)
            if parsed < 0:
                raise ValueError
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid QDRANT_DENSE_VECTOR_CACHE_SIZE=%r; expected a nonnegative integer",
                raw,
            )
        else:
            provider_cfg["dense_vector_cache_size"] = parsed

    bm25_required = _parse_optional_bool_env("QDRANT_BM25_REQUIRED")
    if bm25_required is not None:
        provider_cfg["bm25_required"] = bm25_required
    return cfg


def _apply_env_overrides(cfg: dict) -> dict:
    return _apply_sparse_omp_reranker_env_overrides(_apply_retrieval_env_overrides(_apply_qdrant_env_overrides(cfg)))


def _sparse_omp_reranker_status(cfg: dict) -> tuple[int | None, int | None]:
    reranker = cfg.get("reranker", {}) or {}
    if reranker.get("provider") != "sparse_omp":
        return None, None
    provider_cfg = reranker.get("config", {}) or {}
    max_support = provider_cfg.get("max_support")
    return (
        int(provider_cfg.get("candidate_pool_size", 200)),
        int(max_support) if max_support is not None else None,
    )


def _bm25_status() -> dict[str, Any]:
    """Return Qdrant BM25 readiness without scanning collection points."""
    vector_store_cfg = config.get("vector_store", {}).get("config") or {}
    fallback = {
        "required": bool(vector_store_cfg.get("bm25_required", False)),
        "slot_present": None,
        "encoder_ready": None,
        "ready": None,
        "error": None,
    }
    if memory_instance is None:
        return fallback
    status_fn = getattr(memory_instance.vector_store, "bm25_status", None)
    if not callable(status_fn):
        fallback["error"] = "configured vector store does not expose BM25 readiness"
        return fallback
    return status_fn()


def _load_config() -> dict:
    """Load the memory configuration with environment-based path overrides."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Memory config does not exist: {CONFIG_PATH}")
    cfg = _expand_env_vars(yaml.safe_load(CONFIG_PATH.read_text()))
    cfg.setdefault("version", "v1.1")
    vector = cfg.setdefault("vector_store", {}).setdefault("config", {})
    if os.getenv("QDRANT_HOST"):
        vector["host"] = os.environ["QDRANT_HOST"]
    if os.getenv("QDRANT_PORT"):
        vector["port"] = int(os.environ["QDRANT_PORT"])
    if os.getenv("COLLECTION_NAME"):
        vector["collection_name"] = os.environ["COLLECTION_NAME"]
    cfg["history_db_path"] = os.getenv("HISTORY_DB_PATH", cfg.get("history_db_path", ".mempursuit/history.db"))
    Path(cfg["history_db_path"]).parent.mkdir(parents=True, exist_ok=True)
    return _apply_env_overrides(cfg)


config = _load_config()
logger.info(
    "Resolved reranker config: provider=%s, sparse_omp.candidate_pool_size=%s, sparse_omp.max_support=%s",
    (config.get("reranker") or {}).get("provider", "none"),
    *_sparse_omp_reranker_status(config),
)

memory_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_instance
    from mem0 import Memory

    memory_instance = Memory.from_config(config)
    logger.info("MemPursuit memory server ready")
    try:
        yield
    finally:
        memory_instance.close()
        memory_instance = None


app = FastAPI(
    title="MemPursuit",
    description="Open-source memory server backed by Qdrant",
    lifespan=lifespan,
)


def _get_memory():
    if memory_instance is None:
        raise HTTPException(503, "Memory not initialised")
    return memory_instance


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AddRequest(BaseModel):
    messages: list[dict[str, Any]]
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] | None = None
    timestamp: int | float | str | None = None
    observation_date: str | None = None
    custom_instructions: str | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)
    threshold: float | None = Field(default=None, le=1.0)
    filters: dict[str, Any] | None = None
    rerank: bool = False
    rerank_options: dict[str, Any] | None = None
    score_debug: bool = False


class UpdateRequest(BaseModel):
    data: str


def _build_entity_filters(
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build filters dict for mem0 APIs that reject top-level entity params."""
    filters: dict[str, Any] = {}
    if user_id:
        filters["user_id"] = user_id
    if agent_id:
        filters["agent_id"] = agent_id
    if run_id:
        filters["run_id"] = run_id
    if extra:
        filters.update(extra)
    return filters


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/memories")
def add_memories(req: AddRequest):
    """Add memories extracted from a conversation."""
    mem = _get_memory()
    params: dict[str, Any] = {}
    if req.user_id:
        params["user_id"] = req.user_id
    if req.agent_id:
        params["agent_id"] = req.agent_id
    if req.run_id:
        params["run_id"] = req.run_id
    metadata = dict(req.metadata or {})
    if req.timestamp is not None:
        try:
            timestamp_value = float(req.timestamp)
            metadata["timestamp"] = timestamp_value
            metadata["created_at"] = datetime.fromtimestamp(timestamp_value, tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            metadata["timestamp"] = req.timestamp
    if metadata:
        params["metadata"] = metadata
    # observation_date and custom_instructions: pass through only if
    # the memory implementation supports them
    if req.custom_instructions:
        params["prompt"] = req.custom_instructions

    try:
        result = mem.add(req.messages, **params)
        return result
    except Exception as e:
        logger.exception("add() failed")
        raise HTTPException(500, str(e))


@app.post("/search")
def search_memories(req: SearchRequest):
    """Search memories by semantic similarity + BM25 + entity signals."""
    mem = _get_memory()
    filters = _build_entity_filters(
        user_id=req.user_id,
        agent_id=req.agent_id,
        run_id=req.run_id,
        extra=req.filters,
    )
    if not filters:
        raise HTTPException(400, "Provide at least one of: user_id, agent_id, run_id")

    params: dict[str, Any] = {"top_k": req.limit, "filters": filters}
    fields_set = req.model_fields_set if hasattr(req, "model_fields_set") else req.__fields_set__
    if "threshold" in fields_set:
        params["threshold"] = None if req.threshold is None or req.threshold < 0 else req.threshold
    if req.rerank:
        params["rerank"] = True
        if req.rerank_options is not None:
            params["rerank_options"] = req.rerank_options
    if req.score_debug:
        params["score_debug"] = True

    try:
        result = mem.search(req.query, **params)
        return result
    except Exception as e:
        if getattr(e, "retryable", None) is False and getattr(e, "code", None):
            logger.exception("Non-retryable reranker search() failure")
            raise HTTPException(
                status_code=500,
                detail={
                    "code": e.code,
                    "retryable": e.retryable,
                    "message": str(e),
                    "performance": getattr(e, "performance", {}),
                },
            ) from e
        if isinstance(e, ValueError):
            raise HTTPException(400, str(e)) from e
        logger.exception("search() failed")
        raise HTTPException(500, str(e)) from e


@app.get("/memories")
def get_memories(
    user_id: str | None = Query(None),
    agent_id: str | None = Query(None),
    run_id: str | None = Query(None),
):
    """List all memories for a given user/agent/run."""
    mem = _get_memory()
    filters = _build_entity_filters(user_id=user_id, agent_id=agent_id, run_id=run_id)
    if not filters:
        raise HTTPException(400, "Provide at least one of: user_id, agent_id, run_id")

    try:
        return mem.get_all(filters=filters)
    except Exception as e:
        logger.exception("get_all() failed")
        raise HTTPException(500, str(e))


@app.get("/memories/{memory_id}")
def get_memory(memory_id: str):
    """Get a single memory by ID."""
    mem = _get_memory()
    try:
        return mem.get(memory_id)
    except Exception as e:
        raise HTTPException(404, str(e))


@app.put("/memories/{memory_id}")
def update_memory(memory_id: str, req: UpdateRequest):
    """Update a memory's content."""
    mem = _get_memory()
    try:
        return mem.update(memory_id, data=req.data)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str):
    """Delete a single memory."""
    if DESTRUCTIVE_ENDPOINTS_DISABLED:
        raise HTTPException(403, "Destructive endpoints are disabled for this service")
    mem = _get_memory()
    try:
        mem.delete(memory_id)
        return {"message": "Memory deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/memories")
def delete_all_memories(
    user_id: str | None = Query(None),
    agent_id: str | None = Query(None),
    run_id: str | None = Query(None),
):
    """Delete all memories for a user/agent/run."""
    if DESTRUCTIVE_ENDPOINTS_DISABLED:
        raise HTTPException(403, "Destructive endpoints are disabled for this service")
    mem = _get_memory()
    params: dict[str, Any] = {}
    if user_id:
        params["user_id"] = user_id
    if agent_id:
        params["agent_id"] = agent_id
    if run_id:
        params["run_id"] = run_id

    if not params:
        raise HTTPException(400, "Provide at least one of: user_id, agent_id, run_id")

    try:
        mem.delete_all(**params)
        return {"message": "All memories deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/memories/{memory_id}/history")
def memory_history(memory_id: str):
    """Get mutation history for a memory."""
    mem = _get_memory()
    try:
        return mem.history(memory_id)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/reset")
def reset_all():
    """Reset ALL memories. Use with caution."""
    if DESTRUCTIVE_ENDPOINTS_DISABLED:
        raise HTTPException(403, "Destructive endpoints are disabled for this service")
    mem = _get_memory()
    try:
        mem.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    """Health check."""
    bm25_status = _bm25_status()
    return {
        "status": "ok",
        "llm": config.get("llm", {}).get("provider", "?"),
        "embedder": config.get("embedder", {}).get("provider", "?"),
        "semantic_score_mode": config.get("retrieval", {}).get("semantic_score_mode", "embedding"),
        "enable_bm25": config.get("retrieval", {}).get("enable_bm25", True),
        "enable_rrf": False,
        "bm25_required": bm25_status["required"],
        "bm25_slot_present": bm25_status["slot_present"],
        "bm25_encoder_ready": bm25_status["encoder_ready"],
        "bm25_ready": bm25_status["ready"],
        "bm25_error": bm25_status["error"],
        "enable_entity_boost": config.get("retrieval", {}).get("enable_entity_boost", True),
        "reranker": config.get("reranker", {}).get("provider", "none"),
        "reranker_model": (config.get("reranker", {}).get("config") or {}).get("model"),
        "reranker_candidate_pool_size": (config.get("reranker", {}).get("config") or {}).get("candidate_pool_size"),
        "sparse_omp_candidate_pool_size": _sparse_omp_reranker_status(config)[0],
        "sparse_omp_max_support": _sparse_omp_reranker_status(config)[1],
        "sparse_omp_early_stopping": (config.get("reranker", {}).get("config") or {}).get("early_stopping"),
        "sparse_omp_max_concurrency": (config.get("reranker", {}).get("config") or {}).get("max_concurrency"),
        "sparse_omp_normalize_vectors": (config.get("reranker", {}).get("config") or {}).get("normalize_vectors"),
        "sparse_omp_ridge": (config.get("reranker", {}).get("config") or {}).get("ridge"),
        "sparse_omp_negative_tolerance": (config.get("reranker", {}).get("config") or {}).get("negative_tolerance"),
        "qdrant_dense_vector_cache_size": (config.get("vector_store", {}).get("config") or {}).get(
            "dense_vector_cache_size", 8192
        ),
        "qdrant_host": (config.get("vector_store", {}).get("config") or {}).get("host"),
        "collection_name": (config.get("vector_store", {}).get("config") or {}).get("collection_name"),
        "llm_model": (config.get("llm", {}).get("config") or {}).get("model"),
        "embedder_model": (config.get("embedder", {}).get("config") or {}).get("model"),
        "destructive_endpoints_disabled": DESTRUCTIVE_ENDPOINTS_DISABLED,
        "openblas_num_threads": int(os.getenv("OPENBLAS_NUM_THREADS", "0")),
        "omp_num_threads": int(os.getenv("OMP_NUM_THREADS", "0")),
        "mkl_num_threads": int(os.getenv("MKL_NUM_THREADS", "0")),
        "numexpr_num_threads": int(os.getenv("NUMEXPR_NUM_THREADS", "0")),
    }


@app.get("/")
def root():
    return {"message": "Mem0 OSS Server", "docs": "/docs"}
