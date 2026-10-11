"""
Scoring utilities for hybrid retrieval.

Provides:
- **BM25 normalization**: Sigmoid normalization of raw BM25 scores to [0, 1].
- **BM25 parameter selection**: Query-length-adaptive sigmoid parameters.
- **Additive scoring**: Combined scoring with semantic + BM25 + entity boost.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def get_bm25_params(query: str, *, lemmatized: Optional[str] = None) -> tuple:
    """Get BM25 sigmoid parameters based on query length.

    Longer queries tend to have higher raw BM25 scores, so we adjust
    the sigmoid midpoint and steepness accordingly.

    Returns:
        (midpoint, steepness) for sigmoid normalization.
    """
    if lemmatized is None:
        from mem0.utils.lemmatization import lemmatize_for_bm25

        lemmatized = lemmatize_for_bm25(query)
    num_terms = len(lemmatized.split()) if lemmatized else 1

    if num_terms <= 3:
        return 5.0, 0.7
    elif num_terms <= 6:
        return 7.0, 0.6
    elif num_terms <= 9:
        return 9.0, 0.5
    elif num_terms <= 15:
        return 10.0, 0.5
    else:
        return 12.0, 0.5


def normalize_bm25(raw_score: float, midpoint: float, steepness: float) -> float:
    """Normalize BM25 score to [0, 1] using logistic sigmoid.

    Args:
        raw_score: Raw BM25 score (unbounded, typically 0-20+).
        midpoint: Score at which sigmoid outputs 0.5.
        steepness: Controls how quickly sigmoid transitions.

    Returns:
        Normalized score in range [0, 1].
    """
    return 1.0 / (1.0 + math.exp(-steepness * (raw_score - midpoint)))


ENTITY_BOOST_WEIGHT = 0.5


def score_and_rank(
    semantic_results: List[Dict[str, Any]],
    bm25_scores: Dict[str, float],
    entity_boosts: Dict[str, float],
    threshold: Optional[float],
    top_k: int,
    *,
    bm25_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
    score_debug: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Fuse retrieval signals and return top-k results.

    Candidate ids are the union of semantic and BM25 candidates.
    When set, the semantic threshold only removes candidates that have no other
    active retrieval signal. Set threshold to None to disable this filtering.

    Returns:
        List of scored result dicts sorted by combined score descending.
    """
    bm25_payloads = bm25_payloads or {}
    weights = weights or {}
    semantic_weight = float(weights.get("semantic", 1.0))
    bm25_weight = float(weights.get("bm25", 1.0))
    entity_weight = float(weights.get("entity", 1.0))
    has_semantic = bool(semantic_results)
    has_bm25 = bool(bm25_scores)
    has_entity = bool(entity_boosts)

    max_possible = 0.0
    if has_semantic:
        max_possible += semantic_weight
    if has_bm25:
        max_possible += bm25_weight
    if has_entity:
        max_possible += ENTITY_BOOST_WEIGHT * entity_weight
    if max_possible <= 0:
        max_possible = 1.0

    candidates: Dict[str, Dict[str, Any]] = {}
    for result in semantic_results:
        mem_id = result.get("id")
        if mem_id is None:
            continue
        mem_id_str = str(mem_id)
        candidates[mem_id_str] = {
            "id": mem_id_str,
            "semantic": float(result.get("score", 0.0) or 0.0),
            "payload": result.get("payload"),
            "semantic_debug": result.get("score_breakdown") or {},
        }

    for mem_id, payload in bm25_payloads.items():
        mem_id_str = str(mem_id)
        candidates.setdefault(mem_id_str, {"id": mem_id_str, "semantic": 0.0, "payload": payload})
        if not candidates[mem_id_str].get("payload"):
            candidates[mem_id_str]["payload"] = payload

    scored: List[Dict[str, Any]] = []

    for mem_id_str, result in candidates.items():
        semantic_score = float(result.get("semantic", 0.0) or 0.0)
        bm25_score = bm25_scores.get(mem_id_str, 0.0)
        entity_boost = entity_boosts.get(mem_id_str, 0.0)

        if threshold is not None and semantic_score < threshold and not any((bm25_score, entity_boost)):
            continue

        raw_combined = semantic_weight * semantic_score + bm25_weight * bm25_score + entity_weight * entity_boost
        combined = min(raw_combined / max_possible, 1.0)

        item = {
            "id": mem_id_str,
            "score": combined,
            "payload": result.get("payload"),
        }
        if score_debug:
            semantic_detail = result.get("semantic_debug") or {}
            item["score_breakdown"] = {
                "semantic": semantic_score,
                "bm25": bm25_score,
                "entity_boost": entity_boost,
                "combined": combined,
            }
            if semantic_detail:
                item["score_breakdown"].update(semantic_detail)
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
