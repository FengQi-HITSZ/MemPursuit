"""Standalone implementation of residual-guided memory composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .simplex import solve_simplex_qp


@dataclass(frozen=True)
class PursuitStep:
    """Diagnostics recorded after one select-and-refit step."""

    candidate_index: int
    selection_score: float
    residual_norm: float
    squared_error: float
    relative_improvement: float
    squared_error_reduction: float
    weights: NDArray[np.float64]


@dataclass(frozen=True)
class MemPursuitResult:
    """Selected candidate indices in entry order and their final refit weights."""

    selected_indices: NDArray[np.int64]
    weights: NDArray[np.float64]
    steps: tuple[PursuitStep, ...]
    stop_reason: Literal["budget", "relative_improvement", "frank_wolfe_gap"]


def select(
    query_embedding: ArrayLike,
    candidate_embeddings: ArrayLike,
    budget: int,
    *,
    ridge: float = 1e-10,
    normalize: bool = True,
    adaptive: bool = False,
    relative_improvement_tolerance: float = 1e-5,
    patience: int = 2,
    numerical_floor: float = 1e-12,
    frank_wolfe_gap_tolerance: float | None = None,
    backend: Literal["numpy", "numba"] = "numpy",
) -> MemPursuitResult:
    """Select a residual-guided prefix from a fixed candidate pool.

    Args:
        query_embedding: One query vector with shape ``(dimension,)``.
        candidate_embeddings: Candidate matrix with shape ``(candidates, dimension)``.
        budget: Maximum number of memories to select. Fixed mode returns exactly
            this many entries in greedy entry order.
        ridge: Ridge coefficient in the fully corrective simplex refit.
        normalize: L2-normalize the query and every candidate before selection.
        adaptive: Enable the relative-improvement stopping rule.
        relative_improvement_tolerance: Threshold for squared-error improvement.
        patience: Number of consecutive low-improvement additions before stopping.
            The low-improvement patience window is discarded from the returned
            prefix, matching the experimental implementation.
        numerical_floor: Denominator floor in the relative-improvement rule.
        frank_wolfe_gap_tolerance: Optional full-simplex optimality threshold.
            ``None`` disables this additional stopping rule.
        backend: ``"numpy"`` for the reference solver or ``"numba"`` for the
            optional accelerated simplex kernel.
    """

    query, candidates = _prepare_embeddings(query_embedding, candidate_embeddings, normalize=normalize)
    candidate_count = candidates.shape[0]
    _validate_options(
        budget=budget,
        candidate_count=candidate_count,
        ridge=ridge,
        adaptive=adaptive,
        relative_improvement_tolerance=relative_improvement_tolerance,
        patience=patience,
        numerical_floor=numerical_floor,
        frank_wolfe_gap_tolerance=frank_wolfe_gap_tolerance,
    )

    # G_ij = <f(m_i), f(m_j)> and b_i = <f(m_i), f(q)> (Appendix A.1).
    gram = np.asarray(candidates @ candidates.T, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    linear = np.asarray(candidates @ query, dtype=np.float64)
    query_norm_squared = float(query @ query)

    # Reuse contiguous buffers across pursuit steps. This avoids repeated
    # full-Gram fancy-index copies in the accelerated setting.
    selected_columns = np.empty((budget, candidate_count), dtype=np.float64)
    selected_gram = np.empty((budget, budget), dtype=np.float64)
    selected_linear = np.empty(budget, dtype=np.float64)
    warm_buffer = np.zeros(budget, dtype=np.float64)
    selected: list[int] = []
    weights = np.zeros(0, dtype=np.float64)
    residual_scores = linear.copy()
    unavailable = np.zeros(candidate_count, dtype=np.bool_)
    previous_squared_error = max(query_norm_squared, 0.0)
    previous_regularized_objective: float | None = None
    low_improvement_count = 0
    steps: list[PursuitStep] = []
    snapshots: list[tuple[list[int], NDArray[np.float64], list[PursuitStep]]] = []
    stop_reason: Literal["budget", "relative_improvement", "frank_wolfe_gap"] = "budget"

    for _ in range(budget):
        conditional_scores = residual_scores.copy()
        conditional_scores[unavailable] = -np.inf
        candidate_index = int(np.argmax(conditional_scores))
        if not np.isfinite(conditional_scores[candidate_index]):
            raise RuntimeError("candidate pool was exhausted before reaching the requested budget")

        next_selected = [*selected, candidate_index]
        selected_array = np.asarray(next_selected, dtype=np.int64)
        previous_size = len(selected)
        next_size = previous_size + 1
        selected_columns[previous_size] = gram[:, candidate_index]
        selected_gram[previous_size, :next_size] = gram[candidate_index, selected_array]
        selected_gram[:next_size, previous_size] = selected_gram[previous_size, :next_size]
        selected_linear[previous_size] = linear[candidate_index]
        active_gram = selected_gram[:next_size, :next_size]
        active_linear = selected_linear[:next_size]
        warm_buffer[:previous_size] = weights
        warm_buffer[previous_size] = 0.0
        warm_start = warm_buffer[:next_size] if selected else None
        next_weights = solve_simplex_qp(
            active_gram,
            active_linear,
            ridge=ridge,
            warm_start=warm_start,
            backend=backend,
        )

        squared_error = (
            query_norm_squared
            - 2.0 * float(next_weights @ active_linear)
            + float(next_weights @ active_gram @ next_weights)
        )
        squared_error = max(squared_error, 0.0)
        regularized_objective = squared_error + ridge * float(next_weights @ next_weights)
        if previous_regularized_objective is not None:
            tolerance = 1e-9 * max(1.0, abs(previous_regularized_objective))
            if regularized_objective > previous_regularized_objective + tolerance:
                raise RuntimeError("fully corrective objective increased beyond numerical tolerance")

        relative_improvement = (previous_squared_error - squared_error) / max(
            previous_squared_error,
            numerical_floor,
        )
        step = PursuitStep(
            candidate_index=candidate_index,
            selection_score=float(conditional_scores[candidate_index]),
            residual_norm=float(np.sqrt(squared_error)),
            squared_error=squared_error,
            relative_improvement=relative_improvement,
            squared_error_reduction=previous_squared_error - squared_error,
            weights=next_weights.copy(),
        )

        selected = next_selected
        weights = next_weights
        unavailable[candidate_index] = True
        steps.append(step)
        snapshots.append((list(selected), weights.copy(), list(steps)))

        # Eq. 9: every candidate is rescored against the updated reconstruction.
        residual_scores = linear - selected_columns[:next_size].T @ weights

        if frank_wolfe_gap_tolerance is not None:
            full_weights = np.zeros(candidate_count, dtype=np.float64)
            full_weights[selected_array] = weights
            gradient = -residual_scores + ridge * full_weights
            frank_wolfe_gap = max(float(full_weights @ gradient - np.min(gradient)), 0.0)
            if frank_wolfe_gap <= frank_wolfe_gap_tolerance:
                stop_reason = "frank_wolfe_gap"
                break

        if adaptive and len(steps) > 1:
            if relative_improvement <= relative_improvement_tolerance:
                low_improvement_count += 1
            else:
                low_improvement_count = 0
            if low_improvement_count >= patience:
                # Return the prefix immediately before the stalled patience window.
                selected, weights, steps = snapshots[-(patience + 1)]
                stop_reason = "relative_improvement"
                break

        previous_squared_error = squared_error
        previous_regularized_objective = regularized_objective

    return MemPursuitResult(
        selected_indices=np.asarray(selected, dtype=np.int64),
        weights=weights.copy(),
        steps=tuple(steps),
        stop_reason=stop_reason,
    )


def _prepare_embeddings(
    query_embedding: ArrayLike,
    candidate_embeddings: ArrayLike,
    *,
    normalize: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    query = np.asarray(query_embedding, dtype=np.float64)
    candidates = np.asarray(candidate_embeddings, dtype=np.float64)
    if query.ndim != 1 or query.size == 0:
        raise ValueError("query_embedding must be a nonempty one-dimensional vector")
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("candidate_embeddings must be a nonempty two-dimensional matrix")
    if candidates.shape[1] != query.shape[0]:
        raise ValueError("query and candidate embedding dimensions must match")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(candidates)):
        raise ValueError("embeddings must contain only finite values")

    query_norm = float(np.linalg.norm(query))
    candidate_norms = np.linalg.norm(candidates, axis=1)
    if query_norm <= 1e-12 or np.any(candidate_norms <= 1e-12):
        raise ValueError("query and candidate embeddings must be nonzero")
    if normalize:
        query = query / query_norm
        candidates = candidates / candidate_norms[:, None]
    return np.ascontiguousarray(query), np.ascontiguousarray(candidates)


def _validate_options(
    *,
    budget: int,
    candidate_count: int,
    ridge: float,
    adaptive: bool,
    relative_improvement_tolerance: float,
    patience: int,
    numerical_floor: float,
    frank_wolfe_gap_tolerance: float | None,
) -> None:
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError("budget must be an integer")
    if budget < 1 or budget > candidate_count:
        raise ValueError("budget must satisfy 1 <= budget <= number of candidates")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if not isinstance(adaptive, bool):
        raise TypeError("adaptive must be a boolean")
    if relative_improvement_tolerance < 0:
        raise ValueError("relative_improvement_tolerance must be nonnegative")
    if isinstance(patience, bool) or not isinstance(patience, (int, np.integer)) or patience < 1:
        raise ValueError("patience must be a positive integer")
    if numerical_floor <= 0:
        raise ValueError("numerical_floor must be positive")
    if frank_wolfe_gap_tolerance is not None and frank_wolfe_gap_tolerance < 0:
        raise ValueError("frank_wolfe_gap_tolerance must be nonnegative or None")
