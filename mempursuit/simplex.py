"""Active-set solver for the simplex-constrained refit used by MemPursuit."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


_NUMERICAL_EPS = 1e-12


def solve_simplex_qp(
    gram: ArrayLike,
    linear: ArrayLike,
    *,
    ridge: float = 1e-10,
    warm_start: ArrayLike | None = None,
    feasibility_tolerance: float = 1e-10,
    backend: Literal["numpy", "numba"] = "numpy",
) -> NDArray[np.float64]:
    """Solve the fully corrective MemPursuit refit.

    This minimizes

        0.5 * alpha.T @ (gram + ridge * I) @ alpha - linear.T @ alpha

    subject to ``alpha >= 0`` and ``sum(alpha) == 1``. The omitted constant
    is half the squared norm of the query embedding.
    """

    matrix = np.asarray(gram, dtype=np.float64)
    vector = np.asarray(linear, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be a square matrix")
    if vector.shape != (matrix.shape[0],):
        raise ValueError("linear must have shape (gram.shape[0],)")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("gram and linear must contain only finite values")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    if feasibility_tolerance < 0:
        raise ValueError("feasibility_tolerance must be nonnegative")
    if backend not in {"numpy", "numba"}:
        raise ValueError("backend must be 'numpy' or 'numba'")

    size = matrix.shape[0]
    if size == 0:
        return np.zeros(0, dtype=np.float64)
    if size == 1:
        return np.ones(1, dtype=np.float64)

    hessian = np.ascontiguousarray(0.5 * (matrix + matrix.T))
    hessian.flat[:: size + 1] += float(ridge)
    tolerance = max(float(feasibility_tolerance), _NUMERICAL_EPS)
    kkt_tolerance = max(10.0 * tolerance, 1e-9)

    solution = _feasible_warm_start(warm_start, size)
    use_warm_start = solution is not None
    if solution is None:
        solution = np.zeros(size, dtype=np.float64)
        vertex_objectives = 0.5 * np.diag(hessian) - vector
        solution[int(np.argmin(vertex_objectives))] = 1.0

    if backend == "numba":
        from ._numba_backend import solve_simplex

        result, status = solve_simplex(
            hessian,
            np.ascontiguousarray(vector),
            solution.copy(),
            use_warm_start,
            tolerance,
            kkt_tolerance,
            max(100, 10 * size),
        )
        if status == 0:
            return result
        # Singular or numerically difficult systems use the robust NumPy path.

    return _solve_numpy(
        hessian,
        vector,
        solution,
        use_warm_start=use_warm_start,
        feasibility_tolerance=tolerance,
        kkt_tolerance=kkt_tolerance,
    )


def _feasible_warm_start(warm_start: ArrayLike | None, size: int) -> NDArray[np.float64] | None:
    if warm_start is None:
        return None
    value = np.asarray(warm_start, dtype=np.float64)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        return None
    value = np.maximum(value, 0.0)
    total = float(value.sum())
    if total <= _NUMERICAL_EPS:
        return None
    return value / total


def _kkt_status(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64],
    solution: NDArray[np.float64],
    active: NDArray[np.bool_],
    feasibility_tolerance: float,
    kkt_tolerance: float,
) -> tuple[bool, int]:
    gradient = hessian @ solution - linear
    multiplier = -float(np.mean(gradient[active]))
    active_stationary = (
        float(np.max(np.abs(gradient[active] + multiplier))) <= kkt_tolerance
        and abs(float(solution.sum()) - 1.0) <= kkt_tolerance
        and float(np.min(solution)) >= -feasibility_tolerance
    )
    excluded = np.flatnonzero(~active)
    violating = -1
    if excluded.size:
        reduced_gradient = gradient[excluded] + multiplier
        position = int(np.argmin(reduced_gradient))
        if reduced_gradient[position] < -kkt_tolerance:
            violating = int(excluded[position])
    return active_stationary, violating


def _solve_numpy(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64],
    solution: NDArray[np.float64],
    *,
    use_warm_start: bool,
    feasibility_tolerance: float,
    kkt_tolerance: float,
) -> NDArray[np.float64]:
    size = solution.size
    active = solution > feasibility_tolerance
    if not np.any(active):
        active[int(np.argmax(solution))] = True

    if use_warm_start and np.all(solution[~active] == 0.0):
        stationary, violating = _kkt_status(
            hessian,
            linear,
            solution,
            active,
            feasibility_tolerance,
            kkt_tolerance,
        )
        if stationary:
            if violating < 0:
                return solution
            active[violating] = True

    workspace = (
        np.empty((size + 1) ** 2, dtype=np.float64),
        np.empty(size + 1, dtype=np.float64),
    )
    max_iterations = max(100, 10 * size)
    for _ in range(max_iterations):
        active_indices = np.flatnonzero(active)
        active_solution = _solve_equality_kkt(
            hessian[np.ix_(active_indices, active_indices)],
            linear[active_indices],
            workspace=workspace,
        )

        if np.any(active_solution < -feasibility_tolerance):
            current = solution[active_indices]
            direction = active_solution - current
            decreasing = direction < -feasibility_tolerance
            if not np.any(decreasing):
                raise RuntimeError("simplex working set produced an infeasible direction")
            step_size = min(float(np.min(current[decreasing] / -direction[decreasing])), 1.0)
            next_active = current + step_size * direction
            next_active[np.abs(next_active) <= feasibility_tolerance] = 0.0
            solution.fill(0.0)
            solution[active_indices] = np.maximum(next_active, 0.0)
            _renormalize(solution)
            active = solution > feasibility_tolerance
            continue

        solution.fill(0.0)
        solution[active_indices] = np.maximum(active_solution, 0.0)
        _renormalize(solution)
        active = solution > feasibility_tolerance
        stationary, violating = _kkt_status(
            hessian,
            linear,
            solution,
            active,
            feasibility_tolerance,
            kkt_tolerance,
        )
        if violating >= 0:
            active[violating] = True
            continue
        if stationary:
            return solution

    raise RuntimeError(f"simplex working set did not converge in {max_iterations} iterations")


def _renormalize(solution: NDArray[np.float64]) -> None:
    total = float(solution.sum())
    if total <= _NUMERICAL_EPS:
        raise RuntimeError("simplex working set lost primal feasibility")
    solution /= total


def _solve_equality_kkt(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64],
    *,
    workspace: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Solve an equality-constrained QP on the current positive support."""

    size = hessian.shape[0]
    kkt = workspace[0][: (size + 1) ** 2].reshape(size + 1, size + 1)
    kkt[:size, :size] = hessian
    kkt[:size, size] = 1.0
    kkt[size, :size] = 1.0
    kkt[size, size] = 0.0
    rhs = workspace[1][: size + 1]
    rhs[:size] = linear
    rhs[size] = 1.0
    try:
        return np.linalg.solve(kkt, rhs)[:-1]
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]

