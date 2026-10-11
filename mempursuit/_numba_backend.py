"""Optional float64 simplex kernel used for the reported latency setting."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    import scipy.linalg  # noqa: F401 -- enables Numba's LAPACK bindings
except ImportError as exc:
    raise ImportError(
        "The Numba backend requires the optional dependencies: "
        "pip install 'mempursuit[fast]'"
    ) from exc


@njit(cache=True, fastmath=False)
def _kkt_status(hessian, linear, solution, active, feasibility_tolerance, kkt_tolerance):
    gradient = hessian @ solution - linear
    multiplier = -np.mean(gradient[active])
    stationary = (
        np.max(np.abs(gradient[active] + multiplier)) <= kkt_tolerance
        and abs(solution.sum() - 1.0) <= kkt_tolerance
        and np.min(solution) >= -feasibility_tolerance
    )
    violating = -1
    minimum = -kkt_tolerance
    for index in range(solution.size):
        reduced_gradient = gradient[index] + multiplier
        if not active[index] and reduced_gradient < minimum:
            minimum = reduced_gradient
            violating = index
    return stationary, violating


@njit(cache=True, nogil=True, fastmath=False)
def solve_simplex(
    hessian,
    linear,
    solution,
    use_warm_start,
    feasibility_tolerance,
    kkt_tolerance,
    max_iterations,
):
    """Return ``(solution, status)``; status 1 requests the NumPy fallback."""

    size = solution.size
    active = solution > feasibility_tolerance
    if not np.any(active):
        active[np.argmax(solution)] = True
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
                return solution, 0
            active[violating] = True

    kkt_buffer = np.empty((size + 1) ** 2, dtype=np.float64)
    rhs_buffer = np.empty(size + 1, dtype=np.float64)
    direction = np.empty(size, dtype=np.float64)
    for _ in range(max_iterations):
        indices = np.flatnonzero(active)
        count = indices.size
        kkt = kkt_buffer[: (count + 1) ** 2].reshape((count + 1, count + 1))
        rhs = rhs_buffer[: count + 1]
        for row in range(count):
            rhs[row] = linear[indices[row]]
            for column in range(count):
                kkt[row, column] = hessian[indices[row], indices[column]]
            kkt[row, count] = 1.0
            kkt[count, row] = 1.0
        kkt[count, count] = 0.0
        rhs[count] = 1.0
        try:
            solved = np.linalg.solve(kkt, rhs)
        except Exception:
            return solution, 1
        candidate = solved[:count]
        if not np.all(np.isfinite(candidate)):
            return solution, 1

        if np.any(candidate < -feasibility_tolerance):
            step_size = 1.0
            decreasing = False
            for position in range(count):
                current = solution[indices[position]]
                direction[position] = candidate[position] - current
                if direction[position] < -feasibility_tolerance:
                    decreasing = True
                    step_size = min(step_size, current / -direction[position])
            if not decreasing:
                return solution, 1
            for position in range(count):
                value = solution[indices[position]] + step_size * direction[position]
                direction[position] = 0.0 if abs(value) <= feasibility_tolerance else max(value, 0.0)
            solution.fill(0.0)
            for position in range(count):
                solution[indices[position]] = direction[position]
            total = solution.sum()
            if total <= 1e-12:
                return solution, 1
            solution /= total
            active = solution > feasibility_tolerance
            continue

        solution.fill(0.0)
        for position in range(count):
            solution[indices[position]] = max(candidate[position], 0.0)
        total = solution.sum()
        if total <= 1e-12:
            return solution, 1
        solution /= total
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
            return solution, 0
    return solution, 1


def warmup() -> None:
    """Compile or load the float64 kernel before latency measurement."""

    _, status = solve_simplex(
        np.eye(2),
        np.asarray([0.7, 0.3]),
        np.asarray([1.0, 0.0]),
        True,
        1e-10,
        1e-9,
        100,
    )
    if status:
        raise RuntimeError("Numba simplex warmup failed")

