"""Optional, float64-only simplex working-set kernel (no document objects or logging)."""

from functools import lru_cache

import numpy as np

try:
    from numba import njit
    import scipy.linalg  # noqa: F401 -- Numba's LAPACK bindings require SciPy.
except ImportError as exc:
    raise ImportError("The numba sparse OMP backend requires the mempursuit[fast] extra.") from exc


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
    for i in range(solution.size):
        if not active[i] and gradient[i] + multiplier < minimum:
            minimum = gradient[i] + multiplier
            violating = i
    return stationary, violating


@njit(cache=True, nogil=True, fastmath=False)
def solve_simplex(hessian, linear, solution, use_warm, feasibility_tolerance, kkt_tolerance, max_iterations):
    """Return solution, [iterations, warm skips], KKT size histogram, and status.

    Status 1 asks the Python caller to retry the robust NumPy KKT/lstsq solver.
    All scratch space belongs to this call, including for concurrent requests.
    """
    size = solution.size
    counters = np.zeros(2, dtype=np.int64)
    sizes = np.zeros(size + 1, dtype=np.int64)
    active = solution > feasibility_tolerance
    if not np.any(active):
        active[np.argmax(solution)] = True
    if use_warm and np.all(solution[~active] == 0.0):
        stationary, violating = _kkt_status(hessian, linear, solution, active, feasibility_tolerance, kkt_tolerance)
        if stationary:
            counters[1] += 1
            if violating < 0:
                return solution, counters, sizes, 0
            active[violating] = True

    kkt_buffer = np.empty((size + 1) ** 2, dtype=np.float64)
    rhs_buffer = np.empty(size + 1, dtype=np.float64)
    direction = np.empty(size, dtype=np.float64)
    for _ in range(max_iterations):
        counters[0] += 1
        indices = np.flatnonzero(active)
        count = indices.size
        sizes[count] += 1
        kkt = kkt_buffer[: (count + 1) ** 2].reshape((count + 1, count + 1))
        rhs = rhs_buffer[: count + 1]
        for i in range(count):
            rhs[i] = linear[indices[i]]
            for j in range(count):
                kkt[i, j] = hessian[indices[i], indices[j]]
            kkt[i, count] = 1.0
            kkt[count, i] = 1.0
        kkt[count, count] = 0.0
        rhs[count] = 1.0
        try:
            solved = np.linalg.solve(kkt, rhs)
        except Exception:
            return solution, counters, sizes, 1
        candidate = solved[:count]
        if not np.all(np.isfinite(candidate)):
            return solution, counters, sizes, 1

        if np.any(candidate < -feasibility_tolerance):
            alpha = 1.0
            decreasing = False
            for i in range(count):
                current = solution[indices[i]]
                direction[i] = candidate[i] - current
                if direction[i] < -feasibility_tolerance:
                    decreasing = True
                    alpha = min(alpha, current / -direction[i])
            if not decreasing:
                return solution, counters, sizes, 1
            for i in range(count):
                value = solution[indices[i]] + alpha * direction[i]
                direction[i] = 0.0 if abs(value) <= feasibility_tolerance else max(value, 0.0)
            solution.fill(0.0)
            for i in range(count):
                solution[indices[i]] = direction[i]
            total = solution.sum()
            if total <= 1e-12:
                return solution, counters, sizes, 1
            solution /= total
            active = solution > feasibility_tolerance
            continue

        solution.fill(0.0)
        for i in range(count):
            solution[indices[i]] = max(candidate[i], 0.0)
        total = solution.sum()
        if total <= 1e-12:
            return solution, counters, sizes, 1
        solution /= total
        active = solution > feasibility_tolerance
        stationary, violating = _kkt_status(hessian, linear, solution, active, feasibility_tolerance, kkt_tolerance)
        if violating >= 0:
            active[violating] = True
            continue
        if stationary:
            return solution, counters, sizes, 0
    return solution, counters, sizes, 1


@lru_cache(maxsize=1)
def warmup():
    """Compile/load the C-contiguous float64 signature before serving requests."""
    _, _, _, status = solve_simplex(np.eye(2), np.asarray([0.7, 0.3]), np.asarray([1.0, 0.0]), True, 1e-10, 1e-9, 100)
    if status:
        raise RuntimeError("Numba sparse OMP warmup failed")
