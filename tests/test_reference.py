import itertools

import numpy as np
import pytest

from mempursuit import select
from mempursuit.simplex import solve_simplex_qp


def test_residual_selection_prefers_complementary_evidence():
    query = np.asarray([1.0, 1.0])
    candidates = np.asarray(
        [
            [1.0, 0.0],
            [0.995, -0.1],
            [0.0, 1.0],
        ]
    )

    result = select(query, candidates, budget=2)

    assert result.selected_indices.tolist() == [0, 2]
    assert result.weights.sum() == pytest.approx(1.0)
    assert np.all(result.weights >= 0.0)


def test_fixed_budget_keeps_entry_order_and_zero_weight_items():
    query = np.asarray([1.0, 0.0])
    candidates = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    result = select(query, candidates, budget=3)

    assert result.selected_indices.tolist() == [0, 1, 2]
    assert result.weights[0] == pytest.approx(1.0)
    assert result.weights[1:].tolist() == pytest.approx([0.0, 0.0], abs=1e-9)


def test_one_path_supplies_stable_smaller_budget_prefixes():
    query = np.asarray([0.8, 0.4, 0.2, 0.1])
    candidates = np.eye(4)

    full = select(query, candidates, budget=4)
    prefix = select(query, candidates, budget=2)

    assert full.selected_indices[:2].tolist() == prefix.selected_indices.tolist()


def test_adaptive_patience_discards_the_stalled_window():
    result = select(
        np.asarray([0.8, 0.4, 0.2, 0.1]),
        np.eye(4),
        budget=4,
        adaptive=True,
        relative_improvement_tolerance=1.0,
        patience=2,
    )

    assert result.stop_reason == "relative_improvement"
    assert result.selected_indices.tolist() == [0]


def test_simplex_solver_matches_enumerated_active_sets():
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(5, 4))
    gram = vectors @ vectors.T
    linear = rng.normal(size=5)
    ridge = 1e-8
    hessian = gram + ridge * np.eye(5)

    solution = solve_simplex_qp(gram, linear, ridge=ridge)

    def objective(value):
        return 0.5 * value @ hessian @ value - linear @ value

    feasible_objectives = []
    for size in range(1, 6):
        for support in itertools.combinations(range(5), size):
            indices = np.asarray(support)
            local_hessian = hessian[np.ix_(indices, indices)]
            kkt = np.block(
                [
                    [local_hessian, np.ones((size, 1))],
                    [np.ones((1, size)), np.zeros((1, 1))],
                ]
            )
            rhs = np.concatenate([linear[indices], np.ones(1)])
            local = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]
            if np.min(local) >= -1e-9:
                candidate = np.zeros(5)
                candidate[indices] = local
                feasible_objectives.append(objective(candidate))

    assert solution.sum() == pytest.approx(1.0)
    assert np.min(solution) >= -1e-10
    assert objective(solution) == pytest.approx(min(feasible_objectives), abs=1e-7)


def test_invalid_embeddings_and_budget_are_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        select([1.0, 0.0], [[1.0, 0.0, 0.0]], budget=1)
    with pytest.raises(ValueError, match="nonzero"):
        select([0.0, 0.0], [[1.0, 0.0]], budget=1)
    with pytest.raises(ValueError, match="budget"):
        select([1.0, 0.0], [[1.0, 0.0]], budget=2)


def test_numba_backend_matches_numpy_when_available():
    pytest.importorskip("numba")
    pytest.importorskip("scipy")
    rng = np.random.default_rng(31)
    query = rng.normal(size=12)
    candidates = rng.normal(size=(20, 12))

    numpy_result = select(query, candidates, budget=8, backend="numpy")
    numba_result = select(query, candidates, budget=8, backend="numba")

    assert numba_result.selected_indices.tolist() == numpy_result.selected_indices.tolist()
    assert numba_result.weights == pytest.approx(numpy_result.weights, abs=1e-8)
