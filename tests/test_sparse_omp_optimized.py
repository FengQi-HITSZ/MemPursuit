import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pydantic import ValidationError

from mem0.configs.rerankers.sparse_omp import SparseOmpRerankerConfig
from mem0.reranker.sparse_omp_reranker import SparseOmpReranker


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
        pytest.importorskip("scipy")
    return request.param


def test_backend_config_is_explicit():
    assert SparseOmpRerankerConfig().backend == "numpy"
    with pytest.raises(ValidationError):
        SparseOmpRerankerConfig(backend="gpu")


def test_warm_start_checks_optimality_before_skipping(backend):
    solve = SparseOmpReranker.solve_simplex_least_squares
    stats = SparseOmpReranker._new_solver_stats(backend)
    expected = np.asarray([0.3, 0.7])
    optimal = solve(np.eye(2), expected, ridge=0, warm_start=expected, backend=backend, stats=stats)
    assert optimal == pytest.approx(expected)
    assert stats["kkt_solve_count"] == 0
    assert stats["warm_start_skips"] == 1

    stats = SparseOmpReranker._new_solver_stats(backend)
    suboptimal = solve(np.eye(2), expected, ridge=0, warm_start=[0.8, 0.2], backend=backend, stats=stats)
    assert suboptimal == pytest.approx(expected)
    assert stats["kkt_solve_count"] > 0
    assert stats["warm_start_skips"] == 0


def test_warm_start_enters_violating_variable_without_resolving_old_set(backend):
    stats = SparseOmpReranker._new_solver_stats(backend)
    result = SparseOmpReranker.solve_simplex_least_squares(
        np.eye(2), [0.6, 0.4], ridge=0, warm_start=[1.0, 0.0], backend=backend, stats=stats
    )
    assert result == pytest.approx([0.6, 0.4])
    assert stats["kkt_solve_count"] == 1
    assert stats["kkt_size_counts"] == {"2": 1}


@pytest.mark.parametrize("ridge", [0.0, 1e-10])
def test_solver_matches_enumerated_optimum_with_duplicate_vectors(backend, ridge):
    rng = np.random.default_rng(913)
    for perturbation in [0.0, 1e-8, 0.1]:
        matrix = rng.normal(size=(5, 4))
        matrix[1] = matrix[0] + perturbation
        gram = matrix @ matrix.T
        linear = matrix @ rng.normal(size=4)
        hessian = gram + ridge * np.eye(5)
        result = SparseOmpReranker.solve_simplex_least_squares(
            gram, linear, ridge=ridge, warm_start=np.ones(5) / 5, backend=backend
        )
        objectives = []
        for count in range(1, 6):
            for indices in itertools.combinations(range(5), count):
                indices = np.asarray(indices)
                values = SparseOmpReranker._solve_simplex_equality(hessian[np.ix_(indices, indices)], linear[indices])
                if np.min(values) >= -1e-9:
                    candidate = np.zeros(5)
                    candidate[indices] = values
                    objectives.append(0.5 * candidate @ hessian @ candidate - linear @ candidate)
        assert np.min(result) >= 0.0
        assert result.sum() == pytest.approx(1.0, abs=1e-12)
        assert 0.5 * result @ hessian @ result - linear @ result == pytest.approx(min(objectives), abs=1e-8)
        gradient = hessian @ result - linear
        active = result > 1e-10
        multiplier = -np.mean(gradient[active])
        assert np.max(np.abs(gradient[active] + multiplier)) <= 1e-9
        assert np.min(gradient[~active] + multiplier, initial=0.0) >= -1e-9


def test_singular_kkt_retains_lstsq_fallback(backend):
    matrix = np.asarray([[1.0, 0], [1.0, 0], [0, 1.0]])
    stats = SparseOmpReranker._new_solver_stats(backend)
    result = SparseOmpReranker.solve_simplex_least_squares(
        matrix @ matrix.T,
        [0.8, 0.8, 0.2],
        ridge=0,
        warm_start=[0.2, 0.2, 0.6],
        backend=backend,
        stats=stats,
    )
    assert result @ matrix == pytest.approx([0.8, 0.2], abs=1e-10)
    assert stats["kkt_lstsq_count"] >= 1
    assert stats["numba_fallback_count"] == int(backend == "numba")


@pytest.mark.parametrize("patience", [1, 2, 4])
def test_bounded_snapshots_preserve_full_patience_rollback(backend, patience):
    reranker = SparseOmpReranker(
        {"backend": backend, "fw_gap_tol": 0.0, "relative_improvement_tol": 1.0, "early_stop_patience": patience}
    )
    result = reranker.select_support_detailed(
        np.linspace(0.3, 0.2, 10), np.eye(10), max_support=10, early_stopping=True
    )
    assert result["stop_reason"] == "relative_improvement"
    assert result["iterations"] == 1
    assert result["qp_solve_count"] == patience + 1
    assert [item["candidate_index"] for item in result["support"]] == [0]


def test_rejected_candidate_does_not_pollute_cached_correlations(monkeypatch, backend):
    reranker = SparseOmpReranker({"backend": backend, "fw_gap_tol": 0.0})
    responses = iter([np.asarray([1.0]), np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])])
    monkeypatch.setattr(reranker, "solve_simplex_least_squares", lambda *a, **kw: next(responses))
    result = reranker.select_support_detailed(
        np.asarray([1.0, 0.0]),
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
        max_support=3,
        early_stopping=True,
    )
    assert result["qp_solve_count"] == 3
    assert result["stop_reason"] == "candidate_exhausted"
    assert [item["candidate_index"] for item in result["support"]] == [0]


def test_concurrent_selection_has_request_local_buffers_and_stats(backend):
    rng = np.random.default_rng(910)
    matrix = rng.normal(size=(32, 20))
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    queries = rng.normal(size=(8, 20))
    queries /= np.linalg.norm(queries, axis=1)[:, None]
    reranker = SparseOmpReranker({"backend": backend})

    def select(query):
        return reranker.select_support_detailed(query, matrix, max_support=12, early_stopping=False)

    expected = [select(query) for query in queries]
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(select, queries))
    for query, first, second in zip(queries, expected, actual):
        assert first["support"] == second["support"]
        assert first["qp_solve_count"] == second["qp_solve_count"] == 12
        assert first["solver_stats"]["kkt_solve_count"] == second["solver_stats"]["kkt_solve_count"]
        coefficients = np.zeros(len(matrix))
        for item in first["support"]:
            coefficients[item["candidate_index"]] = item["coefficient"]
        gradient = matrix @ matrix.T @ coefficients - matrix @ query + reranker.config.ridge * coefficients
        gap = max(coefficients @ gradient - gradient.min(), 0.0)
        assert first["final_fw_gap"] == pytest.approx(gap, abs=1e-12)


def test_performance_exposes_solver_stats_without_changing_metadata(backend):
    reranker = SparseOmpReranker({"backend": backend})
    result = reranker.rerank_with_vectors_detailed([1, 0], [{"id": "a"}], {"a": [1, 0]})
    assert result["performance"]["solver_stats"]["backend"] == backend
    assert result["performance"]["solver_stats"]["kkt_solve_count"] == 0
    assert result["rerank_metadata"]["support_ids"] == ["a"]
