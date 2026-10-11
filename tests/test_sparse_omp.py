import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from mem0.configs.rerankers.sparse_omp import SparseOmpRerankerConfig
from mem0.reranker.sparse_omp_reranker import SparseOmpRerankError, SparseOmpReranker
from mem0.utils.factory import RerankerFactory


def test_invalid_sparse_omp_constraint_is_rejected():
    with pytest.raises(ValidationError):
        SparseOmpRerankerConfig(constraint="nonnegative")


def test_factory_creates_sparse_omp_reranker():
    reranker = RerankerFactory.create("sparse_omp", {"constraint": "simplex"})

    assert isinstance(reranker, SparseOmpReranker)


def test_max_support_defaults_to_none_and_accepts_explicit_none():
    assert SparseOmpRerankerConfig().max_support is None
    assert SparseOmpRerankerConfig(max_support=None).max_support is None


def test_max_concurrency_defaults_to_two():
    assert SparseOmpRerankerConfig().max_concurrency == 2


def test_execution_slot_bounds_sparse_omp_concurrency():
    reranker = SparseOmpReranker({"max_concurrency": 2})
    lock = threading.Lock()
    active = 0
    max_active = 0

    def work():
        nonlocal active, max_active
        with reranker.execution_slot():
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: work(), range(4)))

    assert max_active == 2


def test_prepared_normalized_vector_fast_path_skips_norm_and_copy():
    reranker = SparseOmpReranker({})
    vector = np.asarray([0.6, 0.8], dtype=np.float64)
    vector.setflags(write=False)

    with patch("mem0.reranker.sparse_omp_reranker.np.linalg.norm") as norm:
        prepared = reranker._prepare_vector(vector, already_normalized=True)

    assert prepared is vector
    norm.assert_not_called()


def test_simplex_solver_outputs_nonnegative_unit_sum():
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    query = np.asarray([0.75, 0.25], dtype=np.float64)
    gram = vectors @ vectors.T
    linear = vectors @ query

    coefficients = SparseOmpReranker.solve_simplex_least_squares(gram, linear)

    assert np.all(coefficients >= -1e-12)
    assert coefficients.sum() == pytest.approx(1.0)


def test_simplex_solver_reenters_kkt_violating_variable_regression():
    gram = np.asarray(
        [
            [0.78346657, -0.99540001, -0.03050055],
            [-0.99540001, 5.62528481, 0.58272368],
            [-0.03050055, 0.58272368, 0.19911903],
        ]
    )
    linear = np.asarray([-0.00230971, -0.70811777, 0.1873516])

    coefficients = SparseOmpReranker.solve_simplex_least_squares(gram, linear)

    assert coefficients == pytest.approx([0.03828936, 0.0, 0.96171064], abs=1e-7)


def test_simplex_solver_matches_seeded_active_subset_enumeration_and_warm_start():
    rng = np.random.default_rng(41)
    for _ in range(12):
        vectors = rng.normal(size=(5, 4))
        gram = vectors @ vectors.T
        linear = rng.normal(size=5)
        ridge = 1e-8
        hessian = gram + ridge * np.eye(5)

        cold = SparseOmpReranker.solve_simplex_least_squares(gram, linear, ridge=ridge)
        warm = SparseOmpReranker.solve_simplex_least_squares(
            gram,
            linear,
            ridge=ridge,
            warm_start=np.asarray([0.05, 0.15, 0.3, 0.2, 0.3]),
        )
        def objective(value):
            return 0.5 * value @ hessian @ value - linear @ value

        subset_objectives = []
        for size in range(1, 6):
            for support in itertools.combinations(range(5), size):
                indices = np.asarray(support)
                candidate = SparseOmpReranker._solve_simplex_equality(
                    hessian[np.ix_(indices, indices)], linear[indices]
                )
                if np.min(candidate) >= -1e-9:
                    full = np.zeros(5)
                    full[indices] = candidate
                    subset_objectives.append(objective(full))

        assert objective(cold) == pytest.approx(min(subset_objectives), abs=1e-7)
        assert warm == pytest.approx(cold, abs=1e-7)
        assert cold.sum() == pytest.approx(1.0)
        gradient = hessian @ cold - linear
        active = cold > 1e-9
        multiplier = -float(np.mean(gradient[active]))
        assert np.max(np.abs(gradient[active] + multiplier)) <= 1e-7
        assert np.min(gradient[~active] + multiplier, initial=0.0) >= -1e-7


def test_select_support_returns_final_simplex_coefficients():
    reranker = SparseOmpReranker({"constraint": "simplex"})
    query = np.asarray([0.75, 0.25], dtype=np.float64)
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    selected = reranker.select_support(query, matrix, top_k=2)

    assert [item["candidate_index"] for item in selected] == [0, 1]
    assert [item["coefficient"] for item in selected] == pytest.approx([0.75, 0.25])
    assert sum(item["coefficient"] for item in selected) == pytest.approx(1.0)


def test_omp_selection_is_deterministic_with_ties():
    reranker = SparseOmpReranker({"constraint": "simplex", "max_support": 3, "early_stopping": False})
    docs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    vectors = {
        "a": [1.0, 0.0],
        "b": [1.0, 0.0],
        "c": [0.0, 1.0],
    }

    result = reranker.rerank_with_vectors([1.0, 0.0], docs, vectors, top_k=3)

    assert [item["id"] for item in result] == ["a", "b", "c"]
    assert [item["rerank_score"] for item in result] == pytest.approx([1.0, 2 / 3, 1 / 3])
    assert result[-1]["score_breakdown"]["sparse_omp_coefficient"] == pytest.approx(0.0, abs=1e-9)


def test_adaptive_sparse_omp_preserves_original_score_and_returns_only_positive_support():
    reranker = SparseOmpReranker({"constraint": "simplex"})
    docs = [{"id": "x", "score": 0.2}, {"id": "y", "score": 0.9}, {"id": "z", "score": 0.4}]
    vectors = {
        "x": [1.0, 0.0],
        "y": [0.0, 1.0],
        "z": [-1.0, 0.0],
    }

    result = reranker.rerank_with_vectors([1.0, 0.0], docs, vectors, top_k=3)
    assert [item["id"] for item in result] == ["x"]
    assert [item["score"] for item in result] == pytest.approx([0.2])
    assert [item["rerank_score"] for item in result] == pytest.approx([1.0])
    assert result[0]["score_breakdown"]["pre_sparse_omp_score"] == 0.2
    assert result[0]["score_breakdown"]["pre_sparse_omp_rank"] == 1
    assert result[0]["score_breakdown"]["sparse_omp_step"] == 1


def test_sparse_omp_records_exact_original_candidate_rank_after_reordering():
    reranker = SparseOmpReranker({"constraint": "simplex", "max_support": 3, "early_stopping": False})
    docs = [{"id": "first"}, {"id": "second"}, {"id": "third"}]
    vectors = {
        "first": [0.0, 1.0],
        "second": [1.0, 0.0],
        "third": [-1.0, 0.0],
    }

    result = reranker.rerank_with_vectors([1.0, 0.0], docs, vectors, top_k=3)

    assert result[0]["id"] == "second"
    assert result[0]["score_breakdown"]["pre_sparse_omp_rank"] == 2
    assert sorted(item["score_breakdown"]["pre_sparse_omp_rank"] for item in result) == [1, 2, 3]


def test_larger_k_prefix_matches_smaller_k_on_same_candidate_pool():
    reranker = SparseOmpReranker({"constraint": "simplex", "max_support": 4, "early_stopping": False})
    docs = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    vectors = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "c": [0.0, 0.0, 1.0],
        "d": [-1.0, 0.0, 0.0],
    }

    top_4 = reranker.rerank_with_vectors([0.8, 0.4, 0.2], docs, vectors, top_k=4)
    top_2 = reranker.rerank_with_vectors([0.8, 0.4, 0.2], docs, vectors, top_k=2)

    assert [item["id"] for item in top_4[:2]] == [item["id"] for item in top_2]


def test_adaptive_stop_reasons_and_fixed_iteration_reason():
    matrix = np.eye(3)
    query = np.asarray([0.8, 0.4, 0.2])

    residual = SparseOmpReranker({"residual_tol": 2.0}).select_support_detailed(
        query, matrix, max_support=3, early_stopping=True
    )
    fw_gap = SparseOmpReranker({"residual_tol": None}).select_support_detailed(
        np.asarray([1.0, 0.0, 0.0]), matrix, max_support=3, early_stopping=True
    )
    capped = SparseOmpReranker({"fw_gap_tol": 0.0}).select_support_detailed(
        query, matrix, max_support=1, early_stopping=True
    )
    exhausted = SparseOmpReranker({"fw_gap_tol": 0.0}).select_support_detailed(
        query, matrix, max_support=3, early_stopping=False
    )

    assert residual["stop_reason"] == "residual_target"
    assert fw_gap["stop_reason"] == "fw_gap"
    assert capped["stop_reason"] == "max_support"
    assert exhausted["stop_reason"] == "fixed_iterations"
    assert exhausted["qp_solve_count"] == 3


def test_relative_improvement_rolls_back_entire_patience_window():
    reranker = SparseOmpReranker(
        {
            "fw_gap_tol": 0.0,
            "relative_improvement_tol": 1.0,
            "early_stop_patience": 2,
        }
    )
    selected = reranker.select_support_detailed(
        np.asarray([0.8, 0.4, 0.2, 0.1]),
        np.eye(4),
        max_support=4,
        early_stopping=True,
    )

    assert selected["stop_reason"] == "relative_improvement"
    assert selected["iterations"] == 1
    assert [item["candidate_index"] for item in selected["support"]] == [0]


def test_fixed_metadata_retains_zero_coefficients_and_exact_qp_count():
    reranker = SparseOmpReranker({"max_support": 2, "early_stopping": False})
    docs = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}, {"id": "c", "score": 0.7}]
    detailed = reranker.rerank_with_vectors_detailed(
        [0.8, 0.4, 0.2],
        docs,
        {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]},
        top_k=3,
    )

    assert detailed["rerank_metadata"]["support_ids"] == ["a", "b"]
    assert detailed["rerank_metadata"]["support_count"] == 2
    assert detailed["rerank_metadata"]["stop_reason"] == "fixed_iterations"
    assert detailed["rerank_metadata"]["budget_capped"] is False
    assert detailed["rerank_metadata"]["mode"] == "fixed"
    assert detailed["rerank_metadata"]["early_stopping"] is False
    assert detailed["rerank_metadata"]["qp_solve_count"] == 2
    assert detailed["rerank_metadata"]["retain_zero_coefficients"] is True
    assert detailed["rerank_metadata"]["monotonicity_warning_count"] == 0
    assert detailed["rerank_metadata"]["monotonicity_warnings"] == []
    assert [item["id"] for item in detailed["results"]] == ["a", "b"]


def test_none_max_support_uses_request_minimum_for_fixed_and_pool_for_adaptive():
    docs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    vectors = {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}
    reranker = SparseOmpReranker({"max_support": None, "candidate_pool_size": 3})

    fixed = reranker.rerank_with_vectors_detailed(
        [0.8, 0.4, 0.2],
        docs,
        vectors,
        top_k=3,
        early_stopping=False,
        min_iterations=2,
    )
    adaptive = reranker.rerank_with_vectors_detailed(
        [0.8, 0.4, 0.2],
        docs,
        vectors,
        top_k=3,
        early_stopping=True,
    )

    assert fixed["rerank_metadata"]["effective_iteration_limit"] == 2
    assert fixed["rerank_metadata"]["qp_solve_count"] == 2
    assert adaptive["rerank_metadata"]["effective_iteration_limit"] == 3


def test_fixed_short_pool_use_available_returns_every_candidate_and_records_cap():
    reranker = SparseOmpReranker({"max_support": 200, "candidate_pool_size": 200, "early_stopping": False})
    documents = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]

    detailed = reranker.rerank_with_vectors_detailed(
        [1.0, 0.0],
        documents,
        {"a": [1.0, 0.0], "b": [0.0, 1.0]},
        top_k=200,
        early_stopping=False,
        min_iterations=200,
        short_pool_policy="use_available",
    )

    assert {item["id"] for item in detailed["results"]} == {"a", "b"}
    assert detailed["rerank_metadata"]["requested_iteration_limit"] == 200
    assert detailed["rerank_metadata"]["effective_iteration_limit"] == 2
    assert detailed["rerank_metadata"]["short_pool_policy"] == "use_available"
    assert detailed["rerank_metadata"]["short_pool_capped"] is True


def test_residual_tol_requires_normalized_vectors():
    with pytest.raises(ValidationError, match="normalize_vectors"):
        SparseOmpRerankerConfig(normalize_vectors=False, residual_tol=0.5)


def test_adaptive_returns_only_positive_support_within_budget():
    reranker = SparseOmpReranker({"max_support": 32})
    docs = [{"id": str(index), "score": 1.0 - index / 200} for index in range(200)]
    vectors = {str(index): [1.0, 0.0] if index == 0 else [0.0, 1.0] for index in range(200)}

    detailed = reranker.rerank_with_vectors_detailed([1.0, 0.0], docs, vectors, top_k=200)

    assert len(detailed["results"]) == 1
    assert detailed["rerank_metadata"]["support_count"] <= 32
    assert detailed["rerank_metadata"]["support_ids"] == ["0"]


def test_partial_bad_vectors_raise_structured_error_with_real_timing():
    reranker = SparseOmpReranker({})
    docs = [{"id": "bad", "score": 0.9}, {"id": "missing", "score": 0.8}]

    with pytest.raises(SparseOmpRerankError) as raised:
        reranker.rerank_with_vectors_detailed([1.0, 0.0], docs, {"bad": [0.0, 0.0]}, top_k=2)

    assert raised.value.code == "sparse_omp_rerank_failed"
    assert raised.value.performance["rerank_total"] > 0.0
    assert raised.value.performance["gram_build"] is None
    assert raised.value.performance["support_selection"] is None


def test_zero_candidates_return_empty_without_fallback():
    reranker = SparseOmpReranker({})

    detailed = reranker.rerank_with_vectors_detailed(None, [], {}, top_k=2)

    assert detailed["results"] == []
    assert detailed["rerank_metadata"]["stop_reason"] == "no_candidates"
    assert "fallback_used" not in detailed["rerank_metadata"]
    assert detailed["performance"]["rerank_total"] >= 0.0


def test_fixed_candidate_shortage_and_qp_failure_are_structured(monkeypatch):
    reranker = SparseOmpReranker({"max_support": 2, "early_stopping": False})
    with pytest.raises(SparseOmpRerankError, match="requires at least 2 candidates") as empty_error:
        reranker.rerank_with_vectors_detailed(None, [], {})
    assert empty_error.value.performance["rerank_total"] >= 0.0
    assert empty_error.value.performance["vector_prepare"] is None
    assert empty_error.value.performance["qp_solve_count"] == 0

    with pytest.raises(SparseOmpRerankError, match="requires at least 2 candidates"):
        reranker.rerank_with_vectors_detailed([1.0, 0.0], [{"id": "one"}], {"one": [1.0, 0.0]})

    def fail_qp(*args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(reranker, "solve_simplex_least_squares", fail_qp)
    with pytest.raises(SparseOmpRerankError, match="QP failed") as raised:
        reranker.rerank_with_vectors_detailed(
            [1.0, 0.0],
            [{"id": "one"}, {"id": "two"}],
            {"one": [1.0, 0.0], "two": [0.0, 1.0]},
        )
    assert raised.value.performance["qp_solve_count"] == 1
    assert raised.value.performance["support_selection"] is not None


def test_fixed_first_iteration_does_not_compare_against_empty_support():
    reranker = SparseOmpReranker({"max_support": 1, "early_stopping": False})

    detailed = reranker.rerank_with_vectors_detailed(
        [1.0, 0.0],
        [{"id": "orthogonal"}],
        {"orthogonal": [0.0, 1.0]},
    )

    assert detailed["rerank_metadata"]["qp_solve_count"] == 1
    assert detailed["rerank_metadata"]["monotonicity_warning_count"] == 0
    assert detailed["results"][0]["id"] == "orthogonal"


def test_fixed_monotonicity_warning_keeps_incumbent_and_continues(monkeypatch, caplog):
    reranker = SparseOmpReranker({"max_support": 3, "early_stopping": False})
    solutions = iter(
        [
            np.asarray([1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0, 0.0]),
        ]
    )
    monkeypatch.setattr(reranker, "solve_simplex_least_squares", lambda *args, **kwargs: next(solutions))

    with caplog.at_level("WARNING", logger="mem0.reranker.sparse_omp_reranker"):
        detailed = reranker.rerank_with_vectors_detailed(
            [1.0, 0.0],
            [{"id": "one"}, {"id": "two"}, {"id": "three"}],
            {"one": [1.0, 0.0], "two": [0.0, 1.0], "three": [-1.0, 0.0]},
        )

    assert detailed["rerank_metadata"]["qp_solve_count"] == 3
    assert detailed["rerank_metadata"]["monotonicity_warning_count"] == 1
    assert detailed["rerank_metadata"]["monotonicity_warnings"][0]["action"] == "kept_incumbent"
    assert [item["id"] for item in detailed["results"]] == ["one", "two", "three"]
    assert detailed["results"][1]["score_breakdown"]["sparse_omp_coefficient"] == pytest.approx(0.0)
    assert "keeping the incumbent coefficients and continuing" in caplog.text
