from unittest.mock import Mock, call, patch
import pytest
from mem0.configs.base import MemoryConfig
from mem0.memory.main import Memory
from mem0.reranker.sparse_omp_reranker import SparseOmpRerankError


def test_search_sparse_omp_rerank_uses_only_the_hard_candidate_pool():
    with (
        patch("mem0.memory.main.EmbedderFactory") as mock_embedder,
        patch("mem0.memory.main.VectorStoreFactory") as mock_vector_store,
        patch("mem0.memory.main.LlmFactory") as mock_llm,
    ):
        embedding_model = Mock()
        embedding_model.embed.return_value = [1.0, 0.0]
        vector_store = Mock()
        vector_store.search = Mock(
            return_value=[
                Mock(id="far", payload={"data": "far memory", "user_id": "test_user"}, score=0.9),
                Mock(id="near", payload={"data": "near memory", "user_id": "test_user"}, score=0.8),
                Mock(id="wrong-dim", payload={"data": "wrong dim memory", "user_id": "test_user"}, score=0.7),
                Mock(id="zero", payload={"data": "zero memory", "user_id": "test_user"}, score=0.6),
            ]
        )
        vector_store.keyword_search = Mock(return_value=None)
        vector_store.get_dense_vectors = Mock(
            return_value={
                "far": [0.0, 1.0],
                "near": [1.0, 0.0],
                "wrong-dim": [-1.0, 0.0],
                "zero": [0.0, -1.0],
            }
        )
        mock_embedder.create.return_value = embedding_model
        mock_vector_store.create.return_value = vector_store
        mock_llm.create.return_value = Mock()

        config = MemoryConfig(
            version="v1.1",
            history_db_path=":memory:",
            reranker={
                "provider": "sparse_omp",
                "config": {"constraint": "simplex", "candidate_pool_size": 4, "max_support": 4},
            },
        )
        memory = Memory(config)

        with (
            patch("mem0.memory.main.lemmatize_for_bm25", return_value="test query"),
            patch("mem0.memory.main.extract_entities", return_value=[]),
        ):
            result = memory.search(
                "test query",
                filters={"user_id": "test_user"},
                top_k=4,
                rerank=True,
                score_debug=True,
            )

    assert [item["id"] for item in result["results"]] == ["near"]
    assert [item["score"] for item in result["results"]] == pytest.approx([0.8])
    assert [item["rerank_score"] for item in result["results"]] == pytest.approx([1.0])
    vector_store.get_dense_vectors.assert_called_once_with(["far", "near", "wrong-dim", "zero"])
    assert embedding_model.embed.call_args_list == [call("test query", "search")]
    assert vector_store.search.call_args.kwargs["top_k"] == 60
    assert result["rerank_metadata"]["support_ids"] == ["near"]
    assert result["rerank_metadata"]["candidate_pool_size"] == 4
    assert result["rerank_metadata"]["candidate_count"] == 4
    assert result["rerank_metadata"]["mode"] == "adaptive"
    assert "fallback_used" not in result["rerank_metadata"]
    performance = result["query_debug"]["sparse_omp_performance"]
    assert performance["embedding_reused"] is True
    assert performance["candidate_count"] == 4
    assert performance["solver_stats"]["backend"] == "numpy"
    assert performance["solver_stats"]["kkt_solve_count"] >= 0
    unified_performance = result["query_debug"]["search_performance"]
    assert unified_performance["server_search_total"] >= unified_performance["rerank_total"]
    assert unified_performance["query_embedding"] >= 0.0
    assert unified_performance["hybrid_candidate_retrieval"] >= 0.0

    breakdown = result["results"][0]["score_breakdown"]
    assert breakdown["pre_sparse_omp_score"] == pytest.approx(0.8)
    assert breakdown["sparse_omp_score_source"] == "sparse_omp"
    assert breakdown["sparse_omp_constraint"] == "simplex"
    assert breakdown["sparse_omp_step"] == 1
    assert breakdown["sparse_omp_candidate_count"] == 4
    assert breakdown["sparse_omp_selected_count"] == 1

    with (
        patch("mem0.memory.main.lemmatize_for_bm25", return_value="test query"),
        patch("mem0.memory.main.extract_entities", return_value=[]),
    ):
        fixed = memory.search(
            "test query",
            filters={"user_id": "test_user"},
            top_k=4,
            rerank=True,
            rerank_options={"early_stopping": False, "min_iterations": 4},
        )
    assert len(fixed["results"]) == 4
    assert fixed["rerank_metadata"]["mode"] == "fixed"
    assert fixed["rerank_metadata"]["qp_solve_count"] == 4
    assert fixed["rerank_metadata"]["retain_zero_coefficients"] is True

    vector_store.get_dense_vectors.return_value = {
        "far": [0.0, 1.0],
        "near": [1.0, 0.0],
        "wrong-dim": [1.0],
    }
    with (
        patch("mem0.memory.main.lemmatize_for_bm25", return_value="test query"),
        patch("mem0.memory.main.extract_entities", return_value=[]),
        pytest.raises(SparseOmpRerankError) as raised,
    ):
        memory.search(
            "test query",
            filters={"user_id": "test_user"},
            top_k=4,
            rerank=True,
            score_debug=True,
        )
    assert raised.value.performance["rerank_total"] > 0.0
    assert raised.value.performance["support_selection"] is None
