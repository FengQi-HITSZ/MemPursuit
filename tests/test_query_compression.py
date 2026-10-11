from unittest.mock import Mock

import pytest

from mem0.memory.main import Memory, _prepare_retrieval_query_for_embedding


def _embedding_model(token_counts, *, truncated="truncated-query"):
    model = Mock()
    model.max_input_tokens = 8000
    model.count_tokens.side_effect = lambda text: token_counts[text]
    model.truncate_tokens.return_value = truncated
    return model


def test_retrieval_query_within_limit_skips_llm():
    embedding_model = _embedding_model({"original": 7999})
    llm = Mock()

    result = _prepare_retrieval_query_for_embedding("original", embedding_model, llm)

    assert result == "original"
    llm.generate_response.assert_not_called()
    embedding_model.truncate_tokens.assert_not_called()


def test_overlong_retrieval_query_uses_first_llm_result():
    embedding_model = _embedding_model({"original": 9000, "compressed": 8000})
    llm = Mock()
    llm.generate_response.return_value = "compressed"

    result = _prepare_retrieval_query_for_embedding("original", embedding_model, llm)

    assert result == "compressed"
    llm.generate_response.assert_called_once()
    assert llm.generate_response.call_args.kwargs["max_tokens"] == 8000
    embedding_model.truncate_tokens.assert_not_called()


def test_overlong_llm_result_retries_then_truncates():
    embedding_model = _embedding_model(
        {
            "original": 12000,
            "compressed-one": 9000,
            "compressed-two": 8500,
            "truncated-query": 8000,
        }
    )
    llm = Mock()
    llm.generate_response.side_effect = ["compressed-one", "compressed-two"]

    result = _prepare_retrieval_query_for_embedding("original", embedding_model, llm)

    assert result == "truncated-query"
    assert llm.generate_response.call_count == 2
    embedding_model.truncate_tokens.assert_called_once_with("compressed-two", 8000)


def test_compression_llm_failure_propagates_without_fallback():
    embedding_model = _embedding_model({"original": 9000})
    llm = Mock()
    llm.generate_response.side_effect = RuntimeError("compression unavailable")

    with pytest.raises(RuntimeError, match="compression unavailable"):
        _prepare_retrieval_query_for_embedding("original", embedding_model, llm)

    embedding_model.truncate_tokens.assert_not_called()


def test_embedder_without_token_interface_is_unchanged():
    llm = Mock()

    result = _prepare_retrieval_query_for_embedding("original", Mock(spec=[]), llm)

    assert result == "original"
    llm.generate_response.assert_not_called()


def test_memory_add_uses_compressed_query_only_for_retrieval():
    memory = object.__new__(Memory)
    memory.embedding_model = _embedding_model(
        {
            "user: ORIGINAL MESSAGE\n": 9000,
            "COMPRESSED QUERY": 7900,
        }
    )
    memory.embedding_model.embed.return_value = [0.1, 0.2]
    memory.vector_store = Mock()
    memory.vector_store.search.return_value = []
    memory.db = Mock()
    memory.db.get_last_messages.return_value = []
    memory.llm = Mock()
    memory.llm.generate_response.side_effect = ["COMPRESSED QUERY", '{"memory": []}']
    memory.custom_instructions = None
    messages = [{"role": "user", "content": "ORIGINAL MESSAGE"}]

    result = memory._add_to_vector_store(messages, metadata={}, filters={"user_id": "user"}, infer=True)

    assert result == []
    memory.embedding_model.embed.assert_called_once_with("COMPRESSED QUERY", "search")
    memory.vector_store.search.assert_called_once_with(
        query="COMPRESSED QUERY",
        vectors=[0.1, 0.2],
        top_k=10,
        filters={"user_id": "user"},
    )
    extraction_prompt = memory.llm.generate_response.call_args_list[1].kwargs["messages"][1]["content"]
    assert "ORIGINAL MESSAGE" in extraction_prompt
    assert "COMPRESSED QUERY" not in extraction_prompt
