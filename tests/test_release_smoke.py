"""Offline checks of memory ingestion and the evaluation pipeline."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from benchmarks.locomo.run import session_to_chunks
from benchmarks.pipeline.adapter import MemPursuitAdapter
from benchmarks.pipeline.models import CanonicalEvent, ScopeSpec, BenchmarkScope
from benchmarks.pipeline.run import scope_routes, build_parser
from mem0.configs.base import MemoryConfig
from mem0.memory.main import Memory


def test_benchmarks_use_dataset_specific_write_granularity(monkeypatch):
    turns = [{"speaker": "A", "text": "one"}, {"speaker": "B", "text": "two"}]
    assert [len(chunk) for chunk in session_to_chunks(turns, "A", "B")] == [1, 1]
    adapter = MemPursuitAdapter("http://unused")
    payloads = []

    def request(method, path, payload=None, **kw):
        if method == "POST":
            payloads.append(payload)
        return {"results": []}

    monkeypatch.setattr(adapter, "_request", request)
    monkeypatch.setattr(adapter, "_validated_health", lambda: {})

    def event(session, order, date):
        return CanonicalEvent(
            "longmemeval", "scope", session, date, date, "speaker", "user", f"{session}-{order}", order, order
        )

    # Intentionally unordered sessions, with an odd-length earlier session.
    events = [
        event("late", 0, "2024-02-01"),
        event("early", 2, "2024-01-01"),
        event("early", 0, "2024-01-01"),
        event("early", 1, "2024-01-01"),
    ]
    result = adapter._build(ScopeSpec("longmemeval", "scope", "run"), events)
    assert result["batch_size"] == 2
    assert result["native_add_calls"] == 3
    assert [[m["content"] for m in p["messages"]] for p in payloads] == [
        ["early-0", "early-1"],
        ["early-2"],
        ["late-0"],
    ]
    assert [p["metadata"]["pair_offset"] for p in payloads] == [0, 2, 0]
    assert [p["timestamp"] for p in payloads] == ["2024-01-01", "2024-01-01", "2024-02-01"]


def test_native_execution_defaults_and_shard_routes():
    scopes = [BenchmarkScope(ScopeSpec("beam", str(i), "run"), [], []) for i in range(5)]
    assert scope_routes(scopes, ["s1", "s2"]) == {"0": "s1", "1": "s2", "2": "s1", "3": "s2", "4": "s1"}
    for benchmark, workers in [("longmemeval", 12), ("beam", 4)]:
        args = build_parser(benchmark).parse_args(["--project-name", "test"])
        assert args.ingest_workers == args.retrieval_workers == workers
        assert args.max_workers == 16


def test_api_add_then_sparse_omp_search_preserves_facts_and_timestamp(monkeypatch):
    import mem0.memory.main as implementation
    import server.main as server

    # Only the external LLM, embedder and vector database are replaced.
    vectors = {"Alice lives in Paris.": [1.0, 0.0], "Alice likes tea.": [0.0, 1.0]}
    embedder = Mock()
    embedder.embed.side_effect = lambda text, *args: vectors.get(text, [1.0, 0.0])
    embedder.embed_batch.side_effect = lambda texts, *args: [vectors[text] for text in texts]
    store = Mock()
    points = {}

    def insert(vectors, ids, payloads):
        for vector, identifier, payload in zip(vectors, ids, payloads):
            points[identifier] = SimpleNamespace(id=identifier, payload=payload, vector=vector, score=0.8)

    store.insert.side_effect = insert
    store.search.side_effect = lambda **kw: [
        v for v in points.values() if v.payload["user_id"] == kw["filters"]["user_id"]
    ]
    store.keyword_search.return_value = []
    store.get_dense_vectors.side_effect = lambda ids: {i: points[i].vector for i in ids}
    llm = Mock()
    llm.generate_response.return_value = '{"memory": [{"text": "Alice lives in Paris."}, {"text": "Alice likes tea."}, {"text": "Alice lives in Paris."}]}'
    monkeypatch.setattr(implementation.EmbedderFactory, "create", lambda *a, **kw: embedder)
    monkeypatch.setattr(implementation.VectorStoreFactory, "create", lambda *a, **kw: store)
    monkeypatch.setattr(implementation.LlmFactory, "create", lambda *a, **kw: llm)
    monkeypatch.setattr(implementation, "lemmatize_for_bm25", lambda text: text)
    monkeypatch.setattr(implementation, "extract_entities", lambda text: [])
    monkeypatch.setattr(implementation, "extract_entities_batch", lambda texts: [[] for _ in texts])
    config = MemoryConfig(
        history_db_path=":memory:",
        reranker={
            "provider": "sparse_omp",
            "config": {"candidate_pool_size": 200, "max_support": 200, "early_stopping": False},
        },
    )
    memory = Memory(config)
    monkeypatch.setattr(server, "memory_instance", memory)
    try:
        added = server.add_memories(
            server.AddRequest(
                **{
                    "messages": [{"role": "user", "content": "Alice lives in Paris and likes tea."}],
                    "user_id": "scope",
                    "timestamp": 1704067200,
                }
            )
        )
        assert len(added["results"]) == 2  # exact duplicate removed
        assert all(p.payload["created_at"] == "2024-01-01T00:00:00+00:00" for p in points.values())
        value = server.search_memories(
            server.SearchRequest(
                **{
                    "query": "Where does Alice live?",
                    "user_id": "scope",
                    "limit": 200,
                    "rerank": True,
                    "threshold": None,
                    "rerank_options": {
                        "early_stopping": False,
                        "min_iterations": 200,
                        "short_pool_policy": "use_available",
                    },
                }
            )
        )
        assert [row["memory"] for row in value["results"]] == ["Alice lives in Paris.", "Alice likes tea."]
        assert value["rerank_metadata"]["short_pool_capped"] is True
        assert value["rerank_metadata"]["retain_zero_coefficients"] is True
        with pytest.raises(HTTPException) as exc:
            server.search_memories(server.SearchRequest(query="test"))
        assert exc.value.status_code == 400
        assert memory.history(added["results"][0]["id"])[0]["event"] == "ADD"
    finally:
        memory.close()


def test_native_stages_resume_without_repeating_ingestion_or_retrieval(tmp_path, monkeypatch):
    from benchmarks.pipeline import run as runner
    from benchmarks.pipeline.models import BenchmarkQuestion

    source = tmp_path / "dataset.json"
    source.write_text("synthetic dataset fingerprint")
    event = CanonicalEvent(
        "longmemeval", "scope", "session", "2024-01-01", "2024-01-01", "user", "user", "A fact", 0, 0
    )
    scope = BenchmarkScope(
        ScopeSpec("longmemeval", "scope", "test"),
        [[event]],
        [BenchmarkQuestion("q", "scope", "Is this true?", "yes", "single-session-user")],
    )
    monkeypatch.setattr(runner, "load_longmemeval", lambda *a, **kw: [scope])
    monkeypatch.setattr(MemPursuitAdapter, "_validated_health", lambda self: {})
    calls = []

    def request(self, method, path, payload=None, **kw):
        calls.append((method, path))
        if path == "/search":
            return {"results": [{"id": "m", "memory": "A fact", "score": 1}], "rerank_metadata": {"mode": "fixed"}}
        return {"results": [{"id": "m", "memory": "A fact"}]}

    monkeypatch.setattr(MemPursuitAdapter, "_request", request)
    gateway = Mock()
    gateway.text.return_value = ("yes", {"usage": {}})
    monkeypatch.setattr(runner, "OpenAIGateway", lambda *a, **kw: gateway)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-placeholder")
    args = runner.build_parser("longmemeval").parse_args(
        [
            "--project-name",
            "test",
            "--dataset-path",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--top-k-cutoffs",
            "1,2",
            "--resume",
        ]
    )
    metrics = runner.run(args, "longmemeval")
    assert [row["overall"]["accuracy"] for row in metrics["tables"]] == [1.0, 1.0]
    assert calls.count(("POST", "/memories")) == 1
    assert calls.count(("POST", "/search")) == 1
    assert gateway.text.call_count == 4
    first_calls = list(calls)
    runner.run(args, "longmemeval")
    assert calls == first_calls
    assert gateway.text.call_count == 4
    args.judge_model = "changed-judge"
    with pytest.raises(RuntimeError, match="changed"):
        runner.run(args, "longmemeval")


def test_locomo_default_cli_settings(monkeypatch):
    from benchmarks.locomo.run import parse_args

    monkeypatch.setattr("sys.argv", ["locomo", "--project-name", "test"])
    args = parse_args()
    assert args.max_workers == 2
    assert args.mem0_timeout == args.answer_timeout == args.judge_timeout == 900
    assert args.rerank and args.score_debug
    assert args.top_k == 200 and args.threshold == 0.1
    assert args.top_k_cutoffs == "10,20,50,200"
    assert args.categories == "1,2,3,4"
