from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
import yaml
from mempursuit import select
from mem0.configs.base import MemoryConfig
from mem0.reranker.sparse_omp_reranker import SparseOmpReranker
from mem0.memory.storage import SQLiteManager
from benchmarks.pipeline.adapter import MemPursuitAdapter
from benchmarks.pipeline.models import BenchmarkQuestion, CanonicalEvent, ScopeSpec, RetrievalResult, MemoryHit
from benchmarks.pipeline.run import CheckpointAdapter, prefix, evaluate_one
from benchmarks.pipeline.prompts import PromptSnapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_standalone_and_runtime_select_identical_fixed_prefixes(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    rng = np.random.default_rng(42)
    for pool_size, budget in [(8, 5), (20, 10), (40, 20)]:
        query = rng.normal(size=12)
        vectors = rng.normal(size=(pool_size, 12))
        reference = select(query, vectors, budget=budget, backend=backend)
        reranker = SparseOmpReranker(
            {"backend": backend, "max_support": budget, "candidate_pool_size": pool_size, "early_stopping": False}
        )
        docs = [{"id": str(i), "memory": str(i)} for i in range(pool_size)]
        result = reranker.rerank_with_vectors_detailed(
            query, docs, {str(i): v for i, v in enumerate(vectors)}, top_k=budget
        )
        assert [int(row["id"]) for row in result["results"]] == reference.selected_indices.tolist()


def test_sqlite_history_and_recent_messages_survive_without_extra_tables():
    db = SQLiteManager(":memory:")
    db.add_history("id", None, "fact", "ADD")
    db.save_messages([{"role": "user", "content": "hi"}], "scope")
    assert db.get_history("id")[0]["new_memory"] == "fact"
    assert db.get_last_messages("scope")[0]["content"] == "hi"
    names = {r[0] for r in db.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == {"history", "messages"}
    db.reset()
    assert db.get_history("id") == []
    db.close()


def test_add_checkpoint_retries_failed_pair_without_repeating_success(tmp_path, monkeypatch):
    calls = []

    def request(self, method, path, payload=None, **kw):
        calls.append(payload)
        if len(calls) == 2:
            raise RuntimeError("temporary transport failure")
        return {"results": [{"id": "one"}]}

    monkeypatch.setattr(MemPursuitAdapter, "_request", request)
    adapter = CheckpointAdapter("http://unused", tmp_path)
    adapter._request("POST", "/memories", {"messages": ["first"]})
    with pytest.raises(RuntimeError):
        adapter._request("POST", "/memories", {"messages": ["second"]})
    resumed = CheckpointAdapter("http://unused", tmp_path)
    resumed._request("POST", "/memories", {"messages": ["first"]})
    resumed._request("POST", "/memories", {"messages": ["second"]})
    assert calls == [{"messages": ["first"]}, {"messages": ["second"]}, {"messages": ["second"]}]


def test_native_retrieval_has_paper_threshold_pool_and_prefix_contract(monkeypatch):
    adapter = MemPursuitAdapter("http://unused")
    calls = []
    monkeypatch.setattr(adapter, "_validated_health", lambda: {})

    def request(method, path, payload=None):
        calls.append(payload)
        return {
            "results": [
                {"id": "a", "memory": "A", "rerank_score": 1.0},
                {"id": "b", "memory": "B", "rerank_score": 0.5},
            ],
            "rerank_metadata": {"mode": "fixed"},
        }

    monkeypatch.setattr(adapter, "_request", request)
    result = adapter._retrieve(ScopeSpec("longmemeval", "s", "r"), "query", 200)
    assert calls[0]["threshold"] is None
    assert calls[0]["rerank_options"] == {
        "early_stopping": False,
        "min_iterations": 200,
        "short_pool_policy": "use_available",
    }
    assert [x.text for x in prefix(result, 1).main_hits] == ["A"]
    assert len(result.main_hits) == 2


def test_longmem_evaluation_uses_original_prompts_and_strict_judge(tmp_path):
    prompts = PromptSnapshot(ROOT, tmp_path, "longmemeval")
    question = BenchmarkQuestion("q", "s", "Where?", "Paris", "single-session-user", "2024-01-01")
    result = RetrievalResult("s", "Where?", 1, main_hits=[MemoryHit("Paris", timestamp="2023-01-01")])
    answerer = Mock()
    judge = Mock()
    answerer.text.return_value = ("<mem_thinking>reason</mem_thinking>Paris", {"usage": {}})
    judge.text.return_value = ("<judge_thinking>reason</judge_thinking>yes", {"usage": {}})
    record = evaluate_one(question, result, 1, "longmemeval", prompts, answerer, judge)
    assert record["score"] == 1.0
    assert record["generated_answer"] == "Paris"
    assert answerer.text.call_args.kwargs == {"temperature": 0.0, "max_tokens": 4096}
    assert judge.text.call_args.kwargs == {"temperature": 0.0, "max_tokens": 1024}


def test_release_configs_validate_and_enable_the_paper_path():
    for name in ["mempursuit", "adaptive", "numba"]:
        cfg = MemoryConfig(**yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text()))
        assert cfg.retrieval.semantic_score_mode == "embedding"
        assert cfg.reranker.provider == "sparse_omp"
        assert cfg.reranker.config["candidate_pool_size"] == 200
        assert cfg.reranker.config["early_stopping"] == (name == "adaptive")
