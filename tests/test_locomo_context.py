import json

import pytest

from benchmarks.locomo.run import (
    ADAPTIVE_CONTEXT_LABEL,
    _resume_result_complete,
    _retryable_question_failure,
    apply_locomo_judge_to_saved_result,
    compatible_sparse_omp_metadata,
    compute_locomo_metrics,
    context_policy_labels,
    evaluate_locomo_contexts,
    labels_to_evaluate,
    process_question,
    resolve_locomo_context,
    update_reranker_performance_report,
    update_sparse_omp_performance_report,
    validate_context_cutoffs,
)


def _metadata(**overrides):
    metadata = {
        "provider": "sparse_omp",
        "policy_version": "sparse_omp_v2",
        "mode": "adaptive",
        "early_stopping": True,
        "effective_iteration_limit": 200,
        "qp_solve_count": 2,
        "retain_zero_coefficients": False,
        "support_count": 2,
        "support_ids": ["m3", "m1"],
        "stop_reason": "fw_gap",
        "converged": True,
        "budget_capped": False,
    }
    metadata.update(overrides)
    return metadata


def test_resume_result_requires_every_valid_cutoff_for_full_runs():
    result = {
        "question_id": "conv0_q0",
        "retrieval": {"search_results": [{"id": "m1", "memory": "memory"}]},
        "cutoff_results": {
            "top_10": {"generated_answer": "answer", "judgment": "CORRECT", "score": 1.0},
            "top_20": {"generated_answer": "answer", "judgment": "WRONG", "score": 0.0},
        },
    }

    assert _resume_result_complete(result, ["top_10", "top_20"], predict_only=False) is True
    assert _resume_result_complete(result, ["top_10", "top_20", "top_50"], predict_only=False) is False

    result["cutoff_results"]["top_20"]["generated_answer"] = ""
    assert _resume_result_complete(result, ["top_10", "top_20"], predict_only=False) is False


def test_resume_result_accepts_retrieval_only_checkpoint_for_predict_only():
    result = {
        "question_id": "conv0_q0",
        "retrieval": {"search_results": []},
    }

    assert _resume_result_complete(result, ["top_10"], predict_only=True) is True
    assert _resume_result_complete(result, ["top_10"], predict_only=False) is False


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Answer generation failed for top_10: empty response after retries"),
        RuntimeError("Judge failed for top_50: invalid label None"),
        RuntimeError("Server disconnected"),
    ],
)
def test_retryable_question_failure_recognizes_transient_pipeline_errors(error):
    assert _retryable_question_failure(error) is True


def test_retryable_question_failure_rejects_configuration_errors():
    assert _retryable_question_failure(ValueError("top_k must be positive")) is False


def test_adaptive_context_ids_exactly_follow_final_support_order_without_anchor():
    formatted = [
        {"id": "m1", "memory": "baseline first"},
        {"id": "m2", "memory": "not selected"},
        {"id": "m3", "memory": "selected first"},
    ]

    context, diagnostics = resolve_locomo_context(formatted, ADAPTIVE_CONTEXT_LABEL, _metadata())

    assert [item["id"] for item in context] == ["m3", "m1"]
    assert diagnostics["support_count"] == 2
    assert diagnostics["context_policy"] == ADAPTIVE_CONTEXT_LABEL
    assert diagnostics["sparse_omp_policy_version"] == "sparse_omp_v2"


def test_adaptive_context_keeps_budget_capped_support_and_allows_single_memory():
    context, diagnostics = resolve_locomo_context(
        [{"id": "only", "memory": "one"}, {"id": "anchor", "memory": "unused"}],
        ADAPTIVE_CONTEXT_LABEL,
        _metadata(
            support_count=1,
            support_ids=["only"],
            stop_reason="max_support",
            converged=False,
            budget_capped=True,
        ),
    )

    assert [item["id"] for item in context] == ["only"]
    assert diagnostics["budget_capped"] is True
    assert diagnostics["converged"] is False


def test_fallback_era_and_wrong_mode_metadata_are_rejected():
    assert compatible_sparse_omp_metadata(_metadata(), expected_mode="adaptive") is True
    assert compatible_sparse_omp_metadata(_metadata(fallback_used=False), expected_mode="adaptive") is False
    assert compatible_sparse_omp_metadata(_metadata(policy_version="old"), expected_mode="adaptive") is False
    assert compatible_sparse_omp_metadata(_metadata(mode="fixed"), expected_mode="adaptive") is False


def test_adaptive_context_rejects_missing_or_unreproducible_metadata():
    with pytest.raises(RuntimeError, match="requires sparse_omp"):
        resolve_locomo_context([], ADAPTIVE_CONTEXT_LABEL, None)
    with pytest.raises(RuntimeError, match="support memories are missing"):
        resolve_locomo_context([{"id": "m1"}], ADAPTIVE_CONTEXT_LABEL, _metadata())


def test_sparse_omp_performance_report_deduplicates_and_uses_nearest_rank(tmp_path):
    path = tmp_path / "_sparse_omp_performance.json"
    base = {
        "query_embedding": 1.0,
        "support_selection": 10.0,
        "search_total": 20.0,
        "stop_reason": "fw_gap",
        "embedding_reused": True,
        "converged": True,
        "budget_capped": False,
        "concurrency_wait": 2.0,
        "vector_cache_hits": 150,
        "vector_cache_misses": 50,
        "support_count": 2,
        "tail_count": 8,
        "adaptive_context_count": 2,
    }
    update_sparse_omp_performance_report(str(path), "q1", base)
    update_sparse_omp_performance_report(str(path), "q2", {**base, "support_selection": 30.0})
    update_sparse_omp_performance_report(str(path), "q1", {**base, "support_selection": 20.0})

    report = json.loads(path.read_text())
    assert list(report["samples"]) == ["q1", "q2"]
    assert report["summary"]["sample_count"] == 2
    assert report["summary"]["timings_ms"]["support_selection"]["median"] == 25.0
    assert report["summary"]["timings_ms"]["support_selection"]["p90"] == 30.0
    assert report["summary"]["timings_ms"]["concurrency_wait"]["median"] == 2.0
    assert report["summary"]["vector_cache_hits"]["mean"] == 150.0
    assert report["summary"]["vector_cache_misses"]["mean"] == 50.0
    assert report["summary"]["vector_cache_hit_rate"] == 0.75


def test_performance_report_excludes_none_stages_but_keeps_failed_total(tmp_path):
    path = tmp_path / "_sparse_omp_performance.json"
    update_sparse_omp_performance_report(
        str(path),
        "failed",
        {
            "vector_prepare": None,
            "support_selection": None,
            "rerank_total": 7.5,
            "search_total": 9.0,
            "outcome": "error",
            "error_code": "sparse_omp_rerank_failed",
        },
    )

    summary = json.loads(path.read_text())["summary"]
    assert summary["timings_ms"]["vector_prepare"]["count"] == 0
    assert summary["timings_ms"]["rerank_total"]["mean"] == 7.5
    assert summary["outcome"] == {"error": 1}


def test_transformers_reranker_report_uses_sparse_omp_percentile_method(tmp_path):
    path = tmp_path / "_reranker_performance.json"
    base = {
        "provider": "transformers_llm",
        "model": "Qwen/Qwen3.5-0.8B",
        "candidate_count": 20,
        "batch_count": 3,
        "input_tokens": 500,
        "output_tokens": 20,
        "pair_prepare": 1.0,
        "inference": 10.0,
        "score_postprocess": 2.0,
        "rerank_total": 12.0,
        "search_total": 15.0,
        "outcome": "success",
        "error_code": None,
    }
    update_reranker_performance_report(str(path), "q1", base)
    update_reranker_performance_report(str(path), "q2", {**base, "inference": 30.0})

    summary = json.loads(path.read_text())["summary"]
    assert summary["sample_count"] == 2
    assert summary["model"] == {"Qwen/Qwen3.5-0.8B": 2}
    assert summary["timings_ms"]["pair_prepare"]["median"] == 1.0
    assert summary["timings_ms"]["inference"]["median"] == 20.0
    assert summary["timings_ms"]["inference"]["p90"] == 30.0
    assert summary["timings_ms"]["score_postprocess"]["median"] == 2.0
    assert summary["docs_per_second"]["count"] == 2


def test_context_policy_labels_are_mutually_exclusive():
    assert context_policy_labels([10, 20], adaptive_context=False) == ["top_10", "top_20"]
    assert context_policy_labels([10, 20], adaptive_context=True) == [ADAPTIVE_CONTEXT_LABEL]


def test_adaptive_context_ignores_fixed_cutoff_bounds():
    validate_context_cutoffs([10, 20, 50, 200], top_k=50, adaptive_context=True)

    with pytest.raises(ValueError, match="maximum --top-k-cutoffs"):
        validate_context_cutoffs([10, 20, 50, 200], top_k=50, adaptive_context=False)


def test_evaluate_only_selects_only_missing_labels_unless_rejudge():
    existing = {"top_10": {"score": 1.0}, "legacy": {"score": 0.0}}
    assert labels_to_evaluate(existing, ["top_10", "top_20"], rejudge=False) == ["top_20"]
    assert labels_to_evaluate(existing, ["top_10", "top_20"], rejudge=True) == ["top_10", "top_20"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sparse_omp_rerank", "expected_options"),
    [
        (False, None),
        (
            True,
            {
                "early_stopping": False,
                "min_iterations": 20,
                "short_pool_policy": "use_available",
            },
        ),
    ],
)
async def test_process_question_only_sends_options_for_configured_sparse_omp(
    sparse_omp_rerank,
    expected_options,
):
    class Mem0:
        search_kwargs = None

        async def search(self, *args, **kwargs):
            self.search_kwargs = kwargs
            return []

    mem0 = Mem0()
    await process_question(
        qa={"question": "question", "category": 1, "answer": "answer"},
        qa_idx=0,
        conv_idx=0,
        user_id="user",
        mem0=mem0,
        answerer=None,
        judge_llm=None,
        cutoffs=[10, 20],
        top_k=20,
        threshold=None,
        reference_date_human=None,
        user_profile=None,
        evidence_lookup=None,
        predict_only=True,
        logger=None,
        rerank=True,
        sparse_omp_rerank=sparse_omp_rerank,
    )

    assert mem0.search_kwargs["rerank_options"] == expected_options


@pytest.mark.asyncio
async def test_process_question_captures_server_internal_transformers_timing():
    class Mem0:
        async def search(self, *args, **kwargs):
            return {
                "results": [{"id": "m1", "memory": "memory", "score": 0.5, "rerank_score": 0.9}],
                "rerank_metadata": {
                    "provider": "transformers_llm",
                    "model": "Qwen/Qwen3.5-0.8B",
                },
                "query_debug": {
                    "reranker_performance": {
                        "candidate_count": 1,
                        "batch_count": 1,
                        "inference": 10.0,
                        "rerank_total": 12.0,
                    }
                },
            }

    result = await process_question(
        qa={"question": "question", "category": 1, "answer": "answer"},
        qa_idx=0,
        conv_idx=0,
        user_id="user",
        mem0=Mem0(),
        answerer=None,
        judge_llm=None,
        cutoffs=[1],
        top_k=1,
        threshold=None,
        reference_date_human=None,
        user_profile=None,
        evidence_lookup=None,
        predict_only=True,
        logger=None,
        score_debug=True,
        rerank=True,
    )

    performance = result["_reranker_performance"]
    assert performance["provider"] == "transformers_llm"
    assert performance["model"] == "Qwen/Qwen3.5-0.8B"
    assert performance["inference"] == 10.0
    assert performance["rerank_total"] == 12.0
    assert performance["search_total"] >= 0.0


@pytest.mark.asyncio
async def test_label_evaluation_merges_existing_results_and_checkpoints_each_label():
    class Answerer:
        async def generate(self, **kwargs):
            return "answer"

    class Judge:
        async def generate_structured(self, **kwargs):
            return {"label": "CORRECT", "reasoning": "ok"}

    checkpoints = []
    results = await evaluate_locomo_contexts(
        formatted=[{"id": "m1", "memory": "memory"}],
        rerank_metadata=_metadata(mode="fixed", early_stopping=False, retain_zero_coefficients=True),
        labels=["top_20"],
        question="question",
        category=1,
        processed_answer="answer",
        reference_date_human=None,
        user_profile=None,
        evidence_context="",
        answerer=Answerer(),
        judge_llm=Judge(),
        existing_results={"top_10": {"score": 0.0, "generated_answer": "old"}},
        on_label_complete=lambda label, result: checkpoints.append((label, result["score"])),
    )

    assert results["top_10"]["generated_answer"] == "old"
    assert results["top_20"]["score"] == 1.0
    assert checkpoints == [("top_20", 1.0)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "judge_response", "error"),
    [
        ("", {"label": "CORRECT"}, "empty response after retries"),
        ("answer", None, "no structured response after retries"),
        ("answer", {}, "invalid label"),
    ],
)
async def test_label_evaluation_rejects_exhausted_or_invalid_llm_responses(
    answer,
    judge_response,
    error,
):
    class Answerer:
        async def generate(self, **kwargs):
            return answer

    class Judge:
        async def generate_structured(self, **kwargs):
            return judge_response

    checkpoints = []
    with pytest.raises(RuntimeError, match=error):
        await evaluate_locomo_contexts(
            formatted=[{"id": "m1", "memory": "memory"}],
            rerank_metadata=_metadata(mode="fixed", early_stopping=False, retain_zero_coefficients=True),
            labels=["top_20"],
            question="question",
            category=1,
            processed_answer="answer",
            reference_date_human=None,
            user_profile=None,
            evidence_context="",
            answerer=Answerer(),
            judge_llm=Judge(),
            on_label_complete=lambda label, result: checkpoints.append((label, result)),
        )

    assert checkpoints == []


@pytest.mark.asyncio
async def test_label_interruption_resumes_from_next_missing_label(tmp_path):
    class InterruptingAnswerer:
        def __init__(self):
            self.calls = 0

        async def generate(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interrupted")
            return "answer"

    class Answerer:
        async def generate(self, **kwargs):
            return "answer"

    class Judge:
        async def generate_structured(self, **kwargs):
            return {"label": "CORRECT", "reasoning": "ok"}

    path = tmp_path / "conv0_q0.json"
    result = {
        "question": "question",
        "reference_date": None,
        "retrieval": {
            "search_results": [{"id": "m1", "memory": "memory"}],
            "rerank_metadata": _metadata(
                mode="fixed",
                early_stopping=False,
                retain_zero_coefficients=True,
            ),
        },
        "cutoff_results": {"legacy": {"score": 0.25}},
    }
    qa = {"category": 1, "answer": "answer", "evidence": []}

    with pytest.raises(RuntimeError, match="interrupted"):
        await apply_locomo_judge_to_saved_result(
            result,
            qa,
            0,
            InterruptingAnswerer(),
            Judge(),
            [10, 20],
            None,
            labels=["top_10", "top_20"],
            result_path=str(path),
        )

    saved = json.loads(path.read_text())
    assert set(saved["cutoff_results"]) == {"legacy", "top_10"}
    missing = labels_to_evaluate(saved["cutoff_results"], ["top_10", "top_20"], rejudge=False)
    assert missing == ["top_20"]

    await apply_locomo_judge_to_saved_result(
        saved,
        qa,
        0,
        Answerer(),
        Judge(),
        [10, 20],
        None,
        labels=missing,
        result_path=str(path),
    )
    completed = json.loads(path.read_text())
    assert set(completed["cutoff_results"]) == {"legacy", "top_10", "top_20"}


def test_adaptive_metrics_accept_string_policy_and_report_context_sizes_and_stop_reason():
    evaluations = [
        {
            "category_name": "single-hop",
            "cutoff_results": {
                ADAPTIVE_CONTEXT_LABEL: {"score": 1.0, "memories_evaluated": 1, "stop_reason": "fw_gap"}
            },
        },
        {
            "category_name": "single-hop",
            "cutoff_results": {
                ADAPTIVE_CONTEXT_LABEL: {"score": 0.0, "memories_evaluated": 3, "stop_reason": "max_support"}
            },
        },
    ]

    metrics = compute_locomo_metrics(evaluations, [ADAPTIVE_CONTEXT_LABEL])[ADAPTIVE_CONTEXT_LABEL]

    assert metrics["overall"]["accuracy"] == 50.0
    assert metrics["context_size"] == {"mean": 2, "median": 2.0, "p90": 3.0}
    assert metrics["by_stop_reason"]["fw_gap"]["accuracy"] == 100.0
