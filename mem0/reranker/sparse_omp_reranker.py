from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Any, Dict, List, Mapping, Union

import numpy as np

from mem0.configs.rerankers.base import BaseRerankerConfig
from mem0.configs.rerankers.sparse_omp import SparseOmpRerankerConfig
from mem0.reranker.base import BaseReranker


_EPS = 1e-12
SPARSE_OMP_POLICY_VERSION = "sparse_omp_v2"
logger = logging.getLogger(__name__)


class SparseOmpRerankError(RuntimeError):
    """A non-retryable sparse OMP failure with request-local timing data."""

    code = "sparse_omp_rerank_failed"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
        performance: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.cause = cause
        self.performance = dict(performance or {})


class SparseOmpReranker(BaseReranker):
    """Rerank a retrieved candidate pool with simplex-constrained OMP."""

    def __init__(self, config: Union[BaseRerankerConfig, SparseOmpRerankerConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = SparseOmpRerankerConfig(**config)
        elif isinstance(config, BaseRerankerConfig) and not isinstance(config, SparseOmpRerankerConfig):
            config = SparseOmpRerankerConfig(
                provider=getattr(config, "provider", "sparse_omp"),
                model=getattr(config, "model", None),
                api_key=getattr(config, "api_key", None),
                top_k=getattr(config, "top_k", None),
            )

        self.config = config
        if self.config.constraint != "simplex":
            raise ValueError("Sparse OMP reranker only supports constraint='simplex'")
        self._execution_semaphore = BoundedSemaphore(value=int(self.config.max_concurrency))
        if self.config.backend == "numba":
            from mem0.reranker._sparse_omp_numba import warmup

            warmup()

    @contextmanager
    def execution_slot(self):
        """Bound concurrent Qdrant fetch, vector preparation, and OMP work."""
        wait_started = time.perf_counter()
        self._execution_semaphore.acquire()
        wait_ms = (time.perf_counter() - wait_started) * 1000.0
        try:
            yield wait_ms
        finally:
            self._execution_semaphore.release()

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Reject calls that cannot supply the dense vectors required by sparse OMP."""
        started = time.perf_counter()
        raise SparseOmpRerankError(
            "sparse OMP reranking requires query and candidate dense vectors",
            performance={
                "vector_prepare": None,
                "gram_build": None,
                "support_selection": None,
                "tail_fill": None,
                "rerank_total": (time.perf_counter() - started) * 1000.0,
            },
        )

    def rerank_with_vectors(
        self,
        query_vector: Any,
        documents: List[Dict[str, Any]],
        document_vectors: Mapping[str, Any],
        top_k: int = None,
        *,
        early_stopping: bool | None = None,
        min_iterations: int | None = None,
        short_pool_policy: str = "error",
    ) -> List[Dict[str, Any]]:
        return self.rerank_with_vectors_detailed(
            query_vector,
            documents,
            document_vectors,
            top_k=top_k,
            early_stopping=early_stopping,
            min_iterations=min_iterations,
            short_pool_policy=short_pool_policy,
        )["results"]

    def rerank_with_vectors_detailed(
        self,
        query_vector: Any,
        documents: List[Dict[str, Any]],
        document_vectors: Mapping[str, Any],
        top_k: int = None,
        *,
        early_stopping: bool | None = None,
        min_iterations: int | None = None,
        short_pool_policy: str = "error",
    ) -> Dict[str, Any]:
        """Rerank while returning stable metadata and request-local timing details."""
        started = time.perf_counter()
        final_top_k = top_k if top_k is not None else self.config.top_k
        if final_top_k is None:
            final_top_k = len(documents)
        final_top_k = max(int(final_top_k), 0)
        candidate_count = len(documents)
        request_early_stopping = bool(self.config.early_stopping) if early_stopping is None else early_stopping
        if not isinstance(request_early_stopping, bool):
            raise ValueError("early_stopping must be a boolean")
        if min_iterations is not None and (isinstance(min_iterations, bool) or int(min_iterations) < 1):
            raise ValueError("min_iterations must be a positive integer")
        if request_early_stopping and min_iterations is not None:
            raise ValueError("min_iterations is only valid when early_stopping=false")
        if short_pool_policy not in {"error", "use_available"}:
            raise ValueError("short_pool_policy must be 'error' or 'use_available'")

        mode = "adaptive" if request_early_stopping else "fixed"
        configured_support = self.config.max_support
        if request_early_stopping:
            effective_iteration_limit = (
                int(configured_support) if configured_support is not None else int(self.config.candidate_pool_size)
            )
        else:
            effective_iteration_limit = max(int(configured_support or 0), int(min_iterations or 0))
        requested_iteration_limit = effective_iteration_limit
        short_pool_capped = False
        if not request_early_stopping and short_pool_policy == "use_available":
            effective_iteration_limit = min(effective_iteration_limit, candidate_count)
            short_pool_capped = effective_iteration_limit < requested_iteration_limit
        retain_zero_coefficients = not request_early_stopping

        performance: dict[str, Any] = {
            "vector_prepare": None,
            "gram_build": None,
            "support_selection": None,
            "tail_fill": None,
            "rerank_total": None,
            "valid_vector_count": 0,
            "support_count": 0,
            "tail_count": 0,
            "qp_solve_count": 0,
        }

        if not documents:
            performance["rerank_total"] = (time.perf_counter() - started) * 1000.0
            if not request_early_stopping and effective_iteration_limit > 0:
                cause = ValueError(
                    "fixed sparse OMP requires at least "
                    f"{effective_iteration_limit} candidates, but retrieval returned 0"
                )
                raise SparseOmpRerankError(
                    f"sparse OMP reranking failed: {cause}",
                    cause=cause,
                    performance=performance,
                ) from cause
            return {
                "results": [],
                "rerank_metadata": self._metadata(
                    candidate_count=0,
                    support=[],
                    stop_reason="no_candidates",
                    mode=mode,
                    early_stopping=request_early_stopping,
                    effective_iteration_limit=effective_iteration_limit,
                    retain_zero_coefficients=retain_zero_coefficients,
                    requested_iteration_limit=requested_iteration_limit,
                    short_pool_policy=short_pool_policy,
                    short_pool_capped=short_pool_capped,
                ),
                "performance": performance,
            }

        prepare_started = time.perf_counter()
        try:
            if not request_early_stopping and effective_iteration_limit < 1:
                raise ValueError("fixed sparse OMP requires max_support or request-level min_iterations to be set")
            if not request_early_stopping and candidate_count < effective_iteration_limit:
                raise ValueError(
                    "fixed sparse OMP requires at least "
                    f"{effective_iteration_limit} candidates, but retrieval returned {candidate_count}"
                )

            query = self._prepare_vector(query_vector)
            if query is None:
                raise ValueError("sparse OMP query dense vector is missing, zero, or invalid")

            document_vectors_normalized = bool(getattr(document_vectors, "normalized", False))
            vectors_by_id = {str(key): value for key, value in (document_vectors or {}).items()}
            valid_documents: list[Dict[str, Any]] = []
            valid_vectors: list[np.ndarray] = []
            for position, doc in enumerate(documents):
                mem_id = doc.get("id")
                if mem_id is None:
                    raise ValueError(f"sparse OMP candidate at position {position} has no id")
                key = str(mem_id)
                if key not in vectors_by_id:
                    raise ValueError(f"sparse OMP candidate {key!r} has no stored dense vector")
                vector = self._prepare_vector(
                    vectors_by_id[key],
                    already_normalized=document_vectors_normalized,
                )
                if vector is None:
                    raise ValueError(f"sparse OMP candidate {key!r} has a zero or invalid dense vector")
                if vector.shape[0] != query.shape[0]:
                    raise ValueError(
                        f"sparse OMP candidate {key!r} dense vector dimension {vector.shape[0]} "
                        f"does not match query dimension {query.shape[0]}"
                    )
                valid_documents.append(doc)
                valid_vectors.append(vector)

            matrix = np.vstack(valid_vectors).astype(np.float64, copy=False)
            performance["vector_prepare"] = (time.perf_counter() - prepare_started) * 1000.0
            performance["valid_vector_count"] = len(valid_documents)

            detailed_selection = self.select_support_detailed(
                query,
                matrix,
                max_support=effective_iteration_limit,
                early_stopping=request_early_stopping,
                retain_zero_coefficients=retain_zero_coefficients,
            )
            performance["gram_build"] = detailed_selection["timings_ms"]["gram_build"]
            performance["support_selection"] = detailed_selection["timings_ms"]["support_selection"]
            performance["qp_solve_count"] = int(detailed_selection["qp_solve_count"])
            performance["solver_stats"] = detailed_selection.get("solver_stats", {})

            selected = detailed_selection["support"]
            selected_count = len(selected)
            if request_early_stopping and selected_count == 0:
                raise RuntimeError("adaptive sparse OMP produced no positive-coefficient support")

            tail_started = time.perf_counter()
            head: list[Dict[str, Any]] = []
            for rank, support_item in enumerate(selected, start=1):
                valid_index = int(support_item["candidate_index"])
                doc = valid_documents[valid_index].copy()
                pre_score = doc.get("score")
                score = float((selected_count - rank + 1) / selected_count)
                score_breakdown = dict(doc.get("score_breakdown") or {})
                score_breakdown.update(
                    {
                        "pre_sparse_omp_score": pre_score,
                        "pre_sparse_omp_rank": valid_index + 1,
                        "sparse_omp_score_source": "sparse_omp",
                        "sparse_omp_constraint": self.config.constraint,
                        "sparse_omp_step": rank,
                        "sparse_omp_coefficient": float(support_item["coefficient"]),
                        "sparse_omp_candidate_count": len(valid_documents),
                        "sparse_omp_selected_count": selected_count,
                        "sparse_omp_residual_norm": float(support_item["residual_norm"]),
                        "sparse_omp_marginal_gain": float(support_item["marginal_gain"]),
                    }
                )
                doc["rerank_score"] = score
                doc["score_breakdown"] = score_breakdown
                head.append(doc)
            results = head[:final_top_k]
            performance["tail_fill"] = (time.perf_counter() - tail_started) * 1000.0
            performance["support_count"] = selected_count

            metadata = self._metadata(
                candidate_count=candidate_count,
                support=selected,
                stop_reason=str(detailed_selection["stop_reason"]),
                mode=mode,
                early_stopping=request_early_stopping,
                effective_iteration_limit=effective_iteration_limit,
                retain_zero_coefficients=retain_zero_coefficients,
                requested_iteration_limit=requested_iteration_limit,
                short_pool_policy=short_pool_policy,
                short_pool_capped=short_pool_capped,
                iterations=int(detailed_selection["iterations"]),
                qp_solve_count=int(detailed_selection["qp_solve_count"]),
                relative_residual=detailed_selection["relative_residual"],
                final_fw_gap=detailed_selection["final_fw_gap"],
                monotonicity_warnings=detailed_selection["monotonicity_warnings"],
            )
            metadata["support_ids"] = [str(item["id"]) for item in head]
            performance["rerank_total"] = (time.perf_counter() - started) * 1000.0
            return {"results": results, "rerank_metadata": metadata, "performance": performance}
        except SparseOmpRerankError as exc:
            performance.update(exc.performance)
            if performance["vector_prepare"] is None:
                performance["vector_prepare"] = (time.perf_counter() - prepare_started) * 1000.0
            performance["rerank_total"] = (time.perf_counter() - started) * 1000.0
            exc.performance = performance
            raise
        except Exception as exc:
            if performance["vector_prepare"] is None:
                performance["vector_prepare"] = (time.perf_counter() - prepare_started) * 1000.0
            performance["rerank_total"] = (time.perf_counter() - started) * 1000.0
            raise SparseOmpRerankError(
                f"sparse OMP reranking failed: {exc}",
                cause=exc,
                performance=performance,
            ) from exc

    def _metadata(
        self,
        *,
        candidate_count: int,
        support: list[dict[str, Any]],
        stop_reason: str,
        mode: str,
        early_stopping: bool,
        effective_iteration_limit: int,
        retain_zero_coefficients: bool,
        requested_iteration_limit: int | None = None,
        short_pool_policy: str = "error",
        short_pool_capped: bool = False,
        iterations: int = 0,
        qp_solve_count: int = 0,
        relative_residual: float | None = None,
        final_fw_gap: float | None = None,
        monotonicity_warnings: list[dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        converged = mode == "adaptive" and stop_reason in {
            "residual_target",
            "fw_gap",
            "relative_improvement",
            "candidate_exhausted",
        }
        return {
            "provider": "sparse_omp",
            "policy_version": SPARSE_OMP_POLICY_VERSION,
            "mode": mode,
            "early_stopping": bool(early_stopping),
            "effective_iteration_limit": int(effective_iteration_limit),
            "requested_iteration_limit": int(
                effective_iteration_limit if requested_iteration_limit is None else requested_iteration_limit
            ),
            "short_pool_policy": short_pool_policy,
            "short_pool_capped": bool(short_pool_capped),
            "qp_solve_count": int(qp_solve_count),
            "retain_zero_coefficients": bool(retain_zero_coefficients),
            "candidate_pool_size": int(self.config.candidate_pool_size),
            "candidate_count": int(candidate_count),
            "support_count": len(support),
            "support_ids": [],
            "stop_reason": stop_reason,
            "converged": converged,
            "budget_capped": mode == "adaptive" and stop_reason == "max_support",
            "iterations": int(iterations),
            "max_support": int(self.config.max_support) if self.config.max_support is not None else None,
            "relative_residual": relative_residual,
            "final_fw_gap": final_fw_gap,
            "monotonicity_warning_count": len(monotonicity_warnings or []),
            "monotonicity_warnings": list(monotonicity_warnings or []),
        }

    def select_support(
        self,
        query: np.ndarray,
        matrix: np.ndarray,
        top_k: int,
        *,
        early_stopping: bool = False,
    ) -> list[dict[str, float | int]]:
        """Compatibility wrapper; semantic sparse OMP explicitly retains its selection limit."""
        return self.select_support_detailed(
            query,
            matrix,
            max_support=top_k,
            early_stopping=early_stopping,
            retain_zero_coefficients=not early_stopping,
        )["support"]

    def select_support_detailed(
        self,
        query: np.ndarray,
        matrix: np.ndarray,
        *,
        max_support: int,
        early_stopping: bool,
        retain_zero_coefficients: bool = False,
    ) -> Dict[str, Any]:
        selection_started = time.perf_counter()
        if max_support <= 0 or matrix.size == 0:
            return self._empty_selection("candidate_exhausted", selection_started)

        strict_iterations = not early_stopping
        if strict_iterations and int(max_support) > matrix.shape[0]:
            raise SparseOmpRerankError(
                f"fixed sparse OMP requires {int(max_support)} candidates, but only {matrix.shape[0]} are available",
                performance={
                    "gram_build": None,
                    "support_selection": None,
                    "qp_solve_count": 0,
                },
            )

        gram_started = time.perf_counter()
        try:
            full_gram = np.asarray(np.matmul(matrix, matrix.T), dtype=np.float64)
            full_gram = 0.5 * (full_gram + full_gram.T)
            full_linear = np.asarray(np.matmul(matrix, query), dtype=np.float64)
            query_norm_sq = float(np.dot(query, query))
        except Exception as exc:
            raise SparseOmpRerankError(
                f"sparse OMP Gram construction failed: {exc}",
                cause=exc,
                performance={
                    "gram_build": (time.perf_counter() - gram_started) * 1000.0,
                    "support_selection": None,
                    "qp_solve_count": 0,
                },
            ) from exc
        gram_ms = (time.perf_counter() - gram_started) * 1000.0

        addition_limit = int(max_support) if strict_iterations else min(int(max_support), matrix.shape[0])
        attempted: list[int] = []
        rejected: set[int] = set()
        unavailable_mask = np.zeros(matrix.shape[0], dtype=np.bool_)
        # Store selected Gram columns as contiguous rows, avoiding repeated fancy indexing.
        selected_columns = np.empty((addition_limit, matrix.shape[0]), dtype=np.float64)
        selected_gram = np.empty((addition_limit, addition_limit), dtype=np.float64)
        selected_linear = np.empty(addition_limit, dtype=np.float64)
        warm_buffer = np.zeros(addition_limit, dtype=np.float64)
        full_coefficients = np.zeros(matrix.shape[0], dtype=np.float64)
        residual_correlations = full_linear.copy()
        coefficients = np.zeros(0, dtype=np.float64)
        atom_metrics: dict[int, dict[str, float | int]] = {}
        previous_residual_sq = max(query_norm_sq, 0.0)
        previous_residual_norm = float(np.sqrt(previous_residual_sq))
        previous_regularized_objective: float | None = None
        snapshot_limit = int(self.config.early_stop_patience) + 1 if early_stopping else 1
        snapshots: deque[dict[str, Any]] = deque(maxlen=snapshot_limit)
        monotonicity_warnings: list[dict[str, Any]] = []
        low_improvement_count = 0
        stop_reason: str | None = None
        qp_solve_count = 0
        solver_stats = self._new_solver_stats(self.config.backend)
        support_started = time.perf_counter()

        while len(attempted) < addition_limit:
            correlations = residual_correlations.copy()
            correlations[unavailable_mask] = -np.inf
            best_index = int(np.argmax(correlations))
            if not np.isfinite(correlations[best_index]):
                if strict_iterations:
                    raise SparseOmpRerankError(
                        f"fixed sparse OMP exhausted candidates after {len(attempted)} of {addition_limit} iterations",
                        performance={
                            "gram_build": gram_ms,
                            "support_selection": (time.perf_counter() - support_started) * 1000.0,
                            "qp_solve_count": qp_solve_count,
                        },
                    )
                stop_reason = "candidate_exhausted"
                break

            next_attempted = [*attempted, best_index]
            next_array = np.asarray(next_attempted, dtype=np.int64)
            previous_size = len(attempted)
            next_size = previous_size + 1
            selected_columns[previous_size] = full_gram[:, best_index]
            selected_gram[previous_size, :next_size] = full_gram[best_index, next_array]
            selected_gram[:next_size, previous_size] = selected_gram[previous_size, :next_size]
            selected_linear[previous_size] = full_linear[best_index]
            active_gram = selected_gram[:next_size, :next_size]
            active_linear = selected_linear[:next_size]
            warm_buffer[:previous_size] = coefficients
            warm_buffer[previous_size] = 0.0
            warm_start = warm_buffer[:next_size] if attempted else None
            qp_solve_count += 1
            try:
                next_coefficients = self.solve_simplex_least_squares(
                    active_gram,
                    active_linear,
                    ridge=float(self.config.ridge),
                    negative_tolerance=float(self.config.negative_tolerance),
                    warm_start=warm_start,
                    backend=self.config.backend,
                    stats=solver_stats,
                )
            except Exception as exc:
                raise SparseOmpRerankError(
                    f"sparse OMP simplex QP failed at iteration {qp_solve_count}: {exc}",
                    cause=exc,
                    performance={
                        "gram_build": gram_ms,
                        "support_selection": (time.perf_counter() - support_started) * 1000.0,
                        "qp_solve_count": qp_solve_count,
                        "solver_stats": solver_stats,
                    },
                ) from exc
            next_coefficients = np.asarray(next_coefficients, dtype=np.float64)
            feasibility_tolerance = max(float(self.config.negative_tolerance), 1e-12)
            simplex_tolerance = max(10.0 * feasibility_tolerance, 1e-9)
            if (
                next_coefficients.shape != (len(next_attempted),)
                or not np.all(np.isfinite(next_coefficients))
                or float(np.min(next_coefficients)) < -feasibility_tolerance
                or abs(float(np.sum(next_coefficients)) - 1.0) > simplex_tolerance
            ):
                raise SparseOmpRerankError(
                    f"sparse OMP simplex QP returned an infeasible solution at iteration {qp_solve_count}",
                    performance={
                        "gram_build": gram_ms,
                        "support_selection": (time.perf_counter() - support_started) * 1000.0,
                        "qp_solve_count": qp_solve_count,
                    },
                )

            solved_residual_sq = (
                query_norm_sq
                - 2.0 * float(np.dot(next_coefficients, active_linear))
                + float(np.dot(next_coefficients, active_gram @ next_coefficients))
            )
            if not np.isfinite(solved_residual_sq):
                raise SparseOmpRerankError(
                    f"sparse OMP simplex QP produced a non-finite objective at iteration {qp_solve_count}",
                    performance={
                        "gram_build": gram_ms,
                        "support_selection": (time.perf_counter() - support_started) * 1000.0,
                        "qp_solve_count": qp_solve_count,
                    },
                )
            solved_residual_sq = max(float(solved_residual_sq), 0.0)
            solved_regularized_objective = solved_residual_sq + float(self.config.ridge) * float(
                np.dot(next_coefficients, next_coefficients)
            )
            residual_sq = solved_residual_sq
            regularized_objective = solved_regularized_objective

            # The empty support is not simplex-feasible, so the first QP solve has no
            # comparable incumbent. On later iterations, compare the same ridge-
            # regularized objective that the QP solver actually minimizes.
            if attempted and previous_regularized_objective is not None:
                monotonic_tolerance = simplex_tolerance * max(1.0, abs(previous_regularized_objective))
            else:
                monotonic_tolerance = None
            if (
                monotonic_tolerance is not None
                and solved_regularized_objective > previous_regularized_objective + monotonic_tolerance
            ):
                if strict_iterations:
                    warning = {
                        "iteration": qp_solve_count,
                        "candidate_index": best_index,
                        "previous_regularized_objective": float(previous_regularized_objective),
                        "solved_regularized_objective": float(solved_regularized_objective),
                        "increase": float(solved_regularized_objective - previous_regularized_objective),
                        "tolerance": float(monotonic_tolerance),
                        "action": "kept_incumbent",
                    }
                    monotonicity_warnings.append(warning)
                    logger.warning(
                        "Fixed sparse OMP regularized objective increased at iteration %d "
                        "after adding candidate %d (previous=%.12g, solved=%.12g, tolerance=%.12g); "
                        "keeping the incumbent coefficients and continuing",
                        qp_solve_count,
                        best_index,
                        previous_regularized_objective,
                        solved_regularized_objective,
                        monotonic_tolerance,
                    )
                    next_coefficients = np.concatenate([coefficients, np.zeros(1, dtype=np.float64)])
                    residual_sq = previous_residual_sq
                    regularized_objective = previous_regularized_objective
                else:
                    rejected.add(best_index)
                    unavailable_mask[best_index] = True
                    continue

            residual_norm = float(np.sqrt(residual_sq))
            relative_improvement = (previous_residual_sq - residual_sq) / max(previous_residual_sq, _EPS)
            attempted = next_attempted
            coefficients = next_coefficients
            unavailable_mask[best_index] = True
            atom_metrics[best_index] = {
                "candidate_index": best_index,
                "residual_norm": residual_norm,
                "marginal_gain": previous_residual_norm - residual_norm,
            }

            gradient_started = time.perf_counter()
            prediction = selected_columns[:next_size].T @ coefficients
            residual_correlations = full_linear - prediction
            full_coefficients.fill(0.0)
            full_coefficients[next_array] = coefficients
            objective_gradient = -residual_correlations
            if self.config.ridge:
                objective_gradient += float(self.config.ridge) * full_coefficients
            fw_gap = max(float(np.dot(full_coefficients, objective_gradient) - np.min(objective_gradient)), 0.0)
            solver_stats["gradient_ms"] += (time.perf_counter() - gradient_started) * 1000.0
            relative_residual = residual_norm / max(float(np.sqrt(query_norm_sq)), _EPS)
            snapshot = {
                "attempted": list(attempted),
                "coefficients": coefficients.copy(),
                "residual_sq": residual_sq,
                "regularized_objective": regularized_objective,
                "relative_residual": relative_residual,
                "fw_gap": fw_gap,
                "iterations": len(attempted),
            }
            snapshots.append(snapshot)

            if (
                early_stopping
                and self.config.residual_tol is not None
                and relative_residual <= self.config.residual_tol
            ):
                stop_reason = "residual_target"
                break
            if early_stopping and fw_gap <= self.config.fw_gap_tol:
                stop_reason = "fw_gap"
                break
            if early_stopping and len(snapshots) > 1:
                if relative_improvement <= self.config.relative_improvement_tol:
                    low_improvement_count += 1
                else:
                    low_improvement_count = 0
                if low_improvement_count >= self.config.early_stop_patience:
                    chosen = snapshots[0]
                    attempted = list(chosen["attempted"])
                    coefficients = chosen["coefficients"].copy()
                    snapshots.clear()
                    snapshots.append(chosen)
                    stop_reason = "relative_improvement"
                    break

            previous_residual_sq = residual_sq
            previous_residual_norm = residual_norm
            previous_regularized_objective = regularized_objective

        if not snapshots:
            return self._empty_selection(
                stop_reason or "candidate_exhausted",
                selection_started,
                gram_ms=gram_ms,
                qp_solve_count=qp_solve_count,
            )
        chosen = snapshots[-1]
        attempted = list(chosen["attempted"])
        coefficients = chosen["coefficients"]
        positive_tolerance = max(float(self.config.negative_tolerance), _EPS)
        support: list[dict[str, float | int]] = []
        for candidate_index, coefficient in zip(attempted, coefficients):
            if not retain_zero_coefficients and coefficient <= positive_tolerance:
                continue
            item = dict(atom_metrics[candidate_index])
            item["coefficient"] = float(coefficient)
            support.append(item)

        if stop_reason is None:
            if strict_iterations:
                stop_reason = "fixed_iterations"
            else:
                unavailable = len(attempted) + len(rejected) >= matrix.shape[0]
                stop_reason = "candidate_exhausted" if unavailable else "max_support"
        if strict_iterations and (len(attempted) != addition_limit or qp_solve_count != addition_limit):
            raise SparseOmpRerankError(
                "fixed sparse OMP did not complete the requested number of iterations",
                performance={
                    "gram_build": gram_ms,
                    "support_selection": (time.perf_counter() - support_started) * 1000.0,
                    "qp_solve_count": qp_solve_count,
                },
            )
        return {
            "support": support,
            "stop_reason": stop_reason,
            "iterations": int(chosen["iterations"]),
            "relative_residual": float(chosen["relative_residual"]),
            "final_fw_gap": float(chosen["fw_gap"]),
            "qp_solve_count": qp_solve_count,
            "monotonicity_warnings": monotonicity_warnings,
            "solver_stats": solver_stats,
            "timings_ms": {
                "gram_build": gram_ms,
                "support_selection": (time.perf_counter() - support_started) * 1000.0,
            },
        }

    @staticmethod
    def _empty_selection(
        stop_reason: str,
        started: float,
        *,
        gram_ms: float = 0.0,
        qp_solve_count: int = 0,
    ) -> Dict[str, Any]:
        return {
            "support": [],
            "stop_reason": stop_reason,
            "iterations": 0,
            "relative_residual": None,
            "final_fw_gap": None,
            "qp_solve_count": qp_solve_count,
            "monotonicity_warnings": [],
            "timings_ms": {
                "gram_build": gram_ms,
                "support_selection": max((time.perf_counter() - started) * 1000.0 - gram_ms, 0.0),
            },
        }

    @staticmethod
    def _new_solver_stats(backend: str) -> Dict[str, Any]:
        return {
            "backend": backend,
            "qp_iterations": 0,
            "kkt_solve_count": 0,
            "kkt_lstsq_count": 0,
            "max_working_set": 0,
            "kkt_size_counts": {},
            "warm_start_skips": 0,
            "numba_fallback_count": 0,
            "qp_kernel_ms": 0.0,
            # Fine-grained clocks run only in the NumPy backend; no object-mode calls in Numba.
            "numpy_kkt_solve_ms": 0.0,
            "gradient_ms": 0.0,
        }

    @staticmethod
    def solve_simplex_least_squares(
        gram: Any,
        linear: Any,
        *,
        ridge: float = 1e-10,
        negative_tolerance: float = 1e-10,
        warm_start: Any = None,
        backend: str = "numpy",
        stats: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Solve a convex simplex QP with a primal-dual working set."""
        if backend not in {"numpy", "numba"}:
            raise ValueError("backend must be 'numpy' or 'numba'")
        gram_array = np.asarray(gram, dtype=np.float64)
        linear_array = np.asarray(linear, dtype=np.float64)
        if gram_array.ndim != 2 or gram_array.shape[0] != gram_array.shape[1]:
            raise ValueError("gram must be a square matrix")
        if linear_array.shape != (gram_array.shape[0],):
            raise ValueError("linear must have shape (gram.shape[0],)")
        if not np.all(np.isfinite(gram_array)) or not np.all(np.isfinite(linear_array)):
            raise ValueError("gram and linear must contain finite values")

        size = gram_array.shape[0]
        if size == 0:
            return np.zeros(0, dtype=np.float64)

        ridge = max(float(ridge), 0.0)
        negative_tolerance = max(float(negative_tolerance), 0.0)
        hessian = np.ascontiguousarray(0.5 * (gram_array + gram_array.T))
        if ridge:
            hessian.flat[:: size + 1] += ridge

        warm = np.asarray(warm_start, dtype=np.float64) if warm_start is not None else None
        if warm is not None and warm.shape == (size,) and np.all(np.isfinite(warm)):
            solution = np.maximum(warm, 0.0)
            total = float(solution.sum())
            if total <= _EPS:
                solution = None
            else:
                solution /= total
        else:
            solution = None
        use_warm = solution is not None
        if solution is None:
            solution = np.zeros(size, dtype=np.float64)
            vertex_objectives = 0.5 * np.diag(hessian) - linear_array
            solution[int(np.argmin(vertex_objectives))] = 1.0

        feasibility_tolerance = max(negative_tolerance, 1e-12)
        kkt_tolerance = max(10.0 * feasibility_tolerance, 1e-9)
        if stats is None:
            stats = SparseOmpReranker._new_solver_stats(backend)
        solve_started = time.perf_counter()
        try:
            if size == 1:
                return np.ones(1, dtype=np.float64)
            if backend == "numba":
                from mem0.reranker._sparse_omp_numba import solve_simplex

                result, counters, sizes, status = solve_simplex(
                    hessian,
                    np.ascontiguousarray(linear_array),
                    solution.copy(),
                    use_warm,
                    feasibility_tolerance,
                    kkt_tolerance,
                    max(100, 10 * size),
                )
                stats["qp_iterations"] += int(counters[0])
                stats["warm_start_skips"] += int(counters[1])
                for active_size, count in enumerate(sizes):
                    if count:
                        SparseOmpReranker._record_kkt_size(stats, active_size, int(count))
                if status == 0:
                    return result
                # Singular systems and numerical failures retain the direct KKT/lstsq path.
                stats["numba_fallback_count"] += 1
            return SparseOmpReranker._solve_simplex_numpy(
                hessian, linear_array, solution, use_warm, feasibility_tolerance, kkt_tolerance, stats
            )
        finally:
            stats["qp_kernel_ms"] += (time.perf_counter() - solve_started) * 1000.0

    @staticmethod
    def _record_kkt_size(stats: Dict[str, Any], size: int, count: int = 1) -> None:
        stats["kkt_solve_count"] += count
        stats["max_working_set"] = max(stats["max_working_set"], size)
        key = str(size)
        stats["kkt_size_counts"][key] = stats["kkt_size_counts"].get(key, 0) + count

    @staticmethod
    def _simplex_kkt_status(hessian, linear, solution, active, feasibility_tolerance, kkt_tolerance):
        gradient = hessian @ solution - linear
        multiplier = -float(np.mean(gradient[active]))
        stationary = (
            float(np.max(np.abs(gradient[active] + multiplier))) <= kkt_tolerance
            and abs(float(solution.sum()) - 1.0) <= kkt_tolerance
            and float(np.min(solution)) >= -feasibility_tolerance
        )
        excluded = np.flatnonzero(~active)
        violating = -1
        if excluded.size:
            reduced = gradient[excluded] + multiplier
            pos = int(np.argmin(reduced))
            if reduced[pos] < -kkt_tolerance:
                violating = int(excluded[pos])
        return stationary, violating

    @staticmethod
    def _solve_simplex_numpy(hessian, linear, solution, use_warm, feasibility_tolerance, kkt_tolerance, stats):
        size = solution.size
        max_iterations = max(100, 10 * size)
        active = solution > feasibility_tolerance
        if not np.any(active):
            active[int(np.argmax(solution))] = True

        if use_warm and np.all(solution[~active] == 0.0):
            stationary, violating = SparseOmpReranker._simplex_kkt_status(
                hessian, linear, solution, active, feasibility_tolerance, kkt_tolerance
            )
            if stationary:
                stats["warm_start_skips"] += 1
                if violating < 0:
                    return solution
                active[violating] = True

        workspace = (np.empty((size + 1) ** 2, dtype=np.float64), np.empty(size + 1, dtype=np.float64))

        for _ in range(max_iterations):
            stats["qp_iterations"] += 1
            active_array = np.flatnonzero(active)
            SparseOmpReranker._record_kkt_size(stats, active_array.size)
            active_solution = SparseOmpReranker._solve_simplex_equality_kkt(
                hessian[np.ix_(active_array, active_array)],
                linear[active_array],
                workspace=workspace,
                stats=stats,
            )
            if np.any(active_solution < -feasibility_tolerance):
                current = solution[active_array]
                direction = active_solution - current
                decreasing = direction < -feasibility_tolerance
                if not np.any(decreasing):
                    raise RuntimeError("Simplex QP working set produced an infeasible direction")
                alpha = min(float(np.min(current[decreasing] / -direction[decreasing])), 1.0)
                next_active_solution = current + alpha * direction
                next_active_solution[np.abs(next_active_solution) <= feasibility_tolerance] = 0.0
                solution.fill(0.0)
                solution[active_array] = np.maximum(next_active_solution, 0.0)
                total = float(solution.sum())
                if total <= _EPS:
                    raise RuntimeError("Simplex QP working set lost primal feasibility")
                solution /= total
                active = solution > feasibility_tolerance
                continue

            solution.fill(0.0)
            solution[active_array] = np.maximum(active_solution, 0.0)
            total = float(solution.sum())
            if total <= _EPS:
                raise RuntimeError("Simplex QP equality solution has zero mass")
            solution /= total
            active = solution > feasibility_tolerance
            stationary, violating = SparseOmpReranker._simplex_kkt_status(
                hessian, linear, solution, active, feasibility_tolerance, kkt_tolerance
            )
            if violating >= 0:
                active[violating] = True
                continue
            if stationary:
                return solution

        raise RuntimeError(f"Simplex QP working set did not converge in {max_iterations} iterations")

    @staticmethod
    def _solve_simplex_equality(gram: np.ndarray, linear: np.ndarray) -> np.ndarray:
        """Solve the equality-constrained simplex LS system before nonnegative clipping."""
        # Solving the augmented KKT system directly is materially more stable
        # than forming two inverse actions when the Gram matrix is singular or
        # nearly singular (common with duplicate memory embeddings).
        return SparseOmpReranker._solve_simplex_equality_kkt(gram, linear)

    @staticmethod
    def _solve_simplex_equality_kkt(gram: np.ndarray, linear: np.ndarray, *, workspace=None, stats=None) -> np.ndarray:
        size = gram.shape[0]
        if workspace is None:
            workspace = (np.empty((size + 1) ** 2), np.empty(size + 1))
        kkt = workspace[0][: (size + 1) ** 2].reshape(size + 1, size + 1)
        kkt[:size, :size] = gram
        kkt[:size, size] = 1.0
        kkt[size, :size] = 1.0
        kkt[size, size] = 0.0
        rhs = workspace[1][: size + 1]
        rhs[:size] = linear
        rhs[size] = 1.0
        started = time.perf_counter()
        try:
            return np.linalg.solve(kkt, rhs)[:-1]
        except np.linalg.LinAlgError:
            if stats is not None:
                stats["kkt_lstsq_count"] += 1
            return np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]
        finally:
            if stats is not None:
                stats["numpy_kkt_solve_ms"] += (time.perf_counter() - started) * 1000.0

    def _prepare_vector(self, value: Any, *, already_normalized: bool = False) -> np.ndarray | None:
        if (
            already_normalized
            and isinstance(value, np.ndarray)
            and value.dtype == np.float64
            and value.ndim == 1
            and value.flags.c_contiguous
            and not value.flags.writeable
        ):
            return value

        vector = self._to_1d_array(value)
        if vector is None or vector.size == 0 or not np.all(np.isfinite(vector)):
            return None
        if already_normalized:
            return vector.astype(np.float64, copy=False)
        norm = float(np.linalg.norm(vector))
        if norm <= _EPS:
            return None
        if self.config.normalize_vectors:
            vector = vector / norm
        return vector.astype(np.float64, copy=False)

    @staticmethod
    def _to_1d_array(value: Any) -> np.ndarray | None:
        if value is None or isinstance(value, (str, bytes, dict)):
            return None
        if hasattr(value, "indices") and hasattr(value, "values"):
            return None
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if vector.ndim != 1:
            return None
        return vector
