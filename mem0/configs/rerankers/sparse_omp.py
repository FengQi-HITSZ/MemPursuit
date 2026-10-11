from typing import Literal, Optional

from pydantic import Field, model_validator

from mem0.configs.rerankers.base import BaseRerankerConfig


class SparseOmpRerankerConfig(BaseRerankerConfig):
    """
    Configuration for sparse OMP reranking over an existing retrieval pool.
    """

    constraint: Literal["simplex"] = Field(
        default="simplex",
        description="Sparse coefficient constraint. Only simplex is supported.",
    )
    backend: Literal["numpy", "numba"] = Field(
        default="numpy",
        description="Simplex QP backend. numba requires the fast extra and is warmed at construction.",
    )
    normalize_vectors: bool = Field(
        default=True,
        description="L2-normalize query and memory vectors before OMP selection.",
    )
    ridge: float = Field(
        default=1e-10,
        ge=0.0,
        description="Small diagonal ridge used for stable KKT solves.",
    )
    negative_tolerance: float = Field(
        default=1e-10,
        ge=0.0,
        description="Tolerance for active-set simplex coefficient clipping.",
    )
    candidate_pool_size: int = Field(
        default=200,
        ge=1,
        description="Number of baseline retrieval candidates optimized by sparse OMP.",
    )
    max_concurrency: int = Field(
        default=2,
        ge=1,
        description=(
            "Maximum number of concurrent sparse OMP fetch/prepare/solve sections per process. "
            "This prevents Qdrant and BLAS contention without serializing answer generation or judging."
        ),
    )
    max_support: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional hard limit on greedy atom additions and simplex QP size. "
            "Adaptive requests use candidate_pool_size when this is unset."
        ),
    )
    early_stopping: bool = Field(
        default=True,
        description="Enable residual, Frank-Wolfe gap, and relative-improvement stopping.",
    )
    fw_gap_tol: float = Field(
        default=1e-5,
        ge=0.0,
        description="Stop when the all-candidate simplex Frank-Wolfe gap is at most this value.",
    )
    relative_improvement_tol: float = Field(
        default=1e-5,
        ge=0.0,
        description="Minimum relative squared-residual improvement considered useful.",
    )
    early_stop_patience: int = Field(
        default=2,
        ge=1,
        description="Consecutive low-improvement additions required before rolling the whole window back.",
    )
    residual_tol: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional relative reconstruction-residual target; disabled by default.",
    )

    @model_validator(mode="after")
    def _validate_residual_normalization(self):
        if self.residual_tol is not None and not self.normalize_vectors:
            raise ValueError("residual_tol requires normalize_vectors=true")
        return self

    model_config = {"extra": "forbid"}
