from typing import Optional, Literal

from pydantic import BaseModel, Field


class RerankerConfig(BaseModel):
    """Configuration for rerankers."""

    provider: Literal["sparse_omp"] = Field(description="Paper reranker", default="sparse_omp")
    config: Optional[dict] = Field(description="Provider-specific reranker configuration", default=None)

    model_config = {"extra": "forbid"}
