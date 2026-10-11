from typing import Literal
from pydantic import BaseModel, Field


class EmbedderConfig(BaseModel):
    provider: Literal["openai", "huggingface"] = "openai"
    config: dict = Field(default_factory=dict, description="Embedding provider")
    model_config = {"extra": "forbid"}
