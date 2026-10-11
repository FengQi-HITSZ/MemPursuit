from typing import Literal
from pydantic import BaseModel, Field


class LlmConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    config: dict = Field(default_factory=dict, description="OpenAI-compatible extraction LLM")
    model_config = {"extra": "forbid"}
