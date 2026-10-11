from abc import ABC


class BaseEmbedderConfig(ABC):
    """Settings for the paper's OpenAI-compatible and local embedding providers."""

    def __init__(
        self,
        model=None,
        api_key=None,
        embedding_dims=None,
        openai_base_url=None,
        model_kwargs=None,
        huggingface_base_url=None,
    ):
        self.model = model
        self.api_key = api_key
        self.embedding_dims = embedding_dims
        self.openai_base_url = openai_base_url
        self.model_kwargs = model_kwargs or {}
        self.huggingface_base_url = huggingface_base_url
