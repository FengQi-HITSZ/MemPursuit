import os
import warnings
from typing import Literal, Optional

import tiktoken
from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class OpenAIEmbedding(EmbeddingBase):
    # OpenAI embedding models accept at most 8192 input tokens. Keep a small
    # margin for tokenizer/API boundary differences.
    max_input_tokens = 8000

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-3-small"
        # Only pass `dimensions` to the API when the user set embedding_dims; non-matryoshka
        # OpenAI-compatible backends (vLLM, Voyage, etc.) reject the parameter
        self._pass_dimensions_to_api = self.config.embedding_dims is not None
        self.config.embedding_dims = self.config.embedding_dims or 1536
        self._token_encoding = None

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            self.config.openai_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if os.environ.get("OPENAI_API_BASE"):
            warnings.warn(
                "The environment variable 'OPENAI_API_BASE' is deprecated and will be removed in the 0.1.80. "
                "Please use 'OPENAI_BASE_URL' instead.",
                DeprecationWarning,
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _get_token_encoding(self):
        if self._token_encoding is None:
            try:
                self._token_encoding = tiktoken.encoding_for_model(self.config.model)
            except KeyError:
                self._token_encoding = tiktoken.get_encoding("cl100k_base")
        return self._token_encoding

    def count_tokens(self, text: str) -> int:
        """Count tokens exactly as the normalized text is sent to OpenAI."""
        normalized = text.replace("\n", " ")
        return len(self._get_token_encoding().encode(normalized, disallowed_special=()))

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        """Return the longest prefix whose OpenAI token count is at most max_tokens."""
        normalized = text.replace("\n", " ")
        encoding = self._get_token_encoding()
        tokens = encoding.encode(normalized, disallowed_special=())
        return encoding.decode(tokens[:max_tokens])

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        kwargs = {
            "input": [text],
            "model": self.config.model,
            "encoding_format": "float",
        }
        if self._pass_dimensions_to_api:
            kwargs["dimensions"] = self.config.embedding_dims
        return self.client.embeddings.create(**kwargs).data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        """Embed multiple texts in a single OpenAI API call.

        Automatically chunks into batches of 100 to stay within API limits.
        """
        MAX_BATCH = 100
        texts = [text.replace("\n", " ") for text in texts]
        all_embeddings = []
        for i in range(0, len(texts), MAX_BATCH):
            chunk = texts[i : i + MAX_BATCH]
            kwargs = {
                "input": chunk,
                "model": self.config.model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api:
                kwargs["dimensions"] = self.config.embedding_dims
            response = self.client.embeddings.create(**kwargs)
            all_embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return all_embeddings
