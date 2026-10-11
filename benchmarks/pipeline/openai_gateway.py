from __future__ import annotations

import json

import os

import time

import urllib.error

import urllib.parse

import urllib.request

from dataclasses import dataclass, field

from typing import Any


class GatewayError(RuntimeError):
    pass


@dataclass
class OpenAIGateway:
    base_url: str
    api_key: str
    # Set ANSWER_MODEL to the model identifier accepted by the API provider.
    model: str = field(default_factory=lambda: os.environ.get("ANSWER_MODEL", "gpt-4o-mini"))
    timeout: float = 300.0
    max_attempts: int = 4
    provider: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "mempursuit/1",
            },
        )
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value: Any = json.loads(response.read().decode("utf-8"))
                if isinstance(value, dict):
                    if endpoint.strip("/") == "chat/completions":
                        choices = value.get("choices")
                        valid_chat = bool(
                            isinstance(choices, list)
                            and choices
                            and isinstance(choices[0], dict)
                            and isinstance(choices[0].get("message"), dict)
                        )
                        if valid_chat:
                            return value
                        if attempt + 1 >= self.max_attempts:
                            raise GatewayError(
                                "OpenAI-compatible endpoint repeatedly returned a chat response without choices"
                            )
                    else:
                        return value
                if attempt + 1 >= self.max_attempts:
                    raise GatewayError("OpenAI-compatible endpoint repeatedly returned a non-object response")
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # Retry empty or truncated JSON bodies even when the HTTP status is 200.
                if attempt + 1 >= self.max_attempts:
                    raise GatewayError("OpenAI-compatible endpoint repeatedly returned invalid JSON") from exc
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt + 1 >= self.max_attempts:
                    raise GatewayError(f"OpenAI-compatible endpoint failed ({exc.code}): {body}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt + 1 >= self.max_attempts:
                    raise GatewayError(f"OpenAI-compatible endpoint failed: {exc}") from exc
            time.sleep(min(0.5 * (2**attempt), 2.0))
        raise GatewayError("OpenAI-compatible endpoint exhausted retries")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if self.provider is not None:
            # OpenRouter's provider-routing object is intentionally attached to
            # chat requests only. Embeddings may use a different model/vendor.
            payload["provider"] = self.provider
        return self._post("chat/completions", payload)

    def text(self, messages: list[dict[str, str]], **kwargs: Any) -> tuple[str, dict[str, Any]]:
        for attempt in range(self.max_attempts):
            response = self.chat(messages, **kwargs)
            try:
                content = response["choices"][0]["message"].get("content") or ""
            except (KeyError, IndexError, TypeError) as exc:
                if attempt + 1 >= self.max_attempts:
                    raise GatewayError("chat response repeatedly lacks choices[0].message.content") from exc
            else:
                text = str(content)
                if text.strip():
                    return text, response
                if attempt + 1 >= self.max_attempts:
                    raise GatewayError("chat response repeatedly returned empty choices[0].message.content")
            time.sleep(min(0.5 * (2**attempt), 2.0))
        raise GatewayError("chat text response exhausted retries")

    def embedding(self, text: str, model: str = "text-embedding-3-small") -> dict[str, Any]:
        return self._post("embeddings", {"model": model, "input": [text]})
