"""Async OpenAI-compatible evaluation client with retries and rate limiting."""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re
from collections import deque
from typing import Any, TypeVar

try:
    from aiolimiter import AsyncLimiter
except ImportError:

    class AsyncLimiter:
        """Small stdlib fallback matching the async context-manager API."""

        def __init__(self, max_rate: int, time_period: float = 60.0):
            if max_rate <= 0 or time_period <= 0:
                raise ValueError("rate limit and time period must be positive")
            self.max_rate = int(max_rate)
            self.time_period = float(time_period)
            self._timestamps: deque[float] = deque()
            self._lock = asyncio.Lock()

        async def acquire(self) -> None:
            loop = asyncio.get_running_loop()
            while True:
                async with self._lock:
                    now = loop.time()
                    while self._timestamps and now - self._timestamps[0] >= self.time_period:
                        self._timestamps.popleft()
                    if len(self._timestamps) < self.max_rate:
                        self._timestamps.append(now)
                        return
                    delay = self.time_period - (now - self._timestamps[0])
                await asyncio.sleep(max(delay, 0.001))

        async def __aenter__(self) -> "AsyncLimiter":
            await self.acquire()
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None


logger = logging.getLogger(__name__)
T = TypeVar("T")


class LLMClient:
    """Async LLM client with retry logic and rate limiting.

    Supports OpenAI-compatible providers. For OpenAI-compatible
    APIs (e.g., vLLM, Parasail), set provider="openai" with a custom base_url.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
        provider: Use "openai" for GPT or Qwen endpoints.
        api_key: API key. Falls back to provider-specific env vars.
        base_url: Custom base URL (OpenAI-compatible providers).
        max_retries: Maximum retry attempts per call.
        rpm: Requests per minute rate limit.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        rpm: int = 200,
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.provider = provider.lower()
        if self.provider != "openai":
            raise ValueError("Use an OpenAI-compatible endpoint for GPT or Qwen")
        self.max_retries = max_retries
        self.limiter = AsyncLimiter(rpm, 60)
        self.timeout = timeout
        self.extra_body = extra_body
        self._client: Any = None
        self._init_openai(api_key, base_url, timeout, **kwargs)

    def _openai_chat_token_limit_kwargs(self, max_tokens: int) -> dict[str, Any]:
        """Chat Completions: gpt-5 / o-series reject ``max_tokens``; use ``max_completion_tokens``."""
        m = self.model.lower()
        if m.startswith(("gpt-5", "o1", "o3", "o4")):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def _openai_chat_temperature_kwargs(self, temperature: float) -> dict[str, Any]:
        """gpt-5 / o-series only accept the default temperature (1); omit the param for those models."""
        m = self.model.lower()
        if m.startswith(("gpt-5", "o1", "o3", "o4")):
            return {}
        return {"temperature": temperature}

    @staticmethod
    def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
        default = float(2 * (attempt + 1))
        status_code = getattr(exc, "status_code", None)
        message = str(exc)
        if status_code != 402 or "in_flight_budget_exhausted" not in message:
            return default
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        retry_after: Any = None
        if headers is not None:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is None:
            match = re.search("Retry-After['\\\"]?\\s*[:=]\\s*['\\\"]?(\\d+(?:\\.\\d+)?)", message)
            retry_after = match.group(1) if match else None
        try:
            return max(default, float(retry_after))
        except (TypeError, ValueError):
            return max(default, 120.0)

    def _parse_yes_no_judgment(self, raw: str) -> bool:
        """Extract the final yes/no verdict from judge output."""
        text = raw.strip()
        if not text:
            return False
        after_cot = re.split("</judge_thinking>|</thinking>", text, flags=re.IGNORECASE)
        verdict_region = after_cot[-1].strip() if after_cot else text
        verdict_lines = [line.strip().lower() for line in verdict_region.splitlines() if line.strip()]
        for line in reversed(verdict_lines):
            if line == "yes":
                return True
            if line == "no":
                return False
        token_matches = re.findall("\\b(yes|no)\\b", verdict_region.lower())
        if token_matches:
            return token_matches[-1] == "yes"
        return text.lower().startswith("yes")

    def _init_openai(self, api_key: str | None, base_url: str | None, timeout: float, **kwargs: Any) -> None:
        import openai

        client_kwargs: dict[str, Any] = {"timeout": openai.Timeout(timeout, connect=10.0)}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)

    async def generate(self, system: str, user: str, temperature: float = 0, max_tokens: int = 4096) -> str:
        """Generate a text response.

        Args:
            system: System prompt.
            user: User message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            Generated text string.
        """
        return await self._generate_openai(system, user, temperature, max_tokens)

    async def _generate_openai(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        for attempt in range(self.max_retries):
            retry_delay = float(2 * (attempt + 1))
            try:
                async with self.limiter:
                    resp = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **self._openai_chat_temperature_kwargs(temperature),
                            **self._openai_chat_token_limit_kwargs(max_tokens),
                            **{"extra_body": self.extra_body} if self.extra_body is not None else {},
                        ),
                        timeout=self.timeout,
                    )
                content = resp.choices[0].message.content
                if content is None:
                    logger.warning("Generation returned None (finish_reason=%s)", resp.choices[0].finish_reason)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    return ""
                return content.strip()
            except asyncio.TimeoutError:
                logger.warning("Generation attempt %d/%d timed out", attempt + 1, self.max_retries)
            except Exception as exc:
                logger.warning("Generation attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc)
                retry_delay = self._retry_delay_seconds(exc, attempt)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(retry_delay)
        return ""

    async def generate_structured(
        self,
        system: str,
        user: str,
        response_format: type[T] | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
    ) -> Any:
        """Generate a structured response.

        For OpenAI-compatible endpoints, uses response_format={"type": "json_object"}.
        For Anthropic, instructs the model to return JSON and parses it.

        Args:
            system: System prompt.
            user: User message.
            response_format: Pydantic model class for validation (optional).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            Parsed dict or Pydantic model instance.
        """
        return await self._generate_structured_openai(system, user, response_format, temperature, max_tokens)

    async def _generate_structured_openai(
        self, system: str, user: str, response_format: type[T] | None, temperature: float, max_tokens: int
    ) -> Any:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        for attempt in range(self.max_retries):
            retry_delay = float(2 * (attempt + 1))
            try:
                async with self.limiter:
                    resp = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **self._openai_chat_temperature_kwargs(temperature),
                            response_format={"type": "json_object"},
                            **self._openai_chat_token_limit_kwargs(max_tokens),
                            **{"extra_body": self.extra_body} if self.extra_body is not None else {},
                        ),
                        timeout=self.timeout,
                    )
                raw = resp.choices[0].message.content
                if not raw:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    return {}
                parsed = json.loads(raw.strip())
                if "final" in parsed and len(parsed) == 1:
                    inner = parsed["final"]
                    if isinstance(inner, str):
                        parsed = json.loads(inner)
                    elif isinstance(inner, dict):
                        parsed = inner
                if response_format is not None:
                    return response_format(**parsed)
                return parsed
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Structured output parse error attempt %d/%d: %s", attempt + 1, self.max_retries, exc)
            except asyncio.TimeoutError:
                logger.warning("Structured output attempt %d/%d timed out", attempt + 1, self.max_retries)
            except Exception as exc:
                logger.warning("Structured output attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc)
                retry_delay = self._retry_delay_seconds(exc, attempt)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(retry_delay)
        return {} if response_format is None else None

    async def judge_yes_no(self, prompt: str) -> tuple[bool, str]:
        """Run a yes/no judge prompt. Returns (correct, raw_response)."""
        raw = await self.generate(system="", user=prompt, temperature=0)
        return (self._parse_yes_no_judgment(raw), raw)
