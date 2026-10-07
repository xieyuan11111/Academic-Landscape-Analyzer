"""Official OpenAI SDK adapter used for DPSK-compatible endpoints."""

from __future__ import annotations

import json
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


class ModelClientError(RuntimeError):
    """Raised when a model call cannot return a JSON object."""


class ModelTruncationError(ModelClientError):
    """Raised when the model hit the completion ceiling mid-response.

    Kept distinct from a generic parse failure because it is actionable: the
    same prompt frequently succeeds on a retry (the overflow is provider-side
    variance, measured 2026-09-24: identical request, temperature 0, once
    truncated at 8192 tokens and once finished cleanly in 3645). It also means
    a caller lost a whole work unit rather than one malformed field.
    """


def is_retryable_completion(error: BaseException) -> bool:
    """Whether another draw is worth paying for.

    Truncation and unparseable JSON are worth retrying; transport/HTTP/auth
    failures are not, because they will simply fail again. Shared by every
    bounded-retry caller so the policy cannot drift per module.
    """
    if isinstance(error, ModelTruncationError):
        return True
    if isinstance(error, ModelClientError):
        message = str(error)
        return "not valid JSON" in message or "not an object" in message
    return False


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 120.0,
        *,
        max_tokens: int = 8192,
        thinking: dict[str, Any] | None = None,
        send_thinking: bool = True,
        client: Any | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        # Sent explicitly: relying on the provider default is what made the
        # completion ceiling invisible when it silently truncated responses.
        self.max_tokens = max_tokens
        # DeepSeek-style endpoints accept and benefit from the thinking switch;
        # other gateways (OpenCode Zen) reject the unknown body field with a
        # 400 "Upstream request failed", so the caller decides per provider.
        self._thinking = thinking
        if send_thinking:
            self._extra_body: dict[str, Any] | None = {
                "thinking": self._thinking or {"type": "disabled"}
            }
        else:
            self._extra_body = None
        self._client = client or OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            max_retries=0,
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                **({"extra_body": self._extra_body} if self._extra_body else {}),
            )
        except APITimeoutError as error:
            raise ModelClientError("model request timed out") from error
        except APIConnectionError as error:
            raise ModelClientError(f"model connection failed: {error}") from error
        except APIStatusError as error:
            detail = str(error.response.text)[:1000] if error.response is not None else str(error)
            raise ModelClientError(f"model HTTP {error.status_code}: {detail}") from error
        except Exception as error:  # noqa: BLE001 - provider SDK errors vary by backend
            raise ModelClientError(f"model request failed: {error}") from error

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as error:
            raise ModelClientError("model response did not contain choices[0].message.content") from error
        if not isinstance(content, str):
            raise ModelClientError("model message content was not a string")

        # Checked before parsing: a response cut off at the ceiling is by
        # definition incomplete, and reporting it as "not valid JSON" hides the
        # one fact needed to fix it.
        if getattr(response.choices[0], "finish_reason", None) == "length":
            completion = getattr(getattr(response, "usage", None), "completion_tokens", None)
            raise ModelTruncationError(
                "model completion was truncated at "
                f"{completion if completion is not None else self.max_tokens} tokens"
            )

        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise ModelClientError("model content was not valid JSON") from error
        if not isinstance(parsed, dict):
            raise ModelClientError("model JSON root was not an object")
        return parsed
