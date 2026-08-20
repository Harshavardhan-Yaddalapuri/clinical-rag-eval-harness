"""OpenAI-compatible chat client with retry, timeout, and JSON parsing.

Talks to any OpenAI-compatible /chat/completions endpoint. Defaults to the
Ollama cloud gateway (https://ollama.com/v1) with credentials from the
environment (OLLAMA_API_KEY / OLLAMA_BASE_URL). Never reads or writes secrets
to disk; the key is pulled from the process environment only.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import requests

DEFAULT_BASE_URL = "https://ollama.com/v1"


class LLMError(Exception):
    """Non-retryable LLM request failure (4xx, network, bad payload)."""


class LLMRateLimit(LLMError):
    """Retryable rate-limit or server error (429 / 5xx)."""


class LLMJSONError(LLMError):
    """Assistant response was not valid JSON (or not a JSON object)."""


class LLMClient:
    """Minimal OpenAI-compatible chat client that returns parsed JSON."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "",
        timeout_s: int = 60,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @classmethod
    def from_env(
        cls,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "LLMClient":
        """Build a client from environment variables.

        OLLAMA_BASE_URL defaults to https://ollama.com/v1; OLLAMA_API_KEY is
        optional (some local backends need no key).
        """
        base_url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        return cls(base_url=base_url, api_key=api_key, model=model, **kwargs)

    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> dict:
        """POST /chat/completions and parse the assistant content as JSON.

        Retries 429/5xx with exponential backoff. Never retries other 4xx.
        Raises LLMError, LLMRateLimit, or LLMJSONError.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout_s
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"request failed: {exc}") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMRateLimit(
                    f"status {resp.status_code}: {resp.text[:200]}"
                )

            if resp.status_code >= 400:
                raise LLMError(f"status {resp.status_code}: {resp.text[:200]}")

            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise LLMError(f"non-JSON response: {resp.text[:200]}") from exc

            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"unexpected response shape: {data}") from exc

            return self._parse_json(content)

        raise LLMError(f"request failed: {last_exc}")

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Parse assistant content as a JSON object, tolerating code fences."""
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMJSONError(f"invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMJSONError("expected a JSON object")
        return parsed
