"""Unit tests for the LLM client (no network; requests mocked).

Covers:
  - from_env (base url + api key resolution)
  - chat_json success, JSON parse, code-fence stripping
  - retry on 429/5xx, no retry on 4xx
  - error classes: LLMError, LLMRateLimit, LLMJSONError
  - _parse_json edge cases
"""

import json
from unittest.mock import patch, MagicMock

import requests

from harness.llm_client import (
    LLMClient,
    LLMError,
    LLMRateLimit,
    LLMJSONError,
    DEFAULT_BASE_URL,
)


def _resp(status_code, json_data=None, text=""):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    if json_data is not None:
        r.json.return_value = json_data
    return r


class TestFromEnv:
    def test_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            c = LLMClient.from_env("glm-5.2")
            assert c.base_url == DEFAULT_BASE_URL
            assert c.api_key is None
            assert c.model == "glm-5.2"

    def test_env_override(self):
        env = {"OLLAMA_BASE_URL": "http://local/v1", "OLLAMA_API_KEY": "sekret"}
        with patch.dict("os.environ", env, clear=True):
            c = LLMClient.from_env("glm-5.2")
            assert c.base_url == "http://local/v1"
            assert c.api_key == "sekret"

    def test_explicit_args_win(self):
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "envkey"}, clear=True):
            c = LLMClient.from_env("m", base_url="http://x", api_key="argkey")
            assert c.base_url == "http://x"
            assert c.api_key == "argkey"


class TestChatJson:
    def test_success(self):
        c = LLMClient(base_url="http://x", model="m")
        with patch("harness.llm_client.requests.post") as post:
            post.return_value = _resp(200, {"choices": [{"message": {"content": '{"a": 1}'}}]})
            out = c.chat_json("sys", "user")
            assert out == {"a": 1}

    def test_code_fence_stripped(self):
        c = LLMClient(base_url="http://x", model="m")
        with patch("harness.llm_client.requests.post") as post:
            post.return_value = _resp(200, {"choices": [{"message": {"content": '```json\n{"a": 1}\n```'}}]})
            out = c.chat_json("sys", "user")
            assert out == {"a": 1}

    def test_retries_on_429_then_succeeds(self):
        c = LLMClient(base_url="http://x", model="m", max_retries=2)
        with patch("harness.llm_client.requests.post") as post, \
             patch("harness.llm_client.time.sleep") as sleep:
            post.side_effect = [
                _resp(429, text="rate limited"),
                _resp(200, {"choices": [{"message": {"content": '{"ok": true}'}}]}),
            ]
            out = c.chat_json("sys", "user")
            assert out == {"ok": True}
            assert post.call_count == 2
            sleep.assert_called()

    def test_retries_on_500_then_succeeds(self):
        c = LLMClient(base_url="http://x", model="m", max_retries=1)
        with patch("harness.llm_client.requests.post") as post, \
             patch("harness.llm_client.time.sleep"):
            post.side_effect = [
                _resp(500, text="server error"),
                _resp(200, {"choices": [{"message": {"content": '{"ok": true}'}}]}),
            ]
            out = c.chat_json("sys", "user")
            assert out == {"ok": True}

    def test_no_retry_on_400(self):
        c = LLMClient(base_url="http://x", model="m", max_retries=2)
        with patch("harness.llm_client.requests.post") as post:
            post.return_value = _resp(400, text="bad request")
            try:
                c.chat_json("sys", "user")
                assert False, "expected LLMError"
            except LLMError:
                pass
            assert post.call_count == 1

    def test_rate_limit_exhausted(self):
        c = LLMClient(base_url="http://x", model="m", max_retries=1)
        with patch("harness.llm_client.requests.post") as post, \
             patch("harness.llm_client.time.sleep"):
            post.return_value = _resp(429, text="rate limited")
            try:
                c.chat_json("sys", "user")
                assert False, "expected LLMRateLimit"
            except LLMRateLimit:
                pass

    def test_network_error_retries_then_raises(self):
        c = LLMClient(base_url="http://x", model="m", max_retries=1)
        with patch("harness.llm_client.requests.post",
                   side_effect=requests.RequestException("conn")) as post, \
             patch("harness.llm_client.time.sleep"):
            try:
                c.chat_json("sys", "user")
                assert False, "expected LLMError"
            except LLMError:
                pass
            assert post.call_count == 2

    def test_non_json_response_raises(self):
        c = LLMClient(base_url="http://x", model="m")
        with patch("harness.llm_client.requests.post") as post:
            post.return_value = _resp(200, text="not json")
            post.return_value.json.side_effect = json.JSONDecodeError("bad", "doc", 0)
            try:
                c.chat_json("sys", "user")
                assert False, "expected LLMError"
            except LLMError:
                pass

    def test_unexpected_shape_raises(self):
        c = LLMClient(base_url="http://x", model="m")
        with patch("harness.llm_client.requests.post") as post:
            post.return_value = _resp(200, {"no": "choices"})
            try:
                c.chat_json("sys", "user")
                assert False, "expected LLMError"
            except LLMError:
                pass

    def test_auth_header_sent_when_key(self):
        c = LLMClient(base_url="http://x", model="m", api_key="k")
        with patch("harness.llm_client.requests.post") as post:
            post.return_value = _resp(200, {"choices": [{"message": {"content": '{}'}}]})
            c.chat_json("sys", "user")
            headers = post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer k"


class TestParseJson:
    def test_plain_object(self):
        assert LLMClient._parse_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert LLMClient._parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_no_lang(self):
        assert LLMClient._parse_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_invalid_json_raises(self):
        try:
            LLMClient._parse_json("not json")
            assert False, "expected LLMJSONError"
        except LLMJSONError:
            pass

    def test_non_object_raises(self):
        try:
            LLMClient._parse_json("[1, 2, 3]")
            assert False, "expected LLMJSONError"
        except LLMJSONError:
            pass

    def test_empty_content_raises(self):
        try:
            LLMClient._parse_json("")
            assert False, "expected LLMJSONError"
        except LLMJSONError:
            pass
