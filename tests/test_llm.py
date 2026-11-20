"""LLMClient: tag parsing, retry classification, and failure behaviour."""

import pytest

from services.worker.llm import LLMClient


# ── tag parsing ──────────────────────────────────────────────────────────────


def test_parse_tags_reads_each_field():
    parsed = LLMClient._parse_tags(
        "<name>LinearAttention</name>\n"
        "<motivation>Replace softmax with a linear kernel</motivation>\n"
        "<code>def forward(self, x):\n    return x</code>"
    )
    assert parsed["name"] == "LinearAttention"
    assert parsed["motivation"] == "Replace softmax with a linear kernel"
    assert parsed["code"].startswith("def forward")


def test_tags_inside_code_do_not_clobber_other_fields():
    """Regression: the old scanner let code content overwrite the name."""
    parsed = LLMClient._parse_tags(
        "<name>RealName</name>\n"
        "<code>config = {'name': 'inner'}\n"
        "assert '<name>Evil</name>' != ''</code>"
    )
    assert parsed["name"] == "RealName"
    assert "Evil" not in parsed["name"]


def test_mismatched_closing_tag_is_not_accepted():
    parsed = LLMClient._parse_tags("<name>Alpha</motivation> stray")
    assert parsed == {}


def test_first_value_wins_for_repeated_tags():
    parsed = LLMClient._parse_tags("<name>first</name><name>second</name>")
    assert parsed["name"] == "first"


def test_parse_tags_returns_empty_for_untagged_text():
    assert LLMClient._parse_tags("no tags at all") == {}


# ── retry classification ─────────────────────────────────────────────────────


class _Transient(Exception):
    """A generic failure with no status code — worth another attempt."""


class _RateLimit(Exception):
    status_code = 429


class _BadKey(Exception):
    status_code = 401


def test_unknown_failures_are_retried():
    assert LLMClient._is_retryable(_Transient("connection reset")) is True


def test_client_side_http_errors_are_not_retried():
    assert LLMClient._is_retryable(_BadKey("bad key")) is False
    assert LLMClient._is_retryable(_RateLimit("slow down")) is True


def test_named_provider_errors_are_not_retried():
    """LiteLLM reports auth and request errors by class name."""
    AuthenticationError = type("AuthenticationError", (Exception,), {})
    InvalidRequestError = type("InvalidRequestError", (Exception,), {})
    assert LLMClient._is_retryable(AuthenticationError("bad credentials")) is False
    assert LLMClient._is_retryable(InvalidRequestError("unknown model")) is False


# ── failure behaviour ────────────────────────────────────────────────────────


def test_empty_provider_list_raises_a_clear_error():
    client = LLMClient(providers=[])
    client.retry_times = 1
    client.retry_delay = 0

    with pytest.raises(RuntimeError, match="No LLM provider"):
        client.chat([{"role": "user", "content": "hi"}])
