"""Tests for Session: model switching, LLM client management."""

import pytest
from unittest.mock import MagicMock, patch
from ct.agent.session import Session
from ct.agent.config import Config


class TestModelSwitch:
    def test_set_model_changes_config(self):
        cfg = Config(data={"llm.provider": "anthropic", "llm.model": "claude-sonnet-4-5-20250929"})
        session = Session(config=cfg)

        session.set_model("claude-haiku-4-5-20251001")

        assert session.config.get("llm.model") == "claude-haiku-4-5-20251001"

    def test_set_model_resets_llm_client(self):
        cfg = Config(data={"llm.provider": "anthropic", "llm.model": "claude-sonnet-4-5-20250929"})
        session = Session(config=cfg)

        # Simulate having a cached LLM
        session._llm = MagicMock()
        assert session._llm is not None

        session.set_model("claude-opus-4-6")

        # Client should be reset
        assert session._llm is None

    def test_set_model_with_provider(self):
        cfg = Config(data={"llm.provider": "anthropic", "llm.model": "claude-sonnet-4-5-20250929"})
        session = Session(config=cfg)

        session.set_model("gpt-4o", provider="openai")

        assert session.config.get("llm.provider") == "openai"
        assert session.config.get("llm.model") == "gpt-4o"

    def test_current_model_from_config(self):
        cfg = Config(data={"llm.model": "claude-opus-4-6"})
        session = Session(config=cfg)

        assert session.current_model == "claude-opus-4-6"

    def test_current_model_from_llm(self):
        cfg = Config(data={"llm.model": "claude-sonnet-4-5-20250929"})
        session = Session(config=cfg)

        mock_llm = MagicMock()
        mock_llm.model = "claude-haiku-4-5-20251001"
        session._llm = mock_llm

        assert session.current_model == "claude-haiku-4-5-20251001"

    def test_current_model_default(self):
        cfg = Config(data={})
        session = Session(config=cfg)

        # Should return a sensible default
        assert "claude" in session.current_model or "gpt" in session.current_model


class TestSessionBasics:
    def test_log_to_scratchpad(self):
        session = Session(config=Config(data={}))
        session.log("test message")

        assert "test message" in session._scratchpad

    def test_verbose_logging(self):
        session = Session(config=Config(data={}), verbose=True)
        session.console = MagicMock()
        session.log("verbose msg")

        session.console.print.assert_called_once()

    def test_save_scratchpad(self, tmp_path):
        session = Session(config=Config(data={}))
        session.log("line 1")
        session.log("line 2")

        out = tmp_path / "scratch.txt"
        session.save_scratchpad(out)

        assert out.exists()
        content = out.read_text()
        assert "line 1" in content
        assert "line 2" in content

    def test_openai_provider_uses_openai_api_key(self):
        cfg = Config(data={
            "llm.provider": "openai",
            "llm.model": "gpt-4o",
            "llm.api_key": "anthropic-key",
            "llm.openai_api_key": "openai-key",
        })
        session = Session(config=cfg)

        with patch("ct.models.llm.LLMClient") as mock_llm_client:
            session._create_llm()
            _, kwargs = mock_llm_client.call_args

        assert kwargs["provider"] == "openai"
        assert kwargs["api_key"] == "openai-key"


class TestToolHealth:
    def test_transient_failures_trigger_suppression(self):
        cfg = Config(data={
            "agent.tool_health_enabled": True,
            "agent.tool_health_fail_threshold": 2,
            "agent.tool_health_failure_window_s": 3600,
            "agent.tool_health_suppress_seconds": 600,
        })
        session = Session(config=cfg)

        session.record_tool_failure("clinical.trial_search", "HTTP 503: service unavailable")
        assert "clinical.trial_search" not in session.tool_health_suppressed_tools()

        session.record_tool_failure("clinical.trial_search", "timeout during request")
        assert "clinical.trial_search" in session.tool_health_suppressed_tools()

    def test_non_transient_errors_do_not_trigger_suppression(self):
        cfg = Config(data={"agent.tool_health_enabled": True})
        session = Session(config=cfg)

        session.record_tool_failure("chemistry.sar_analyze", "invalid smiles input")
        assert "chemistry.sar_analyze" not in session.tool_health_suppressed_tools()

    def test_success_clears_suppression(self):
        cfg = Config(data={
            "agent.tool_health_enabled": True,
            "agent.tool_health_fail_threshold": 1,
            "agent.tool_health_suppress_seconds": 600,
        })
        session = Session(config=cfg)

        session.record_tool_failure("clinical.trial_search", "HTTP 503")
        assert "clinical.trial_search" in session.tool_health_suppressed_tools()

        session.record_tool_success("clinical.trial_search")
        assert "clinical.trial_search" not in session.tool_health_suppressed_tools()
