"""Tests for configuration loading, LLM preflight validation, and schema validation."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ct.agent import config as config_mod
from ct.agent.config import Config, _validate_config


def test_load_invalid_json_does_not_crash(monkeypatch, tmp_path):
    bad_config = tmp_path / "config.json"
    bad_config.write_text("{ not valid json")

    monkeypatch.setattr(config_mod, "CONFIG_FILE", bad_config)

    cfg = Config.load()
    # Should fall back to defaults when config file is corrupt
    assert cfg.get("llm.provider") == "anthropic"


def test_llm_preflight_requires_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    cfg = Config(data={"llm.provider": "anthropic"})
    issue = cfg.llm_preflight_issue()
    assert issue is not None
    assert "Anthropic API key" in issue


def test_llm_preflight_accepts_foundry_env_vars(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "test-resource")
    cfg = Config(data={"llm.provider": "anthropic"})
    assert cfg.llm_preflight_issue() is None


def test_llm_preflight_error_mentions_foundry(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    cfg = Config(data={"llm.provider": "anthropic"})
    issue = cfg.llm_preflight_issue()
    assert "Foundry" in issue


def test_llm_preflight_requires_openai_key():
    cfg = Config(data={"llm.provider": "openai"})
    issue = cfg.llm_preflight_issue()
    assert issue is not None
    assert "OpenAI API key" in issue


def test_llm_preflight_accepts_openai_key():
    cfg = Config(
        data={
            "llm.provider": "openai",
            "llm.openai_api_key": "test-openai-key",
            "llm.model": "gpt-4o",
        }
    )
    assert cfg.llm_preflight_issue() is None


def test_llm_preflight_requires_model_for_local():
    cfg = Config(data={"llm.provider": "local", "llm.model": None})
    issue = cfg.llm_preflight_issue()
    assert issue is not None
    assert "llm.model is required" in issue


def test_llm_preflight_rejects_unknown_provider():
    cfg = Config(data={"llm.provider": "mystery_provider"})
    issue = cfg.llm_preflight_issue()
    assert issue is not None
    assert "Unsupported llm.provider" in issue


def test_claude_code_tool_disabled_by_default():
    cfg = Config(data={})
    assert cfg.get("agent.enable_claude_code_tool") is False


def test_memory_retrieval_enabled_by_default():
    cfg = Config(data={})
    assert cfg.get("agent.memory_retrieval_enabled") is True


def test_quality_gate_defaults():
    cfg = Config(data={})
    assert cfg.get("agent.quality_gate_enabled") is True
    assert cfg.get("agent.quality_gate_strict") is False


def test_agent_profile_enterprise_applies_preset():
    cfg = Config(data={})
    cfg.set("agent.profile", "enterprise")
    assert cfg.get("agent.profile") == "enterprise"
    assert cfg.get("agent.quality_gate_strict") is True
    assert cfg.get("agent.allow_creative_hypotheses") is False
    assert cfg.get("agent.enable_claude_code_tool") is False
    assert cfg.get("enterprise.enforce_policy") is True


def test_agent_profile_research_applies_preset():
    cfg = Config(data={})
    cfg.set("agent.profile", "research")
    assert cfg.get("agent.profile") == "research"
    assert cfg.get("agent.quality_gate_strict") is False
    assert cfg.get("agent.allow_creative_hypotheses") is True
    assert cfg.get("enterprise.enforce_policy") is False


def test_agent_profile_pharma_applies_preset():
    cfg = Config(data={})
    cfg.set("agent.profile", "pharma")
    assert cfg.get("agent.profile") == "pharma"
    assert cfg.get("agent.quality_gate_strict") is True
    assert cfg.get("agent.allow_creative_hypotheses") is False
    assert cfg.get("agent.synthesis_style") == "pharma"
    assert cfg.get("agent.quality_gate_min_next_steps") == 3
    assert cfg.get("agent.quality_gate_max_next_steps") == 3
    assert cfg.get("enterprise.enforce_policy") is False


def test_agent_profile_invalid_raises():
    cfg = Config(data={})
    with pytest.raises(ValueError):
        cfg.set("agent.profile", "unknown-profile")


def test_default_output_dir_is_cwd_outputs_when_missing(tmp_path, monkeypatch):
    cfg_path = tmp_path / "missing_config.json"
    monkeypatch.setattr(config_mod, "CONFIG_FILE", cfg_path)
    cfg = Config.load()
    assert cfg.get("sandbox.output_dir") == config_mod.DEFAULTS["sandbox.output_dir"]


def test_load_migrates_legacy_home_output_dir(tmp_path, monkeypatch):
    legacy = str(Path.home() / ".ct" / "outputs")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"sandbox.output_dir": legacy}), encoding="utf-8")
    monkeypatch.setattr(config_mod, "CONFIG_FILE", cfg_path)

    with patch("ct.agent.config.Path.cwd", return_value=tmp_path):
        cfg = Config.load()

    assert cfg.get("sandbox.output_dir") == str(tmp_path / "outputs")


# ── Config Schema Validation ──────────────────────────────────────────


class TestValidateConfig:
    """Tests for _validate_config()."""

    def test_valid_config_returns_empty(self):
        issues = _validate_config({})
        assert issues == []

    def test_unknown_key_warns(self):
        issues = _validate_config({"agent.typo_key": True})
        assert any("Unknown config key" in i and "typo_key" in i for i in issues)

    def test_type_error_bool_as_string(self):
        issues = _validate_config({"output.verbose": "yes"})
        assert any("Type error" in i and "output.verbose" in i for i in issues)

    def test_type_error_int_as_string(self):
        issues = _validate_config({"agent.max_iterations": "three"})
        assert any("Type error" in i and "agent.max_iterations" in i for i in issues)

    def test_type_error_float_as_string(self):
        issues = _validate_config({"llm.temperature": "hot"})
        assert any("Type error" in i and "llm.temperature" in i for i in issues)

    def test_correct_types_no_warnings(self):
        issues = _validate_config({
            "agent.max_iterations": 5,
            "output.verbose": False,
            "llm.temperature": 0.3,
            "llm.provider": "anthropic",
        })
        assert issues == []

    def test_range_max_iterations_zero(self):
        issues = _validate_config({"agent.max_iterations": 0})
        assert any("Range error" in i and "max_iterations" in i for i in issues)

    def test_range_max_iterations_negative(self):
        issues = _validate_config({"agent.max_iterations": -1})
        assert any("Range error" in i for i in issues)

    def test_range_synthesis_max_tokens_too_low(self):
        issues = _validate_config({"agent.synthesis_max_tokens": 100})
        assert any("Range error" in i and "synthesis_max_tokens" in i for i in issues)

    def test_range_synthesis_max_tokens_valid(self):
        issues = _validate_config({"agent.synthesis_max_tokens": 512})
        assert not any("synthesis_max_tokens" in i for i in issues)

    def test_range_sandbox_timeout_zero(self):
        issues = _validate_config({"sandbox.timeout": 0})
        assert any("Range error" in i and "sandbox.timeout" in i for i in issues)

    def test_interdependency_pharma_strict_false(self):
        issues = _validate_config({
            "agent.profile": "pharma",
            "agent.quality_gate_strict": False,
        })
        assert any("Interdependency warning" in i and "pharma" in i for i in issues)

    def test_interdependency_pharma_strict_true(self):
        issues = _validate_config({
            "agent.profile": "pharma",
            "agent.quality_gate_strict": True,
        })
        assert not any("Interdependency" in i for i in issues)

    def test_interdependency_research_strict_false_no_warning(self):
        issues = _validate_config({
            "agent.profile": "research",
            "agent.quality_gate_strict": False,
        })
        assert not any("Interdependency" in i for i in issues)

    def test_none_values_skipped(self):
        """None values should not trigger type errors."""
        issues = _validate_config({"llm.api_key": None})
        assert not any("Type error" in i and "llm.api_key" in i for i in issues)

    def test_validate_method_on_config_instance(self):
        cfg = Config(data={"agent.max_iterations": -1, "bogus_key": True})
        issues = cfg.validate()
        assert any("Range error" in i for i in issues)
        assert any("Unknown config key" in i for i in issues)

    def test_load_logs_validation_warnings(self, tmp_path, monkeypatch):
        """Config.load() should run validation and log warnings."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"agent.max_iterations": -1}))
        monkeypatch.setattr(config_mod, "CONFIG_FILE", cfg_path)
        with patch("ct.agent.config.logger") as mock_logger:
            Config.load()
            # Should have logged at least one warning about negative max_iterations
            assert mock_logger.warning.called
            calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("max_iterations" in c for c in calls)
