"""Tests for Claude reasoning tools and Claude Code integration."""

import pytest
from unittest.mock import MagicMock, patch
from ct.tools.claude import reason, compare, summarize, code, _build_context_section


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.verbose = False
    session.console = MagicMock()
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(
        content="Expert analysis here.",
        model="claude-sonnet-4-5-20250929",
        usage={"input": 100, "output": 50},
    )
    session.get_llm.return_value = mock_llm
    return session


class TestReason:
    def test_basic_reasoning(self, mock_session):
        result = reason(goal="Why is TP53 a poor degradation target?",
                        _session=mock_session)
        assert "summary" in result
        assert result["summary"] == "Expert analysis here."
        assert "error" not in result

    def test_with_context(self, mock_session):
        result = reason(
            goal="Interpret these findings",
            context="CRISPR screen showed TP53 dependency in AML",
            _session=mock_session,
        )
        assert "summary" in result
        # Verify context was passed to the LLM
        call_args = mock_session.get_llm().chat.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "CRISPR screen" in user_msg

    def test_with_prior_results(self, mock_session):
        prior = {1: {"summary": "Found 15 co-essential genes"}}
        result = reason(
            goal="What do these co-essentiality results mean?",
            _session=mock_session,
            _prior_results=prior,
        )
        assert "summary" in result
        # Verify prior results injected into system prompt
        call_args = mock_session.get_llm().chat.call_args
        system = call_args[1]["system"]
        assert "co-essential" in system

    def test_no_session(self):
        result = reason(goal="test", _session=None)
        assert "error" in result


class TestCompare:
    def test_basic_comparison(self, mock_session):
        result = compare(
            goal="Which compound to advance?",
            options="lenalidomide, pomalidomide, iberdomide",
            _session=mock_session,
        )
        assert "summary" in result
        assert "error" not in result

    def test_with_criteria(self, mock_session):
        result = compare(
            goal="Best target for degradation",
            options="IKZF1, GSPT1, SALL4",
            criteria="selectivity, safety, clinical precedent",
            _session=mock_session,
        )
        assert "summary" in result
        call_args = mock_session.get_llm().chat.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "selectivity" in user_msg

    def test_no_session(self):
        result = compare(goal="test", _session=None)
        assert "error" in result


class TestSummarize:
    def test_basic_summary(self, mock_session):
        result = summarize(
            goal="Summarize the safety assessment for the project lead",
            _session=mock_session,
        )
        assert "summary" in result
        assert "error" not in result

    def test_with_content(self, mock_session):
        result = summarize(
            goal="Distill key findings",
            content="BRCA1 shows selective dependency in TNBC...",
            _session=mock_session,
        )
        assert "summary" in result
        call_args = mock_session.get_llm().chat.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "BRCA1" in user_msg

    def test_no_session(self):
        result = summarize(goal="test", _session=None)
        assert "error" in result


class TestBuildContextSection:
    def test_empty(self):
        assert _build_context_section(None) == ""
        assert _build_context_section({}) == ""

    def test_with_results(self):
        prior = {
            1: {"summary": "Found 10 synergy partners"},
            2: {"summary": "Safety profile is clean"},
        }
        ctx = _build_context_section(prior)
        assert "Step 1" in ctx
        assert "synergy partners" in ctx
        assert "Safety profile" in ctx

    def test_non_dict_results(self):
        """Non-dict results should be stringified and truncated."""
        prior = {1: "x" * 1000}
        ctx = _build_context_section(prior)
        # str(result)[:500] truncation kicks in for non-dict results
        assert len(ctx) < 600


class TestClaudeCode:
    @pytest.fixture
    def mock_code_session(self):
        session = MagicMock()
        session.config = MagicMock()
        session.config.get.side_effect = lambda key, default=None: {
            "agent.enable_claude_code_tool": True,
        }.get(key, default)
        session.console = MagicMock()
        return session

    def test_disabled_by_policy(self):
        result = code(task="refactor the tests")
        assert result["error"] == "disabled_by_policy"

    def test_claude_not_found(self, mock_code_session):
        with patch("shutil.which", return_value=None):
            result = code(task="refactor the tests", _session=mock_code_session)
        assert result["error"] == "claude_not_found"

    def test_successful_invocation(self, mock_code_session):
        mock_result = MagicMock(
            returncode=0,
            stdout="I've refactored the tests successfully. Changes:\n- Split into 3 files\n- Added fixtures",
            stderr="",
        )
        with patch("shutil.which", return_value="/usr/local/bin/claude"), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            result = code(task="refactor the tests", max_budget=0.50, _session=mock_code_session)

        assert "error" not in result
        assert "refactored" in result["summary"]
        assert result["exit_code"] == 0

        # Verify CLI flags
        cmd = mock_run.call_args[0][0]
        assert "-p" in cmd
        assert "--permission-mode" in cmd
        assert "bypassPermissions" in cmd
        assert "--max-budget-usd" in cmd
        assert "0.5" in cmd

    def test_with_prior_results(self, mock_code_session):
        mock_result = MagicMock(returncode=0, stdout="Done.", stderr="")
        with patch("shutil.which", return_value="/usr/local/bin/claude"), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            result = code(
                task="write tests for the new tool",
                _prior_results={1: {"summary": "Added similarity_search tool"}},
                _session=mock_code_session,
            )

        # Verify context was passed in the prompt
        cmd = mock_run.call_args[0][0]
        prompt_idx = cmd.index("-p") + 1
        prompt = cmd[prompt_idx]
        assert "similarity_search" in prompt

    def test_timeout(self, mock_code_session):
        import subprocess as sp
        with patch("shutil.which", return_value="/usr/local/bin/claude"), \
             patch("subprocess.run", side_effect=sp.TimeoutExpired("claude", 300)):
            result = code(task="big refactor", _session=mock_code_session)

        assert result["error"] == "timeout"

    def test_no_output(self, mock_code_session):
        mock_result = MagicMock(returncode=1, stdout="", stderr="Error: auth failed")
        with patch("shutil.which", return_value="/usr/local/bin/claude"), \
             patch("subprocess.run", return_value=mock_result):
            result = code(task="do something", _session=mock_code_session)

        assert result["error"] == "no_output"
        assert "auth failed" in result["stderr"]

    def test_custom_allowed_tools(self, mock_code_session):
        mock_result = MagicMock(returncode=0, stdout="Done.", stderr="")
        with patch("shutil.which", return_value="/usr/local/bin/claude"), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            code(task="just read files", allowed_tools="Read,Glob", _session=mock_code_session)

        cmd = mock_run.call_args[0][0]
        tools_idx = cmd.index("--allowed-tools") + 1
        assert cmd[tools_idx] == "Read,Glob"
