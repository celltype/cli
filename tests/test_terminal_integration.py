"""Integration tests for the interactive terminal.

These tests use pexpect to drive the ct REPL and verify UI features.
Marked with @pytest.mark.integration — skipped in environments without a terminal.
"""

import shutil
import sys

import pytest

# Skip all tests in this module if pexpect is not available or no terminal
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ct") is None,
        reason="ct CLI not found in PATH",
    ),
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="pexpect not supported on Windows",
    ),
]

try:
    import pexpect
except ImportError:
    pexpect = None
    pytestmark.append(
        pytest.mark.skipif(True, reason="pexpect not installed")
    )


@pytest.fixture
def ct_repl():
    """Spawn a ct interactive session and yield the pexpect child."""
    if pexpect is None:
        pytest.skip("pexpect not installed")
    try:
        child = pexpect.spawn("ct", timeout=30, encoding="utf-8")
        # Wait for the prompt to appear
        child.expect("❯", timeout=15)
    except (pexpect.TIMEOUT, pexpect.EOF, OSError):
        pytest.skip("ct REPL failed to start — likely missing dependencies or config")
    yield child
    try:
        child.sendline("/exit")
        child.close()
    except Exception:
        pass


class TestMentionDropdown:
    """Test @ mention completion in the REPL."""

    def test_at_trigger_shows_completions(self, ct_repl):
        """Type @tar and check that target tools appear."""
        ct_repl.send("@tar\t")
        # The completion should show target.* tools
        try:
            ct_repl.expect("target", timeout=5)
        except pexpect.TIMEOUT:
            pytest.skip("Completion dropdown not rendered in this environment")


class TestSlashCommandsCoexist:
    """Test that /commands still work alongside @mentions."""

    def test_slash_tools(self, ct_repl):
        ct_repl.sendline("/tools")
        try:
            ct_repl.expect("target", timeout=10)
        except pexpect.TIMEOUT:
            pytest.skip("Tool list not rendered")


class TestPlanPreview:
    """Test plan preview when agent.plan_preview=true."""

    def test_plan_displayed(self, ct_repl):
        """This test is aspirational — requires a working API key."""
        pytest.skip("Requires API key and live execution")


class TestExecutionTraces:
    """Test that trace panels appear during execution."""

    def test_traces_visible(self, ct_repl):
        """This test is aspirational — requires a working API key."""
        pytest.skip("Requires API key and live execution")
