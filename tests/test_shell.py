"""Tests for shell.run tool."""

import pytest
from unittest.mock import patch, MagicMock
from ct.tools.shell import shell_run, _is_blocked


class TestIsBlocked:
    def test_sudo_blocked(self):
        assert _is_blocked("sudo rm -rf /") is not None

    def test_rm_rf_root_blocked(self):
        assert _is_blocked("rm -rf /") is not None

    def test_chmod_777_blocked(self):
        assert _is_blocked("chmod 777 /etc") is not None

    def test_fork_bomb_blocked(self):
        assert _is_blocked(":(){ :|:& };:") is not None

    def test_safe_commands_allowed(self):
        assert _is_blocked("ls -la") is None
        assert _is_blocked("git status") is None
        assert _is_blocked("python script.py") is None
        assert _is_blocked("echo hello") is None
        assert _is_blocked("pip install numpy") is None

    def test_shell_operators_blocked(self):
        assert _is_blocked("echo hello | wc -c") is not None
        assert _is_blocked("echo hello && pwd") is not None

    def test_destructive_binary_blocked(self):
        assert _is_blocked("rm -rf tempdir") is not None
        assert _is_blocked("chmod 755 file.txt") is not None


class TestShellRun:
    def test_simple_command(self):
        result = shell_run(command="echo hello")
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    def test_failed_command(self):
        result = shell_run(command="false")
        assert result["exit_code"] != 0

    def test_blocked_command(self):
        result = shell_run(command="sudo echo test")
        assert result["error"] == "blocked_command"

    def test_timeout(self):
        result = shell_run(command="sleep 10", timeout=1)
        assert result["error"] == "timeout"

    def test_timeout_cap(self):
        """Timeout should be capped at 300s."""
        with patch("ct.tools.shell.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="ok", stderr=""
            )
            shell_run(command="echo test", timeout=999)
            _, kwargs = mock_run.call_args
            assert kwargs["timeout"] == 300

    def test_stderr_captured(self):
        result = shell_run(command="ls /definitely_missing_ct_shell_test_path")
        assert result["exit_code"] != 0
        assert "No such file or directory" in result["stderr"] or "cannot access" in result["stderr"]

    def test_multiline_output(self):
        result = shell_run(command="echo 'line1\nline2\nline3'")
        assert result["exit_code"] == 0
        assert "line1" in result["stdout"]
