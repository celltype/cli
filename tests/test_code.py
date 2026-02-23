"""Tests for the code.execute tool."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from ct.models.llm import LLMResponse


def _make_llm_response(code: str) -> LLMResponse:
    """Helper to create a mock LLM response with code."""
    return LLMResponse(content=code, model="test")


@pytest.fixture
def mock_session(tmp_path):
    """Create a mock session with LLM and config."""
    session = MagicMock()
    session.config.get.side_effect = lambda key, default=None: {
        "sandbox.timeout": 5,
        "sandbox.output_dir": str(tmp_path),
        "sandbox.max_retries": 2,
    }.get(key, default)
    session.console.status.return_value.__enter__ = MagicMock()
    session.console.status.return_value.__exit__ = MagicMock()
    return session


class TestCodeExecute:
    @patch("ct.agent.sandbox.Sandbox.load_datasets", return_value={})
    def test_script_authoring_goal_writes_python_file(self, mock_load, mock_session, tmp_path, monkeypatch):
        from ct.tools.code import execute

        monkeypatch.chdir(tmp_path)
        mock_session.get_llm.return_value.chat.return_value = _make_llm_response(
            "def main():\n"
            "    print('ok')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

        result = execute(
            goal="Write a Python script that prints ok and save it as hello_script.py",
            _session=mock_session,
        )

        target = tmp_path / "hello_script.py"
        assert result.get("error") is None
        assert result["path"] == str(target)
        assert target.exists()
        assert "def main()" in target.read_text(encoding="utf-8")
        # Script-authoring path should bypass sandbox dataset loading.
        mock_load.assert_not_called()

    @patch("ct.agent.sandbox.Sandbox.load_datasets", return_value={})
    def test_script_authoring_retries_on_syntax_error(self, mock_load, mock_session, tmp_path, monkeypatch):
        from ct.tools.code import execute

        monkeypatch.chdir(tmp_path)
        llm = mock_session.get_llm.return_value
        llm.chat.side_effect = [
            _make_llm_response("def main(\n    pass\n"),  # syntax error
            _make_llm_response(
                "def main():\n"
                "    print('fixed')\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
        ]

        result = execute(
            goal="Create a standalone Python script saved as retry_script.py",
            _session=mock_session,
        )

        target = tmp_path / "retry_script.py"
        assert result.get("error") is None
        assert target.exists()
        assert llm.chat.call_count == 2
        mock_load.assert_not_called()

    @patch("ct.agent.sandbox.Sandbox.load_datasets", return_value={})
    def test_simple_execution(self, mock_load, mock_session):
        from ct.tools.code import execute

        mock_session.get_llm.return_value.chat.return_value = _make_llm_response(
            "result = {'summary': 'The answer is 42', 'value': 42}"
        )

        result = execute(goal="compute 42", _session=mock_session)
        assert "42" in result["summary"]
        assert result.get("error") is None

    @patch("ct.agent.sandbox.Sandbox.load_datasets", return_value={})
    def test_retry_on_error(self, mock_load, mock_session):
        from ct.tools.code import execute

        llm = mock_session.get_llm.return_value
        # First attempt: bad code. Second attempt: fixed code.
        llm.chat.side_effect = [
            _make_llm_response("x = 1 / 0"),  # will fail
            _make_llm_response("result = {'summary': 'fixed', 'value': 1}"),  # will succeed
        ]

        result = execute(goal="divide something", _session=mock_session)
        assert result["summary"] == "fixed"
        assert result.get("error") is None
        assert llm.chat.call_count == 2

    @patch("ct.agent.sandbox.Sandbox.load_datasets", return_value={})
    def test_all_retries_exhausted(self, mock_load, mock_session):
        from ct.tools.code import execute

        llm = mock_session.get_llm.return_value
        # All attempts fail
        llm.chat.side_effect = [
            _make_llm_response("x = 1 / 0"),
            _make_llm_response("x = 1 / 0"),
            _make_llm_response("x = 1 / 0"),
        ]

        result = execute(goal="impossible task", _session=mock_session)
        assert "error" in result
        assert result["error"] is not None

    def test_no_session_returns_error(self):
        from ct.tools.code import execute

        result = execute(goal="anything", _session=None)
        assert "unavailable" in result["summary"].lower()

    @patch("ct.agent.sandbox.Sandbox.load_datasets", return_value={})
    def test_with_prior_results(self, mock_load, mock_session):
        from ct.tools.code import execute

        mock_session.get_llm.return_value.chat.return_value = _make_llm_response(
            "genes = step_1['genes']\n"
            "result = {'summary': f'Got {len(genes)} genes', 'count': len(genes)}"
        )

        result = execute(
            goal="process prior results",
            _session=mock_session,
            _prior_results={1: {"genes": ["TP53", "BRCA1", "KRAS"]}},
        )
        assert result.get("error") is None
        assert "3" in result["summary"]

    @patch("ct.agent.sandbox.Sandbox.load_datasets", return_value={})
    def test_plot_generation(self, mock_load, mock_session):
        from ct.tools.code import execute

        mock_session.get_llm.return_value.chat.return_value = _make_llm_response(
            "fig, ax = plt.subplots()\n"
            "ax.plot([1,2,3], [1,4,9])\n"
            "plt.savefig(OUTPUT_DIR / 'test.png', dpi=72)\n"
            "plt.close()\n"
            "result = {'summary': 'Plot created'}"
        )

        result = execute(goal="make a plot", _session=mock_session)
        assert result.get("error") is None
        assert len(result["plots"]) == 1

    def test_extracts_code_from_markdown_fences(self):
        from ct.tools.code import _extract_code

        raw = "```python\nresult = {'summary': 'ok'}\n```"
        assert _extract_code(raw) == "result = {'summary': 'ok'}"

        # No fences
        raw2 = "result = {'summary': 'ok'}"
        assert _extract_code(raw2) == "result = {'summary': 'ok'}"

        # Bare ``` fences
        raw3 = "```\nx = 1\n```"
        assert _extract_code(raw3) == "x = 1"
