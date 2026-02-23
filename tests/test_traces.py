"""Tests for the TraceRenderer and helper functions."""

from io import StringIO

import pytest
from rich.console import Console

from ct.ui.traces import TraceRenderer, format_args, truncate_output


# ---------------------------------------------------------------------------
# format_args
# ---------------------------------------------------------------------------

class TestFormatArgs:
    def test_empty(self):
        assert format_args(None) == ""
        assert format_args({}) == ""

    def test_basic(self):
        result = format_args({"gene": "CRBN", "top_n": 20})
        assert 'gene="CRBN"' in result
        assert "top_n=20" in result

    def test_omits_private_keys(self):
        result = format_args({"gene": "CRBN", "_session": "obj", "_prior": {}})
        assert "gene" in result
        assert "_session" not in result
        assert "_prior" not in result

    def test_truncates_long_values(self):
        long_val = "C" * 100
        result = format_args({"smiles": long_val}, max_value_len=50)
        assert "..." in result
        assert len(result) < 200

    def test_non_string_values(self):
        result = format_args({"n": 10, "flag": True})
        assert "n=10" in result
        assert "flag=True" in result


# ---------------------------------------------------------------------------
# truncate_output
# ---------------------------------------------------------------------------

class TestTruncateOutput:
    def test_empty(self):
        assert truncate_output("") == ""

    def test_short_text_unchanged(self):
        text = "Found 20 co-essential genes."
        assert truncate_output(text) == text

    def test_long_text_truncated(self):
        text = "A" * 500
        result = truncate_output(text, max_chars=200)
        assert result.startswith("A" * 200)
        assert "500 chars total" in result

    def test_exact_boundary(self):
        text = "B" * 200
        assert truncate_output(text, max_chars=200) == text


# ---------------------------------------------------------------------------
# TraceRenderer
# ---------------------------------------------------------------------------

def _captured_console():
    """Create a Console that captures output to a StringIO buffer.

    Uses ``no_color=True`` so assertions can match plain text.
    """
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120)
    return console, buf


class TestTraceRendererStart:
    def test_contains_tool_name(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_start("target.coessentiality", {"gene": "CRBN"})
        output = buf.getvalue()
        assert "target.coessentiality" in output

    def test_contains_args(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_start("target.coessentiality", {"gene": "CRBN"})
        output = buf.getvalue()
        assert "CRBN" in output

    def test_running_indicator(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_start("target.coessentiality")
        output = buf.getvalue()
        assert "\u25b8" in output  # ▸

    def test_strips_mcp_prefix(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_start("mcp__ct-tools__target.coessentiality")
        output = buf.getvalue()
        assert "mcp__ct-tools__" not in output
        assert "target.coessentiality" in output


class TestTraceRendererComplete:
    def test_contains_tool_name_and_timing(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_complete(
            "target.coessentiality",
            {"gene": "CRBN"},
            "Found 20 co-essential genes.",
            2.3,
        )
        output = buf.getvalue()
        assert "target.coessentiality" in output
        assert "2.3s" in output

    def test_contains_result_summary(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_complete(
            "target.coessentiality",
            {},
            "Found 20 co-essential genes. Top: COPS5, DDB1",
            1.0,
        )
        output = buf.getvalue()
        assert "Found 20" in output

    def test_checkmark_indicator(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_complete("tool.name", {}, "result", 1.0)
        output = buf.getvalue()
        assert "\u2713" in output  # ✓

    def test_truncates_long_result(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_complete("tool.name", {}, "X" * 500, 1.0)
        output = buf.getvalue()
        assert "500 chars total" in output


class TestTraceRendererError:
    def test_contains_tool_name_and_error(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_error(
            "expression.pathway_enrichment",
            "API key missing",
        )
        output = buf.getvalue()
        assert "expression.pathway_enrichment" in output
        assert "API key missing" in output

    def test_error_indicator(self):
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_error("tool.name", "error")
        output = buf.getvalue()
        assert "\u2717" in output  # ✗


class TestTraceRendererSnapshot:
    """Snapshot test: render a sequence of traces and compare to golden file."""

    @staticmethod
    def _render_sequence():
        console, buf = _captured_console()
        renderer = TraceRenderer(console)
        renderer.render_tool_start("target.coessentiality", {"gene": "CRBN"})
        renderer.render_tool_complete(
            "target.coessentiality",
            {"gene": "CRBN"},
            "Found 20 co-essential genes.",
            2.3,
        )
        renderer.render_tool_error(
            "expression.pathway_enrichment",
            "API key missing",
        )
        return buf.getvalue()

    def test_snapshot_matches(self, tmp_path):
        """Generate golden file on first run, compare on subsequent runs."""
        import pathlib
        golden_path = pathlib.Path(__file__).parent / "fixtures" / "trace_snapshot.txt"
        output = self._render_sequence()

        if not golden_path.exists():
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(output)
            pytest.skip("Golden file created — rerun to compare")

        expected = golden_path.read_text()
        assert output == expected, (
            f"Trace output changed. Delete {golden_path} to regenerate.\n"
            f"Got:\n{output}\nExpected:\n{expected}"
        )
