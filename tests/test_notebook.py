"""Tests for ct.reports.notebook — trace-to-notebook conversion."""

import base64
import json
import tempfile
from pathlib import Path

import pytest
import nbformat

from ct.reports.notebook import trace_to_notebook, save_notebook


def _write_trace(path: Path, events: list[dict]):
    """Write a list of event dicts as a JSONL trace file."""
    with open(path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# 6.4 trace_to_notebook — full integration
# ---------------------------------------------------------------------------


def test_trace_to_notebook_full(tmp_path):
    """Convert a trace with text, run_python (stdout+plot), run_r, and a
    generic tool call. Verify cell types, sources, and outputs."""
    # Small 1x1 PNG
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    events = [
        {"type": "query_start", "session_id": "t1", "query": "Analyze data", "model": "test-model", "timestamp": 1000},
        {"type": "text", "content": "Let me analyze the data.", "timestamp": 1001},
        # run_python with stdout + plot
        {"type": "tool_start", "tool": "run_python", "input": {"code": "print('hello')"}, "tool_use_id": "tu1", "timestamp": 1002},
        {"type": "tool_result", "tool": "run_python", "tool_use_id": "tu1", "result_text": "hello",
         "is_error": False, "duration_s": 1.0, "code": "print('hello')", "stdout": "hello\n",
         "plots": [], "plots_base64": [{"filename": "fig.png", "mime": "image/png", "data": png_b64}],
         "exports": [], "timestamp": 1003},
        # Text between tools
        {"type": "text", "content": "Now running R analysis.", "timestamp": 1004},
        # run_r
        {"type": "tool_start", "tool": "run_r", "input": {"code": "summary(mtcars)"}, "tool_use_id": "tu2", "timestamp": 1005},
        {"type": "tool_result", "tool": "run_r", "tool_use_id": "tu2", "result_text": "mpg summary",
         "is_error": False, "duration_s": 0.5, "code": "summary(mtcars)", "stdout": "mpg: Min 10.4 Max 33.9\n",
         "timestamp": 1006},
        # Generic tool
        {"type": "tool_start", "tool": "literature.pubmed_search", "input": {"query": "CRBN"}, "tool_use_id": "tu3", "timestamp": 1007},
        {"type": "tool_result", "tool": "literature.pubmed_search", "tool_use_id": "tu3",
         "result_text": "Found 47 results for CRBN", "is_error": False, "duration_s": 2.0, "timestamp": 1008},
        # Final synthesis
        {"type": "text", "content": "## Summary\nThe analysis shows significant results.", "timestamp": 1009},
        {"type": "query_end", "duration_s": 10.0, "cost_usd": 0.05, "timestamp": 1010},
    ]

    trace_path = tmp_path / "test.trace.jsonl"
    _write_trace(trace_path, events)

    nb = trace_to_notebook(trace_path)

    # Validate it's a proper notebook
    assert nb.nbformat == 4
    assert nb.metadata["kernelspec"]["name"] == "python3"

    cells = nb.cells
    assert len(cells) >= 5  # header + text + python + text + r + tool + synthesis

    # Cell 0: header
    assert cells[0].cell_type == "markdown"
    assert "Analyze data" in cells[0].source

    # Cell 1: reasoning text
    assert cells[1].cell_type == "markdown"
    assert "analyze the data" in cells[1].source

    # Cell 2: Python code cell
    assert cells[2].cell_type == "code"
    assert "print('hello')" in cells[2].source
    # Should have stdout + image outputs
    assert len(cells[2].outputs) >= 1

    # Cell 3: text between tools
    assert cells[3].cell_type == "markdown"
    assert "R analysis" in cells[3].source

    # Cell 4: R code cell
    assert cells[4].cell_type == "code"
    assert "%%R" in cells[4].source
    assert "summary(mtcars)" in cells[4].source

    # Cell 5: generic tool (markdown)
    assert cells[5].cell_type == "markdown"
    assert "literature.pubmed_search" in cells[5].source
    assert "47 results" in cells[5].source

    # Cell 6: synthesis
    assert cells[6].cell_type == "markdown"
    assert "Summary" in cells[6].source


def test_trace_to_notebook_empty(tmp_path):
    """Empty trace produces a notebook with 'no activity' message."""
    trace_path = tmp_path / "empty.trace.jsonl"
    trace_path.write_text("")

    nb = trace_to_notebook(trace_path)
    assert len(nb.cells) == 1
    assert "No agent activity" in nb.cells[0].source


def test_trace_to_notebook_code_error(tmp_path):
    """Code cell with error gets error output."""
    events = [
        {"type": "query_start", "session_id": "e1", "query": "test", "timestamp": 1000},
        {"type": "tool_start", "tool": "run_python", "input": {"code": "1/0"}, "tool_use_id": "tu1", "timestamp": 1001},
        {"type": "tool_result", "tool": "run_python", "tool_use_id": "tu1",
         "result_text": "ZeroDivisionError", "is_error": True, "duration_s": 0.1,
         "code": "1/0", "stdout": "", "error": "Traceback:\n  ZeroDivisionError: division by zero",
         "timestamp": 1002},
        {"type": "query_end", "duration_s": 1.0, "timestamp": 1003},
    ]

    trace_path = tmp_path / "error.trace.jsonl"
    _write_trace(trace_path, events)

    nb = trace_to_notebook(trace_path)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) == 1
    error_outputs = [o for o in code_cells[0].outputs if o.output_type == "error"]
    assert len(error_outputs) == 1


# ---------------------------------------------------------------------------
# 6.5 Multi-query notebook
# ---------------------------------------------------------------------------


def test_multi_query_notebook(tmp_path):
    """Trace with two query_start events produces heading separators."""
    events = [
        {"type": "query_start", "session_id": "m1", "query": "First question", "timestamp": 1000},
        {"type": "text", "content": "Answer 1", "timestamp": 1001},
        {"type": "query_end", "duration_s": 5.0, "timestamp": 1002},
        {"type": "query_start", "session_id": "m1", "query": "Second question", "timestamp": 1003},
        {"type": "text", "content": "Answer 2", "timestamp": 1004},
        {"type": "query_end", "duration_s": 3.0, "timestamp": 1005},
    ]

    trace_path = tmp_path / "multi.trace.jsonl"
    _write_trace(trace_path, events)

    nb = trace_to_notebook(trace_path)

    # Should have: header1 + answer1 + header2 + answer2 = 4 cells
    assert len(nb.cells) == 4
    # First query: full header
    assert "First question" in nb.cells[0].source
    # Second query: separator with "Query 2:"
    assert "Query 2:" in nb.cells[2].source
    assert "Second question" in nb.cells[2].source


# ---------------------------------------------------------------------------
# 6.7 save_notebook
# ---------------------------------------------------------------------------


def test_save_notebook(tmp_path):
    """save_notebook writes a valid .ipynb file."""
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("# Test"))

    out_path = tmp_path / "subdir" / "test.ipynb"
    result = save_notebook(nb, out_path)

    assert result.exists()
    # Verify it's valid JSON and can be loaded
    loaded = nbformat.read(str(result), as_version=4)
    assert len(loaded.cells) == 1
    assert loaded.cells[0].source == "# Test"


def test_trace_not_found():
    """trace_to_notebook raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        trace_to_notebook("/nonexistent/trace.jsonl")
