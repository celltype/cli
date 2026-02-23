"""Tests for ct.agent.trace_store — trace capture and persistence."""

import base64
import json
import tempfile
from pathlib import Path

import pytest

from ct.agent.trace_store import TraceStore, parse_trace_meta, TRACE_META_MARKER


# ---------------------------------------------------------------------------
# 6.1 TraceStore round-trip
# ---------------------------------------------------------------------------


def test_trace_store_round_trip(tmp_path):
    """Add events, flush, load, verify fidelity."""
    store = TraceStore(session_id="test-123")
    trace_path = tmp_path / "test-123.trace.jsonl"

    events = [
        {"type": "text", "content": "Hello world", "timestamp": 1000.0},
        {"type": "tool_start", "tool": "run_python", "input": {"code": "1+1"}, "timestamp": 1001.0},
        {"type": "tool_result", "tool": "run_python", "result_text": "2", "is_error": False, "timestamp": 1002.0},
    ]
    store.add_events(events, query="test query", model="test-model", duration_s=5.0)
    store.flush(path=trace_path)

    loaded = TraceStore.load(trace_path)

    # Should have: query_start + 3 events + query_end = 5
    assert len(loaded) == 5
    assert loaded[0]["type"] == "query_start"
    assert loaded[0]["query"] == "test query"
    assert loaded[0]["model"] == "test-model"
    assert loaded[1]["type"] == "text"
    assert loaded[1]["content"] == "Hello world"
    assert loaded[2]["type"] == "tool_start"
    assert loaded[3]["type"] == "tool_result"
    assert loaded[4]["type"] == "query_end"
    assert loaded[4]["duration_s"] == 5.0


def test_trace_store_multi_turn_append(tmp_path):
    """Multiple add_events + flush calls append to the same file."""
    store = TraceStore(session_id="multi")
    trace_path = tmp_path / "multi.trace.jsonl"

    store.add_events(
        [{"type": "text", "content": "Turn 1"}],
        query="Q1",
    )
    store.flush(path=trace_path)

    store.add_events(
        [{"type": "text", "content": "Turn 2"}],
        query="Q2",
    )
    store.flush(path=trace_path)

    loaded = TraceStore.load(trace_path)
    query_starts = [e for e in loaded if e["type"] == "query_start"]
    assert len(query_starts) == 2
    assert query_starts[0]["query"] == "Q1"
    assert query_starts[1]["query"] == "Q2"


def test_trace_store_base64_embedding(tmp_path):
    """Plot files are base64-encoded when added via add_events."""
    # Create a small test PNG (1x1 pixel)
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    plot_path = tmp_path / "test_plot.png"
    plot_path.write_bytes(png_data)

    store = TraceStore(session_id="embed-test")
    trace_path = tmp_path / "embed-test.trace.jsonl"

    events = [
        {
            "type": "tool_result",
            "tool": "run_python",
            "plots": [str(plot_path)],
            "timestamp": 1000.0,
        },
    ]
    store.add_events(events, query="plot test")
    store.flush(path=trace_path)

    loaded = TraceStore.load(trace_path)
    tool_result = [e for e in loaded if e["type"] == "tool_result"][0]
    assert "plots_base64" in tool_result
    assert len(tool_result["plots_base64"]) == 1
    assert tool_result["plots_base64"][0]["filename"] == "test_plot.png"
    assert tool_result["plots_base64"][0]["mime"] == "image/png"
    # Verify round-trip: decode and compare
    decoded = base64.b64decode(tool_result["plots_base64"][0]["data"])
    assert decoded == png_data


def test_trace_store_missing_plot(tmp_path):
    """Missing plot files are skipped with a warning (not an error)."""
    store = TraceStore(session_id="missing")
    trace_path = tmp_path / "missing.trace.jsonl"

    events = [
        {
            "type": "tool_result",
            "tool": "run_python",
            "plots": ["/nonexistent/path/plot.png"],
            "timestamp": 1000.0,
        },
    ]
    store.add_events(events, query="missing plot test")
    store.flush(path=trace_path)

    loaded = TraceStore.load(trace_path)
    tool_result = [e for e in loaded if e["type"] == "tool_result"][0]
    # plots_base64 should be empty (missing file) or not present
    assert tool_result.get("plots_base64", []) == []
    # Original paths still preserved
    assert tool_result["plots"] == ["/nonexistent/path/plot.png"]


# ---------------------------------------------------------------------------
# 6.3 __CT_TRACE_META__ parsing
# ---------------------------------------------------------------------------


def test_parse_trace_meta_basic():
    """Extract metadata from result text with marker."""
    meta = {"code": "print(1)", "stdout": "1\n", "plots": [], "exports": [], "error": None}
    text = f"1\nPlots saved: []{TRACE_META_MARKER}{json.dumps(meta)}"

    parsed = parse_trace_meta(text)
    assert parsed is not None
    assert parsed["code"] == "print(1)"
    assert parsed["stdout"] == "1\n"
    assert parsed["error"] is None


def test_parse_trace_meta_no_marker():
    """No marker returns None."""
    assert parse_trace_meta("just some text") is None


def test_parse_trace_meta_invalid_json():
    """Invalid JSON after marker returns None."""
    text = f"output{TRACE_META_MARKER}not valid json"
    assert parse_trace_meta(text) is None


# ---------------------------------------------------------------------------
# 6.2 find_trace
# ---------------------------------------------------------------------------


def test_find_trace_most_recent(tmp_path, monkeypatch):
    """find_trace with no session_id returns most recent."""
    import ct.agent.trace_store as ts_mod
    monkeypatch.setattr(ts_mod, "_sessions_dir", lambda: tmp_path)

    # Create two trace files with different mtimes
    (tmp_path / "old.trace.jsonl").write_text('{"type":"query_start"}\n')
    import time; time.sleep(0.05)
    (tmp_path / "new.trace.jsonl").write_text('{"type":"query_start"}\n')

    result = TraceStore.find_trace()
    assert result is not None
    assert result.name == "new.trace.jsonl"


def test_find_trace_by_prefix(tmp_path, monkeypatch):
    """find_trace with prefix matches the right file."""
    import ct.agent.trace_store as ts_mod
    monkeypatch.setattr(ts_mod, "_sessions_dir", lambda: tmp_path)

    (tmp_path / "abc-123.trace.jsonl").write_text('{"type":"query_start"}\n')
    (tmp_path / "def-456.trace.jsonl").write_text('{"type":"query_start"}\n')

    result = TraceStore.find_trace("abc")
    assert result is not None
    assert "abc-123" in result.name
