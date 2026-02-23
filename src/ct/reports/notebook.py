"""
Notebook exporter: converts agent trace JSONL to Jupyter notebook (.ipynb).

Maps trace events to notebook cells:
  - text → markdown cell
  - tool_start + tool_result (run_python) → Python code cell with outputs
  - tool_start + tool_result (run_r) → code cell with %%R magic
  - tool_start + tool_result (other) → markdown cell with tool summary
  - query_start → header markdown cell
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("ct.reports.notebook")

# Code tool names that map to code cells (not markdown)
_CODE_TOOLS = {"run_python", "run_r"}


def _make_markdown_cell(source: str) -> dict:
    """Create a nbformat-compatible markdown cell dict."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def _make_code_cell(
    source: str,
    outputs: list[dict] | None = None,
) -> dict:
    """Create a nbformat-compatible code cell dict."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": source,
        "outputs": outputs or [],
    }


def _stdout_output(text: str) -> dict:
    """Create a stream stdout output."""
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text,
    }


def _image_output(base64_data: str, mime: str = "image/png") -> dict:
    """Create a display_data output with an embedded image."""
    return {
        "output_type": "display_data",
        "data": {mime: base64_data},
        "metadata": {},
    }


def _error_output(traceback_text: str) -> dict:
    """Create an error output from a traceback string."""
    # Split traceback into lines for the notebook format
    lines = traceback_text.splitlines(keepends=True)
    return {
        "output_type": "error",
        "ename": "Error",
        "evalue": lines[-1].strip() if lines else "Error",
        "traceback": lines,
    }


def _format_tool_args(input_args: dict, max_value_len: int = 80) -> str:
    """Format tool arguments as a readable string for markdown cells."""
    if not input_args:
        return ""
    parts = []
    for k, v in input_args.items():
        if k.startswith("_"):
            continue
        v_str = str(v)
        if len(v_str) > max_value_len:
            v_str = v_str[:max_value_len] + "..."
        parts.append(f"- `{k}`: {v_str}")
    return "\n".join(parts)


def _format_timestamp(ts: float) -> str:
    """Format a Unix timestamp as a human-readable string."""
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (OSError, ValueError, OverflowError):
        return "unknown"


def _build_code_cell_outputs(event: dict) -> list[dict]:
    """Build notebook outputs list from a tool_result trace event."""
    outputs = []

    # Stdout
    stdout = event.get("stdout", "")
    if stdout and stdout.strip():
        outputs.append(_stdout_output(stdout))

    # Embedded plots
    for plot in event.get("plots_base64", []):
        mime = plot.get("mime", "image/png")
        data = plot.get("data", "")
        if data:
            outputs.append(_image_output(data, mime))

    # Error
    if event.get("is_error") and event.get("error"):
        outputs.append(_error_output(event["error"]))

    return outputs


def trace_to_notebook(trace_path: Path | str) -> Any:
    """Convert a trace JSONL file to a Jupyter NotebookNode.

    Args:
        trace_path: Path to the .trace.jsonl file.

    Returns:
        An nbformat.NotebookNode (v4) object.

    Raises:
        ImportError: If nbformat is not installed.
        FileNotFoundError: If the trace file does not exist.
    """
    import nbformat

    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    # Load events
    events = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    if not events:
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_markdown_cell("*No agent activity recorded*"))
        return nb

    # Build notebook
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells: list[dict] = []
    pending_text: list[str] = []  # For merging consecutive text events
    query_count = 0

    # Index tool_result events by tool_use_id for pairing
    result_by_id: dict[str, dict] = {}
    for ev in events:
        if ev.get("type") == "tool_result" and ev.get("tool_use_id"):
            result_by_id[ev["tool_use_id"]] = ev

    def _flush_text():
        """Merge and emit accumulated text events as a single markdown cell."""
        if pending_text:
            merged = "\n".join(pending_text)
            if merged.strip():
                cells.append(_make_markdown_cell(merged))
            pending_text.clear()

    for event in events:
        etype = event.get("type", "")

        if etype == "query_start":
            _flush_text()
            query_count += 1
            query = event.get("query", "")
            model = event.get("model", "")
            ts = _format_timestamp(event.get("timestamp", 0))

            if query_count == 1:
                # First query: full header
                header = f"# {query}\n\n"
                header += f"*Generated by ct on {ts}*"
                if model:
                    header += f" *| Model: {model}*"
            else:
                # Subsequent queries: separator heading
                header = f"# Query {query_count}: {query}"
            cells.append(_make_markdown_cell(header))

        elif etype == "text":
            content = event.get("content", "")
            if content.strip():
                pending_text.append(content)

        elif etype == "tool_start":
            tool = event.get("tool", "")
            tool_use_id = event.get("tool_use_id", "")
            input_args = event.get("input", {})
            result_event = result_by_id.get(tool_use_id, {})

            if tool in _CODE_TOOLS:
                _flush_text()
                # Code cell
                code = result_event.get("code", "")
                if not code and "code" in input_args:
                    code = input_args["code"]

                if tool == "run_r":
                    code = "%%R\n" + code

                outputs = _build_code_cell_outputs(result_event)
                cells.append(_make_code_cell(code, outputs))
            else:
                _flush_text()
                # Non-code tool: markdown cell
                args_str = _format_tool_args(input_args)
                result_text = result_event.get("result_text", "")
                # Truncate long results
                if len(result_text) > 2000:
                    result_text = result_text[:2000] + "\n\n*... truncated*"

                md_parts = [f"**{tool}**"]
                if args_str:
                    md_parts.append(args_str)
                if result_text.strip():
                    md_parts.append(f"\n> {result_text.strip()}")

                cells.append(_make_markdown_cell("\n".join(md_parts)))

        elif etype == "tool_result":
            # Already consumed via result_by_id pairing — skip
            pass

        elif etype == "query_end":
            _flush_text()
            # No cell needed — metadata only

    # Flush any trailing text
    _flush_text()

    # Handle empty notebook (only had metadata events)
    if not cells:
        cells.append(_make_markdown_cell("*No agent activity recorded*"))

    # Convert cell dicts to nbformat cells
    for cell_dict in cells:
        if cell_dict["cell_type"] == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(cell_dict["source"]))
        elif cell_dict["cell_type"] == "code":
            code_cell = nbformat.v4.new_code_cell(cell_dict["source"])
            code_cell.outputs = [
                nbformat.v4.new_output(**out) if "output_type" in out else out
                for out in cell_dict.get("outputs", [])
            ]
            nb.cells.append(code_cell)

    return nb


def events_to_notebook(events: list[dict], title: str = "", model: str = "") -> Any:
    """Convert a list of trace event dicts to a Jupyter NotebookNode.

    Like ``trace_to_notebook`` but takes events directly instead of
    loading from a JSONL file. Used by the benchmark runner.
    """
    import nbformat

    if not events:
        nb = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_markdown_cell("*No agent activity recorded*"))
        return nb

    # Inject a synthetic query_start if not present
    if events[0].get("type") != "query_start" and title:
        events = [{"type": "query_start", "query": title, "model": model, "timestamp": 0}] + events

    # Write to temp JSONL and reuse trace_to_notebook
    import tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
        return trace_to_notebook(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def save_notebook(nb: Any, path: Path | str) -> Path:
    """Write a NotebookNode to disk.

    Args:
        nb: An nbformat.NotebookNode.
        path: Output file path (should end in .ipynb).

    Returns:
        The resolved output path.
    """
    import nbformat

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    logger.info("Saved notebook to %s", path)
    return path
