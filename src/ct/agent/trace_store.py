"""
Trace store: full-fidelity capture of agent message streams.

Captures the complete sequence of text blocks, tool calls, and tool results
from each query execution. Used by the notebook exporter and other
downstream consumers that need richer data than the compact Trajectory.

Operates independently of Trajectory — Trajectory stores compact turn
summaries for LLM context injection; TraceStore captures the full stream
for export/replay.
"""

import base64
import json
import logging
import mimetypes
import time
from pathlib import Path

logger = logging.getLogger("ct.trace_store")

# Marker used by MCP handlers to embed structured metadata in tool results
TRACE_META_MARKER = "\n__CT_TRACE_META__\n"

# Maximum file size for base64 embedding (10 MB)
_MAX_EMBED_BYTES = 10 * 1024 * 1024


def _sessions_dir() -> Path:
    """Return the sessions directory, creating it if needed."""
    d = Path.home() / ".ct" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_trace_meta(result_text: str) -> dict | None:
    """Extract __CT_TRACE_META__ JSON from a tool result string.

    Returns the parsed dict, or None if no marker found.
    """
    if TRACE_META_MARKER not in result_text:
        return None
    try:
        _, meta_json = result_text.split(TRACE_META_MARKER, 1)
        return json.loads(meta_json.strip())
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse trace meta: %s", e)
        return None


def _embed_plots(event: dict) -> None:
    """Read plot files and add base64-encoded data to the event in-place."""
    plots = event.get("plots", [])
    if not plots:
        return

    embedded = []
    for plot_path_str in plots:
        plot_path = Path(plot_path_str)
        if not plot_path.exists():
            logger.warning("Plot file missing at capture time: %s", plot_path)
            continue
        if plot_path.stat().st_size > _MAX_EMBED_BYTES:
            logger.warning(
                "Plot file too large for embedding (%d bytes): %s",
                plot_path.stat().st_size,
                plot_path,
            )
            continue

        mime, _ = mimetypes.guess_type(str(plot_path))
        if mime is None:
            suffix_map = {
                ".png": "image/png",
                ".svg": "image/svg+xml",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".pdf": "application/pdf",
            }
            mime = suffix_map.get(plot_path.suffix.lower(), "application/octet-stream")

        try:
            data = base64.b64encode(plot_path.read_bytes()).decode("ascii")
            embedded.append({
                "filename": plot_path.name,
                "mime": mime,
                "data": data,
            })
        except OSError as e:
            logger.warning("Failed to read plot file %s: %s", plot_path, e)

    if embedded:
        event["plots_base64"] = embedded


class TraceStore:
    """Captures and persists full agent message stream traces.

    Usage::

        store = TraceStore(session_id="abc-123")
        events = []

        # ... pass events list to process_messages() ...

        store.add_events(events, query="my question", model="claude-sonnet-4-5")
        store.flush()  # persist to ~/.ct/sessions/abc-123.trace.jsonl
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._events: list[dict] = []
        self._path = _sessions_dir() / f"{session_id}.trace.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    @property
    def events(self) -> list[dict]:
        return self._events

    def add_event(self, event: dict) -> None:
        """Add a single trace event."""
        if "timestamp" not in event:
            event["timestamp"] = time.time()
        self._events.append(event)

    def add_events(
        self,
        events: list[dict],
        query: str = "",
        model: str = "",
        duration_s: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        """Wrap a list of trace events with query_start/query_end and add them.

        Also performs eager base64 embedding of plots.
        """
        now = time.time()

        # query_start
        self._events.append({
            "type": "query_start",
            "session_id": self.session_id,
            "query": query,
            "model": model,
            "timestamp": now,
        })

        # Add all events (with plot embedding)
        for event in events:
            if event.get("type") == "tool_result" and event.get("plots"):
                _embed_plots(event)
            self._events.append(event)

        # query_end
        self._events.append({
            "type": "query_end",
            "duration_s": duration_s,
            "cost_usd": cost_usd,
            "timestamp": time.time(),
        })

    def flush(self, path: Path | None = None) -> Path:
        """Persist all events to a JSONL file. Appends if file exists."""
        out = path or self._path
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "a", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event, default=str) + "\n")

        count = len(self._events)
        self._events.clear()
        logger.info("Flushed %d trace events to %s", count, out)
        return out

    @staticmethod
    def load(path: Path | str) -> list[dict]:
        """Load trace events from a JSONL file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    @staticmethod
    def find_trace(session_id: str | None = None) -> Path | None:
        """Find a trace file by session ID prefix, or return the most recent.

        Args:
            session_id: Session ID or prefix to match. If None, returns
                the most recent trace file.

        Returns:
            Path to the trace file, or None if not found.
        """
        sessions_dir = _sessions_dir()
        traces = sorted(
            sessions_dir.glob("*.trace.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not traces:
            return None

        if session_id is None:
            return traces[0]

        # Exact match first
        exact = sessions_dir / f"{session_id}.trace.jsonl"
        if exact.exists():
            return exact

        # Prefix match
        for t in traces:
            if t.stem.replace(".trace", "").startswith(session_id):
                return t

        return None
