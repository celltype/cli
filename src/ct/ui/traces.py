"""
Execution trace rendering for tool calls.

Renders real-time trace panels showing tool name, arguments, results,
and timing — used by both the SDK runner and legacy executor paths.
"""

import time
from rich.console import Console
from rich.live import Live
from rich.text import Text

from ct.ui.status import SPINNERS, apply_gradient


def format_args(args: dict | None, max_value_len: int = 50) -> str:
    """Format tool arguments as a compact key=value string.

    Omits internal keys (starting with ``_``) and truncates long values.
    """
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if k.startswith("_"):
            continue
        v_str = str(v)
        if len(v_str) > max_value_len:
            v_str = v_str[:max_value_len] + "..."
        if isinstance(v, str):
            parts.append(f'{k}="{v_str}"')
        else:
            parts.append(f"{k}={v_str}")
    return ", ".join(parts)


def truncate_output(text: str, max_chars: int = 200) -> str:
    """Truncate tool output text, appending length info if trimmed."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [{len(text)} chars total]"


class TraceRenderer:
    """Renders tool call traces to a Rich Console.

    Accepts a ``Console`` at construction so tests can pass
    ``Console(file=StringIO())`` to capture output.
    """

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def render_tool_start(self, name: str, args: dict | None = None) -> None:
        """Render a running tool indicator."""
        clean = name.replace("mcp__ct-tools__", "")
        args_str = format_args(args)
        
        line = Text()
        line.append("  \u25b8 ", style="bold cyan")
        line.append(clean, style="cyan")
        if args_str:
            line.append(f"  {args_str}", style="dim")
        self.console.print(line)

    def render_tool_complete(
        self,
        name: str,
        args: dict | None = None,
        result: str = "",
        duration: float = 0.0,
    ) -> None:
        """Render a completed tool trace with result summary and timing."""
        clean = name.replace("mcp__ct-tools__", "")
        args_str = format_args(args)
        summary = truncate_output(result)

        header = Text()
        header.append("  \u2713 ", style="bold green")
        header.append(clean, style="green")
        if duration > 0:
            header.append(f"  {duration:.1f}s", style="dim")
        self.console.print(header)

        if args_str:
            self.console.print(f"    [dim]{args_str}[/dim]")
        if summary:
            self.console.print(f"    [dim]{summary}[/dim]")

    def render_tool_error(self, name: str, error: str = "") -> None:
        """Render a failed tool trace with error message."""
        clean = name.replace("mcp__ct-tools__", "")
        header = Text()
        header.append("  \u2717 ", style="bold red")
        header.append(clean, style="red")
        self.console.print(header)
        if error:
            err_text = truncate_output(error)
            self.console.print(f"    [red]{err_text}[/red]")

    def render_reasoning(self, text: str) -> None:
        """Render Claude's reasoning text between tool calls."""
        if not text or not text.strip():
            return
        
        try:
            from ct.ui.markdown import LeftMarkdown
            self.console.print(LeftMarkdown(text))
        except Exception:
            self.console.print(text)
