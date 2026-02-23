"""
Custom Markdown rendering for ct — left-aligned headings.

Rich's default Markdown renderer centers headings. This module provides
a LeftMarkdown class that renders headings left-aligned instead.
"""

from rich import box
from rich.markdown import Heading, Markdown
from rich.panel import Panel
from rich.text import Text


class _LeftHeading(Heading):
    """Heading with left alignment instead of Rich's default centered."""

    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        if self.tag == "h1":
            yield Panel(text, box=box.HEAVY, style="markdown.h1.border")
        else:
            yield Text("")
            yield text
            yield Text("")


class LeftMarkdown(Markdown):
    """Markdown renderer with left-aligned headings."""

    elements = {**Markdown.elements, "heading_open": _LeftHeading}
