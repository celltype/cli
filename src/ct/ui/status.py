"""
Thinking status display for ct.

Shows a DNA double-helix animation with rotating drug discovery themed words
and an elapsed time counter. The helix scrolls at 8fps while words rotate
every ~3 seconds.

Usage:
    with ThinkingStatus(console, "planning"):
        result = llm.chat(...)
"""

import random
import time
from typing import List

from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

# ---------------------------------------------------------------------------
# Spinner animations
# ---------------------------------------------------------------------------

_BASE = "\u2881\u2822\u2814\u2848\u2814\u2822"  # ⢁⠢⠔⡈⠔⠢
DNA_HELIX_FRAMES: List[str] = [_BASE[i:] + _BASE[:i] for i in range(len(_BASE))]

SPINNERS = {
    "benzene_breathing": {
        "frames": ['⬡', '⎔', '⌬', '⬢', '⌬', '⎔'],
        "interval_ms": 125,
    },
    "dna_helix": {
        "frames": DNA_HELIX_FRAMES,
        "interval_ms": 125,
    },
}

import math

def apply_gradient(text: str, elapsed_s: float = 0.0) -> Text:
    """Apply a #50fa7b (neon green) to #00e5ff (cyan) temporal gradient."""
    if not text:
        return Text("")
    
    # #50fa7b (80, 250, 123) to #00e5ff (0, 229, 255)
    r1, g1, b1 = 80, 250, 123
    r2, g2, b2 = 0, 229, 255
    
    # Cycle over 3 spinner loops (6 frames × 125ms × 3 = 2.25s)
    cycle_duration_s = 2.25
    t = (math.sin((elapsed_s % cycle_duration_s) * (2 * math.pi / cycle_duration_s)) + 1) / 2
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    
    result = Text()
    result.append(text, style=hex_color)
    return result

THINKING_WORDS = {
    "planning": [
        "Hypothesizing", "Mapping pathways", "Reviewing literature",
        "Selecting tools", "Designing strategy", "Evaluating evidence",
        "Prioritizing targets", "Cross-referencing data", "Consulting databases",
        "Analyzing feasibility", "Scanning publications", "Charting biology",
        "Surveying chemical space", "Assessing druggability", "Interrogating targets",
        "Probing mechanisms", "Mining databases", "Scouting leads",
        "Triaging candidates", "Devising experiments", "Calibrating approach",
        "Querying knowledge base", "Formulating hypothesis", "Mapping the landscape",
        "Checking prior art", "Weighing approaches", "Modeling the problem",
    ],
    "synthesizing": [
        "Synthesizing findings", "Connecting pathways", "Weighing evidence",
        "Integrating data", "Analyzing patterns", "Formulating insights",
        "Evaluating significance", "Drafting conclusions", "Assessing confidence",
        "Distilling results", "Building narrative", "Ranking findings",
        "Reconciling data", "Crystallizing insights", "Spotting trends",
        "Interpreting signals", "Assembling the picture", "Triangulating evidence",
        "Parsing results", "Connecting the dots", "Extracting key findings",
        "Gauging clinical relevance", "Framing the story", "Identifying next steps",
    ],
    "evaluating": [
        "Evaluating results", "Checking completeness", "Assessing quality",
        "Reviewing coverage", "Validating findings", "Gauging sufficiency",
        "Scoring confidence", "Auditing data gaps", "Stress-testing conclusions",
        "Verifying consistency", "Checking for blind spots", "Weighing completeness",
    ],
    "reasoning": [
        "Reasoning through mechanisms", "Connecting biology to chemistry",
        "Evaluating hypotheses", "Considering alternatives", "Weighing trade-offs",
        "Modeling interactions", "Analyzing structure-activity", "Exploring mechanisms",
        "Deconvolving signals", "Tracing pathways", "Dissecting mechanisms",
        "Thinking through pharmacology", "Pondering selectivity",
        "Probing binding kinetics", "Assessing off-target risk",
    ],
    "comparing": [
        "Comparing options", "Benchmarking candidates", "Ranking alternatives",
        "Evaluating trade-offs", "Scoring criteria", "Weighing pros and cons",
        "Aligning properties", "Contrasting profiles", "Assessing differentiators",
    ],
    "summarizing": [
        "Distilling key findings", "Extracting insights", "Condensing results",
        "Identifying highlights", "Prioritizing conclusions", "Crystallizing takeaways",
        "Compiling brief", "Summarizing evidence", "Framing recommendations",
    ],
    "coding": [
        "Writing code", "Editing files", "Reading codebase", "Running tests",
        "Debugging", "Refactoring", "Searching files", "Analyzing code",
        "Applying changes", "Iterating on fixes",
    ],
}


class _ThinkingRenderable:
    """Self-updating renderable: Animated spinner + rotating bio-themed word + elapsed time.

    Computes display state from wall-clock time on each refresh, so no
    separate update thread is needed — Rich's Live refresh handles it.
    """

    def __init__(self, words, spinner_style="benzene_breathing"):
        self.words = words
        self.start_time = time.time()
        
        # Load spinner configuration
        spinner_conf = SPINNERS.get(spinner_style, SPINNERS["benzene_breathing"])
        self.frames = spinner_conf["frames"]
        self.interval_ms = spinner_conf["interval_ms"]

    def __rich_console__(self, console, options):
        elapsed = time.time() - self.start_time
        
        # Determine current word (rotates every 3 seconds)
        word_idx = int(elapsed / 3) % len(self.words)
        word = self.words[word_idx]

        # Determine current spinner frame
        frame_idx = int(elapsed * (1000 / self.interval_ms)) % len(self.frames)
        frame_str = self.frames[frame_idx]

        if elapsed < 60:
            time_str = f"{elapsed:.0f}s"
        else:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            time_str = f"{mins}m {secs}s"

        # Apply gradient to spinner frame
        output = apply_gradient(frame_str, elapsed_s=elapsed)
        output.append("  ")
        output.append(f"{word}…", style="cyan")
        output.append("  ")
        output.append(f"({time_str})", style="dim")

        yield output


class ThinkingStatus:
    """Context manager showing a Claude Code-style thinking status.

    Displays a spinner with rotating drug-discovery themed words and an
    elapsed time counter. The status disappears when the context exits.

    The Rich Live daemon thread may stall when the GIL is held by
    CPU-bound tool code running via ``asyncio.to_thread()``.  To keep
    the timer ticking, call :meth:`kick` from the async message loop
    or start :meth:`async_refresh_task` as a background asyncio task.

    Args:
        console: Rich Console instance.
        phase: One of the keys in THINKING_WORDS (planning, synthesizing, etc.).
    """

    def __init__(self, console, phase="planning"):
        from ct.agent.config import Config
        self.console = console
        words = list(THINKING_WORDS.get(phase, THINKING_WORDS["planning"]))
        random.shuffle(words)

        # Determine spinner style from config
        try:
            cfg = Config.load()
            spinner_style = cfg.get("ui.spinner", "benzene_breathing")
        except Exception:
            spinner_style = "benzene_breathing"

        self._renderable = _ThinkingRenderable(words, spinner_style=spinner_style)
        self._live = None
        self._async_task = None

    def __enter__(self):
        self._live = Live(
            self._renderable,
            console=self.console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        self._cancel_async_task()
        if self._live is not None:
            return self._live.__exit__(*args)

    def kick(self):
        """Force a single refresh of the Live display.

        Call from the async message loop to keep the timer updating
        even when the daemon thread is GIL-starved.
        """
        if self._live is not None:
            try:
                self._live.refresh()
            except Exception:
                pass

    def start_async_refresh(self):
        """Start a background asyncio task that refreshes the display.

        This supplements the Rich Live daemon thread: while a tool runs
        in ``asyncio.to_thread()``, the event loop still gets cycles
        during I/O waits and can drive this coroutine to keep the timer
        updating.  Call :meth:`stop` or :meth:`_cancel_async_task` to
        stop.
        """
        import asyncio

        async def _refresh_loop():
            try:
                while True:
                    await asyncio.sleep(0.125)
                    self.kick()
            except asyncio.CancelledError:
                pass

        try:
            loop = asyncio.get_running_loop()
            self._async_task = loop.create_task(_refresh_loop())
        except RuntimeError:
            pass  # No running event loop — daemon thread is the fallback

    def _cancel_async_task(self):
        if self._async_task is not None:
            self._async_task.cancel()
            self._async_task = None

    def stop(self):
        """Programmatically stop the animation (idempotent)."""
        self._cancel_async_task()
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                pass
            self._live = None
