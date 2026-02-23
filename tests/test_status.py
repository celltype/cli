"""Tests for the DNA helix loading animation in ct.ui.status."""

import io

import pytest
from rich.console import Console

from ct.ui.status import DNA_HELIX_FRAMES, ThinkingStatus, _ThinkingRenderable


# ------------------------------------------------------------------
# DNA helix frame properties
# ------------------------------------------------------------------


class TestDNAHelixFrames:
    def test_frame_count(self):
        assert len(DNA_HELIX_FRAMES) == 6

    def test_frame_width(self):
        for frame in DNA_HELIX_FRAMES:
            assert len(frame) == 6

    def test_frames_unique(self):
        assert len(set(DNA_HELIX_FRAMES)) == 6

    def test_all_chars_in_braille_range(self):
        for frame in DNA_HELIX_FRAMES:
            for ch in frame:
                assert 0x2800 <= ord(ch) <= 0x28FF, f"Char {ch!r} (U+{ord(ch):04X}) not in braille range"

    def test_frames_are_rotations(self):
        """Each frame is a left-rotation of the first."""
        base = DNA_HELIX_FRAMES[0]
        for i, frame in enumerate(DNA_HELIX_FRAMES):
            assert frame == base[i:] + base[:i]


# ------------------------------------------------------------------
# ThinkingStatus lifecycle
# ------------------------------------------------------------------


class TestThinkingStatus:
    def _make_console(self):
        return Console(file=io.StringIO(), force_terminal=True)

    def test_enter_exit(self):
        console = self._make_console()
        status = ThinkingStatus(console, phase="planning")
        status.__enter__()
        assert status._live is not None
        status.__exit__(None, None, None)

    def test_context_manager(self):
        console = self._make_console()
        with ThinkingStatus(console, phase="planning") as s:
            assert s._live is not None

    def test_stop(self):
        console = self._make_console()
        status = ThinkingStatus(console, phase="planning")
        status.__enter__()
        assert status._live is not None
        status.stop()
        assert status._live is None

    def test_stop_idempotent(self):
        console = self._make_console()
        status = ThinkingStatus(console, phase="planning")
        status.__enter__()
        status.stop()
        status.stop()  # should not raise
        assert status._live is None

    def test_stop_without_enter(self):
        """stop() before __enter__ should not raise."""
        console = self._make_console()
        status = ThinkingStatus(console, phase="planning")
        status.stop()  # no-op, _live is None

    def test_unknown_phase_falls_back(self):
        console = self._make_console()
        status = ThinkingStatus(console, phase="nonexistent")
        # Should fall back to "planning" words
        assert len(status._renderable.words) > 0


# ------------------------------------------------------------------
# Renderable wiring
# ------------------------------------------------------------------


class TestThinkingRenderable:
    def test_default_spinner_is_benzene(self):
        """Default spinner should be benzene_breathing, not DNA helix."""
        r = _ThinkingRenderable(["Testing"])
        from ct.ui.status import SPINNERS
        assert r.frames == SPINNERS["benzene_breathing"]["frames"]

    def test_dna_spinner_when_requested(self):
        r = _ThinkingRenderable(["Testing"], spinner_style="dna_helix")
        assert r.frames == DNA_HELIX_FRAMES

    def test_spinner_interval(self):
        r = _ThinkingRenderable(["Testing"])
        from ct.ui.status import SPINNERS
        assert r.interval_ms == SPINNERS["benzene_breathing"]["interval_ms"]
