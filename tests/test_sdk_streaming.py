"""Tests for SDK streaming message processing."""

import asyncio
import time
from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from ct.ui.traces import TraceRenderer


# ---------------------------------------------------------------------------
# Mock SDK message types (avoid importing claude_agent_sdk in tests)
# ---------------------------------------------------------------------------

class MockTextBlock:
    def __init__(self, text=""):
        self.text = text


class MockToolUseBlock:
    def __init__(self, id="", name="", input=None):
        self.id = id
        self.name = name
        self.input = input or {}


class MockToolResultBlock:
    def __init__(self, tool_use_id="", content=None, is_error=False):
        self.tool_use_id = tool_use_id
        self.content = content or []
        self.is_error = is_error


class MockAssistantMessage:
    def __init__(self, content=None):
        self.content = content or []


class MockResultMessage:
    def __init__(self, total_cost_usd=0.0, num_turns=0, duration_ms=0):
        self.total_cost_usd = total_cost_usd
        self.num_turns = num_turns
        self.duration_ms = duration_ms


class MockStreamEvent:
    def __init__(self, event=None):
        self.event = event or {}


# ---------------------------------------------------------------------------
# Patched process_messages that uses our mock types
# ---------------------------------------------------------------------------

async def _process_messages_with_mocks(messages, trace_renderer=None, headless=False):
    """Re-implementation of process_messages using mock types for testing.

    This mirrors the logic in ct.agent.runner.process_messages but uses
    our mock types instead of claude_agent_sdk imports.
    """
    full_text = []
    tool_calls = []
    inflight = {}
    result_msg = None
    streamed_len = 0

    async for message in messages:
        if isinstance(message, MockStreamEvent):
            event = message.event or {}
            if isinstance(event, dict):
                delta = event.get("delta", {})
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        streamed_len += len(text)
            continue

        if isinstance(message, MockAssistantMessage):
            for block in (message.content or []):
                if isinstance(block, MockTextBlock):
                    text = block.text or ""
                    full_text.append(text)
                    if not headless and trace_renderer and streamed_len == 0:
                        trace_renderer.render_reasoning(text)

                elif isinstance(block, MockToolUseBlock):
                    inflight[block.id] = {
                        "name": block.name,
                        "input": block.input,
                        "start_time": time.time(),
                    }
                    tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                    })
                    if not headless and trace_renderer:
                        trace_renderer.render_tool_start(block.name, block.input)

                elif isinstance(block, MockToolResultBlock):
                    tool_use_id = block.tool_use_id
                    tracked = inflight.pop(tool_use_id, None)
                    duration = 0.0
                    tool_name = ""
                    tool_input = {}
                    if tracked:
                        duration = time.time() - tracked["start_time"]
                        tool_name = tracked["name"]
                        tool_input = tracked["input"]

                    result_text = ""
                    if isinstance(block.content, list):
                        for item in block.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                result_text += item.get("text", "")
                    elif isinstance(block.content, str):
                        result_text = block.content

                    for tc in reversed(tool_calls):
                        if tc["name"] == tool_name and "result_text" not in tc:
                            tc["result_text"] = result_text
                            tc["duration_s"] = duration
                            break

                    if not headless and trace_renderer:
                        if block.is_error:
                            trace_renderer.render_tool_error(tool_name or "unknown", result_text)
                        else:
                            trace_renderer.render_tool_complete(
                                tool_name or "unknown", tool_input, result_text, duration
                            )

        elif isinstance(message, MockResultMessage):
            result_msg = message

    return {
        "full_text": full_text,
        "tool_calls": tool_calls,
        "result_msg": result_msg,
        "streamed_len": streamed_len,
    }


async def _aiter(items):
    for item in items:
        yield item


def _captured_renderer():
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120)
    return TraceRenderer(console), buf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProcessMessages:
    def test_tool_use_and_result(self):
        messages = [
            MockAssistantMessage(content=[
                MockToolUseBlock(id="t1", name="target.coessentiality", input={"gene": "CRBN"}),
                MockToolResultBlock(
                    tool_use_id="t1",
                    content=[{"type": "text", "text": "Found 20 genes"}],
                ),
            ]),
            MockResultMessage(total_cost_usd=0.01),
        ]
        renderer, buf = _captured_renderer()
        result = asyncio.run(_process_messages_with_mocks(_aiter(messages), renderer))

        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["name"] == "target.coessentiality"
        assert tc["result_text"] == "Found 20 genes"
        assert "duration_s" in tc
        assert result["result_msg"] is not None

    def test_tool_use_id_pairing(self):
        messages = [
            MockAssistantMessage(content=[
                MockToolUseBlock(id="t1", name="tool_a", input={}),
                MockToolUseBlock(id="t2", name="tool_b", input={}),
                MockToolResultBlock(tool_use_id="t2", content=[{"type": "text", "text": "B result"}]),
                MockToolResultBlock(tool_use_id="t1", content=[{"type": "text", "text": "A result"}]),
            ]),
            MockResultMessage(),
        ]
        renderer, buf = _captured_renderer()
        result = asyncio.run(_process_messages_with_mocks(_aiter(messages), renderer))

        assert len(result["tool_calls"]) == 2
        # tool_b should have its result
        tc_b = next(tc for tc in result["tool_calls"] if tc["name"] == "tool_b")
        assert tc_b["result_text"] == "B result"
        # tool_a should have its result
        tc_a = next(tc for tc in result["tool_calls"] if tc["name"] == "tool_a")
        assert tc_a["result_text"] == "A result"

    def test_orphan_result_block(self):
        messages = [
            MockAssistantMessage(content=[
                MockToolResultBlock(
                    tool_use_id="orphan_id",
                    content=[{"type": "text", "text": "orphan result"}],
                ),
            ]),
            MockResultMessage(),
        ]
        renderer, buf = _captured_renderer()
        result = asyncio.run(_process_messages_with_mocks(_aiter(messages), renderer))
        # Should not crash, tool_calls empty (no matching ToolUseBlock)
        assert len(result["tool_calls"]) == 0

    def test_error_result_block(self):
        messages = [
            MockAssistantMessage(content=[
                MockToolUseBlock(id="t1", name="bad.tool", input={}),
                MockToolResultBlock(
                    tool_use_id="t1",
                    content=[{"type": "text", "text": "API key missing"}],
                    is_error=True,
                ),
            ]),
            MockResultMessage(),
        ]
        renderer, buf = _captured_renderer()
        result = asyncio.run(_process_messages_with_mocks(_aiter(messages), renderer))

        output = buf.getvalue()
        assert "API key missing" in output
        assert "\u2717" in output  # ✗ error indicator

    def test_text_block_collected(self):
        messages = [
            MockAssistantMessage(content=[
                MockTextBlock("Hello, I will analyze CRBN."),
            ]),
            MockResultMessage(),
        ]
        renderer, buf = _captured_renderer()
        result = asyncio.run(_process_messages_with_mocks(_aiter(messages), renderer))

        assert len(result["full_text"]) == 1
        assert "Hello" in result["full_text"][0]

    def test_stream_event_deduplication(self):
        messages = [
            MockStreamEvent(event={"delta": {"type": "text_delta", "text": "Hello"}}),
            MockAssistantMessage(content=[
                MockTextBlock("Hello world"),
            ]),
            MockResultMessage(),
        ]
        renderer, buf = _captured_renderer()
        result = asyncio.run(_process_messages_with_mocks(_aiter(messages), renderer))

        # StreamEvent should have been counted
        assert result["streamed_len"] == 5  # "Hello" = 5 chars
        # The full text from AssistantMessage should still be stored
        assert "Hello world" in result["full_text"]

    def test_headless_no_rendering(self):
        messages = [
            MockAssistantMessage(content=[
                MockToolUseBlock(id="t1", name="tool.name", input={}),
                MockToolResultBlock(
                    tool_use_id="t1",
                    content=[{"type": "text", "text": "result"}],
                ),
            ]),
            MockResultMessage(),
        ]
        renderer, buf = _captured_renderer()
        result = asyncio.run(
            _process_messages_with_mocks(_aiter(messages), renderer, headless=True)
        )

        # Tool calls should still be recorded
        assert len(result["tool_calls"]) == 1
        # But nothing rendered
        assert buf.getvalue() == ""

    def test_no_renderer_fallback(self):
        """Process messages without a trace renderer — should not crash."""
        messages = [
            MockAssistantMessage(content=[
                MockTextBlock("text"),
                MockToolUseBlock(id="t1", name="tool", input={}),
            ]),
            MockResultMessage(),
        ]
        result = asyncio.run(
            _process_messages_with_mocks(_aiter(messages), trace_renderer=None)
        )
        assert len(result["full_text"]) == 1
        assert len(result["tool_calls"]) == 1
