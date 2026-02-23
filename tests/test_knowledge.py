"""Tests for knowledge-primer tool consistency."""

from __future__ import annotations

import re

from ct.agent.knowledge import KNOWLEDGE_PRIMER
from ct.tools import ensure_loaded, registry


def _claimed_tools_from_primer(primer: str) -> set[str]:
    claimed: set[str] = set()
    for line in primer.splitlines():
        match = re.match(r"\s*-\s+\*\*([a-z_]+)\*\*:\s+(.+)", line)
        if not match:
            continue
        category = match.group(1)
        names = [n.strip() for n in match.group(2).split(",")]
        for name in names:
            # Keep only the raw tool name before any explanatory parenthetical.
            tool_name = re.sub(r"\s*\(.*?\)\s*", "", name).strip()
            if tool_name:
                claimed.add(f"{category}.{tool_name}")
    return claimed


def test_knowledge_primer_claimed_tools_exist():
    ensure_loaded()
    claimed = _claimed_tools_from_primer(KNOWLEDGE_PRIMER)
    registered = {tool.name for tool in registry.list_tools()}
    missing = sorted(claimed - registered)
    assert missing == [], f"Knowledge primer references unknown tools: {missing}"


def test_knowledge_primer_mentions_viability_suite():
    claimed = _claimed_tools_from_primer(KNOWLEDGE_PRIMER)
    expected = {
        "viability.dose_response",
        "viability.tissue_selectivity",
        "viability.compare_compounds",
    }
    assert expected.issubset(claimed)
