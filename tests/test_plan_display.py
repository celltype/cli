"""Tests for plan preview rendering."""

from io import StringIO

import pytest
from rich.console import Console

from ct.agent.types import Plan, Step
from ct.ui.terminal import render_plan_preview


def _captured_console():
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=120)
    return console, buf


def _make_step(id, tool, description, depends_on=None, tool_args=None):
    s = Step(id=id, tool=tool, description=description)
    s.depends_on = depends_on or []
    s.tool_args = tool_args or {}
    return s


class TestRenderPlanPreview:
    def test_linear_plan(self):
        plan = Plan(
            query="analyze CRBN",
            steps=[
                _make_step(1, "target.coessentiality", "Find co-essential genes"),
                _make_step(2, "expression.pathway_enrichment", "Enrich pathways", depends_on=[1]),
                _make_step(3, "literature.pubmed_search", "Search literature", depends_on=[2]),
            ],
        )
        console, buf = _captured_console()
        render_plan_preview(plan, console)
        output = buf.getvalue()

        assert "target.coessentiality" in output
        assert "expression.pathway_enrichment" in output
        assert "literature.pubmed_search" in output
        assert "Find co-essential genes" in output
        assert "Enrich pathways" in output
        assert "Search literature" in output

    def test_parallel_plan(self):
        plan = Plan(
            query="analyze compound",
            steps=[
                _make_step(1, "target.coessentiality", "Gene analysis"),
                _make_step(2, "viability.dose_response", "Dose response"),
                _make_step(3, "literature.pubmed_search", "Literature", depends_on=[1, 2]),
            ],
        )
        console, buf = _captured_console()
        render_plan_preview(plan, console)
        output = buf.getvalue()

        # Step 3 depends on both 1 and 2
        assert "after step 1, 2" in output

    def test_step_args_shown(self):
        plan = Plan(
            query="find coessential partners",
            steps=[
                _make_step(
                    1,
                    "target.coessentiality",
                    "Find co-essential genes for CRBN",
                    tool_args={"gene": "CRBN", "top_n": 20},
                ),
            ],
        )
        console, buf = _captured_console()
        render_plan_preview(plan, console)
        output = buf.getvalue()

        assert "CRBN" in output
        assert "top_n=20" in output

    def test_plan_preview_panel(self):
        plan = Plan(
            query="test",
            steps=[_make_step(1, "tool.name", "description")],
        )
        console, buf = _captured_console()
        render_plan_preview(plan, console)
        output = buf.getvalue()

        assert "Plan Preview" in output
        assert "Research Plan" in output


class TestPlanSnapshot:
    @staticmethod
    def _render_sample():
        plan = Plan(
            query="analyze CRBN degradation",
            steps=[
                _make_step(1, "target.coessentiality", "Find co-essential genes", tool_args={"gene": "CRBN"}),
                _make_step(2, "expression.pathway_enrichment", "Enrich pathways", depends_on=[1]),
                _make_step(3, "literature.pubmed_search", "Search literature", depends_on=[2], tool_args={"query": "CRBN degradation"}),
            ],
        )
        console, buf = _captured_console()
        render_plan_preview(plan, console)
        return buf.getvalue()

    def test_snapshot(self, tmp_path):
        import pathlib
        golden_path = pathlib.Path(__file__).parent / "fixtures" / "plan_snapshot.txt"
        output = self._render_sample()

        if not golden_path.exists():
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(output)
            pytest.skip("Golden file created — rerun to compare")

        expected = golden_path.read_text()
        assert output == expected, (
            f"Plan output changed. Delete {golden_path} to regenerate.\n"
            f"Got:\n{output}\nExpected:\n{expected}"
        )
