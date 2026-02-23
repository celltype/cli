"""Tests for workflow templates."""

import pytest
from ct.agent.workflows import WORKFLOWS, format_workflows_for_llm
from ct.tools import registry, ensure_loaded


class TestWorkflowDefinitions:
    def test_all_workflows_have_required_fields(self):
        for wf_id, wf in WORKFLOWS.items():
            assert "description" in wf, f"{wf_id} missing description"
            assert "trigger_phrases" in wf, f"{wf_id} missing trigger_phrases"
            assert "steps" in wf, f"{wf_id} missing steps"
            assert len(wf["steps"]) > 0, f"{wf_id} has no steps"

    def test_all_steps_have_tool_and_why(self):
        for wf_id, wf in WORKFLOWS.items():
            for i, step in enumerate(wf["steps"]):
                assert "tool" in step, f"{wf_id} step {i} missing tool"
                assert "why" in step, f"{wf_id} step {i} missing why"

    def test_expected_workflows_exist(self):
        expected = {
            "target_validation",
            "compound_safety",
            "hit_characterization",
            "combination_therapy",
            "clinical_positioning",
            "cro_engagement",
            "structure_prediction",
            "gpu_computation",
            "custom_analysis",
            "script_authoring",
            "report_generation",
            "genetic_evidence",
            "lead_optimization",
            "protein_deep_dive",
            "drug_repurposing",
            "molecular_docking",
            "resistance_analysis",
            "therapeutic_window",
            "competitive_landscape",
            "treatment_landscape",
            "mutation_resistance",
            "protac_design",
            "patient_population",
            "compound_comparison",
            "omics_scrnaseq_analysis",
            "omics_bulk_analysis",
            "omics_data_discovery",
            "omics_methylation_analysis",
            "omics_proteomics_analysis",
            "omics_epigenomics_analysis",
            "omics_multiomics_integration",
            "omics_spatial_analysis",
        }
        assert expected == set(WORKFLOWS.keys())


class TestWorkflowToolReferences:
    @pytest.fixture(autouse=True)
    def load_registry(self):
        ensure_loaded()

    def test_all_workflow_tools_exist_in_registry(self):
        """Every tool referenced in a workflow must exist in the registry."""
        missing = []
        for wf_id, wf in WORKFLOWS.items():
            for step in wf["steps"]:
                tool_name = step["tool"]
                if registry.get_tool(tool_name) is None:
                    missing.append(f"{wf_id}: {tool_name}")
        assert not missing, f"Missing tools referenced in workflows: {missing}"


class TestFormatWorkflows:
    def test_format_returns_string(self):
        result = format_workflows_for_llm()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_contains_workflow_names(self):
        result = format_workflows_for_llm()
        for wf_id in WORKFLOWS:
            assert wf_id in result

    def test_format_contains_tool_names(self):
        result = format_workflows_for_llm()
        for wf in WORKFLOWS.values():
            for step in wf["steps"]:
                assert step["tool"] in result

    def test_format_has_headers(self):
        result = format_workflows_for_llm()
        assert "# Recommended Workflows" in result

    def test_format_respects_allowed_tools_filter(self):
        allowed = {"code.execute", "files.write_report"}
        result = format_workflows_for_llm(allowed_tools=allowed)
        assert "code.execute" in result
        assert "files.write_report" in result
        assert "compute.estimate_cost" not in result
        assert "cro.match_experiment" not in result
