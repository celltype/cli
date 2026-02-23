"""Tests for experiment design tools."""

import pytest
from ct.tools.experiment import (
    list_assays,
    design_assay,
    estimate_timeline,
    ASSAY_TEMPLATES,
)


class TestListAssays:
    def test_lists_all_templates(self):
        result = list_assays()
        assert "summary" in result
        assert len(result["assays"]) == len(ASSAY_TEMPLATES)

    def test_has_categories(self):
        result = list_assays()
        assert "categories" in result
        cats = result["categories"]
        assert "degradation" in cats
        assert "binding" in cats
        assert "viability" in cats

    def test_each_assay_has_fields(self):
        result = list_assays()
        for assay in result["assays"]:
            assert "assay_type" in assay
            assert "name" in assay
            assert "description" in assay
            assert "category" in assay


class TestDesignAssay:
    @pytest.mark.parametrize("assay_type", list(ASSAY_TEMPLATES.keys()))
    def test_design_all_assay_types(self, assay_type):
        result = design_assay(assay_type=assay_type)
        assert "summary" in result
        assert "protocol" in result
        protocol = result["protocol"]
        assert protocol["assay_type"] == assay_type
        assert len(protocol["protocol_steps"]) > 0
        assert len(protocol["reagents"]) > 0
        assert "positive" in protocol["controls"]
        assert "negative" in protocol["controls"]
        assert protocol["readout"]

    def test_design_with_target(self):
        result = design_assay(assay_type="hibit", target="ZNF687")
        assert "ZNF687" in result["summary"]
        assert "ZNF687" in result["protocol"]["context"]

    def test_design_with_compound(self):
        result = design_assay(assay_type="hibit", compound="YU-123")
        assert "YU-123" in result["summary"]

    def test_design_with_cell_line(self):
        result = design_assay(assay_type="ctg_viability", cell_line="HeLa")
        assert "HeLa" in result["summary"]

    def test_design_with_goal(self):
        result = design_assay(
            assay_type="tmt_proteomics",
            goal="Profile degradation selectivity",
        )
        assert result["protocol"]["experimental_goal"] == "Profile degradation selectivity"

    def test_design_unknown_assay(self):
        result = design_assay(assay_type="nonexistent_assay")
        assert "error" in result

    def test_design_has_timing(self):
        result = design_assay(assay_type="hibit")
        protocol = result["protocol"]
        assert protocol["estimated_hands_on_hours"] > 0
        assert protocol["estimated_calendar_days"] > 0
        assert protocol["estimated_cost_per_plate"] > 0


class TestEstimateTimeline:
    def test_basic_estimate(self):
        result = estimate_timeline(assay_type="hibit")
        assert "summary" in result
        assert result["n_compounds"] == 1
        assert result["n_doses"] == 8
        assert result["n_replicates"] == 3
        assert result["hands_on_hours"] > 0
        assert result["calendar_days"] > 0
        assert result["estimated_cost"] > 0

    def test_scaling_with_compounds(self):
        result_1 = estimate_timeline(assay_type="ctg_viability", n_compounds=1)
        result_10 = estimate_timeline(assay_type="ctg_viability", n_compounds=10)
        assert result_10["n_plates"] >= result_1["n_plates"]
        assert result_10["estimated_cost"] >= result_1["estimated_cost"]

    def test_scaling_with_doses(self):
        result_4 = estimate_timeline(assay_type="hibit", n_doses=4)
        result_10 = estimate_timeline(assay_type="hibit", n_doses=10)
        assert result_10["total_wells"] >= result_4["total_wells"]

    def test_scaling_with_replicates(self):
        result_2 = estimate_timeline(assay_type="hibit", n_replicates=2)
        result_5 = estimate_timeline(assay_type="hibit", n_replicates=5)
        assert result_5["total_wells"] >= result_2["total_wells"]

    def test_unknown_assay(self):
        result = estimate_timeline(assay_type="nonexistent")
        assert "error" in result

    @pytest.mark.parametrize("assay_type", list(ASSAY_TEMPLATES.keys()))
    def test_estimate_all_assay_types(self, assay_type):
        result = estimate_timeline(assay_type=assay_type)
        assert "summary" in result
        assert result["estimated_cost"] > 0

    def test_large_scale(self):
        result = estimate_timeline(
            assay_type="ctg_viability",
            n_compounds=100,
            n_replicates=3,
            n_doses=10,
        )
        assert result["n_plates"] > 1
        assert result["estimated_cost"] > 500
