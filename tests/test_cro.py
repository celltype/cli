"""Tests for CRO tools."""

import pytest
from unittest.mock import patch

MOCK_CRO_DB = [
    {
        "id": "reaction-biology",
        "name": "Reaction Biology",
        "website": "https://reactionbiology.com",
        "contact_email": "info@reactionbiology.com",
        "headquarters": "Malvern, PA, USA",
        "services": [
            {"category": "cell_based_assay", "turnaround_days": 21, "price_range": "$10K-$30K"},
            {"category": "biochemical_assay", "turnaround_days": 14, "price_range": "$5K-$15K"},
        ],
        "therapeutic_areas": ["Oncology", "Immunology"],
        "capabilities": ["HiBiT degradation assay", "NanoBRET ternary complex", "dose-response profiling"],
        "specialties": ["Targeted protein degradation", "molecular glue screening"],
        "size": "medium",
    },
    {
        "id": "charles-river",
        "name": "Charles River Laboratories",
        "website": "https://criver.com",
        "contact_email": "info@criver.com",
        "headquarters": "Wilmington, MA, USA",
        "services": [
            {"category": "in_vivo_efficacy", "turnaround_days": 60, "price_range": "$50K-$150K"},
            {"category": "toxicology", "turnaround_days": 90, "price_range": "$100K-$500K"},
            {"category": "ADME_DMPK", "turnaround_days": 21, "price_range": "$15K-$40K"},
        ],
        "therapeutic_areas": ["Oncology", "Neuroscience", "Rare Disease"],
        "capabilities": ["GLP toxicology", "in vivo pharmacology", "DMPK studies"],
        "specialties": ["Preclinical development", "safety pharmacology"],
        "size": "large",
    },
    {
        "id": "promega-services",
        "name": "Promega",
        "website": "https://promega.com",
        "contact_email": "services@promega.com",
        "headquarters": "Madison, WI, USA",
        "services": [
            {"category": "cell_based_assay", "turnaround_days": 14, "price_range": "$8K-$25K"},
        ],
        "therapeutic_areas": ["Oncology"],
        "capabilities": ["HiBiT degradation", "NanoBRET", "luminescent assays"],
        "specialties": ["Protein degradation assays", "HiBiT technology"],
        "size": "large",
    },
]


@pytest.fixture(autouse=True)
def mock_cro_db():
    with patch("ct.tools.cro._load_cro_db", return_value=MOCK_CRO_DB):
        # Reset cache
        import ct.tools.cro as cro_mod
        cro_mod._cro_db_cache = None
        yield


class TestCroSearch:
    def test_search_by_name(self):
        from ct.tools.cro import search
        result = search(query="Promega")
        assert "summary" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "promega-services"

    def test_search_by_capability(self):
        from ct.tools.cro import search
        result = search(query="degradation")
        assert len(result["results"]) >= 2  # reaction-biology and promega

    def test_search_with_service_filter(self):
        from ct.tools.cro import search
        result = search(query="Oncology", service_type="in_vivo_efficacy")
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "charles-river"

    def test_search_with_therapeutic_area_filter(self):
        from ct.tools.cro import search
        result = search(query="assay", therapeutic_area="Immunology")
        assert len(result["results"]) >= 1

    def test_search_no_results(self):
        from ct.tools.cro import search
        result = search(query="nonexistent_xyz_service")
        assert len(result["results"]) == 0

    def test_search_case_insensitive(self):
        from ct.tools.cro import search
        result = search(query="promega")
        assert len(result["results"]) == 1


class TestCroMatchExperiment:
    def test_match_cell_based_assay(self):
        from ct.tools.cro import match_experiment
        result = match_experiment(assay_type="cell_based_assay")
        assert "summary" in result
        assert len(result["ranked_cros"]) >= 2
        # Both reaction-biology and promega offer cell_based_assay
        ids = [c["id"] for c in result["ranked_cros"]]
        assert "reaction-biology" in ids
        assert "promega-services" in ids

    def test_match_returns_scores(self):
        from ct.tools.cro import match_experiment
        result = match_experiment(assay_type="cell_based_assay", target="CRBN")
        for cro in result["ranked_cros"]:
            assert "score" in cro
            assert 0 < cro["score"] <= 1.0

    def test_match_ranking_order(self):
        from ct.tools.cro import match_experiment
        result = match_experiment(assay_type="cell_based_assay")
        scores = [c["score"] for c in result["ranked_cros"]]
        assert scores == sorted(scores, reverse=True)

    def test_match_with_pricing(self):
        from ct.tools.cro import match_experiment
        result = match_experiment(assay_type="cell_based_assay")
        # At least one CRO should have pricing info
        has_pricing = any("price_range" in c for c in result["ranked_cros"])
        assert has_pricing


class TestCroCompare:
    def test_compare_two_cros(self):
        from ct.tools.cro import compare
        result = compare(cro_ids=["reaction-biology", "promega-services"])
        assert "summary" in result
        assert len(result["comparisons"]) == 2
        assert not result["not_found"]

    def test_compare_not_found(self):
        from ct.tools.cro import compare
        result = compare(cro_ids=["reaction-biology", "fake-cro"])
        assert len(result["comparisons"]) == 1
        assert "fake-cro" in result["not_found"]

    def test_compare_has_services(self):
        from ct.tools.cro import compare
        result = compare(cro_ids=["reaction-biology"])
        comp = result["comparisons"][0]
        assert "services" in comp
        assert "cell_based_assay" in comp["services"]


class TestCroDraftInquiry:
    def test_draft_inquiry_basic(self):
        from ct.tools.cro import draft_inquiry
        result = draft_inquiry(
            cro_id="reaction-biology",
            experiment_description="HiBiT degradation assay for ZNF687",
        )
        assert "summary" in result
        assert "subject" in result
        assert "body" in result
        assert "Reaction Biology" in result["body"]
        assert result["to_email"] == "info@reactionbiology.com"

    def test_draft_inquiry_with_context(self):
        from ct.tools.cro import draft_inquiry
        result = draft_inquiry(
            cro_id="reaction-biology",
            experiment_description="HiBiT assay",
            compound="YU-123",
            target="ZNF687",
            timeline="3 months",
        )
        assert "ZNF687" in result["body"]
        assert "YU-123" in result["body"]
        assert "3 months" in result["body"]

    def test_draft_inquiry_not_found(self):
        from ct.tools.cro import draft_inquiry
        result = draft_inquiry(
            cro_id="fake-cro",
            experiment_description="test",
        )
        assert "error" in result


class TestCroSendInquiry:
    def test_send_inquiry_dry_run(self):
        from ct.tools.cro import send_inquiry
        result = send_inquiry(
            cro_id="reaction-biology",
            subject="Test inquiry",
            body="Test body",
            dry_run=True,
        )
        assert "summary" in result
        assert result["dry_run"] is True
        assert "DRY RUN" in result["summary"]

    def test_send_inquiry_not_found(self):
        from ct.tools.cro import send_inquiry
        result = send_inquiry(
            cro_id="fake-cro",
            subject="Test",
            body="Test",
        )
        assert "error" in result
