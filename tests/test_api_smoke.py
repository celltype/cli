"""Optional live API smoke checks for schema drift detection.

These tests hit real public endpoints and are skipped by default.
Enable in CI with: CT_RUN_API_SMOKE=1.
"""

from __future__ import annotations

import os

import pytest

from ct.tools.clinical import (
    competitive_landscape,
    endpoint_benchmark,
    trial_design_benchmark,
    trial_search,
)
from ct.tools.data_api import opentargets_search, uniprot_lookup
from ct.tools.genomics import coloc, gwas_lookup
from ct.tools.intel import pipeline_watch
from ct.tools.literature import openalex_search, pubmed_search
from ct.tools.target import disease_association, druggability, expression_profile
from ct.tools.translational import biomarker_readiness


_RUN_SMOKE = os.environ.get("CT_RUN_API_SMOKE", "").strip().lower() in {"1", "true", "yes"}
_STRICT_SMOKE = os.environ.get("CT_API_SMOKE_STRICT", "").strip().lower() in {"1", "true", "yes"}

pytestmark = [
    pytest.mark.api_smoke,
    pytest.mark.skipif(not _RUN_SMOKE, reason="Set CT_RUN_API_SMOKE=1 to run live smoke tests"),
]


def _assert_no_signature_error(payload: dict):
    text = str(payload)
    assert "unexpected keyword argument 'json'" not in text
    assert "unexpected keyword argument 'data'" not in text


def _skip_if_non_strict_error(payload: dict):
    """Skip local smoke runs on live API errors.

    In strict mode (CI), any API/tool error should fail the job so we detect drift/outages.
    """
    if _STRICT_SMOKE:
        return
    if "error" in payload:
        pytest.skip(f"Live API smoke skipped in non-strict mode: {payload.get('error')}")


def test_pubmed_search_smoke():
    result = pubmed_search("TP53 cancer", max_results=1)
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert isinstance(result.get("articles"), list)
    assert "summary" in result


def test_openalex_search_smoke():
    result = openalex_search("TP53 cancer", max_results=1)
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert isinstance(result.get("articles"), list)
    assert "summary" in result


def test_uniprot_lookup_smoke():
    result = uniprot_lookup("P04637")
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert result.get("accession") == "P04637"
    assert "summary" in result


def test_gwas_lookup_smoke():
    result = gwas_lookup(gene="SNCA", trait="Parkinson disease")
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert result.get("n_associations", 0) >= 1


def test_trial_search_smoke():
    result = trial_search(query="Parkinson disease")
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert result.get("total_count", 0) >= 1


def test_coloc_smoke():
    result = coloc(gene="SNCA")
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert "n_colocalizations" in result


def test_target_druggability_smoke():
    result = druggability(gene="LRRK2")
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert "druggability_score" in result


def test_target_expression_profile_smoke():
    result = expression_profile(gene="SNCA", top_n=5)
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert isinstance(result.get("gtex_expression"), list)


def test_target_disease_association_smoke():
    result = disease_association(gene="PINK1", min_score=0.1)
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert "associations" in result


def test_opentargets_search_smoke():
    result = opentargets_search(query="Parkinson disease", entity_type="disease")
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result


def test_trial_design_benchmark_smoke():
    result = trial_design_benchmark(query="Parkinson disease", max_results=5)
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert "top_primary_endpoints" in result


def test_endpoint_benchmark_smoke():
    result = endpoint_benchmark(query="ulcerative colitis", max_results=5)
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert "endpoint_families" in result


def test_competitive_landscape_smoke():
    result = competitive_landscape(gene="IL23R", indication="ulcerative colitis")
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result


def test_pipeline_watch_smoke():
    result = pipeline_watch(query="LRRK2", indication="Parkinson disease", max_trials=5, max_papers=3)
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert "momentum_score" in result


def test_biomarker_readiness_smoke():
    result = biomarker_readiness(biomarker="PD-L1", indication="NSCLC", max_evidence=3)
    _assert_no_signature_error(result)
    _skip_if_non_strict_error(result)
    assert "error" not in result, result.get("error")
    assert "summary" in result
    assert "readiness_score" in result
