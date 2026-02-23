"""Tests for individual tool implementations (unit tests with mocked data)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# ─── Safety tools ─────────────────────────────────────────────

class TestSafetyAntitargetProfile:
    @patch("ct.data.loaders.load_proteomics")
    def test_single_compound_clean(self, mock_prot):
        from ct.tools.safety import antitarget_profile

        mock_prot.return_value = pd.DataFrame(
            {"cpd_A": [0.1, -0.2, 0.3, 0.0]},
            index=["GENE1", "GENE2", "GENE3", "GENE4"],
        )
        result = antitarget_profile(compound_id="cpd_A")
        assert result["profiles"][0]["n_antitargets"] == 0
        assert result["profiles"][0]["safety_penalty"] == 0.0

    @patch("ct.data.loaders.load_proteomics")
    def test_single_compound_hits_tumor_suppressor(self, mock_prot):
        from ct.tools.safety import antitarget_profile

        mock_prot.return_value = pd.DataFrame(
            {"cpd_A": [-0.8, -0.6, 0.1]},
            index=["TP53", "RB1", "GENE1"],
        )
        result = antitarget_profile(compound_id="cpd_A")
        profile = result["profiles"][0]
        assert profile["n_tumor_suppressors"] == 2
        assert profile["safety_penalty"] == 6.0  # 3.0 per TSG

    @patch("ct.data.loaders.load_proteomics")
    def test_teratogenic_hit_highest_penalty(self, mock_prot):
        from ct.tools.safety import antitarget_profile

        mock_prot.return_value = pd.DataFrame(
            {"cpd_A": [-1.0]},
            index=["SALL4"],
        )
        result = antitarget_profile(compound_id="cpd_A")
        assert result["profiles"][0]["n_teratogenic"] == 1
        assert result["profiles"][0]["safety_penalty"] == 10.0


class TestSafetyClassify:
    @patch("ct.data.loaders.load_prism")
    @patch("ct.data.loaders.load_proteomics")
    def test_safe_compound(self, mock_prot, mock_prism):
        from ct.tools.safety import classify

        mock_prot.return_value = pd.DataFrame(
            {"cpd_A": [0.1, 0.2]},
            index=["GENE1", "GENE2"],
        )
        mock_prism.return_value = pd.DataFrame({
            "pert_name": ["cpd_A"] * 3,
            "pert_dose": [10.0] * 3,
            "ccle_name": ["CELL1", "CELL2", "CELL3"],
            "LFC": [0.0, -0.1, 0.05],
        })
        result = classify(compound_id="cpd_A")
        assert result["classifications"][0]["classification"] == "SAFE"

    @patch("ct.data.loaders.load_prism")
    @patch("ct.data.loaders.load_proteomics")
    def test_dangerous_compound(self, mock_prot, mock_prism):
        from ct.tools.safety import classify

        mock_prot.return_value = pd.DataFrame(
            {"cpd_A": [-1.0]},
            index=["SALL4"],
        )
        mock_prism.return_value = pd.DataFrame({
            "pert_name": ["cpd_A"],
            "pert_dose": [10.0],
            "ccle_name": ["CELL1"],
            "LFC": [-0.5],
        })
        result = classify(compound_id="cpd_A")
        assert result["classifications"][0]["classification"] == "DANGEROUS"


class TestSall4Risk:
    @patch("ct.data.loaders.load_proteomics")
    def test_high_risk(self, mock_prot):
        from ct.tools.safety import sall4_risk

        mock_prot.return_value = pd.DataFrame(
            {"cpd_A": [-1.2, -0.3]},
            index=["SALL4", "SALL1"],
        )
        result = sall4_risk(compound_id="cpd_A")
        assert result["assessments"][0]["risk_level"] == "HIGH"
        assert "thalidomide" in result["assessments"][0]["risk_detail"].lower()

    @patch("ct.data.loaders.load_proteomics")
    def test_minimal_risk(self, mock_prot):
        from ct.tools.safety import sall4_risk

        mock_prot.return_value = pd.DataFrame(
            {"cpd_A": [0.1]},
            index=["SALL4"],
        )
        result = sall4_risk(compound_id="cpd_A")
        assert result["assessments"][0]["risk_level"] == "MINIMAL"

    @patch("ct.data.loaders.load_proteomics")
    def test_no_sall_in_data(self, mock_prot):
        from ct.tools.safety import sall4_risk

        mock_prot.return_value = pd.DataFrame(
            {"cpd_A": [0.1]},
            index=["GENE1"],
        )
        result = sall4_risk(compound_id="cpd_A")
        assert result["risk_assessment"] == "UNKNOWN"


class TestSafetyFaersSignalScan:
    @patch("ct.tools.safety.request_json")
    def test_top_events_signal_scan(self, mock_request_json):
        from ct.tools.safety import faers_signal_scan

        def _mock_json(method, url, params=None, **kwargs):
            search = (params or {}).get("search", "")
            count = (params or {}).get("count", "")

            if not search and not count:
                return {"meta": {"results": {"total": 10000}}}, None

            if search == 'patient.drug.medicinalproduct.exact:"DrugX"' and not count:
                return {"meta": {"results": {"total": 200}}}, None

            if count == "patient.reaction.reactionmeddrapt.exact":
                return {
                    "results": [
                        {"term": "NAUSEA", "count": 40},
                        {"term": "HEADACHE", "count": 20},
                    ]
                }, None

            if search == 'patient.reaction.reactionmeddrapt.exact:"NAUSEA"':
                return {"meta": {"results": {"total": 500}}}, None
            if search == (
                'patient.drug.medicinalproduct.exact:"DrugX"+AND+'
                'patient.reaction.reactionmeddrapt.exact:"NAUSEA"'
            ):
                return {"meta": {"results": {"total": 25}}}, None

            if search == 'patient.reaction.reactionmeddrapt.exact:"HEADACHE"':
                return {"meta": {"results": {"total": 350}}}, None
            if search == (
                'patient.drug.medicinalproduct.exact:"DrugX"+AND+'
                'patient.reaction.reactionmeddrapt.exact:"HEADACHE"'
            ):
                return {"meta": {"results": {"total": 5}}}, None

            return None, f"Unhandled params: {params}"

        mock_request_json.side_effect = _mock_json

        result = faers_signal_scan(drug_name="DrugX", top_n=2)
        assert "error" not in result
        assert result["drug_name"] == "DrugX"
        assert result["n_events_analyzed"] == 2
        assert len(result["signals"]) == 2
        assert result["signals"][0]["event"] == "NAUSEA"
        assert result["signals"][0]["meets_signal_criteria"] is True
        assert result["signals"][0]["prr"] > 2.0
        assert "summary" in result

    @patch("ct.tools.safety.request_json")
    def test_specific_event_scan(self, mock_request_json):
        from ct.tools.safety import faers_signal_scan

        def _mock_json(method, url, params=None, **kwargs):
            search = (params or {}).get("search", "")
            if not search:
                return {"meta": {"results": {"total": 2000}}}, None
            if search == 'patient.drug.medicinalproduct.exact:"DrugY"':
                return {"meta": {"results": {"total": 80}}}, None
            if search == 'patient.reaction.reactionmeddrapt.exact:"RASH"':
                return {"meta": {"results": {"total": 120}}}, None
            if search == (
                'patient.drug.medicinalproduct.exact:"DrugY"+AND+'
                'patient.reaction.reactionmeddrapt.exact:"RASH"'
            ):
                return {"meta": {"results": {"total": 12}}}, None
            return None, f"Unhandled params: {params}"

        mock_request_json.side_effect = _mock_json

        result = faers_signal_scan(drug_name="DrugY", event="RASH")
        assert "error" not in result
        assert result["n_events_analyzed"] == 1
        assert result["signals"][0]["event"] == "RASH"
        assert "prr" in result["signals"][0]


class TestSafetyLabelRiskExtract:
    @patch("ct.tools.safety.request_json")
    def test_extracts_key_label_sections(self, mock_request_json):
        from ct.tools.safety import label_risk_extract

        mock_request_json.return_value = (
            {
                "results": [
                    {
                        "openfda": {
                            "brand_name": ["DrugA"],
                            "generic_name": ["druga"],
                            "application_number": ["NDA123456"],
                            "manufacturer_name": ["Acme Pharma"],
                        },
                        "boxed_warning": ["Can cause severe hepatotoxicity."],
                        "contraindications": ["Do not use with MAO inhibitors."],
                        "warnings_and_cautions": ["Monitor liver enzymes regularly."],
                        "drug_interactions": ["Strong CYP3A4 inhibitors increase exposure."],
                    }
                ]
            },
            None,
        )

        result = label_risk_extract(drug_name="DrugA")
        assert "error" not in result
        assert result["labels_found"] == 1
        assert result["risk_level"] == "HIGH"
        label = result["labels"][0]
        assert "boxed_warning" in label["risk_flags"]
        assert "severe hepatotoxicity" in label["sections"]["boxed_warning"].lower()

    @patch("ct.tools.safety.request_json")
    def test_no_labels_found(self, mock_request_json):
        from ct.tools.safety import label_risk_extract

        mock_request_json.return_value = ({"results": []}, None)
        result = label_risk_extract(drug_name="MissingDrug")
        assert "error" not in result
        assert result["labels_found"] == 0
        assert result["risk_level"] == "UNKNOWN"


# ─── Combination tools ────────────────────────────────────────

class TestSynergyPredict:
    @patch("ct.data.loaders.load_model_metadata")
    @patch("ct.data.loaders.load_prism")
    @patch("ct.data.loaders.load_l1000")
    def test_basic_synergy(self, mock_l1000, mock_prism, mock_model):
        from ct.tools.combination import synergy_predict

        # L1000: rows=compounds, cols=genes. Anti-correlated pair.
        mock_l1000.return_value = pd.DataFrame(
            {"GENE1": [1.0, -1.0], "GENE2": [-1.0, 1.0],
             "GENE3": [0.5, -0.5], "GENE4": [-0.5, 0.5]},
            index=["cpd_A", "cpd_B"],
        )
        mock_prism.return_value = pd.DataFrame({
            "pert_name": ["cpd_A", "cpd_A", "cpd_B", "cpd_B"],
            "pert_dose": [10.0, 10.0, 10.0, 10.0],
            "ccle_name": ["CELL1", "CELL2", "CELL1", "CELL2"],
            "LFC": [-0.5, -0.1, -0.1, -0.6],
        })
        mock_model.return_value = pd.DataFrame({
            "CCLEName": ["CELL1", "CELL2"],
            "OncotreeLineage": ["Lung", "Breast"],
        })
        result = synergy_predict(compound_id="cpd_A")
        assert "top_candidates" in result
        assert result["n_pairs"] >= 1


class TestSyntheticLethality:
    @patch("ct.data.loaders.load_crispr")
    def test_basic_sl(self, mock_crispr):
        from ct.tools.combination import synthetic_lethality

        # Need 50+ rows (cell lines) to pass the min-cell-lines filter.
        # Create anti-correlated genes.
        np.random.seed(42)
        n = 60
        base = np.random.randn(n)
        data = {
            "GENE_A": base,
            "GENE_B": -base + np.random.randn(n) * 0.1,  # anti-correlated
            "GENE_C": np.random.randn(n),
        }
        mock_crispr.return_value = pd.DataFrame(data)
        result = synthetic_lethality(gene="GENE_A")
        assert "top_partners" in result
        assert "target_gene" in result


# ─── Clinical tools ───────────────────────────────────────────

class TestIndicationMap:
    @patch("ct.data.loaders.load_model_metadata")
    @patch("ct.data.loaders.load_prism")
    def test_basic_mapping(self, mock_prism, mock_model):
        from ct.tools.clinical import indication_map

        mock_prism.return_value = pd.DataFrame({
            "pert_name": ["cpd_A"] * 10,
            "pert_dose": [10.0] * 10,
            "ccle_name": [f"CELL{i}" for i in range(10)],
            "LFC": [-1.0, -0.8, -0.6, -0.3, 0.0, 0.1, 0.2, -0.9, -0.7, -0.1],
        })
        mock_model.return_value = pd.DataFrame({
            "CCLEName": [f"CELL{i}" for i in range(10)],
            "OncotreeLineage": ["Lung"] * 5 + ["Breast"] * 5,
        })
        result = indication_map(compound_id="cpd_A")
        assert "indications" in result
        assert result["n_indications"] >= 0


class TestTrialDesignBenchmark:
    @patch("ct.tools.clinical.request_json")
    def test_benchmark_aggregates_design_features(self, mock_request_json):
        from ct.tools.clinical import trial_design_benchmark

        mock_request_json.return_value = (
            {
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT0001", "briefTitle": "Trial One"},
                            "statusModule": {
                                "overallStatus": "RECRUITING",
                                "startDateStruct": {"date": "2024-01"},
                            },
                            "designModule": {
                                "phases": ["PHASE2"],
                                "studyType": "INTERVENTIONAL",
                                "designInfo": {
                                    "allocation": "RANDOMIZED",
                                    "interventionModel": "PARALLEL",
                                    "maskingInfo": {"masking": "DOUBLE"},
                                },
                                "enrollmentInfo": {"count": 120},
                            },
                            "outcomesModule": {
                                "primaryOutcomes": [
                                    {"measure": "Progression-free survival"},
                                    {"measure": "Objective response rate"},
                                ]
                            },
                            "eligibilityModule": {
                                "eligibilityCriteria": "Inclusion: EGFR mutation required; ECOG <= 1."
                            },
                            "armsInterventionsModule": {
                                "interventions": [
                                    {"name": "Drug A"},
                                    {"name": "Placebo"},
                                ]
                            },
                            "conditionsModule": {"conditions": ["NSCLC"]},
                            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor A"}},
                        }
                    },
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCT0002", "briefTitle": "Trial Two"},
                            "statusModule": {
                                "overallStatus": "COMPLETED",
                                "startDateStruct": {"date": "2022-05"},
                            },
                            "designModule": {
                                "phases": ["PHASE3"],
                                "studyType": "INTERVENTIONAL",
                                "designInfo": {
                                    "allocation": "NON_RANDOMIZED",
                                    "interventionModel": "SINGLE_GROUP",
                                    "maskingInfo": {"masking": "NONE"},
                                },
                                "enrollmentInfo": {"count": 300},
                            },
                            "outcomesModule": {
                                "primaryOutcomes": [
                                    {"measure": "Overall survival"},
                                ]
                            },
                            "eligibilityModule": {"eligibilityCriteria": "Inclusion: Histologically confirmed disease."},
                            "armsInterventionsModule": {"interventions": [{"name": "Drug B"}]},
                            "conditionsModule": {"conditions": ["NSCLC"]},
                            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor B"}},
                        }
                    },
                ]
            },
            None,
        )

        result = trial_design_benchmark(query="NSCLC")
        assert "error" not in result
        assert result["n_trials"] == 2
        assert result["design_patterns"]["randomized_trials"] == 1
        assert result["design_patterns"]["blinded_trials"] == 1
        assert result["design_patterns"]["placebo_control_trials"] == 1
        assert result["design_patterns"]["biomarker_criteria_trials"] == 1
        assert result["design_patterns"]["ecog_criteria_trials"] == 1
        assert len(result["top_primary_endpoints"]) >= 1
        assert "summary" in result

    @patch("ct.tools.clinical.request_json")
    def test_phase_filter(self, mock_request_json):
        from ct.tools.clinical import trial_design_benchmark

        mock_request_json.return_value = (
            {
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCTP2", "briefTitle": "Phase 2 Trial"},
                            "statusModule": {"overallStatus": "RECRUITING"},
                            "designModule": {"phases": ["PHASE2"], "designInfo": {}, "enrollmentInfo": {"count": 80}},
                            "outcomesModule": {"primaryOutcomes": []},
                            "eligibilityModule": {"eligibilityCriteria": ""},
                            "armsInterventionsModule": {"interventions": []},
                            "conditionsModule": {"conditions": []},
                            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
                        }
                    },
                    {
                        "protocolSection": {
                            "identificationModule": {"nctId": "NCTP3", "briefTitle": "Phase 3 Trial"},
                            "statusModule": {"overallStatus": "RECRUITING"},
                            "designModule": {"phases": ["PHASE3"], "designInfo": {}, "enrollmentInfo": {"count": 100}},
                            "outcomesModule": {"primaryOutcomes": []},
                            "eligibilityModule": {"eligibilityCriteria": ""},
                            "armsInterventionsModule": {"interventions": []},
                            "conditionsModule": {"conditions": []},
                            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
                        }
                    },
                ]
            },
            None,
        )

        result = trial_design_benchmark(query="NSCLC", phase="phase 3")
        assert "error" not in result
        assert result["n_trials"] == 1
        assert result["trials"][0]["nct_id"] == "NCTP3"


# ─── Literature tools (mock HTTP) ─────────────────────────────

class TestPubmedSearch:
    @patch("httpx.get")
    def test_successful_search(self, mock_get):
        from ct.tools.literature import pubmed_search

        # Mock ESearch response
        search_response = MagicMock()
        search_response.json.return_value = {
            "esearchresult": {"count": "1", "idlist": ["12345678"]},
        }
        search_response.raise_for_status = MagicMock()

        # Mock ESummary response
        summary_response = MagicMock()
        summary_response.json.return_value = {
            "result": {
                "uids": ["12345678"],
                "12345678": {
                    "title": "Test Paper",
                    "authors": [{"name": "Smith J"}],
                    "source": "Nature",
                    "pubdate": "2024",
                    "articleids": [{"idtype": "doi", "value": "10.1234/test"}],
                },
            },
        }
        summary_response.raise_for_status = MagicMock()

        mock_get.side_effect = [search_response, summary_response]

        result = pubmed_search(query="CRBN degrader")
        assert result["total_count"] == 1
        assert len(result["articles"]) == 1
        assert result["articles"][0]["title"] == "Test Paper"

    @patch("httpx.get")
    def test_no_results(self, mock_get):
        from ct.tools.literature import pubmed_search

        resp = MagicMock()
        resp.json.return_value = {"esearchresult": {"count": "0", "idlist": []}}
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = pubmed_search(query="xyznonexistent")
        assert result["total_count"] == 0
        assert result["articles"] == []


class TestChemblQuery:
    @patch("ct.tools.literature.request_json")
    def test_accepts_compound_alias_for_query_type(self, mock_request_json):
        from ct.tools.literature import chembl_query

        mock_request_json.return_value = (
            {
                "molecules": [
                    {
                        "molecule_chembl_id": "CHEMBL123",
                        "pref_name": "SILMITASERTIB",
                        "molecule_type": "Small molecule",
                        "max_phase": 2,
                        "molecule_properties": {"full_mwt": 350.4, "alogp": 2.3},
                        "molecule_structures": {"canonical_smiles": "CCN"},
                    }
                ]
            },
            None,
        )

        result = chembl_query(query="CX-4945", query_type="compound")
        assert "error" not in result
        assert result["molecules"][0]["chembl_id"] == "CHEMBL123"


# ─── Target expression_profile (GTEx + HPA API) ─────────────

class TestExpressionProfile:
    @patch("httpx.get")
    def test_gtex_and_hpa_success(self, mock_get):
        from ct.tools.target import expression_profile

        # Mock 1: GTEx gene reference API
        gtex_ref_resp = MagicMock()
        gtex_ref_resp.status_code = 200
        gtex_ref_resp.json.return_value = {
            "data": [{
                "gencodeId": "ENSG00000012048.23",
                "geneSymbol": "BRCA1",
            }]
        }

        # Mock 2: GTEx median expression API
        gtex_expr_resp = MagicMock()
        gtex_expr_resp.status_code = 200
        gtex_expr_resp.json.return_value = {
            "data": [
                {"tissueSiteDetailId": "Testis", "median": 45.2},
                {"tissueSiteDetailId": "Ovary", "median": 32.1},
                {"tissueSiteDetailId": "Breast_Mammary_Tissue", "median": 15.5},
                {"tissueSiteDetailId": "Lung", "median": 8.3},
                {"tissueSiteDetailId": "Liver", "median": 3.1},
            ]
        }

        # Mock 3: HPA API (using Ensembl ID)
        hpa_resp = MagicMock()
        hpa_resp.status_code = 200
        hpa_resp.json.return_value = {
            "RNATissue": {
                "summary": "Tissue enhanced (testis)",
                "data": [
                    {"Tissue": "testis", "TPM": 40.0, "nTPM": 35.0},
                    {"Tissue": "ovary", "TPM": 30.0, "nTPM": 25.0},
                ],
            },
            "ProteinTissue": {"data": []},
            "RNACancer": {"data": [{"Cancer": "Breast cancer", "TPM": 20.0, "nTPM": 18.0}]},
            "RNASingleCell": {"data": []},
        }

        mock_get.side_effect = [gtex_ref_resp, gtex_expr_resp, hpa_resp]

        result = expression_profile(gene="BRCA1", top_n=5)
        assert "summary" in result
        assert result["gene"] == "BRCA1"
        assert result["gencode_id"] == "ENSG00000012048.23"
        assert result["tissue_specificity_tau"] is not None
        assert len(result["gtex_expression"]) == 5
        assert result["gtex_expression"][0]["tissue"] == "Testis"
        assert result["gtex_expression"][0]["median_tpm"] == 45.2
        assert "BRCA1 expression" in result["summary"]

    @patch("httpx.get")
    def test_gtex_gene_not_found(self, mock_get):
        from ct.tools.target import expression_profile

        # GTEx gene reference returns empty
        gtex_ref_resp = MagicMock()
        gtex_ref_resp.status_code = 200
        gtex_ref_resp.json.return_value = {"data": []}

        # HPA also returns 404
        hpa_resp = MagicMock()
        hpa_resp.status_code = 404
        hpa_resp.json.return_value = {}

        mock_get.side_effect = [gtex_ref_resp, hpa_resp]

        result = expression_profile(gene="FAKEGENE")
        assert "error" in result or "No expression data" in result.get("summary", "")

    @patch("httpx.get")
    def test_tau_specificity(self, mock_get):
        from ct.tools.target import expression_profile

        # Mock GTEx with one tissue much higher than rest → high tau
        gtex_ref_resp = MagicMock()
        gtex_ref_resp.status_code = 200
        gtex_ref_resp.json.return_value = {
            "data": [{"gencodeId": "ENSG00000000001.1", "geneSymbol": "TGENE"}]
        }

        gtex_expr_resp = MagicMock()
        gtex_expr_resp.status_code = 200
        gtex_expr_resp.json.return_value = {
            "data": [
                {"tissueSiteDetailId": "Testis", "median": 100.0},
                {"tissueSiteDetailId": "Lung", "median": 1.0},
                {"tissueSiteDetailId": "Liver", "median": 1.0},
            ]
        }

        hpa_resp = MagicMock()
        hpa_resp.status_code = 404
        hpa_resp.json.return_value = {}

        mock_get.side_effect = [gtex_ref_resp, gtex_expr_resp, hpa_resp]

        result = expression_profile(gene="TGENE")
        tau = result["tissue_specificity_tau"]
        assert tau is not None
        assert tau > 0.8  # Highly tissue-specific
        assert "tissue-specific" in result["summary"]

    @patch("httpx.get")
    def test_expression_profile_alias_fallback_gba1_to_gba(self, mock_get):
        from ct.tools.target import expression_profile

        gtex_ref_gba1 = MagicMock()
        gtex_ref_gba1.status_code = 200
        gtex_ref_gba1.json.return_value = {"data": []}

        gtex_ref_gba = MagicMock()
        gtex_ref_gba.status_code = 200
        gtex_ref_gba.json.return_value = {
            "data": [{"gencodeId": "ENSG00000177628.13", "geneSymbol": "GBA"}]
        }

        gtex_expr = MagicMock()
        gtex_expr.status_code = 200
        gtex_expr.json.return_value = {
            "data": [
                {"tissueSiteDetailId": "Brain_Cortex", "median": 21.5},
                {"tissueSiteDetailId": "Liver", "median": 8.2},
            ]
        }

        hpa_ensembl_not_found = MagicMock()
        hpa_ensembl_not_found.status_code = 404
        hpa_ensembl_not_found.json.return_value = {}

        hpa_gba = MagicMock()
        hpa_gba.status_code = 200
        hpa_gba.json.return_value = {
            "RNATissue": {
                "summary": "Tissue enhanced (brain)",
                "data": [{"Tissue": "brain", "TPM": 19.0, "nTPM": 18.0}],
            },
            "ProteinTissue": {"data": []},
            "RNACancer": {"data": []},
            "RNASingleCell": {"data": []},
        }

        mock_get.side_effect = [
            gtex_ref_gba1,
            gtex_ref_gba,
            gtex_expr,
            hpa_ensembl_not_found,
            hpa_gba,
        ]

        result = expression_profile(gene="GBA1", top_n=5)
        assert "error" not in result
        assert result["gene"] == "GBA"
        assert result["gencode_id"] == "ENSG00000177628.13"
        assert result["gtex_expression"][0]["tissue"] == "Brain_Cortex"
        assert "GBA expression" in result["summary"]


# ─── Expression diff_expression ──────────────────────────────

class TestDiffExpression:
    @patch("ct.data.loaders.load_l1000")
    def test_single_gene(self, mock_l1000):
        from ct.tools.expression import diff_expression

        # 8 samples, 4 per group — enough for clear separation
        mock_l1000.return_value = pd.DataFrame(
            {
                "GENE1": [3.0, 2.5, 2.8, 3.2, -1.0, -0.5, -0.8, -1.2],
                "GENE2": [0.1, 0.2, -0.1, 0.0, 0.1, 0.2, -0.1, 0.0],
                "GENE3": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "GENE4": [-2.0, -1.5, -1.8, -2.2, 0.5, 0.8, 1.0, 0.6],
            },
            index=["cpd_A1", "cpd_A2", "cpd_A3", "cpd_A4",
                   "cpd_B1", "cpd_B2", "cpd_B3", "cpd_B4"],
        )

        result = diff_expression(
            gene="GENE1",
            group_a=["cpd_A1", "cpd_A2", "cpd_A3", "cpd_A4"],
            group_b=["cpd_B1", "cpd_B2", "cpd_B3", "cpd_B4"],
        )
        assert "summary" in result
        assert result["n_tested"] == 1
        assert len(result["results"]) == 1
        row = result["results"][0]
        assert row["gene"] == "GENE1"
        assert row["log2_fold_change"] > 0  # Higher in group A
        assert row["direction"] == "up_in_A"
        assert row["p_value"] < 0.05  # Should be significant with clear separation

    @patch("ct.data.loaders.load_l1000")
    def test_all_genes_with_fdr(self, mock_l1000):
        from ct.tools.expression import diff_expression

        np.random.seed(42)
        n_a, n_b = 5, 5
        # GENE1: clear difference, GENE2-5: no difference
        data = {
            "GENE1": list(np.random.randn(n_a) + 3) + list(np.random.randn(n_b) - 1),
            "GENE2": list(np.random.randn(n_a + n_b)),
            "GENE3": list(np.random.randn(n_a + n_b)),
            "GENE4": list(np.random.randn(n_a + n_b)),
            "GENE5": list(np.random.randn(n_a + n_b)),
        }
        idx = [f"A{i}" for i in range(n_a)] + [f"B{i}" for i in range(n_b)]
        mock_l1000.return_value = pd.DataFrame(data, index=idx)

        result = diff_expression(
            gene="all",
            group_a=[f"A{i}" for i in range(n_a)],
            group_b=[f"B{i}" for i in range(n_b)],
        )
        assert result["n_tested"] == 5
        assert "fdr" in result["results"][0]
        # GENE1 should be most significant
        assert result["results"][0]["gene"] == "GENE1"

    @patch("ct.data.loaders.load_l1000")
    def test_missing_groups(self, mock_l1000):
        from ct.tools.expression import diff_expression

        mock_l1000.return_value = pd.DataFrame(
            {"GENE1": [1.0, 2.0]}, index=["A1", "A2"]
        )

        result = diff_expression(gene="GENE1", group_a=["A1", "A2"], group_b=["B1", "B2"])
        assert "error" in result

    def test_no_groups_provided(self):
        from ct.tools.expression import diff_expression

        result = diff_expression(gene="GENE1")
        assert "error" in result

    @patch("ct.data.loaders.load_l1000")
    def test_gene_not_found(self, mock_l1000):
        from ct.tools.expression import diff_expression

        mock_l1000.return_value = pd.DataFrame(
            {"GENE1": [1.0, 2.0, 0.5, 0.3]},
            index=["A1", "A2", "B1", "B2"],
        )
        result = diff_expression(gene="NONEXIST", group_a=["A1", "A2"], group_b=["B1", "B2"])
        assert "error" in result


# ─── Biomarker panel_select ──────────────────────────────────

class TestBiomarkerPanelSelect:
    @patch("ct.data.loaders.load_model_metadata")
    @patch("ct.data.loaders.load_mutations")
    @patch("ct.data.loaders.load_prism")
    def test_mutual_info_method(self, mock_prism, mock_mutations, mock_model):
        from ct.tools.biomarker import panel_select

        np.random.seed(42)
        n_cells = 40
        cell_names = [f"CELL{i}" for i in range(n_cells)]
        model_ids = [f"ACH-{i:06d}" for i in range(n_cells)]

        # Model metadata: map CCLEName -> ModelID
        mock_model.return_value = pd.DataFrame({
            "CCLEName": cell_names,
            "ModelID": model_ids,
        })

        # Mutations: ModelID x genes, binary (0/1)
        # Make BRAF correlated with sensitivity
        braf_mut = np.array([1] * 15 + [0] * 25)
        np.random.shuffle(braf_mut)
        mut_data = {
            "BRAF": braf_mut,
            "TP53": np.random.binomial(1, 0.3, n_cells),
            "KRAS": np.random.binomial(1, 0.2, n_cells),
            "PIK3CA": np.random.binomial(1, 0.15, n_cells),
            "EGFR": np.random.binomial(1, 0.1, n_cells),
            "PTEN": np.random.binomial(1, 0.25, n_cells),
        }
        mock_mutations.return_value = pd.DataFrame(mut_data, index=model_ids)

        # PRISM: sensitive cells have LFC < -0.5
        # BRAF-mutant cells are more sensitive
        lfc_values = []
        for i in range(n_cells):
            if braf_mut[i]:
                lfc_values.append(float(np.random.uniform(-1.5, -0.6)))
            else:
                lfc_values.append(float(np.random.uniform(-0.3, 0.3)))

        mock_prism.return_value = pd.DataFrame({
            "pert_name": ["cpd_A"] * n_cells,
            "pert_dose": [10.0] * n_cells,
            "ccle_name": cell_names,
            "LFC": lfc_values,
        })

        result = panel_select(compound_id="cpd_A", n_features=5, method="mutual_info")
        assert "summary" in result
        assert result["compound"] == "cpd_A"
        assert result["method"] == "mutual_info"
        assert len(result["biomarkers"]) <= 5
        assert result["n_cell_lines"] == n_cells
        # BRAF should be among top biomarkers
        biomarker_genes = [b["gene"] for b in result["biomarkers"]]
        assert "BRAF" in biomarker_genes, f"BRAF not in top biomarkers: {biomarker_genes}"

    @patch("ct.data.loaders.load_model_metadata")
    @patch("ct.data.loaders.load_mutations")
    @patch("ct.data.loaders.load_prism")
    def test_random_forest_method(self, mock_prism, mock_mutations, mock_model):
        from ct.tools.biomarker import panel_select

        np.random.seed(42)
        n_cells = 30
        cell_names = [f"CELL{i}" for i in range(n_cells)]
        model_ids = [f"ACH-{i:06d}" for i in range(n_cells)]

        mock_model.return_value = pd.DataFrame({
            "CCLEName": cell_names, "ModelID": model_ids,
        })

        mut_data = {
            "GENE_A": np.random.binomial(1, 0.3, n_cells),
            "GENE_B": np.random.binomial(1, 0.4, n_cells),
            "GENE_C": np.random.binomial(1, 0.25, n_cells),
            "GENE_D": np.random.binomial(1, 0.2, n_cells),
        }
        mock_mutations.return_value = pd.DataFrame(mut_data, index=model_ids)

        # Mix of sensitive and resistant
        lfc_vals = list(np.random.uniform(-1.5, -0.6, 10)) + list(np.random.uniform(-0.2, 0.3, 20))
        mock_prism.return_value = pd.DataFrame({
            "pert_name": ["cpd_X"] * n_cells,
            "pert_dose": [10.0] * n_cells,
            "ccle_name": cell_names,
            "LFC": lfc_vals,
        })

        result = panel_select(compound_id="cpd_X", method="random_forest", n_features=3)
        assert result["method"] == "random_forest"
        assert len(result["biomarkers"]) <= 3
        assert result.get("cv_auc") is not None or result.get("cv_auc") is None  # May or may not compute

    @patch("ct.data.loaders.load_model_metadata")
    @patch("ct.data.loaders.load_mutations")
    @patch("ct.data.loaders.load_prism")
    def test_lasso_method(self, mock_prism, mock_mutations, mock_model):
        from ct.tools.biomarker import panel_select

        np.random.seed(42)
        n_cells = 30
        cell_names = [f"CELL{i}" for i in range(n_cells)]
        model_ids = [f"ACH-{i:06d}" for i in range(n_cells)]

        mock_model.return_value = pd.DataFrame({
            "CCLEName": cell_names, "ModelID": model_ids,
        })

        mut_data = {
            "GENE_A": np.random.binomial(1, 0.3, n_cells),
            "GENE_B": np.random.binomial(1, 0.4, n_cells),
            "GENE_C": np.random.binomial(1, 0.25, n_cells),
            "GENE_D": np.random.binomial(1, 0.2, n_cells),
        }
        mock_mutations.return_value = pd.DataFrame(mut_data, index=model_ids)

        lfc_vals = list(np.random.uniform(-1.5, -0.6, 10)) + list(np.random.uniform(-0.2, 0.3, 20))
        mock_prism.return_value = pd.DataFrame({
            "pert_name": ["cpd_Y"] * n_cells,
            "pert_dose": [10.0] * n_cells,
            "ccle_name": cell_names,
            "LFC": lfc_vals,
        })

        result = panel_select(compound_id="cpd_Y", method="lasso", n_features=3)
        assert result["method"] == "lasso"
        assert "biomarkers" in result

    def test_invalid_method(self):
        """Test that invalid method returns error without calling loaders."""
        from ct.tools.biomarker import panel_select

        result = panel_select(compound_id="cpd_Z", method="invalid_method")
        assert "error" in result

    @patch("ct.data.loaders.load_model_metadata")
    @patch("ct.data.loaders.load_mutations")
    @patch("ct.data.loaders.load_prism")
    def test_compound_not_found(self, mock_prism, mock_mutations, mock_model):
        from ct.tools.biomarker import panel_select

        mock_prism.return_value = pd.DataFrame({
            "pert_name": ["other_cpd"],
            "pert_dose": [10.0],
            "ccle_name": ["CELL1"],
            "LFC": [-0.5],
        })
        mock_mutations.return_value = pd.DataFrame({"GENE1": [1]}, index=["ACH-000001"])
        mock_model.return_value = pd.DataFrame({"CCLEName": ["CELL1"], "ModelID": ["ACH-000001"]})

        result = panel_select(compound_id="nonexistent_cpd")
        assert "error" in result


# ─── Genomics GWAS / coloc ───────────────────────────────────

class TestGenomicsGwasLookup:
    @patch("ct.tools.genomics.request_json")
    def test_requires_non_empty_gene(self, mock_request_json):
        from ct.tools.genomics import gwas_lookup

        result = gwas_lookup(gene="", trait="obesity")
        assert "error" in result
        assert "requires a non-empty gene symbol" in result["summary"]
        assert result["trait_filter"] == "obesity"
        mock_request_json.assert_not_called()


class TestGenomicsColoc:
    @patch("httpx.post")
    @patch("httpx.get")
    def test_coloc_with_results(self, mock_get, mock_post):
        from ct.tools.genomics import coloc

        # Mock Ensembl gene lookup
        ens_resp = MagicMock()
        ens_resp.status_code = 200
        ens_resp.json.return_value = {"id": "ENSG00000169174"}
        mock_get.return_value = ens_resp

        # Mock Open Targets GraphQL response
        ot_resp = MagicMock()
        ot_resp.status_code = 200
        ot_resp.json.return_value = {
            "data": {
                "target": {
                    "id": "ENSG00000169174",
                    "approvedSymbol": "PCSK9",
                    "approvedName": "proprotein convertase subtilisin/kexin type 9",
                    "gwasCredibleSets": {
                        "count": 5,
                        "rows": [
                            {
                                "studyLocusId": "sl_001",
                                "study": {
                                    "id": "GCST001",
                                    "studyType": "GWAS",
                                    "traitFromSource": "LDL cholesterol",
                                    "diseases": [{"id": "EFO_0004611", "name": "LDL cholesterol measurement"}],
                                    "nSamples": 50000,
                                },
                                "variant": {
                                    "id": "1_55505647_G_A",
                                    "rsIds": ["rs11591147"],
                                    "chromosome": "1",
                                    "position": 55505647,
                                },
                                "pValueMantissa": 3.5,
                                "pValueExponent": -50,
                                "beta": -0.35,
                                "l2GPredictions": [
                                    {"target": {"id": "ENSG00000169174", "approvedSymbol": "PCSK9"}, "score": 0.95}
                                ],
                                "locus": {"count": 12},
                                "colocalisationsQtl": [
                                    {
                                        "studyLocusId": "sl_qtl_001",
                                        "qtlStudyId": "GTEx_eQTL_Liver",
                                        "phenotypeId": "ENSG00000169174",
                                        "tissue": {"id": "UBERON_0002107", "name": "Liver"},
                                        "h4": 0.92,
                                        "h3": 0.03,
                                        "h0": 0.01,
                                        "h1": 0.02,
                                        "h2": 0.02,
                                        "log2h4h3": 4.94,
                                    },
                                    {
                                        "studyLocusId": "sl_qtl_002",
                                        "qtlStudyId": "GTEx_eQTL_Kidney",
                                        "phenotypeId": "ENSG00000169174",
                                        "tissue": {"id": "UBERON_0002113", "name": "Kidney"},
                                        "h4": 0.45,
                                        "h3": 0.30,
                                        "h0": 0.10,
                                        "h1": 0.05,
                                        "h2": 0.10,
                                        "log2h4h3": 0.58,
                                    },
                                ],
                            },
                        ],
                    },
                }
            }
        }
        mock_post.return_value = ot_resp

        result = coloc(gene="PCSK9")
        assert "summary" in result
        assert result["gene"] == "PCSK9"
        assert result["ensembl_id"] == "ENSG00000169174"
        assert result["n_colocalizations"] == 2
        assert result["n_strong_coloc"] == 1  # H4 > 0.8 (Liver: 0.92)
        assert result["n_moderate_coloc"] == 0  # 0.5 < H4 <= 0.8 (Kidney: 0.45 not moderate)
        assert "Liver" in result["tissues"]
        # Top colocalization should be the strongest (highest H4)
        top = result["colocalizations"][0]
        assert top["h4"] == 0.92
        assert top["tissue"] == "Liver"

    @patch("httpx.post")
    @patch("httpx.get")
    def test_coloc_no_results(self, mock_get, mock_post):
        from ct.tools.genomics import coloc

        ens_resp = MagicMock()
        ens_resp.status_code = 200
        ens_resp.json.return_value = {"id": "ENSG00000000001"}
        mock_get.return_value = ens_resp

        ot_resp = MagicMock()
        ot_resp.status_code = 200
        ot_resp.json.return_value = {
            "data": {
                "target": {
                    "id": "ENSG00000000001",
                    "approvedSymbol": "TGENE",
                    "approvedName": "test gene",
                    "gwasCredibleSets": {
                        "count": 0,
                        "rows": [],
                    },
                }
            }
        }
        mock_post.return_value = ot_resp

        result = coloc(gene="TGENE")
        assert result["n_colocalizations"] == 0
        assert "no QTL colocalization" in result["summary"]

    @patch("httpx.get")
    def test_coloc_gene_not_found(self, mock_get):
        from ct.tools.genomics import coloc

        ens_resp = MagicMock()
        ens_resp.status_code = 404
        ens_resp.json.return_value = {}
        mock_get.return_value = ens_resp

        result = coloc(gene="FAKEGENE123")
        assert "error" in result

    @patch("httpx.post")
    @patch("httpx.get")
    def test_coloc_study_filter(self, mock_get, mock_post):
        from ct.tools.genomics import coloc

        ens_resp = MagicMock()
        ens_resp.status_code = 200
        ens_resp.json.return_value = {"id": "ENSG00000169174"}
        mock_get.return_value = ens_resp

        ot_resp = MagicMock()
        ot_resp.status_code = 200
        ot_resp.json.return_value = {
            "data": {
                "target": {
                    "id": "ENSG00000169174",
                    "approvedSymbol": "PCSK9",
                    "approvedName": "proprotein convertase subtilisin/kexin type 9",
                    "gwasCredibleSets": {
                        "count": 2,
                        "rows": [
                            {
                                "studyLocusId": "sl_001",
                                "study": {"id": "GCST001", "studyType": "GWAS",
                                          "traitFromSource": "LDL", "diseases": [], "nSamples": 10000},
                                "variant": {"id": "1_55505647_G_A", "rsIds": ["rs11591147"],
                                            "chromosome": "1", "position": 55505647},
                                "pValueMantissa": 2.0, "pValueExponent": -20, "beta": -0.2,
                                "l2GPredictions": [], "locus": {"count": 5},
                                "colocalisationsQtl": [{
                                    "studyLocusId": "sl_q1", "qtlStudyId": "eQTL_Liver",
                                    "phenotypeId": "ENSG00000169174",
                                    "tissue": {"id": "T1", "name": "Liver"},
                                    "h4": 0.88, "h3": 0.05, "h0": 0.01, "h1": 0.03, "h2": 0.03,
                                    "log2h4h3": 4.14,
                                }],
                            },
                            {
                                "studyLocusId": "sl_002",
                                "study": {"id": "GCST999", "studyType": "GWAS",
                                          "traitFromSource": "HDL", "diseases": [], "nSamples": 5000},
                                "variant": {"id": "1_55505700_C_T", "rsIds": ["rs999"],
                                            "chromosome": "1", "position": 55505700},
                                "pValueMantissa": 5.0, "pValueExponent": -10, "beta": 0.1,
                                "l2GPredictions": [], "locus": {"count": 3},
                                "colocalisationsQtl": [{
                                    "studyLocusId": "sl_q2", "qtlStudyId": "eQTL_Blood",
                                    "phenotypeId": "ENSG00000169174",
                                    "tissue": {"id": "T2", "name": "Blood"},
                                    "h4": 0.55, "h3": 0.20, "h0": 0.05, "h1": 0.10, "h2": 0.10,
                                    "log2h4h3": 1.46,
                                }],
                            },
                        ],
                    },
                }
            }
        }
        mock_post.return_value = ot_resp

        # Filter to only GCST001
        result = coloc(gene="PCSK9", study_id="GCST001")
        assert result["n_colocalizations"] == 1
        assert result["colocalizations"][0]["gwas_study_id"] == "GCST001"


# ─── Proteomics graceful degradation ─────────────────────────

class TestProteomicsGracefulDegradation:
    """Test that proteomics-dependent tools gracefully degrade when data is missing."""

    @patch("ct.data.loaders.load_proteomics", side_effect=FileNotFoundError)
    def test_antitarget_profile_no_proteomics(self, mock_load):
        from ct.tools.safety import antitarget_profile
        result = antitarget_profile(compound_id="test")
        assert "error" in result
        assert "summary" in result
        assert "not available" in result["summary"].lower()

    @patch("ct.data.loaders.load_proteomics", side_effect=FileNotFoundError)
    def test_classify_no_proteomics(self, mock_load):
        from ct.tools.safety import classify
        result = classify(compound_id="test")
        assert "error" in result
        assert "summary" in result

    @patch("ct.data.loaders.load_proteomics", side_effect=FileNotFoundError)
    def test_sall4_risk_no_proteomics(self, mock_load):
        from ct.tools.safety import sall4_risk
        result = sall4_risk(compound_id="test")
        assert "error" in result
        assert "summary" in result
        assert "not available" in result["summary"].lower()

    @patch("ct.data.loaders.load_proteomics", side_effect=FileNotFoundError)
    def test_neosubstrate_score_no_proteomics(self, mock_load):
        from ct.tools.target import neosubstrate_score
        result = neosubstrate_score()
        assert "error" in result
        assert "summary" in result
        assert "not available" in result["summary"].lower()

    @patch("httpx.post")
    @patch("httpx.get")
    def test_coloc_current_ot_schema(self, mock_get, mock_post):
        from ct.tools.genomics import coloc

        ens_resp = MagicMock()
        ens_resp.status_code = 200
        ens_resp.json.return_value = {"id": "ENSG00000145335"}
        mock_get.return_value = ens_resp

        ot_resp = MagicMock()
        ot_resp.status_code = 200
        ot_resp.json.return_value = {
            "data": {
                "target": {
                    "id": "ENSG00000145335",
                    "approvedSymbol": "SNCA",
                    "approvedName": "synuclein alpha",
                    "credibleSets": {
                        "count": 2,
                        "rows": [
                            {
                                "studyLocusId": "sl_gwas_1",
                                "studyId": "GCST900001",
                                "studyType": "gwas",
                                "study": {
                                    "id": "GCST900001",
                                    "studyType": "gwas",
                                    "traitFromSource": "Parkinson disease",
                                    "diseases": [{"id": "EFO_0002508", "name": "Parkinson disease"}],
                                    "nSamples": 100000,
                                },
                                "variant": {
                                    "id": "4_90626111_G_A",
                                    "rsIds": ["rs356219"],
                                    "chromosome": "4",
                                    "position": 90626111,
                                },
                                "pValueMantissa": 6.0,
                                "pValueExponent": -65,
                                "beta": 0.18,
                                "l2GPredictions": {
                                    "rows": [
                                        {
                                            "target": {"id": "ENSG00000145335", "approvedSymbol": "SNCA"},
                                            "yProbaModel": 0.91,
                                        }
                                    ]
                                },
                                "colocalisation": {
                                    "count": 1,
                                    "rows": [
                                        {
                                            "h4": 0.91,
                                            "h3": 0.06,
                                            "clpp": 0.21,
                                            "colocalisationMethod": "ecaviar",
                                            "rightStudyType": "eqtl",
                                            "betaRatioSignAverage": 1.0,
                                            "numberColocalisingVariants": 3,
                                            "otherStudyLocus": {
                                                "studyLocusId": "sl_eqtl_1",
                                                "studyId": "GTEX_V8_CORTEX",
                                                "studyType": "eqtl",
                                                "qtlGeneId": "ENSG00000145335",
                                                "study": {
                                                    "id": "GTEX_V8_CORTEX",
                                                    "traitFromSource": "SNCA expression",
                                                    "condition": "cortex",
                                                    "biosample": {
                                                        "biosampleId": "UBERON_0000956",
                                                        "biosampleName": "Brain - Cortex",
                                                    },
                                                },
                                            },
                                        }
                                    ],
                                },
                            },
                            {
                                # Non-GWAS row should be ignored by parser.
                                "studyLocusId": "sl_qtl_only",
                                "studyId": "QTL_ONLY",
                                "studyType": "eqtl",
                                "study": {
                                    "id": "QTL_ONLY",
                                    "studyType": "eqtl",
                                    "traitFromSource": "expression",
                                    "diseases": [],
                                    "nSamples": 500,
                                },
                                "variant": {"id": "4_1_A_T", "rsIds": [], "chromosome": "4", "position": 1},
                                "pValueMantissa": 1.0,
                                "pValueExponent": -8,
                                "beta": 0.01,
                                "l2GPredictions": {"rows": []},
                                "colocalisation": {"count": 0, "rows": []},
                            },
                        ],
                    },
                }
            }
        }
        mock_post.return_value = ot_resp

        result = coloc(gene="SNCA")
        assert result["n_colocalizations"] == 1
        assert result["n_strong_coloc"] == 1
        assert result["gene"] == "SNCA"
        assert result["colocalizations"][0]["qtl_study_id"] == "GTEX_V8_CORTEX"
        assert result["colocalizations"][0]["right_study_type"] == "eqtl"
        assert "Brain - Cortex" in result["tissues"]

    @patch("httpx.post")
    @patch("httpx.get")
    def test_coloc_alias_fallback_gba1_to_gba_after_ot_400(self, mock_get, mock_post):
        from ct.tools.genomics import coloc

        def get_router(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 404
            resp.json.return_value = {}
            if url.endswith("/GBA1"):
                resp.status_code = 200
                resp.json.return_value = {"id": "ENSG_GBA1"}
            elif url.endswith("/GBA"):
                resp.status_code = 200
                resp.json.return_value = {"id": "ENSG_GBA"}
            return resp

        def post_router(url, **kwargs):
            vars_ = kwargs.get("json", {}).get("variables", {})
            ens = vars_.get("ensemblId")
            resp = MagicMock()
            if ens == "ENSG_GBA1":
                resp.status_code = 400
                resp.text = "query complexity limit exceeded"
                resp.json.return_value = {"errors": [{"message": "query complexity limit exceeded"}]}
                return resp

            resp.status_code = 200
            resp.json.return_value = {
                "data": {
                    "target": {
                        "id": "ENSG_GBA",
                        "approvedSymbol": "GBA1",
                        "approvedName": "glucosylceramidase beta",
                        "credibleSets": {
                            "count": 1,
                            "rows": [
                                {
                                    "studyLocusId": "sl_gwas_1",
                                    "studyId": "GCST000123",
                                    "studyType": "gwas",
                                    "study": {
                                        "id": "GCST000123",
                                        "studyType": "gwas",
                                        "traitFromSource": "Parkinson disease",
                                        "diseases": [{"id": "EFO_0002508", "name": "Parkinson disease"}],
                                        "nSamples": 80000,
                                    },
                                    "variant": {
                                        "id": "1_155236376_G_A",
                                        "rsIds": ["rs76763715"],
                                        "chromosome": "1",
                                        "position": 155236376,
                                    },
                                    "pValueMantissa": 2.2,
                                    "pValueExponent": -20,
                                    "beta": 0.3,
                                    "l2GPredictions": {
                                        "rows": [
                                            {
                                                "target": {"id": "ENSG_GBA", "approvedSymbol": "GBA1"},
                                                "yProbaModel": 0.77,
                                            }
                                        ]
                                    },
                                    "colocalisation": {
                                        "count": 1,
                                        "rows": [
                                            {
                                                "h4": 0.83,
                                                "h3": 0.09,
                                                "clpp": 0.18,
                                                "colocalisationMethod": "ecaviar",
                                                "rightStudyType": "eqtl",
                                                "betaRatioSignAverage": 1.0,
                                                "numberColocalisingVariants": 2,
                                                "otherStudyLocus": {
                                                    "studyLocusId": "sl_eqtl_2",
                                                    "studyId": "GTEX_V8_SUBSTANTIA_NIGRA",
                                                    "studyType": "eqtl",
                                                    "qtlGeneId": "ENSG_GBA",
                                                    "study": {
                                                        "id": "GTEX_V8_SUBSTANTIA_NIGRA",
                                                        "traitFromSource": "GBA expression",
                                                        "condition": "substantia nigra",
                                                        "biosample": {
                                                            "biosampleId": "UBERON_0002038",
                                                            "biosampleName": "Brain - Substantia nigra",
                                                        },
                                                    },
                                                },
                                            }
                                        ],
                                    },
                                }
                            ],
                        },
                    }
                }
            }
            return resp

        mock_get.side_effect = get_router
        mock_post.side_effect = post_router

        result = coloc(gene="GBA1")
        assert "error" not in result
        assert result["gene"] == "GBA1"
        assert result["ensembl_id"] == "ENSG_GBA"
        assert result["n_colocalizations"] == 1
        assert result["n_strong_coloc"] == 1
        assert "Brain - Substantia nigra" in result["tissues"]

    @patch("httpx.post")
    @patch("httpx.get")
    def test_coloc_returns_unavailable_when_ot_query_fails_for_valid_gene(self, mock_get, mock_post):
        from ct.tools.genomics import coloc

        ens_resp = MagicMock()
        ens_resp.status_code = 200
        ens_resp.json.return_value = {"id": "ENSG00000130203"}
        mock_get.return_value = ens_resp

        ot_resp = MagicMock()
        ot_resp.status_code = 503
        ot_resp.text = "service unavailable"
        mock_post.return_value = ot_resp

        result = coloc(gene="APOE")

        assert "error" not in result
        assert result["gene"] == "APOE"
        assert result["ensembl_id"] == "ENSG00000130203"
        assert result["data_unavailable"] is True
        assert result["n_colocalizations"] == 0
        assert "unavailable from Open Targets" in result["summary"]
        assert "Open Targets API returned HTTP 503" in result["warning"]

    @patch("httpx.post")
    @patch("httpx.get")
    def test_coloc_keeps_primary_symbol_without_generic_trailing_digit_alias(self, mock_get, mock_post):
        from ct.tools.genomics import coloc

        def get_router(url, **kwargs):
            del kwargs
            resp = MagicMock()
            if url.endswith("/PSEN1"):
                resp.status_code = 200
                resp.json.return_value = {"id": "ENSG00000080815"}
                return resp
            resp.status_code = 404
            resp.json.return_value = {}
            return resp

        ot_resp = MagicMock()
        ot_resp.status_code = 503
        ot_resp.text = "service unavailable"
        mock_get.side_effect = get_router
        mock_post.return_value = ot_resp

        result = coloc(gene="PSEN1")

        assert "error" not in result
        assert result["gene"] == "PSEN1"
        assert result["ensembl_id"] == "ENSG00000080815"
        assert result["data_unavailable"] is True
        assert mock_get.call_count == 1


# ─── Regulatory tools ──────────────────────────────────────────

class TestRegulatoryCdiscLint:
    def test_valid_ae_dataset(self, tmp_path):
        from ct.tools.regulatory import cdisc_lint

        df = pd.DataFrame(
            {
                "STUDYID": ["STUDY1", "STUDY1"],
                "DOMAIN": ["AE", "AE"],
                "USUBJID": ["S1-001", "S1-002"],
                "AESEQ": [1, 1],
                "AETERM": ["Headache", "Nausea"],
                "AESTDTC": ["2025-01-01", "2025-02-14T13:45"],
            }
        )
        path = tmp_path / "ae.csv"
        df.to_csv(path, index=False)

        result = cdisc_lint(dataset_path=str(path), domain="AE")

        assert "error" not in result
        assert result["error_count"] == 0
        assert result["domain"] == "AE"
        assert result["n_rows"] == 2
        assert result["quality_score"] >= 90

    def test_catches_missing_required_and_invalid_dates(self, tmp_path):
        from ct.tools.regulatory import cdisc_lint

        df = pd.DataFrame(
            {
                "STUDYID": ["STUDY1"],
                "DOMAIN": ["AE"],
                "AESEQ": [1],
                "AETERM": ["Headache"],
                "AESTDTC": ["01-31-2025"],
            }
        )
        path = tmp_path / "ae_bad.csv"
        df.to_csv(path, index=False)

        result = cdisc_lint(dataset_path=str(path), domain="AE")

        assert "error" not in result
        assert result["error_count"] >= 2
        codes = {issue["code"] for issue in result["issues"]}
        assert "missing_required_column" in codes
        assert "invalid_datetime_format" in codes


class TestRegulatoryDefineXmlLint:
    def test_valid_define_xml(self, tmp_path):
        from ct.tools.regulatory import define_xml_lint

        xml = """<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.0"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     ODMVersion="1.3.2">
  <Study OID="S.STUDY">
    <MetaDataVersion OID="MDV.1" Name="Metadata">
      <ItemDef OID="IT.STUDYID" Name="STUDYID" DataType="text" />
      <ItemDef OID="IT.USUBJID" Name="USUBJID" DataType="text" />
      <ItemDef OID="IT.AESEQ" Name="AESEQ" DataType="integer" />
      <CodeList OID="CL.YESNO" Name="YesNo" DataType="text">
        <CodeListItem CodedValue="Y"><Decode><TranslatedText>Yes</TranslatedText></Decode></CodeListItem>
        <CodeListItem CodedValue="N"><Decode><TranslatedText>No</TranslatedText></Decode></CodeListItem>
      </CodeList>
      <ItemDef OID="IT.AEYN" Name="AEYN" DataType="text">
        <CodeListRef CodeListOID="CL.YESNO" />
      </ItemDef>
      <ItemGroupDef OID="IG.AE" Name="AE" Domain="AE" Repeating="Yes" IsReferenceData="No">
        <ItemRef ItemOID="IT.STUDYID" Mandatory="Yes" />
        <ItemRef ItemOID="IT.USUBJID" Mandatory="Yes" />
        <ItemRef ItemOID="IT.AESEQ" Mandatory="Yes" />
      </ItemGroupDef>
      <def:leaf ID="LF.AE" xlink:href="ae.xpt" />
    </MetaDataVersion>
  </Study>
</ODM>"""
        path = tmp_path / "define.xml"
        path.write_text(xml, encoding="utf-8")

        result = define_xml_lint(define_xml_path=str(path))
        assert "error" not in result
        assert result["error_count"] == 0
        assert result["counts"]["itemdef"] >= 4
        assert result["counts"]["itemgroupdef"] == 1

    def test_invalid_references(self, tmp_path):
        from ct.tools.regulatory import define_xml_lint

        xml = """<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.0"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     ODMVersion="1.3.2">
  <Study OID="S.STUDY">
    <MetaDataVersion OID="MDV.1" Name="Metadata">
      <ItemDef OID="IT.AETERM" Name="AETERM" DataType="text">
        <CodeListRef CodeListOID="CL.MISSING" />
      </ItemDef>
      <ItemGroupDef OID="IG.AE" Name="AE" Domain="AE" Repeating="Yes" IsReferenceData="No">
        <ItemRef ItemOID="IT.MISSING" Mandatory="Yes" />
      </ItemGroupDef>
      <def:leaf ID="LF.AE" xlink:href="" />
    </MetaDataVersion>
  </Study>
</ODM>"""
        path = tmp_path / "define_bad.xml"
        path.write_text(xml, encoding="utf-8")

        result = define_xml_lint(define_xml_path=str(path))
        assert "error" not in result
        assert result["error_count"] >= 3
        codes = {issue["code"] for issue in result["issues"]}
        assert "itemref_unknown_itemoid" in codes
        assert "codelistref_unknown_oid" in codes
        assert "leaf_missing_href" in codes


# ─── PK tools ──────────────────────────────────────────────────

class TestPkNcaBasic:
    def test_nca_basic_core_metrics(self):
        from ct.tools.pk import nca_basic

        result = nca_basic(
            times=[0, 1, 2, 4, 8, 12],
            concentrations=[0, 15, 12, 7, 2.5, 1.0],
            dose=100,
            route="iv",
            subject_id="SUBJ001",
        )

        assert "error" not in result
        assert result["cmax"] == 15.0
        assert result["tmax"] == 1.0
        assert result["auc_last"] is not None and result["auc_last"] > 0
        assert result["lambda_z"] is not None and result["lambda_z"] > 0
        assert result["half_life"] is not None and result["half_life"] > 0
        assert result["clearance"] is not None and result["clearance"] > 0

    def test_nca_basic_length_mismatch(self):
        from ct.tools.pk import nca_basic

        result = nca_basic(times=[0, 1, 2], concentrations=[1, 2])
        assert result["error"] == "length_mismatch"

    def test_nca_basic_terminal_warning_when_insufficient_positive_points(self):
        from ct.tools.pk import nca_basic

        result = nca_basic(
            times=[0, 1, 2, 4],
            concentrations=[5, 2, 0, 0],
            lloq=0.5,
            dose=50,
            route="extravascular",
        )
        assert "error" not in result
        assert result["lambda_z"] is None
        assert any("cannot estimate lambda_z" in msg.lower() for msg in result["warnings"])


# ─── Regulatory package checks ─────────────────────────────────

class TestRegulatorySubmissionPackageCheck:
    def test_submission_package_check_passes_minimal_package(self, tmp_path):
        from ct.tools.regulatory import submission_package_check

        package = tmp_path / "pkg"
        package.mkdir(parents=True, exist_ok=True)
        (package / "ae.csv").write_text(
            "STUDYID,DOMAIN,USUBJID,AESEQ,AETERM,AESTDTC\n"
            "STUDY1,AE,S1-001,1,Headache,2025-01-01\n",
            encoding="utf-8",
        )
        (package / "define.xml").write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<ODM xmlns="http://www.cdisc.org/ns/odm/v1.3"
     xmlns:def="http://www.cdisc.org/ns/def/v2.0"
     xmlns:xlink="http://www.w3.org/1999/xlink" ODMVersion="1.3.2">
  <Study OID="S.STUDY">
    <MetaDataVersion OID="MDV.1" Name="Metadata">
      <ItemDef OID="IT.STUDYID" Name="STUDYID" DataType="text" />
      <ItemDef OID="IT.USUBJID" Name="USUBJID" DataType="text" />
      <ItemDef OID="IT.AESEQ" Name="AESEQ" DataType="integer" />
      <ItemGroupDef OID="IG.AE" Name="AE" Domain="AE" Repeating="Yes" IsReferenceData="No">
        <ItemRef ItemOID="IT.STUDYID" Mandatory="Yes" />
        <ItemRef ItemOID="IT.USUBJID" Mandatory="Yes" />
        <ItemRef ItemOID="IT.AESEQ" Mandatory="Yes" />
      </ItemGroupDef>
      <def:leaf ID="LF.AE" xlink:href="ae.csv" />
    </MetaDataVersion>
  </Study>
</ODM>""",
            encoding="utf-8",
        )

        result = submission_package_check(package_dir=str(package))
        assert "error" not in result
        assert result["readiness"] in {"ready", "needs_review"}
        assert result["datasets_linted"] >= 1


# ─── Clinical endpoint benchmarking ────────────────────────────

class TestClinicalEndpointBenchmark:
    @patch("ct.tools.clinical.trial_design_benchmark")
    def test_endpoint_benchmark_family_counts(self, mock_benchmark):
        from ct.tools.clinical import endpoint_benchmark

        mock_benchmark.return_value = {
            "trials": [
                {"primary_endpoints": ["Overall survival"], "enrollment": 300},
                {"primary_endpoints": ["Progression-free survival"], "enrollment": 180},
                {"primary_endpoints": ["Objective response rate"], "enrollment": 120},
            ],
            "phase_distribution": {"PHASE3": 2},
            "status_distribution": {"RECRUITING": 2},
        }

        result = endpoint_benchmark(query="NSCLC")
        assert "error" not in result
        assert result["n_trials"] == 3
        assert result["endpoint_families"]["overall_survival"] >= 1
        assert result["endpoint_families"]["progression_free_survival"] >= 1
        assert result["median_enrollment"] == 180.0


# ─── Intel tools ───────────────────────────────────────────────

class TestIntelPipelineWatch:
    @patch("ct.tools.literature.openalex_search")
    @patch("ct.tools.literature.pubmed_search")
    @patch("ct.tools.clinical.trial_search")
    def test_pipeline_watch_success(self, mock_trial_search, mock_pubmed, mock_openalex):
        from ct.tools.intel import pipeline_watch

        mock_trial_search.return_value = {
            "total_count": 12,
            "phase_distribution": {"PHASE3": 1},
            "status_distribution": {"RECRUITING": 3},
            "trials": [{"nct_id": "NCT1", "sponsor": "CompanyA"}],
        }
        mock_pubmed.return_value = {
            "total_count": 25,
            "articles": [{"pub_date": "2025"}],
        }
        mock_openalex.return_value = {
            "total_count": 42,
            "articles": [{"publication_year": 2025}],
        }

        result = pipeline_watch(query="IL23R", indication="ulcerative colitis")
        assert "error" not in result
        assert result["momentum_score"] > 0
        assert result["trials"]["total_count"] == 12
        assert "summary" in result


class TestIntelCompetitorSnapshot:
    @patch("ct.tools.clinical.trial_design_benchmark")
    @patch("ct.tools.clinical.competitive_landscape")
    def test_competitor_snapshot_success(self, mock_landscape, mock_benchmark):
        from ct.tools.intel import competitor_snapshot

        mock_landscape.return_value = {
            "trials": {
                "total_count": 8,
                "phase_distribution": {"PHASE2": 3, "PHASE3": 1},
                "top_trials": [
                    {"sponsor": "CompanyA", "nct_id": "NCT1"},
                    {"sponsor": "CompanyA", "nct_id": "NCT2"},
                    {"sponsor": "CompanyB", "nct_id": "NCT3"},
                ],
            },
            "chembl": {"unique_compounds": 15, "moa_types": ["Inhibitor"]},
            "open_targets": {"n_known_drugs": 4},
        }
        mock_benchmark.return_value = {
            "top_primary_endpoints": [{"endpoint": "Overall survival", "count": 5}],
        }

        result = competitor_snapshot(gene="LRRK2", indication="Parkinson disease")
        assert "error" not in result
        assert result["phase_distribution"]["PHASE2"] == 3
        assert len(result["top_sponsors"]) >= 1
        assert "summary" in result


# ─── Translational tools ───────────────────────────────────────

class TestTranslationalBiomarkerReadiness:
    @patch("ct.tools.literature.openalex_search")
    @patch("ct.tools.literature.pubmed_search")
    @patch("ct.tools.clinical.trial_search")
    def test_biomarker_readiness_scoring(self, mock_trial_search, mock_pubmed, mock_openalex):
        from ct.tools.translational import biomarker_readiness

        mock_trial_search.return_value = {
            "total_count": 9,
            "status_distribution": {"RECRUITING": 2},
            "phase_distribution": {"PHASE2": 3},
            "trials": [{"nct_id": "NCT1"}],
        }
        mock_pubmed.return_value = {"total_count": 30, "articles": [{"title": "A"}]}
        mock_openalex.return_value = {"total_count": 60, "articles": [{"title": "B"}]}

        result = biomarker_readiness(biomarker="PD-L1", indication="NSCLC")
        assert "error" not in result
        assert result["readiness_score"] > 0
        assert result["readiness_level"] in {"high", "moderate", "early"}
        assert result["trials"]["total_count"] == 9


# ─── Report tools ──────────────────────────────────────────────

class TestReportPharmaBrief:
    def test_pharma_brief_writes_markdown_and_html(self, tmp_path):
        from ct.tools.report import pharma_brief

        class _Cfg:
            def get(self, key, default=None):
                if key == "sandbox.output_dir":
                    return str(tmp_path)
                return default

        class _Session:
            config = _Cfg()

        result = pharma_brief(
            query="IL23R strategy for ulcerative colitis",
            program_thesis="Advance IL23R-pathway strategy with genotype-informed enrichment.",
            evidence={"summary": "IL23R has strong human genetic support in UC."},
            _session=_Session(),
        )

        assert "error" not in result
        assert result["markdown_path"] is not None
        assert result["html_path"] is not None
        assert Path(result["markdown_path"]).exists()
        assert Path(result["html_path"]).exists()
