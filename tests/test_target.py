"""Tests for target discovery tools: neosubstrate_score, degron_predict, coessentiality."""

import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


class TestNeosubstrateScore:
    def _make_proteomics(self, n_proteins=100, n_compounds=20):
        np.random.seed(42)
        proteins = [f"PROT_{i}" for i in range(n_proteins)]
        compounds = [f"COMP_{i}" for i in range(n_compounds)]
        data = np.random.randn(n_proteins, n_compounds) * 0.3
        # Make a few proteins strongly degraded by few compounds (selective)
        data[0, :3] = -1.5  # PROT_0: degraded by 3 compounds
        data[1, :1] = -2.0  # PROT_1: strongly degraded by 1 compound
        return pd.DataFrame(data, index=proteins, columns=compounds)

    @patch("ct.data.loaders.load_proteomics")
    def test_returns_top_targets(self, mock_load):
        from ct.tools.target import neosubstrate_score
        mock_load.return_value = self._make_proteomics()

        result = neosubstrate_score(top_n=10)

        assert "summary" in result
        assert "top_targets" in result
        assert len(result["top_targets"]) <= 10
        assert result["n_proteins_scored"] > 0

    @patch("ct.data.loaders.load_proteomics")
    def test_selective_degradation_scores_high(self, mock_load):
        from ct.tools.target import neosubstrate_score
        mock_load.return_value = self._make_proteomics()

        result = neosubstrate_score(top_n=5)

        # PROT_0 and PROT_1 should appear in top targets
        top_names = [t["protein"] for t in result["top_targets"]]
        assert "PROT_0" in top_names or "PROT_1" in top_names

    @patch("ct.data.loaders.load_proteomics")
    def test_scoring_fields_present(self, mock_load):
        from ct.tools.target import neosubstrate_score
        mock_load.return_value = self._make_proteomics()

        result = neosubstrate_score(top_n=5)

        for target in result["top_targets"]:
            assert "protein" in target
            assert "score" in target
            assert "n_degraders" in target
            assert "mean_degradation" in target
            assert "selectivity" in target
            assert target["score"] > 0

    @patch("ct.data.loaders.load_proteomics")
    def test_no_degradation(self, mock_load):
        from ct.tools.target import neosubstrate_score
        # All values above -0.5 threshold → no degradation detected
        data = pd.DataFrame(
            np.ones((10, 5)) * 0.1,
            index=[f"P{i}" for i in range(10)],
            columns=[f"C{i}" for i in range(5)],
        )
        mock_load.return_value = data

        result = neosubstrate_score()

        assert result["n_proteins_scored"] == 0
        assert result["top_targets"] == []

    def test_custom_proteomics_path(self, tmp_path):
        from ct.tools.target import neosubstrate_score
        csv_path = tmp_path / "prot.csv"
        data = pd.DataFrame(
            [[-1.5, 0.1, 0.2], [-0.1, -0.2, 0.0]],
            index=["TARGET1", "TARGET2"],
            columns=["C1", "C2", "C3"],
        )
        data.to_csv(csv_path)

        result = neosubstrate_score(proteomics_path=str(csv_path), top_n=5)

        assert result["n_proteins_scored"] >= 1
        top_names = [t["protein"] for t in result["top_targets"]]
        assert "TARGET1" in top_names


class TestDegronPredict:
    """Test degron_predict with mocked UniProt API responses."""

    def _mock_uniprot_response(self, zinc_fingers=0, disordered=0,
                                seq_length=400, lysines=30, ub_sites=0,
                                gene="TEST", protein="Test protein"):
        """Build a mock UniProt JSON response."""
        features = []

        for i in range(zinc_fingers):
            features.append({
                "type": "Zinc finger",
                "description": f"C2H2-type {i+1}",
                "location": {"start": {"value": 100 + i*30}, "end": {"value": 120 + i*30}},
            })

        if disordered > 0:
            features.append({
                "type": "Region",
                "description": "Disordered",
                "location": {"start": {"value": 1}, "end": {"value": disordered}},
            })

        for i in range(ub_sites):
            features.append({
                "type": "Modified residue",
                "description": f"Ubiquitin conjugation site K{50+i}",
                "location": {"start": {"value": 50 + i}, "end": {"value": 50 + i}},
            })

        # Build a sequence with the specified number of lysines
        seq = "A" * (seq_length - lysines) + "K" * lysines

        return {
            "proteinDescription": {"recommendedName": {"fullName": {"value": protein}}},
            "genes": [{"geneName": {"value": gene}}],
            "sequence": {"length": seq_length, "value": seq},
            "features": features,
        }

    @patch("httpx.get")
    def test_high_degradability_zinc_fingers(self, mock_get):
        from ct.tools.target import degron_predict

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = self._mock_uniprot_response(
            zinc_fingers=3, disordered=100, seq_length=500, lysines=50,
            gene="IKZF1", protein="Ikaros",
        )
        mock_get.return_value = resp

        result = degron_predict("Q13422")

        assert result["classification"] in ("high", "moderate")
        assert result["degradability_score"] > 0.3
        assert result["features"]["zinc_fingers"] == 3
        assert result["gene"] == "IKZF1"
        assert "summary" in result

    @patch("httpx.get")
    def test_low_degradability_no_features(self, mock_get):
        from ct.tools.target import degron_predict

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = self._mock_uniprot_response(
            zinc_fingers=0, disordered=0, seq_length=2000, lysines=10,
            gene="TITIN", protein="Titin",
        )
        mock_get.return_value = resp

        result = degron_predict("Q8WZ42")

        assert result["classification"] == "low"
        assert result["degradability_score"] < 0.25
        assert result["features"]["zinc_fingers"] == 0

    @patch("httpx.get")
    def test_score_breakdown_present(self, mock_get):
        from ct.tools.target import degron_predict

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = self._mock_uniprot_response(
            zinc_fingers=1, disordered=50, seq_length=400, lysines=30,
        )
        mock_get.return_value = resp

        result = degron_predict("P12345")

        assert "score_breakdown" in result
        breakdown = result["score_breakdown"]
        assert "zinc_fingers" in breakdown
        assert "disorder" in breakdown
        assert "lysine_accessibility" in breakdown
        assert "known_ub_sites" in breakdown
        assert "protein_size" in breakdown

    @patch("httpx.get")
    def test_uniprot_not_found(self, mock_get):
        from ct.tools.target import degron_predict

        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp

        result = degron_predict("INVALID")

        assert "error" in result

    @patch("httpx.get")
    def test_network_error(self, mock_get):
        import httpx as httpx_mod
        from ct.tools.target import degron_predict

        mock_get.side_effect = httpx_mod.ConnectError("Connection refused")

        result = degron_predict("P04637")

        assert "error" in result

    @patch("httpx.get")
    def test_ubiquitination_sites_boost_score(self, mock_get):
        from ct.tools.target import degron_predict

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = self._mock_uniprot_response(
            zinc_fingers=0, disordered=50, seq_length=400, lysines=40,
            ub_sites=5,
        )
        mock_get.return_value = resp

        result = degron_predict("P12345")

        assert result["features"]["known_ub_sites"] == 5
        assert result["score_breakdown"]["known_ub_sites"] > 0

    @patch("httpx.get")
    def test_small_protein_size_bonus(self, mock_get):
        from ct.tools.target import degron_predict

        # Small protein
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = self._mock_uniprot_response(seq_length=300, lysines=20)
        mock_get.return_value = resp1
        result_small = degron_predict("P_SMALL")

        # Large protein (same features otherwise)
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = self._mock_uniprot_response(seq_length=3000, lysines=20)
        mock_get.return_value = resp2
        result_large = degron_predict("P_LARGE")

        assert result_small["score_breakdown"]["protein_size"] > result_large["score_breakdown"]["protein_size"]


class TestCoessentiality:
    def _make_crispr(self, n_lines=80, n_genes=50):
        np.random.seed(42)
        lines = [f"LINE_{i}" for i in range(n_lines)]
        genes = [f"GENE_{i}" for i in range(n_genes)]
        data = np.random.randn(n_lines, n_genes)
        # Make GENE_0 and GENE_1 perfectly correlated (co-essential)
        data[:, 1] = data[:, 0] + np.random.randn(n_lines) * 0.01
        # Make GENE_0 and GENE_2 anti-correlated (synthetic lethal)
        data[:, 2] = -data[:, 0] + np.random.randn(n_lines) * 0.01
        df = pd.DataFrame(data, index=lines, columns=genes)
        return df

    @patch("ct.data.loaders.load_crispr")
    def test_finds_coessential_partners(self, mock_load):
        from ct.tools.target import coessentiality
        mock_load.return_value = self._make_crispr()

        result = coessentiality("GENE_0", top_n=5)

        assert "summary" in result
        assert "co_essential" in result
        assert "synthetic_lethal" in result
        assert len(result["co_essential"]) <= 5

        # GENE_1 should be top co-essential partner
        co_genes = [p["gene"] for p in result["co_essential"]]
        assert "GENE_1" in co_genes

    @patch("ct.data.loaders.load_crispr")
    def test_finds_synthetic_lethal_partners(self, mock_load):
        from ct.tools.target import coessentiality
        mock_load.return_value = self._make_crispr()

        result = coessentiality("GENE_0", top_n=5)

        # GENE_2 should be top synthetic lethal partner
        sl_genes = [p["gene"] for p in result["synthetic_lethal"]]
        assert "GENE_2" in sl_genes

    @patch("ct.data.loaders.load_crispr")
    def test_gene_not_found(self, mock_load):
        from ct.tools.target import coessentiality
        mock_load.return_value = self._make_crispr()

        result = coessentiality("NONEXISTENT_GENE")

        assert "error" in result

    @patch("ct.data.loaders.load_crispr")
    def test_correlation_fields(self, mock_load):
        from ct.tools.target import coessentiality
        mock_load.return_value = self._make_crispr()

        result = coessentiality("GENE_0", top_n=3)

        for partner in result["co_essential"]:
            assert "gene" in partner
            assert "r" in partner
            assert "p" in partner
            assert partner["r"] > 0  # co-essential = positive correlation

        for partner in result["synthetic_lethal"]:
            assert partner["r"] < 0  # synthetic lethal = negative correlation


class TestDiseaseAssociation:
    @patch("httpx.post")
    @patch("httpx.get")
    def test_parses_datasource_scores_with_schema_id_field(self, mock_get, mock_post):
        from ct.tools.target import disease_association

        ens_resp = MagicMock()
        ens_resp.status_code = 200
        ens_resp.json.return_value = {"id": "ENSG00000111540"}
        mock_get.return_value = ens_resp

        ot_resp = MagicMock()
        ot_resp.status_code = 200
        ot_resp.json.return_value = {
            "data": {
                "target": {
                    "approvedSymbol": "RAB5A",
                    "approvedName": "RAB5A, member RAS oncogene family",
                    "associatedDiseases": {
                        "count": 2,
                        "rows": [
                            {
                                "disease": {"id": "EFO_0002508", "name": "Parkinson disease"},
                                "score": 0.71,
                                "datasourceScores": [
                                    {"id": "ot_genetics_portal", "score": 0.82},
                                    {"id": "chembl", "score": 0.22},
                                ],
                            },
                            {
                                "disease": {"id": "EFO_0001379", "name": "Alzheimer disease"},
                                "score": 0.09,
                                "datasourceScores": [{"id": "ot_genetics_portal", "score": 0.1}],
                            },
                        ],
                    },
                }
            }
        }
        mock_post.return_value = ot_resp

        result = disease_association(gene="RAB5A", min_score=0.1)
        assert "error" not in result
        assert result["gene"] == "RAB5A"
        assert result["total_associations"] == 2
        assert result["filtered_associations"] == 1
        assert result["associations"][0]["disease_name"] == "Parkinson disease"
        assert result["associations"][0]["genetic_association"] == 0.82

    @patch("httpx.post")
    @patch("httpx.get")
    def test_handles_no_associations_above_threshold(self, mock_get, mock_post):
        from ct.tools.target import disease_association

        ens_resp = MagicMock()
        ens_resp.status_code = 200
        ens_resp.json.return_value = {"id": "ENSG00000141510"}
        mock_get.return_value = ens_resp

        ot_resp = MagicMock()
        ot_resp.status_code = 200
        ot_resp.json.return_value = {
            "data": {
                "target": {
                    "approvedSymbol": "TP53",
                    "approvedName": "tumor protein p53",
                    "associatedDiseases": {
                        "count": 1,
                        "rows": [
                            {
                                "disease": {"id": "EFO_0000001", "name": "example disease"},
                                "score": 0.05,
                                "datasourceScores": [{"id": "ot_genetics_portal", "score": 0.05}],
                            }
                        ],
                    },
                }
            }
        }
        mock_post.return_value = ot_resp

        result = disease_association(gene="TP53", min_score=0.2)
        assert result["filtered_associations"] == 0
        assert "Top: none." in result["summary"]
