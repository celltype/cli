"""Tests for chemistry.retrosynthesis and chemistry.pharmacophore tools."""

import pytest
from unittest.mock import patch, MagicMock


# ─── Retrosynthesis (heuristic) ──────────────────────────────────

class TestRetrosynthesisHeuristic:
    """Test heuristic retrosynthesis with RDKit (no API key)."""

    def test_amide_disconnection(self):
        """Aspirin-like amide should find amide bond disconnection."""
        from ct.tools.chemistry import retrosynthesis

        # Acetaminophen (paracetamol) — has an amide bond
        result = retrosynthesis(smiles="CC(=O)Nc1ccc(O)cc1", max_steps=3)

        assert "summary" in result
        assert result["source"] == "heuristic"
        assert result["n_routes"] > 0
        # Should find the amide disconnection
        transform_names = [d["transform_name"] for d in result.get("disconnections", [])]
        assert "Amide bond disconnection" in transform_names

    def test_ester_disconnection(self):
        """Ethyl acetate — has an ester bond."""
        from ct.tools.chemistry import retrosynthesis

        result = retrosynthesis(smiles="CCOC(C)=O", max_steps=3)

        assert "summary" in result
        assert result["n_routes"] > 0
        transform_names = [d["transform_name"] for d in result.get("disconnections", [])]
        assert "Ester hydrolysis" in transform_names

    def test_biaryl_suzuki(self):
        """Biphenyl — should find Suzuki coupling disconnection."""
        from ct.tools.chemistry import retrosynthesis

        result = retrosynthesis(smiles="c1ccc(-c2ccccc2)cc1", max_steps=3)

        assert "summary" in result
        assert result["n_routes"] > 0
        transform_names = [d["transform_name"] for d in result.get("disconnections", [])]
        assert "Suzuki coupling" in transform_names

    def test_invalid_smiles(self):
        """Invalid SMILES should return error."""
        from ct.tools.chemistry import retrosynthesis

        result = retrosynthesis(smiles="INVALID_SMILES_XYZ")

        assert "error" in result

    def test_simple_alkane_no_disconnections(self):
        """Methane — no disconnections possible."""
        from ct.tools.chemistry import retrosynthesis

        result = retrosynthesis(smiles="C", max_steps=3)

        assert "summary" in result
        assert result["n_routes"] == 0

    def test_complex_molecule_multiple_disconnections(self):
        """Lenalidomide — complex molecule with multiple disconnectable bonds."""
        from ct.tools.chemistry import retrosynthesis

        # Lenalidomide SMILES
        result = retrosynthesis(
            smiles="O=C1CCC(N2C(=O)c3cccc(N)c3C2=O)C(=O)N1",
            max_steps=3,
        )

        assert "summary" in result
        assert result["n_routes"] >= 1
        assert "formula" in result
        assert "molecular_weight" in result

    def test_brics_fragments_present(self):
        """BRICS decomposition should appear for decomposable molecules."""
        from ct.tools.chemistry import retrosynthesis

        # A molecule with BRICS-decomposable bonds
        result = retrosynthesis(smiles="CC(=O)Nc1ccc(O)cc1", max_steps=3)

        assert "brics_fragments" in result

    def test_routes_sorted_by_steps(self):
        """Routes should be sorted by number of steps (shortest first)."""
        from ct.tools.chemistry import retrosynthesis

        result = retrosynthesis(smiles="CC(=O)Nc1ccc(O)cc1", max_steps=3)

        if result["n_routes"] >= 2:
            for i in range(len(result["routes"]) - 1):
                assert result["routes"][i]["n_steps"] <= result["routes"][i + 1]["n_steps"]


class TestRetrosynthesisIBMRXN:
    """Test IBM RXN API path (mocked)."""

    @patch("httpx.get")
    @patch("httpx.post")
    def test_ibm_rxn_success(self, mock_post, mock_get):
        from ct.tools.chemistry import retrosynthesis

        # Mock POST: submit prediction
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"prediction_id": "test-123"}
        mock_post.return_value = post_resp

        # Mock GET: poll results (SUCCESS)
        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = {
            "status": "SUCCESS",
            "retrosynthetic_paths": [
                {
                    "confidence": 0.85,
                    "steps": [
                        {
                            "reaction": "CC(=O)Cl.Nc1ccccc1>>CC(=O)Nc1ccccc1",
                            "reactants": ["CC(=O)Cl", "Nc1ccccc1"],
                            "confidence": 0.9,
                        },
                    ],
                },
            ],
        }
        mock_get.return_value = get_resp

        # Provide a mock session with API key
        session = MagicMock()
        session.config.get.return_value = "fake-api-key"

        result = retrosynthesis(
            smiles="CC(=O)Nc1ccccc1",
            max_steps=3,
            _session=session,
        )

        assert result["source"] == "ibm_rxn"
        assert result["n_routes"] == 1
        assert "summary" in result

    @patch("httpx.post")
    def test_ibm_rxn_failure_falls_back(self, mock_post):
        """If IBM RXN fails, should fall back to heuristic."""
        from ct.tools.chemistry import retrosynthesis

        mock_post.side_effect = Exception("Connection refused")

        session = MagicMock()
        session.config.get.return_value = "fake-api-key"

        result = retrosynthesis(
            smiles="CC(=O)Nc1ccccc1",
            max_steps=3,
            _session=session,
        )

        # Should fall back to heuristic
        assert result["source"] == "heuristic"

    def test_no_api_key_uses_heuristic(self):
        """Without API key, should use heuristic directly."""
        from ct.tools.chemistry import retrosynthesis

        result = retrosynthesis(smiles="CC(=O)Nc1ccccc1", max_steps=3)

        assert result["source"] == "heuristic"


# ─── Pharmacophore ───────────────────────────────────────────────

class TestPharmacophore:
    """Test pharmacophore model generation."""

    def test_basic_common_features(self):
        """Test common feature detection across a simple set."""
        from ct.tools.chemistry import pharmacophore

        # Three compounds with HBA and aromatic features
        smiles_list = [
            "c1ccccc1O",      # phenol: aromatic + HBD/HBA (OH)
            "c1ccccc1N",      # aniline: aromatic + HBD (NH2)
            "c1ccc(O)cc1O",   # catechol: aromatic + HBD/HBA (2x OH)
        ]

        result = pharmacophore(smiles_list=smiles_list)

        assert "summary" in result
        assert result["n_compounds"] == 3
        assert "common_features" in result
        assert "feature_distribution" in result
        assert "consensus_score" in result

        # Aromatic should be common to all
        common_types = [f["type"] for f in result["common_features"]]
        assert "Aromatic" in common_types

    def test_feature_distribution(self):
        """Feature distribution should have entries for all feature types."""
        from ct.tools.chemistry import pharmacophore

        smiles_list = [
            "CC(=O)Nc1ccc(O)cc1",  # acetaminophen
            "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        ]

        result = pharmacophore(smiles_list=smiles_list)

        assert "feature_distribution" in result
        dist = result["feature_distribution"]
        for feat_type in ["HBD", "HBA", "Aromatic", "Hydrophobic", "PosIonizable", "NegIonizable"]:
            assert feat_type in dist
            assert "frequency" in dist[feat_type]
            assert "mean_count" in dist[feat_type]
            assert "present_in_n" in dist[feat_type]

    def test_per_molecule_features(self):
        """Per-molecule features should be present for each compound."""
        from ct.tools.chemistry import pharmacophore

        smiles_list = ["c1ccccc1", "c1ccncc1", "c1ccoc1"]

        result = pharmacophore(smiles_list=smiles_list)

        assert "per_molecule_features" in result
        assert len(result["per_molecule_features"]) == 3
        for entry in result["per_molecule_features"]:
            assert "smiles" in entry
            assert "features" in entry

    def test_consensus_score_range(self):
        """Consensus score should be between 0 and 1."""
        from ct.tools.chemistry import pharmacophore

        smiles_list = [
            "c1ccccc1O",
            "c1ccccc1N",
        ]

        result = pharmacophore(smiles_list=smiles_list)

        assert 0.0 <= result["consensus_score"] <= 1.0

    def test_too_few_compounds(self):
        """Should error with fewer than 2 compounds."""
        from ct.tools.chemistry import pharmacophore

        result = pharmacophore(smiles_list=["c1ccccc1"])
        assert "error" in result

    def test_empty_list(self):
        """Should error with empty list."""
        from ct.tools.chemistry import pharmacophore

        result = pharmacophore(smiles_list=[])
        assert "error" in result

    def test_none_list(self):
        """Should error with None."""
        from ct.tools.chemistry import pharmacophore

        result = pharmacophore(smiles_list=None)
        assert "error" in result

    def test_invalid_smiles_in_list(self):
        """Invalid SMILES should be reported but not crash."""
        from ct.tools.chemistry import pharmacophore

        smiles_list = [
            "c1ccccc1O",
            "INVALID_SMILES",
            "c1ccccc1N",
        ]

        result = pharmacophore(smiles_list=smiles_list)

        # Should succeed with 2 valid molecules
        assert "error" not in result
        assert result["n_compounds"] == 2
        assert "invalid_smiles" in result
        assert "INVALID_SMILES" in result["invalid_smiles"]

    def test_all_invalid_smiles(self):
        """All invalid SMILES should error."""
        from ct.tools.chemistry import pharmacophore

        result = pharmacophore(smiles_list=["INVALID1", "INVALID2", "INVALID3"])
        assert "error" in result

    def test_identical_compounds_high_consensus(self):
        """Identical compounds should give perfect consensus."""
        from ct.tools.chemistry import pharmacophore

        smiles_list = [
            "c1ccccc1O",
            "c1ccccc1O",
            "c1ccccc1O",
        ]

        result = pharmacophore(smiles_list=smiles_list)

        # All features should be identical, so consensus should be high
        assert result["consensus_score"] >= 0.5

    def test_diverse_compounds_lower_consensus(self):
        """Very different compounds should have lower consensus."""
        from ct.tools.chemistry import pharmacophore

        smiles_list = [
            "CCCCCCCCCC",            # decane: just hydrophobic
            "O=C(O)c1ccccc1O",       # salicylic acid: aromatic, HBD, HBA, NegIonizable
        ]

        result = pharmacophore(smiles_list=smiles_list)

        # Different compounds, fewer common features
        assert len(result["common_features"]) < 6  # not all feature types common
