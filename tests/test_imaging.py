"""Tests for imaging tools: cellpainting_lookup, morphology_similarity."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ─── imaging.cellpainting_lookup ──────────────────────────────


class TestCellpaintingLookup:
    """Tests for imaging.cellpainting_lookup."""

    @patch("httpx.get")
    @patch("httpx.post")
    def test_lookup_by_name(self, mock_post, mock_get):
        """Look up compound by name via PubChem."""
        from ct.tools.imaging import cellpainting_lookup

        # PubChem compound lookup
        pubchem_props = MagicMock()
        pubchem_props.status_code = 200
        pubchem_props.json.return_value = {
            "PropertyTable": {
                "Properties": [{
                    "CID": 216326,
                    "CanonicalSMILES": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
                    "InChIKey": "UEJJHQNUAWNRDK-UHFFFAOYSA-N",
                    "IUPACName": "lenalidomide",
                }]
            }
        }

        # PubChem bioassay response
        bioassay_resp = MagicMock()
        bioassay_resp.status_code = 200
        bioassay_resp.json.return_value = {"Table": {"Row": []}}

        # JUMP S3 response (just check availability)
        jump_resp = MagicMock()
        jump_resp.status_code = 200

        mock_get.side_effect = [pubchem_props, jump_resp, bioassay_resp]

        result = cellpainting_lookup(compound="lenalidomide")

        assert "summary" in result
        assert result["compound_info"]["cid"] == 216326

    @patch("httpx.get")
    @patch("httpx.post")
    def test_lookup_by_smiles(self, mock_post, mock_get):
        """Look up compound by SMILES."""
        from ct.tools.imaging import cellpainting_lookup

        pubchem_props = MagicMock()
        pubchem_props.status_code = 200
        pubchem_props.json.return_value = {
            "PropertyTable": {
                "Properties": [{
                    "CID": 2244,
                    "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                    "IUPACName": "aspirin",
                }]
            }
        }

        # POST for SMILES
        mock_post.return_value = pubchem_props

        # GET for JUMP + bioassay
        jump_resp = MagicMock()
        jump_resp.status_code = 200
        bioassay_resp = MagicMock()
        bioassay_resp.status_code = 200
        bioassay_resp.json.return_value = {"Table": {"Row": []}}
        mock_get.side_effect = [jump_resp, bioassay_resp]

        result = cellpainting_lookup(compound="CC(=O)OC1=CC=CC=C1C(=O)O", source="jump")

        assert "summary" in result
        assert result["compound_info"]["cid"] == 2244

    @patch("httpx.get")
    def test_pubchem_not_found(self, mock_get):
        """Compound not found in PubChem."""
        from ct.tools.imaging import cellpainting_lookup

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = cellpainting_lookup(compound="nonexistent_compound_xyz")

        assert "summary" in result
        # Should still return a result, just without data
        assert result["compound_info"]["cid"] is None

    @patch("httpx.get")
    def test_api_timeout_handled(self, mock_get):
        """API timeouts are handled gracefully."""
        import httpx
        from ct.tools.imaging import cellpainting_lookup

        mock_get.side_effect = httpx.TimeoutException("timeout")

        result = cellpainting_lookup(compound="aspirin")

        assert "summary" in result
        # Should not crash, just have empty data


# ─── imaging.morphology_similarity ────────────────────────────


class TestMorphologySimilarity:
    """Tests for imaging.morphology_similarity."""

    def test_identical_compounds(self):
        """Same SMILES should give similarity ~1.0."""
        from ct.tools.imaging import morphology_similarity

        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        result = morphology_similarity(smiles_a=smiles, smiles_b=smiles)

        assert "summary" in result
        assert result["similarity_scores"]["morgan_tanimoto"] == 1.0
        assert result["similarity_class"] == "highly similar"

    def test_different_compounds(self):
        """Different compounds should have <1.0 similarity."""
        from ct.tools.imaging import morphology_similarity

        aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

        result = morphology_similarity(smiles_a=aspirin, smiles_b=caffeine)

        assert "summary" in result
        assert result["similarity_scores"]["morgan_tanimoto"] < 1.0
        assert "compound_a" in result
        assert "compound_b" in result
        assert "property_differences" in result

    def test_invalid_smiles_a(self):
        """Invalid SMILES for compound A returns error."""
        from ct.tools.imaging import morphology_similarity

        result = morphology_similarity(smiles_a="INVALID", smiles_b="CC(=O)O")

        assert "error" in result

    def test_invalid_smiles_b(self):
        """Invalid SMILES for compound B returns error."""
        from ct.tools.imaging import morphology_similarity

        result = morphology_similarity(smiles_a="CC(=O)O", smiles_b="INVALID")

        assert "error" in result

    def test_similar_compounds(self):
        """Structurally similar compounds should have high similarity."""
        from ct.tools.imaging import morphology_similarity

        # Lenalidomide and pomalidomide (close analogs)
        lenalidomide = "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"
        pomalidomide = "O=C1CCC(N2C(=O)c3cc(N)ccc3C2=O)C(=O)N1"

        result = morphology_similarity(smiles_a=lenalidomide, smiles_b=pomalidomide)

        assert "summary" in result
        assert result["similarity_scores"]["morgan_tanimoto"] > 0.5
        assert "shared_features" in result

    def test_output_structure(self):
        """Verify all expected fields are present."""
        from ct.tools.imaging import morphology_similarity

        result = morphology_similarity(
            smiles_a="c1ccccc1",
            smiles_b="c1ccncc1",
        )

        assert "summary" in result
        assert "similarity_scores" in result
        assert "morgan_tanimoto" in result["similarity_scores"]
        assert "maccs_tanimoto" in result["similarity_scores"]
        assert "dice" in result["similarity_scores"]
        assert "combined" in result["similarity_scores"]
        assert "similarity_class" in result
        assert "morphology_prediction" in result
        assert "compound_a" in result
        assert "compound_b" in result
        assert "properties" in result["compound_a"]
        assert "properties" in result["compound_b"]

    def test_rdkit_not_installed(self):
        """When RDKit not installed, return helpful error."""
        from ct.tools.imaging import morphology_similarity

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if "rdkit" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = morphology_similarity(smiles_a="CC", smiles_b="CCC")
            assert "error" in result
            assert "rdkit" in result["error"].lower() or "RDKit" in result["error"]
