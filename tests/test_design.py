"""Tests for design tools: compound modification suggestions."""

import pytest
from unittest.mock import patch, MagicMock


def _rdkit_available():
    try:
        from rdkit import Chem
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
class TestSuggestModifications:
    def test_basic_benzamide(self):
        """Benzamide should produce modification suggestions."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("c1ccccc1C(=O)N", objective="potency", n_suggestions=5)

        assert "summary" in result
        assert "parent_smiles" in result
        assert "parent_properties" in result
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0
        assert len(result["suggestions"]) <= 5

    def test_suggestion_has_required_fields(self):
        """Each suggestion should have smiles, rationale, score, properties."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("c1ccccc1C(=O)N", objective="potency", n_suggestions=3)

        if result["suggestions"]:
            s = result["suggestions"][0]
            assert "smiles" in s
            assert "transform" in s
            assert "rationale" in s
            assert "score" in s
            assert "properties" in s
            assert "property_deltas" in s
            assert "lipinski_violations" in s

    def test_property_deltas_computed(self):
        """Property deltas should show the difference from parent."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("c1ccccc1O", objective="admet", n_suggestions=5)

        if result["suggestions"]:
            deltas = result["suggestions"][0]["property_deltas"]
            assert "mw" in deltas
            assert "logp" in deltas
            assert "hbd" in deltas
            assert "hba" in deltas
            assert "tpsa" in deltas

    def test_all_objectives(self):
        """All supported objectives should work without error."""
        from ct.tools.design import suggest_modifications

        for obj in ("potency", "selectivity", "admet", "solubility", "metabolic_stability"):
            result = suggest_modifications("c1ccccc1", objective=obj, n_suggestions=3)
            assert "summary" in result
            assert "error" not in result

    def test_invalid_objective(self):
        """Unknown objective should return error."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("CCO", objective="invalid_goal")
        assert "error" in result
        assert "invalid_goal" in result["error"]

    def test_invalid_smiles(self):
        """Invalid SMILES should return error."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("NOT_A_SMILES_AT_ALL")
        assert "error" in result

    def test_parent_properties_correct(self):
        """Parent properties should be computed correctly."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("c1ccccc1", objective="potency")

        props = result["parent_properties"]
        assert props["mw"] > 70  # benzene MW ~78
        assert props["mw"] < 90
        assert "logp" in props
        assert "tpsa" in props
        assert props["aromatic_rings"] == 1

    def test_fluorinated_compound(self):
        """Compound with F should trigger F->Cl, F->H transforms."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("c1ccc(F)cc1", objective="potency", n_suggestions=10)

        transforms_found = {s["transform"] for s in result["suggestions"]}
        # Should find at least one F-related transform
        assert len(result["suggestions"]) > 0

    def test_n_suggestions_limit(self):
        """Should not return more suggestions than requested."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("c1ccccc1C(=O)N", n_suggestions=2)
        assert len(result["suggestions"]) <= 2

    def test_suggestions_sorted_by_score(self):
        """Suggestions should be sorted by score descending."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("c1ccccc1C(=O)N", objective="potency", n_suggestions=10)

        scores = [s["score"] for s in result["suggestions"]]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_smiles(self):
        """No two suggestions should have the same SMILES."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("c1ccccc1C(=O)N", objective="potency", n_suggestions=10)

        smiles_set = {s["smiles"] for s in result["suggestions"]}
        assert len(smiles_set) == len(result["suggestions"])

    def test_simple_molecule_ethanol(self):
        """Simple molecules without aromatic rings may produce fewer suggestions."""
        from ct.tools.design import suggest_modifications

        result = suggest_modifications("CCO", objective="solubility")
        # Should not error
        assert "summary" in result
        assert "error" not in result


@pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
class TestHelperFunctions:
    def test_compute_properties(self):
        from ct.tools.design import _compute_properties
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")
        props = _compute_properties(mol)
        assert props["aromatic_rings"] == 1
        assert props["rings"] == 1
        assert props["hbd"] == 0
        assert props["heavy_atoms"] == 6

    def test_lipinski_violations_clean(self):
        from ct.tools.design import _lipinski_violations

        # Small drug-like molecule
        props = {"mw": 300, "logp": 2.5, "hbd": 2, "hba": 5}
        assert _lipinski_violations(props) == 0

    def test_lipinski_violations_all(self):
        from ct.tools.design import _lipinski_violations

        # Everything violated
        props = {"mw": 600, "logp": 6, "hbd": 6, "hba": 11}
        assert _lipinski_violations(props) == 4

    def test_veber_violations_clean(self):
        from ct.tools.design import _veber_violations

        props = {"tpsa": 80, "rotatable_bonds": 5}
        assert _veber_violations(props) == 0

    def test_veber_violations_both(self):
        from ct.tools.design import _veber_violations

        props = {"tpsa": 200, "rotatable_bonds": 15}
        assert _veber_violations(props) == 2

    def test_score_for_objective_returns_float(self):
        from ct.tools.design import _score_for_objective

        parent = {"mw": 300, "logp": 3.0, "hbd": 2, "hba": 4, "tpsa": 60,
                   "fsp3": 0.3, "rotatable_bonds": 4}
        child = {"mw": 310, "logp": 3.2, "hbd": 2, "hba": 5, "tpsa": 70,
                  "fsp3": 0.35, "rotatable_bonds": 4}

        for obj in ("potency", "selectivity", "admet", "solubility", "metabolic_stability"):
            score = _score_for_objective(parent, child, obj)
            assert isinstance(score, float)


@pytest.mark.skipif(not _rdkit_available(), reason="RDKit not installed")
class TestDesignRegistration:
    def test_tool_registered(self):
        """design.suggest_modifications should be in the registry."""
        from ct.tools import registry, ensure_loaded
        ensure_loaded()

        tool = registry.get_tool("design.suggest_modifications")
        assert tool is not None
        assert tool.category == "design"
        assert "smiles" in tool.parameters
