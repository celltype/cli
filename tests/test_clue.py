"""Tests for CLUE/L1000 compound signature and connectivity tools.

Our clue.py uses local L1000 Level 5 compound profiles (19,811 compounds × 978 genes)
from parquet files. It does NOT require an API key for local data.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestClueConnectivityQuery:
    def test_signature_query_success(self):
        from ct.tools.clue import connectivity_query

        result = connectivity_query(
            gene_list={"up": ["TP53", "CDKN1A"], "down": ["MYC", "CCND1"]}
        )
        assert "summary" in result
        assert "error" not in result
        assert result["n_up_matched"] >= 0
        assert result["n_down_matched"] >= 0
        assert "top_mimickers" in result or "n_compounds_scored" in result

    def test_invalid_gene_list(self):
        from ct.tools.clue import connectivity_query

        result = connectivity_query(gene_list=None)
        assert "error" in result

    def test_empty_gene_list(self):
        from ct.tools.clue import connectivity_query

        result = connectivity_query(gene_list={"up": [], "down": []})
        assert "error" in result

    def test_get_clue_key_exists(self):
        """_get_clue_key function exists for API fallback compatibility."""
        from ct.tools.clue import _get_clue_key

        # Returns None when no key configured (local data doesn't need it)
        result = _get_clue_key()
        assert result is None or isinstance(result, str)


class TestClueCompoundSignature:
    def test_known_compound(self):
        from ct.tools.clue import compound_signature

        result = compound_signature(compound="vorinostat")
        assert "summary" in result
        assert result["compound"] == "vorinostat"
        # Local data returns up_genes and down_genes as lists of dicts with gene/z_score
        assert "up_genes" in result or "error" not in result

    def test_unknown_compound(self):
        from ct.tools.clue import compound_signature

        result = compound_signature(compound="nonexistent_compound_xyz_12345")
        assert "error" in result or "not found" in result.get("summary", "").lower()

    def test_missing_api_key_still_works_locally(self):
        """Local data implementation works without API key."""
        from ct.tools.clue import compound_signature

        with patch("ct.tools.clue._get_clue_key", return_value=None):
            result = compound_signature(compound="vorinostat")
            # Should succeed with local data even without API key
            assert "summary" in result
