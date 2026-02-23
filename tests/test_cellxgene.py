"""Tests for CELLxGENE Census tools."""

import sys
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


# Create a mock cellxgene_census module so tests can run without the real SDK
mock_census_module = MagicMock()
mock_census_module.__name__ = "cellxgene_census"


class TestCellxGeneGeneExpression:
    def test_gene_expression_success(self):
        with patch.dict(sys.modules, {"cellxgene_census": mock_census_module}):
            from ct.tools.cellxgene import gene_expression

            mock_census = MagicMock()
            mock_census_module.open_soma.return_value.__enter__ = MagicMock(return_value=mock_census)
            mock_census_module.open_soma.return_value.__exit__ = MagicMock(return_value=False)

            # Mock obs query
            obs_df = pd.DataFrame({
                "cell_type": ["T cell", "T cell", "B cell", "macrophage"],
                "tissue_general": ["lung", "lung", "lung", "lung"],
                "disease": ["normal"] * 4,
                "assay": ["10x 3' v3"] * 4,
            })
            mock_obs = MagicMock()
            mock_obs.read.return_value.concat.return_value.to_pandas.return_value = obs_df
            mock_census.__getitem__.return_value.__getitem__.return_value.obs = mock_obs

            # Mock var (gene) query
            gene_df = pd.DataFrame({
                "soma_joinid": [123],
                "feature_name": ["TP53"],
            })
            mock_var = MagicMock()
            mock_var.read.return_value.concat.return_value.to_pandas.return_value = gene_df
            mock_ms = MagicMock()
            mock_ms.__getitem__.return_value.var = mock_var
            mock_census.__getitem__.return_value.__getitem__.return_value.ms = mock_ms

            result = gene_expression(gene="TP53", tissue="lung")
            assert "summary" in result
            assert "error" not in result
            assert result["gene"] == "TP53"
            assert result["n_cells_total"] == 4
            assert "lung" in result["tissues"]

    def test_missing_sdk_returns_install_instructions(self):
        """When cellxgene-census not installed, tool returns error dict."""
        from ct.tools.cellxgene import _check_census_sdk
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            err = _check_census_sdk()
            assert err is not None
            assert "error" in err
            assert "cellxgene-census" in err["summary"]

    def test_invalid_gene_returns_error(self):
        with patch.dict(sys.modules, {"cellxgene_census": mock_census_module}):
            from ct.tools.cellxgene import gene_expression

            mock_census = MagicMock()
            mock_census_module.open_soma.return_value.__enter__ = MagicMock(return_value=mock_census)
            mock_census_module.open_soma.return_value.__exit__ = MagicMock(return_value=False)

            obs_df = pd.DataFrame({
                "cell_type": ["T cell"],
                "tissue_general": ["lung"],
                "disease": ["normal"],
                "assay": ["10x"],
            })
            mock_obs = MagicMock()
            mock_obs.read.return_value.concat.return_value.to_pandas.return_value = obs_df
            mock_census.__getitem__.return_value.__getitem__.return_value.obs = mock_obs

            # Empty gene result
            mock_var = MagicMock()
            mock_var.read.return_value.concat.return_value.to_pandas.return_value = pd.DataFrame()
            mock_ms = MagicMock()
            mock_ms.__getitem__.return_value.var = mock_var
            mock_census.__getitem__.return_value.__getitem__.return_value.ms = mock_ms

            result = gene_expression(gene="FAKEGENE123")
            assert "error" in result
            assert "not found" in result["error"]


class TestCellxGeneCellTypeMarkers:
    def test_markers_for_t_cell(self):
        with patch.dict(sys.modules, {"cellxgene_census": mock_census_module}):
            from ct.tools.cellxgene import cell_type_markers

            mock_census = MagicMock()
            mock_census_module.open_soma.return_value.__enter__ = MagicMock(return_value=mock_census)
            mock_census_module.open_soma.return_value.__exit__ = MagicMock(return_value=False)

            obs_df = pd.DataFrame({
                "cell_type": ["T cell"] * 100,
                "tissue_general": ["lung"] * 50 + ["blood"] * 50,
            })
            mock_obs = MagicMock()
            mock_obs.read.return_value.concat.return_value.to_pandas.return_value = obs_df
            mock_census.__getitem__.return_value.__getitem__.return_value.obs = mock_obs

            result = cell_type_markers(cell_type="T cell")
            assert "summary" in result
            assert result["cell_type"] == "T cell"
            assert result["n_cells"] == 100
            assert "lung" in result["tissues"]
            assert "blood" in result["tissues"]

    def test_cell_type_not_found(self):
        with patch.dict(sys.modules, {"cellxgene_census": mock_census_module}):
            from ct.tools.cellxgene import cell_type_markers

            mock_census = MagicMock()
            mock_census_module.open_soma.return_value.__enter__ = MagicMock(return_value=mock_census)
            mock_census_module.open_soma.return_value.__exit__ = MagicMock(return_value=False)

            mock_obs = MagicMock()
            mock_obs.read.return_value.concat.return_value.to_pandas.return_value = pd.DataFrame()
            mock_census.__getitem__.return_value.__getitem__.return_value.obs = mock_obs

            result = cell_type_markers(cell_type="nonexistent_cell_type")
            assert "error" in result


class TestCellxGeneDatasetSearch:
    def test_search_by_tissue(self):
        with patch.dict(sys.modules, {"cellxgene_census": mock_census_module}):
            from ct.tools.cellxgene import dataset_search

            mock_census = MagicMock()
            mock_census_module.open_soma.return_value.__enter__ = MagicMock(return_value=mock_census)
            mock_census_module.open_soma.return_value.__exit__ = MagicMock(return_value=False)

            obs_df = pd.DataFrame({
                "dataset_id": ["ds1", "ds1", "ds2"],
                "tissue_general": ["lung", "lung", "lung"],
                "disease": ["normal", "normal", "COVID-19"],
                "assay": ["10x 3' v3", "10x 3' v3", "Smart-seq2"],
                "cell_type": ["T cell", "B cell", "macrophage"],
            })
            mock_obs = MagicMock()
            mock_obs.read.return_value.concat.return_value.to_pandas.return_value = obs_df
            mock_census.__getitem__.return_value.__getitem__.return_value.obs = mock_obs

            result = dataset_search(tissue="lung")
            assert "summary" in result
            assert result["n_datasets"] == 2
            assert result["n_cells_total"] == 3
            assert len(result["datasets"]) == 2
