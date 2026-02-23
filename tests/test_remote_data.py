"""Tests for remote_data tool (Data API client)."""

import pytest
from unittest.mock import patch, MagicMock
import httpx


class TestRemoteDataQuery:
    @patch("httpx.post")
    @patch("ct.tools.remote_data._get_endpoint", return_value="http://localhost:8000")
    def test_query_success(self, mock_endpoint, mock_post):
        from ct.tools.remote_data import query

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "dataset": "perturbatlas",
            "total_rows": 2,
            "data": [
                {"gene": "TP53", "log2FoldChange": -1.5, "padj": 0.001},
                {"gene": "TP53", "log2FoldChange": -0.8, "padj": 0.01},
            ],
        }
        mock_post.return_value = mock_resp

        result = query(dataset="perturbatlas", gene="TP53")
        assert "summary" in result
        assert "error" not in result
        assert result["total_rows"] == 2
        assert len(result["data"]) == 2
        assert "gene" in result["columns"]

    def test_no_endpoint_configured(self):
        from ct.tools.remote_data import query

        with patch("ct.tools.remote_data._get_endpoint", return_value=None):
            result = query(dataset="perturbatlas")
            assert "error" in result
            assert "not configured" in result["error"].lower()

    @patch("httpx.post", side_effect=httpx.ConnectError("Connection refused"))
    @patch("ct.tools.remote_data._get_endpoint", return_value="http://localhost:8000")
    def test_api_unreachable(self, mock_endpoint, mock_post):
        from ct.tools.remote_data import query

        result = query(dataset="perturbatlas", gene="TP53")
        assert "error" in result
        assert "unreachable" in result["summary"].lower()

    @patch("httpx.post")
    @patch("ct.tools.remote_data._get_endpoint", return_value="http://localhost:8000")
    def test_dataset_not_found(self, mock_endpoint, mock_post):
        from ct.tools.remote_data import query

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_post.return_value = mock_resp

        result = query(dataset="nonexistent")
        assert "error" in result
        assert "not found" in result["error"].lower()

    @patch("httpx.post")
    @patch("ct.tools.remote_data._get_endpoint", return_value="http://localhost:8000")
    def test_query_with_compound_filter(self, mock_endpoint, mock_post):
        from ct.tools.remote_data import query

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "dataset": "chembl",
            "total_rows": 1,
            "data": [{"molecule_chembl_id": "CHEMBL25", "pchembl_value": 6.5}],
        }
        mock_post.return_value = mock_resp

        result = query(dataset="chembl", compound="CHEMBL25")
        assert "error" not in result
        assert result["total_rows"] == 1


class TestRemoteDataListDatasets:
    @patch("httpx.get")
    @patch("ct.tools.remote_data._get_endpoint", return_value="http://localhost:8000")
    def test_list_success(self, mock_endpoint, mock_get):
        from ct.tools.remote_data import list_datasets

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {"name": "perturbatlas", "description": "PerturbAtlas DEGs",
             "format": "csv.gz", "n_files": 2066, "total_size_mb": 500},
            {"name": "chembl", "description": "ChEMBL v36",
             "format": "parquet", "n_files": 5, "total_size_mb": 800},
        ]
        mock_get.return_value = mock_resp

        result = list_datasets()
        assert "summary" in result
        assert "error" not in result
        assert len(result["datasets"]) == 2
        assert "perturbatlas" in result["summary"]

    def test_no_endpoint_configured(self):
        from ct.tools.remote_data import list_datasets

        with patch("ct.tools.remote_data._get_endpoint", return_value=None):
            result = list_datasets()
            assert "error" in result

    @patch("httpx.get", side_effect=httpx.ConnectError("Connection refused"))
    @patch("ct.tools.remote_data._get_endpoint", return_value="http://localhost:8000")
    def test_api_unreachable(self, mock_endpoint, mock_get):
        from ct.tools.remote_data import list_datasets

        result = list_datasets()
        assert "error" in result
        assert "unreachable" in result["summary"].lower()
