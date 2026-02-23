"""Tests for network tools: ppi_analysis and pathway_crosstalk."""

from unittest.mock import MagicMock, patch

import httpx


class TestPpiAnalysis:
    @patch("httpx.get")
    def test_ppi_analysis_success(self, mock_get):
        from ct.tools.network import ppi_analysis

        network_resp = MagicMock()
        network_resp.status_code = 200
        network_resp.raise_for_status = MagicMock()
        network_resp.json.return_value = [
            {
                "preferredName_A": "CRBN",
                "preferredName_B": "DDB1",
                "score": 0.91,
                "nscore": 0.0,
                "fscore": 0.0,
                "pscore": 0.0,
                "ascore": 0.0,
                "escore": 0.0,
                "dscore": 0.0,
                "tscore": 0.0,
            }
        ]

        enrich_resp = MagicMock()
        enrich_resp.status_code = 200
        enrich_resp.raise_for_status = MagicMock()
        enrich_resp.json.return_value = [
            {
                "category": "Process",
                "term": "GO:0008150",
                "description": "protein ubiquitination",
                "p_value": 1.0e-6,
                "fdr": 1.0e-4,
                "number_of_genes": 2,
                "preferredNames": "CRBN,DDB1",
            }
        ]

        mock_get.side_effect = [network_resp, enrich_resp]

        result = ppi_analysis(gene="CRBN")

        assert "summary" in result
        assert result["network_stats"]["node_count"] == 2
        assert result["network_stats"]["edge_count"] == 1
        assert len(result["interactions"]) == 1
        assert result["enrichment"][0]["description"] == "protein ubiquitination"

    @patch("httpx.get")
    def test_ppi_analysis_api_error(self, mock_get):
        from ct.tools.network import ppi_analysis

        mock_get.side_effect = httpx.TimeoutException("timeout")

        result = ppi_analysis(gene="CRBN")

        assert "error" in result
        assert "STRING network query failed" in result["error"]

    @patch("httpx.get")
    def test_ppi_analysis_accepts_list_input(self, mock_get):
        from ct.tools.network import ppi_analysis

        network_resp = MagicMock()
        network_resp.status_code = 200
        network_resp.raise_for_status = MagicMock()
        network_resp.json.return_value = [
            {
                "preferredName_A": "CRBN",
                "preferredName_B": "DDB1",
                "score": 0.91,
                "nscore": 0.0,
                "fscore": 0.0,
                "pscore": 0.0,
                "ascore": 0.0,
                "escore": 0.0,
                "dscore": 0.0,
                "tscore": 0.0,
            }
        ]
        enrich_resp = MagicMock()
        enrich_resp.status_code = 200
        enrich_resp.raise_for_status = MagicMock()
        enrich_resp.json.return_value = []
        mock_get.side_effect = [network_resp, enrich_resp]

        result = ppi_analysis(gene=["CRBN", "DDB1"])
        assert "error" not in result
        assert result["query_genes"] == ["CRBN", "DDB1"]


class TestPathwayCrosstalk:
    @patch("httpx.get")
    @patch("httpx.post")
    def test_pathway_crosstalk_success(self, mock_post, mock_get):
        from ct.tools.network import pathway_crosstalk

        analysis_resp = MagicMock()
        analysis_resp.status_code = 200
        analysis_resp.raise_for_status = MagicMock()
        analysis_resp.json.return_value = {
            "pathways": [
                {
                    "stId": "R-HSA-12345",
                    "name": "Ubiquitin-dependent degradation",
                    "entities": {"pValue": 1.0e-5, "fdr": 1.0e-3, "found": 2, "total": 20, "ratio": 0.1},
                },
            ],
            "identifiersNotFound": 0,
            "foundEntities": 2,
        }
        mock_post.return_value = analysis_resp

        participants_resp = MagicMock()
        participants_resp.status_code = 200
        participants_resp.json.return_value = [
            {"refEntities": [{"displayName": "UniProt:Q96SW2 CRBN"}]},
            {"refEntities": [{"displayName": "UniProt:Q16531 DDB1"}]},
        ]
        mock_get.return_value = participants_resp

        result = pathway_crosstalk(genes="CRBN,DDB1")

        assert "summary" in result
        assert len(result["pathways"]) == 1
        assert result["pathways"][0]["name"] == "Ubiquitin-dependent degradation"
        assert result["genes_not_found"] == 0

    @patch("httpx.post")
    def test_pathway_crosstalk_api_error(self, mock_post):
        from ct.tools.network import pathway_crosstalk

        mock_post.side_effect = httpx.RequestError("network down")

        result = pathway_crosstalk(genes="CRBN,DDB1")

        assert "error" in result
        assert "Reactome analysis failed" in result["error"]

    @patch("httpx.get")
    @patch("httpx.post")
    def test_pathway_crosstalk_accepts_list_input(self, mock_post, mock_get):
        from ct.tools.network import pathway_crosstalk

        analysis_resp = MagicMock()
        analysis_resp.status_code = 200
        analysis_resp.raise_for_status = MagicMock()
        analysis_resp.json.return_value = {
            "pathways": [
                {
                    "stId": "R-HSA-12345",
                    "name": "Ubiquitin-dependent degradation",
                    "entities": {"pValue": 1.0e-5, "fdr": 1.0e-3, "found": 2, "total": 20, "ratio": 0.1},
                },
            ],
            "identifiersNotFound": 0,
            "foundEntities": 2,
        }
        mock_post.return_value = analysis_resp

        participants_resp = MagicMock()
        participants_resp.status_code = 200
        participants_resp.json.return_value = [
            {"refEntities": [{"displayName": "UniProt:Q96SW2 CRBN"}]},
            {"refEntities": [{"displayName": "UniProt:Q16531 DDB1"}]},
        ]
        mock_get.return_value = participants_resp

        result = pathway_crosstalk(genes=["CRBN", "DDB1"])
        assert "error" not in result
        assert result["query_genes"] == ["CRBN", "DDB1"]
