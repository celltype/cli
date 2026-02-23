"""Tests for repurposing tools."""

from unittest.mock import patch


class TestRepurposingCmapQuery:
    def test_remote_fallback_returns_hits_when_l1000_missing(self):
        from ct.tools.repurposing import cmap_query

        with patch("ct.data.loaders.load_l1000", side_effect=FileNotFoundError("missing")):
            with patch("ct.tools.repurposing.request_json") as mock_request_json:
                mock_request_json.side_effect = [
                    ({"result_id": "abc123"}, None),
                    (
                        {
                            "similar": [
                                {"pert_iname": "drug_a", "score": 0.91},
                                {"pert_iname": "drug_b", "score": 0.88},
                            ],
                            "opposite": [
                                {"pert_iname": "drug_x", "score": -0.93},
                                {"pert_iname": "drug_y", "score": -0.90},
                            ],
                        },
                        None,
                    ),
                ]

                result = cmap_query(
                    gene_signature={
                        "TNF": 2.5,
                        "IL1B": 2.1,
                        "IL10": -1.3,
                        "MUC2": -1.1,
                    },
                    mode="reverse",
                    top_n=10,
                )

        assert "error" not in result
        assert result["remote_used"] is True
        assert result["remote_source"] == "L1000FWD"
        assert result["local_data_unavailable"] is True
        assert len(result["hits"]) == 2
        assert result["hits"][0]["compound"] == "drug_x"
        assert "Remote CMap query via L1000FWD" in result["summary"]

    @patch("ct.data.loaders.load_l1000", side_effect=FileNotFoundError("missing"))
    @patch("ct.tools.repurposing.request_json", return_value=(None, "HTTP 503"))
    def test_marks_data_unavailable_when_remote_fallback_fails(self, _mock_request_json, _mock_l1000):
        from ct.tools.repurposing import cmap_query

        result = cmap_query(
            gene_signature={
                "TNF": 2.5,
                "IL1B": 2.1,
                "IL10": -1.3,
                "MUC2": -1.1,
            },
            mode="reverse",
            top_n=10,
        )

        assert "error" not in result
        assert result["data_unavailable"] is True
        assert result["remote_used"] is False
        assert "remote_error" in result
        assert result["hits"] == []
        assert len(result["up_genes"]) == 2
        assert len(result["down_genes"]) == 2
        assert "cannot compute correlations locally" in result["summary"]

    @patch("ct.data.loaders.load_l1000", side_effect=FileNotFoundError("missing"))
    @patch("ct.tools.repurposing.request_json")
    def test_no_remote_call_when_allow_remote_false(self, mock_request_json, _mock_l1000):
        from ct.tools.repurposing import cmap_query

        result = cmap_query(
            gene_signature={
                "TNF": 2.5,
                "IL1B": 2.1,
                "IL10": -1.3,
                "MUC2": -1.1,
            },
            mode="reverse",
            top_n=10,
            allow_remote=False,
        )

        mock_request_json.assert_not_called()
        assert result["data_unavailable"] is True
        assert result["remote_used"] is False
