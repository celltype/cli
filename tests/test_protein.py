"""Tests for protein analysis tools: embed, function_predict, domain_annotate."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ─── protein.embed ────────────────────────────────────────────


class TestProteinEmbed:
    """Tests for protein.embed."""

    def test_empty_sequence_returns_error(self):
        from ct.tools.protein import embed
        result = embed(sequence="")
        assert "error" in result

    def test_invalid_characters_returns_error(self):
        from ct.tools.protein import embed
        result = embed(sequence="MKTL123!!!")
        assert "error" in result
        assert "invalid" in result["error"].lower() or "Invalid" in result["error"]

    def test_too_long_sequence_returns_error(self):
        from ct.tools.protein import embed
        result = embed(sequence="M" * 3000)
        assert "error" in result
        assert "too long" in result["error"].lower() or "exceeds" in result["error"].lower()

    def test_esm_not_installed(self):
        """When torch/esm not installed, returns install instructions."""
        from ct.tools.protein import embed

        # Patch only torch and esm imports inside the function
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name in ("torch", "esm"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = embed(sequence="MKTLLILAVLCLAV")

        assert "error" in result
        assert "torch" in result["error"].lower() or "esm" in result["error"].lower()
        assert result["computed_locally"] is False

    def test_esm_success_mock(self):
        """Successful ESM-2 embedding with mocked torch and esm."""
        import sys

        # Create mock torch and esm modules
        mock_torch = MagicMock()
        mock_torch.no_grad = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        ))

        seq = "MKTLLILAVLCLAV"
        seq_len = len(seq)

        # Mock ESM model output
        mock_repr = np.random.randn(1, seq_len + 2, 1280).astype(np.float32)
        mock_results = {
            "representations": {
                33: MagicMock(numpy=MagicMock(return_value=mock_repr[0]),
                              __getitem__=MagicMock(return_value=MagicMock(
                                  numpy=MagicMock(return_value=mock_repr[0, 1:seq_len + 1])
                              )))
            }
        }

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = mock_results

        mock_alphabet = MagicMock()
        batch_converter = MagicMock(return_value=(["protein"], [seq], MagicMock()))
        mock_alphabet.get_batch_converter = MagicMock(return_value=batch_converter)

        mock_esm = MagicMock()
        mock_esm.pretrained.esm2_t33_650M_UR50D = MagicMock(return_value=(mock_model, mock_alphabet))

        with patch.dict(sys.modules, {"torch": mock_torch, "esm": mock_esm}):
            from ct.tools.protein import embed
            result = embed(sequence=seq)

        assert "summary" in result
        assert result["sequence_length"] == seq_len
        assert result["computed_locally"] is True

    def test_valid_sequence_accepted(self):
        """Valid amino acid sequences should not fail validation."""
        from ct.tools.protein import embed

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name in ("torch", "esm"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = embed(sequence="ACDEFGHIKLMNPQRSTVWY")

        # Should get past validation, fail on import
        assert result["sequence_length"] == 20


# ─── protein.function_predict ─────────────────────────────────


class TestProteinFunctionPredict:
    """Tests for protein.function_predict."""

    @patch("httpx.get")
    def test_gene_search_success(self, mock_get):
        """Query by gene symbol returns function data."""
        from ct.tools.protein import function_predict

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{
                "primaryAccession": "P38398",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Breast cancer type 1 susceptibility protein"}
                    }
                },
                "genes": [{"geneName": {"value": "BRCA1"}}],
                "sequence": {"length": 1863, "value": "M" * 1863},
                "comments": [
                    {
                        "commentType": "FUNCTION",
                        "texts": [{"value": "DNA repair protein involved in homologous recombination"}],
                    },
                    {
                        "commentType": "SUBCELLULAR LOCATION",
                        "subcellularLocations": [{"location": {"value": "Nucleus"}}],
                    },
                    {
                        "commentType": "DISEASE",
                        "disease": {
                            "diseaseId": "Breast cancer",
                            "description": "Hereditary breast cancer",
                            "acronym": "BC",
                        },
                    },
                ],
                "features": [
                    {"type": "Domain", "description": "BRCT 1", "location": {"start": {"value": 1642}, "end": {"value": 1736}}},
                    {"type": "Domain", "description": "BRCT 2", "location": {"start": {"value": 1756}, "end": {"value": 1855}}},
                    {"type": "Modified residue", "description": "Phosphoserine", "location": {"start": {"value": 1524}, "end": {"value": 1524}}},
                ],
                "keywords": [{"name": "DNA repair"}, {"name": "Tumor suppressor"}],
                "uniProtKBCrossReferences": [
                    {"database": "GO", "id": "GO:0006281", "properties": [{"key": "GoTerm", "value": "P:DNA repair"}]},
                ],
            }],
        }
        mock_get.return_value = mock_response

        result = function_predict(gene="BRCA1")

        assert "summary" in result
        assert result["uniprot_id"] == "P38398"
        assert result["gene"] == "BRCA1"
        assert "DNA repair" in result["function"]
        assert "Nucleus" in result["subcellular_locations"]
        assert len(result["domains"]) == 2
        assert len(result["disease_associations"]) == 1

    @patch("httpx.get")
    def test_uniprot_id_direct_lookup(self, mock_get):
        """Query by UniProt ID uses direct endpoint."""
        from ct.tools.protein import function_predict

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "primaryAccession": "P04637",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}
            },
            "genes": [{"geneName": {"value": "TP53"}}],
            "sequence": {"length": 393, "value": "M" * 393},
            "comments": [],
            "features": [],
            "keywords": [],
            "uniProtKBCrossReferences": [],
        }
        mock_get.return_value = mock_response

        result = function_predict(gene="P04637")

        assert "summary" in result
        assert result["uniprot_id"] == "P04637"

    @patch("httpx.get")
    def test_gene_not_found(self, mock_get):
        """Nonexistent gene returns error."""
        from ct.tools.protein import function_predict

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response

        result = function_predict(gene="FAKEGENE123")

        assert "error" in result
        # Error message says "No UniProt entry found..."
        assert "found" in result["error"].lower()

    @patch("httpx.get")
    def test_api_error_handled(self, mock_get):
        """HTTP errors are handled gracefully."""
        import httpx
        from ct.tools.protein import function_predict

        mock_get.side_effect = httpx.HTTPError("Connection failed")

        result = function_predict(gene="BRCA1")

        assert "error" in result


# ─── protein.domain_annotate ──────────────────────────────────


class TestProteinDomainAnnotate:
    """Tests for protein.domain_annotate."""

    def test_no_gene_or_uniprot_returns_error(self):
        """Must provide either gene or uniprot_id."""
        from ct.tools.protein import domain_annotate
        result = domain_annotate()
        assert "error" in result

    @patch("httpx.get")
    def test_domain_annotation_success(self, mock_get):
        """Successful domain annotation from InterPro."""
        from ct.tools.protein import domain_annotate

        # Mock UniProt resolution
        uniprot_response = MagicMock()
        uniprot_response.status_code = 200
        uniprot_response.json.return_value = {
            "results": [{"primaryAccession": "P04637"}]
        }

        # Mock InterPro response
        interpro_response = MagicMock()
        interpro_response.status_code = 200
        interpro_response.json.return_value = {
            "results": [
                {
                    "metadata": {
                        "accession": "IPR002117",
                        "name": "p53 tumour suppressor family",
                        "type": "family",
                        "source_database": "pfam",
                        "description": [{"text": "P53 family"}],
                    },
                    "proteins": [{
                        "entry_protein_locations": [{
                            "fragments": [{"start": 102, "end": 292}]
                        }]
                    }],
                },
                {
                    "metadata": {
                        "accession": "IPR011615",
                        "name": "p53 DNA-binding domain",
                        "type": "domain",
                        "source_database": "pfam",
                        "description": [{"text": "DNA binding domain of p53"}],
                    },
                    "proteins": [{
                        "entry_protein_locations": [{
                            "fragments": [{"start": 102, "end": 292}]
                        }]
                    }],
                },
                {
                    "metadata": {
                        "accession": "IPR010991",
                        "name": "p53 tetramerization domain",
                        "type": "domain",
                        "source_database": "pfam",
                        "description": [{"text": "Tetramerization domain"}],
                    },
                    "proteins": [{
                        "entry_protein_locations": [{
                            "fragments": [{"start": 323, "end": 356}]
                        }]
                    }],
                },
            ]
        }

        mock_get.side_effect = [uniprot_response, interpro_response]

        result = domain_annotate(gene="TP53")

        assert "summary" in result
        assert result["uniprot_id"] == "P04637"
        assert result["n_domains"] == 2
        assert result["n_families"] == 1
        assert "TP53" in result["summary"]

    @patch("httpx.get")
    def test_direct_uniprot_id(self, mock_get):
        """Can use UniProt ID directly without gene resolution."""
        from ct.tools.protein import domain_annotate

        interpro_response = MagicMock()
        interpro_response.status_code = 200
        interpro_response.json.return_value = {"results": []}
        mock_get.return_value = interpro_response

        result = domain_annotate(uniprot_id="P04637")

        assert "summary" in result
        assert result["uniprot_id"] == "P04637"
        assert mock_get.call_count == 1

    @patch("httpx.get")
    def test_gene_not_resolved(self, mock_get):
        """Gene that can't be resolved returns error."""
        from ct.tools.protein import domain_annotate

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response

        result = domain_annotate(gene="NONEXISTENT")

        assert "error" in result
        assert "found" in result["error"].lower() or "resolve" in result["error"].lower()

    @patch("httpx.get")
    def test_interpro_204_returns_empty_annotations(self, mock_get):
        """InterPro HTTP 204 should be treated as no domain annotations, not hard failure."""
        from ct.tools.protein import domain_annotate

        uniprot_response = MagicMock()
        uniprot_response.status_code = 200
        uniprot_response.json.return_value = {
            "results": [{"primaryAccession": "I3L4S0"}]
        }

        interpro_no_content = MagicMock()
        interpro_no_content.status_code = 204
        mock_get.side_effect = [uniprot_response, interpro_no_content]

        result = domain_annotate(gene="VPS35")

        assert "error" not in result
        assert result["uniprot_id"] == "I3L4S0"
        assert result["n_domains"] == 0

    @patch("httpx.get")
    def test_non_human_gene_resolution_tries_unrestricted_search(self, mock_get):
        """Non-human-like queries should try unrestricted UniProt search."""
        from ct.tools.protein import domain_annotate

        uniprot_fail = MagicMock()
        uniprot_fail.status_code = 200
        uniprot_fail.json.return_value = {"results": []}

        uniprot_hit = MagicMock()
        uniprot_hit.status_code = 200
        uniprot_hit.json.return_value = {"results": [{"primaryAccession": "A0A0X0HP01"}]}

        interpro_resp = MagicMock()
        interpro_resp.status_code = 200
        interpro_resp.json.return_value = {"results": []}

        # First UniProt attempt empty; second resolves; then InterPro lookup.
        mock_get.side_effect = [uniprot_fail, uniprot_hit, interpro_resp]

        result = domain_annotate(gene="Heligmosomoides polygyrus")
        assert "error" not in result
        assert result["uniprot_id"] == "A0A0X0HP01"

    @patch("httpx.get")
    def test_keyword_mode_when_gene_not_resolved(self, mock_get):
        """If no UniProt mapping exists, domain keyword search should still return InterPro hits."""
        from ct.tools.protein import domain_annotate

        uniprot_fail = MagicMock()
        uniprot_fail.status_code = 200
        uniprot_fail.json.return_value = {"results": []}

        interpro_keyword = MagicMock()
        interpro_keyword.status_code = 200
        interpro_keyword.json.return_value = {
            "results": [
                {
                    "metadata": {
                        "accession": "IPR000321",
                        "name": "CAP domain",
                        "type": "domain",
                        "source_database": "interpro",
                        "description": [{"text": "CAP/SCP/TAPS domain"}],
                    }
                },
                {
                    "metadata": {
                        "accession": "IPR036390",
                        "name": "SCP/TAPS family",
                        "type": "family",
                        "source_database": "interpro",
                        "description": [{"text": "SCP/TAPS proteins"}],
                    }
                },
            ]
        }

        mock_get.side_effect = [uniprot_fail, uniprot_fail, interpro_keyword]

        result = domain_annotate(gene="SCP/TAPS CAP superfamily")
        assert "error" not in result
        assert result["mode"] == "interpro_keyword_search"
        assert result["n_domains"] >= 1

    @patch("httpx.get")
    def test_interpro_accession_mode(self, mock_get):
        """Direct InterPro accession should bypass UniProt resolution."""
        from ct.tools.protein import domain_annotate

        interpro_resp = MagicMock()
        interpro_resp.status_code = 200
        interpro_resp.json.return_value = {
            "results": [
                {
                    "metadata": {
                        "accession": "IPR014044",
                        "name": "CAP domain",
                        "type": "domain",
                        "source_database": "interpro",
                        "description": [{"text": "CAP superfamily"}],
                    }
                }
            ]
        }
        mock_get.return_value = interpro_resp

        result = domain_annotate(gene="IPR014044")
        assert "error" not in result
        assert result["mode"] == "interpro_accession_lookup"
        assert result["n_domains"] >= 1
