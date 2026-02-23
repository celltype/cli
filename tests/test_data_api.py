"""Tests for data_api tools: DepMap, Open Targets, UniProt, PDB, Ensembl, NCBI, ChEMBL, DrugBank."""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code=200, json_data=None):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    return resp


class TestHttpRetryHelpers:
    @patch("time.sleep")
    @patch("httpx.get")
    def test_http_get_retries_on_transient_status(self, mock_get, mock_sleep):
        from ct.tools.data_api import _http_get

        mock_get.side_effect = [
            _mock_response(503, {}),
            _mock_response(200, {"ok": True}),
        ]

        resp = _http_get("https://example.org", timeout=5, retries=2)
        assert resp.status_code == 200
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("time.sleep")
    @patch("httpx.post")
    def test_http_post_retries_on_timeout(self, mock_post, mock_sleep):
        import httpx
        from ct.tools.data_api import _http_post

        mock_post.side_effect = [
            httpx.TimeoutException("timeout"),
            _mock_response(200, {"ok": True}),
        ]

        resp = _http_post("https://example.org", json={"q": 1}, timeout=5, retries=2)
        assert resp.status_code == 200
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()


# ===========================================================================
# 1. DepMap search
# ===========================================================================

class TestDepMapSearch:
    @patch("ct.data.loaders.load_crispr")
    def test_local_crispr_data(self, mock_crispr):
        import pandas as pd
        import numpy as np
        from ct.tools.data_api import depmap_search

        np.random.seed(42)
        n_lines = 100
        data = pd.DataFrame(
            np.random.randn(n_lines, 3) * 0.3,
            index=[f"LINE_{i}" for i in range(n_lines)],
            columns=["BRCA1", "TP53", "EGFR"],
        )
        # Make BRCA1 essential in some lines
        data.loc["LINE_0":"LINE_9", "BRCA1"] = -1.2
        mock_crispr.return_value = data

        result = depmap_search(gene="BRCA1", dataset="crispr")

        assert "summary" in result
        assert result["gene"] == "BRCA1"
        assert result["n_cell_lines"] == n_lines
        assert result["n_essential"] >= 10
        assert result["mean_score"] < 0

    @patch("ct.data.loaders.load_crispr")
    def test_gene_not_found(self, mock_crispr):
        import pandas as pd
        from ct.tools.data_api import depmap_search

        mock_crispr.return_value = pd.DataFrame(
            {"TP53": [0.1, -0.5]}, index=["L1", "L2"]
        )

        result = depmap_search(gene="FAKEGENE", dataset="crispr")
        assert "error" in result

    def test_invalid_dataset(self):
        from ct.tools.data_api import depmap_search

        result = depmap_search(gene="BRCA1", dataset="invalid")
        assert "error" in result
        assert "invalid" in result["error"].lower() or "Invalid" in result["error"]

    @patch("ct.data.loaders.load_crispr", side_effect=ImportError)
    @patch("httpx.get")
    def test_fallback_to_api(self, mock_get, mock_crispr):
        from ct.tools.data_api import depmap_search

        mock_get.return_value = _mock_response(200, {
            "data": [{"symbol": "BRCA1", "id": 672}]
        })

        result = depmap_search(gene="BRCA1", dataset="crispr")
        assert "summary" in result
        assert result["source"] == "cell_model_passports"

    @patch("ct.data.loaders.load_crispr", side_effect=ImportError)
    @patch("httpx.get")
    def test_api_timeout(self, mock_get, mock_crispr):
        import httpx
        from ct.tools.data_api import depmap_search

        mock_get.side_effect = httpx.TimeoutException("timeout")

        result = depmap_search(gene="BRCA1", dataset="crispr")
        assert "error" in result


# ===========================================================================
# 2. Open Targets search
# ===========================================================================

class TestOpenTargetsSearch:
    @patch("httpx.post")
    def test_target_search(self, mock_post):
        from ct.tools.data_api import opentargets_search

        # First call: search
        search_resp = _mock_response(200, {
            "data": {"search": {
                "total": 1,
                "hits": [{"id": "ENSG00000141510", "entity": "target", "name": "TP53", "description": "tumor protein"}]
            }}
        })
        # Second call: detail
        detail_resp = _mock_response(200, {
            "data": {"target": {
                "id": "ENSG00000141510",
                "approvedSymbol": "TP53",
                "approvedName": "Tumor protein p53",
                "biotype": "protein_coding",
                "functionDescriptions": ["DNA-binding transcription factor"],
                "subcellularLocations": [],
                "tractability": [{"label": "small molecule", "modality": "SM", "value": True}],
                "associatedDiseases": {
                    "count": 1234,
                    "rows": [
                        {"disease": {"id": "EFO1", "name": "lung carcinoma"}, "score": 0.89},
                        {"disease": {"id": "EFO2", "name": "breast carcinoma"}, "score": 0.85},
                    ]
                },
                "knownDrugs": {"uniqueDrugs": 5, "rows": []},
            }}
        })
        mock_post.side_effect = [search_resp, detail_resp]

        result = opentargets_search(query="TP53", entity_type="target")

        assert "summary" in result
        assert result["entity_type"] == "target"
        assert result["approved_symbol"] == "TP53"
        assert result["n_disease_associations"] == 1234
        assert len(result["top_diseases"]) == 2

    @patch("httpx.post")
    def test_disease_search(self, mock_post):
        from ct.tools.data_api import opentargets_search

        search_resp = _mock_response(200, {
            "data": {"search": {
                "total": 1,
                "hits": [{"id": "EFO_0001645", "entity": "disease", "name": "coronary artery disease", "description": ""}]
            }}
        })
        detail_resp = _mock_response(200, {
            "data": {"disease": {
                "id": "EFO_0001645",
                "name": "coronary artery disease",
                "description": "A disease of the heart.",
                "therapeuticAreas": [{"id": "ta1", "name": "cardiovascular disease"}],
                "associatedTargets": {
                    "count": 500,
                    "rows": [
                        {"target": {"id": "E1", "approvedSymbol": "PCSK9"}, "score": 0.95},
                    ]
                },
                "knownDrugs": {"uniqueDrugs": 10, "rows": []},
            }}
        })
        mock_post.side_effect = [search_resp, detail_resp]

        result = opentargets_search(query="coronary artery disease", entity_type="disease")

        assert result["entity_type"] == "disease"
        assert result["n_associated_targets"] == 500

    @patch("httpx.post")
    def test_drug_search(self, mock_post):
        from ct.tools.data_api import opentargets_search

        search_resp = _mock_response(200, {
            "data": {"search": {
                "total": 1,
                "hits": [{"id": "CHEMBL941", "entity": "drug", "name": "imatinib", "description": ""}]
            }}
        })
        detail_resp = _mock_response(200, {
            "data": {"drug": {
                "id": "CHEMBL941",
                "name": "imatinib",
                "drugType": "Small molecule",
                "maximumClinicalTrialPhase": 4,
                "hasBeenWithdrawn": False,
                "description": "BCR-ABL inhibitor",
                "mechanismsOfAction": {"rows": [
                    {"mechanismOfAction": "BCR-ABL inhibitor", "targets": [{"id": "E1", "approvedSymbol": "ABL1"}]},
                ]},
                "indications": {"count": 5, "rows": [
                    {"disease": {"id": "D1", "name": "CML"}, "maxPhaseForIndication": 4},
                ]},
            }}
        })
        mock_post.side_effect = [search_resp, detail_resp]

        result = opentargets_search(query="imatinib", entity_type="drug")

        assert result["entity_type"] == "drug"
        assert result["name"] == "imatinib"
        assert result["n_indications"] == 5

    @patch("httpx.post")
    def test_not_found(self, mock_post):
        from ct.tools.data_api import opentargets_search

        mock_post.return_value = _mock_response(200, {
            "data": {"search": {"total": 0, "hits": []}}
        })

        result = opentargets_search(query="xyznotreal", entity_type="target")
        assert "error" in result

    def test_invalid_entity_type(self):
        from ct.tools.data_api import opentargets_search

        result = opentargets_search(query="TP53", entity_type="bogus")
        assert "error" in result


# ===========================================================================
# 3. UniProt lookup
# ===========================================================================

class TestUniProtLookup:
    @patch("httpx.get")
    def test_accession_lookup(self, mock_get):
        from ct.tools.data_api import uniprot_lookup

        mock_get.return_value = _mock_response(200, {
            "primaryAccession": "P04637",
            "genes": [{"geneName": {"value": "TP53"}, "synonyms": []}],
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}
            },
            "sequence": {"length": 393, "value": "A" * 393},
            "comments": [
                {"commentType": "FUNCTION", "texts": [{"value": "DNA-binding transcription factor."}]},
                {"commentType": "SUBCELLULAR LOCATION", "subcellularLocations": [
                    {"location": {"value": "Nucleus"}},
                ]},
            ],
            "features": [
                {"type": "Domain", "description": "p53 tetramerization"},
                {"type": "Zinc finger", "description": "C2H2-type"},
            ],
            "uniProtKBCrossReferences": [
                {"database": "PDB", "id": "1TUP", "properties": []},
                {"database": "PDB", "id": "2XWR", "properties": []},
                {"database": "GO", "id": "GO:0003700", "properties": [
                    {"key": "GoTerm", "value": "F:DNA-binding transcription factor activity"},
                    {"key": "GoEvidenceType", "value": "IDA"},
                ]},
            ],
            "keywords": [{"name": "Tumor suppressor"}, {"name": "Transcription"}],
        })

        result = uniprot_lookup(query="P04637")

        assert "summary" in result
        assert result["accession"] == "P04637"
        assert "TP53" in result["gene_names"]
        assert result["sequence_length"] == 393
        assert result["n_pdb_structures"] == 2
        assert len(result["go_terms"]) == 1

    @patch("httpx.get")
    def test_gene_symbol_search(self, mock_get):
        from ct.tools.data_api import uniprot_lookup

        # First call: direct lookup fails (not an accession)
        # Second call: search succeeds
        search_resp = _mock_response(200, {
            "results": [{
                "primaryAccession": "P04637",
                "genes": [{"geneName": {"value": "TP53"}, "synonyms": []}],
                "proteinDescription": {
                    "recommendedName": {"fullName": {"value": "Cellular tumor antigen p53"}}
                },
                "sequence": {"length": 393, "value": "A" * 393},
                "comments": [],
                "features": [],
                "uniProtKBCrossReferences": [],
                "keywords": [],
            }]
        })
        mock_get.return_value = search_resp

        result = uniprot_lookup(query="TP53", organism="human")

        assert "summary" in result
        assert result["accession"] == "P04637"

    @patch("httpx.get")
    def test_not_found(self, mock_get):
        from ct.tools.data_api import uniprot_lookup

        mock_get.return_value = _mock_response(200, {"results": []})

        result = uniprot_lookup(query="XYZFAKEGENE")
        assert "error" in result

    @patch("httpx.get")
    def test_timeout(self, mock_get):
        import httpx
        from ct.tools.data_api import uniprot_lookup

        mock_get.side_effect = httpx.TimeoutException("timeout")

        result = uniprot_lookup(query="TP53")
        assert "error" in result

    @patch("httpx.get")
    def test_non_human_query_uses_species_fallback_candidates(self, mock_get):
        from ct.tools.data_api import uniprot_lookup

        def _side_effect(url, **kwargs):
            params = kwargs.get("params", {}) or {}
            q = params.get("query", "")
            if "organism_name:\"Heligmosomoides polygyrus\"" in q:
                return _mock_response(
                    200,
                    {
                        "results": [
                            {
                                "primaryAccession": "A0A0X0HP01",
                                "genes": [{"geneName": {"value": "hp_1234"}, "synonyms": []}],
                                "proteinDescription": {
                                    "recommendedName": {"fullName": {"value": "Secreted protein"}}
                                },
                                "sequence": {"length": 287, "value": "A" * 287},
                                "comments": [],
                                "features": [],
                                "uniProtKBCrossReferences": [],
                                "keywords": [{"name": "Secreted"}],
                            }
                        ]
                    },
                )
            return _mock_response(200, {"results": []})

        mock_get.side_effect = _side_effect

        result = uniprot_lookup(query="Heligmosomoides polygyrus secreted")
        assert "summary" in result
        assert result["accession"] == "A0A0X0HP01"
        assert "organism_name:\"Heligmosomoides polygyrus\"" in result.get("matched_query", "")

    @patch("httpx.get")
    def test_uniprot_lookup_organism_any_disables_organism_filter(self, mock_get):
        from ct.tools.data_api import uniprot_lookup

        captured = {}

        def _side_effect(url, **kwargs):
            params = kwargs.get("params", {}) or {}
            captured["query"] = params.get("query", "")
            return _mock_response(200, {"results": []})

        mock_get.side_effect = _side_effect
        result = uniprot_lookup(query="Fasciola hepatica secreted", organism="any")
        assert "error" in result
        assert "organism_id:9606" not in captured.get("query", "")

    @patch("httpx.get")
    def test_non_human_query_rejects_human_only_hits(self, mock_get):
        from ct.tools.data_api import uniprot_lookup

        mock_get.return_value = _mock_response(
            200,
            {
                "results": [
                    {
                        "primaryAccession": "O75900",
                        "organism": {"scientificName": "Homo sapiens"},
                        "genes": [{"geneName": {"value": "MMP23B"}, "synonyms": []}],
                        "proteinDescription": {
                            "recommendedName": {"fullName": {"value": "Matrix metalloproteinase-23"}}
                        },
                        "sequence": {"length": 390, "value": "A" * 390},
                        "comments": [],
                        "features": [],
                        "uniProtKBCrossReferences": [],
                        "keywords": [],
                    }
                ]
            },
        )

        result = uniprot_lookup(query="SCP TAPS venom allergen-like helminth")
        assert "error" in result
        assert "non-human" in result["summary"].lower()


# ===========================================================================
# 4. PDB search
# ===========================================================================

class TestPDBSearch:
    @patch("httpx.get")
    @patch("httpx.post")
    def test_text_search(self, mock_post, mock_get):
        from ct.tools.data_api import pdb_search

        mock_post.return_value = _mock_response(200, {
            "total_count": 156,
            "result_set": [
                {"identifier": "4HJO", "score": 1.0},
                {"identifier": "1M17", "score": 0.9},
            ]
        })
        # Detail fetches for each PDB ID
        detail_data = {
            "struct": {"title": "Crystal structure of EGFR"},
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "reflns": [{"d_resolution_high": 1.5}],
            "rcsb_entry_info": {"resolution_combined": [1.5]},
            "rcsb_accession_info": {"deposit_date": "2023-01-01"},
            "rcsb_entry_container_identifiers": {"non_polymer_entity_ids": ["1"]},
        }
        mock_get.return_value = _mock_response(200, detail_data)

        result = pdb_search(query="EGFR kinase")

        assert "summary" in result
        assert result["total_count"] == 156
        assert len(result["structures"]) == 2
        assert result["best_resolution"] == 1.5
        assert result["best_pdb_id"] == "4HJO"

    @patch("httpx.get")
    def test_direct_pdb_id(self, mock_get):
        from ct.tools.data_api import pdb_search

        mock_get.return_value = _mock_response(200, {
            "struct": {"title": "Crystal structure of EGFR"},
            "exptl": [{"method": "X-RAY DIFFRACTION"}],
            "reflns": [{"d_resolution_high": 2.0}],
            "rcsb_entry_info": {},
            "rcsb_accession_info": {"deposit_date": "2023-01-01"},
        })

        result = pdb_search(query="4HJO")

        assert "summary" in result
        assert result["pdb_id"] == "4HJO"
        assert result["resolution"] == 2.0

    @patch("httpx.post")
    def test_no_results(self, mock_post):
        from ct.tools.data_api import pdb_search

        mock_post.return_value = _mock_response(200, {
            "total_count": 0,
            "result_set": [],
        })

        result = pdb_search(query="nonexistentprotein12345")
        assert result["total_count"] == 0

    @patch("httpx.post")
    def test_search_timeout(self, mock_post):
        import httpx
        from ct.tools.data_api import pdb_search

        mock_post.side_effect = httpx.TimeoutException("timeout")

        result = pdb_search(query="EGFR")
        assert "error" in result


# ===========================================================================
# 5. Ensembl lookup
# ===========================================================================

class TestEnsemblLookup:
    @patch("httpx.get")
    def test_gene_symbol_lookup(self, mock_get):
        from ct.tools.data_api import ensembl_lookup

        # First call: gene lookup
        gene_resp = _mock_response(200, {
            "id": "ENSG00000012048",
            "display_name": "BRCA1",
            "description": "BRCA1 DNA repair associated",
            "biotype": "protein_coding",
            "seq_region_name": "17",
            "start": 43044295,
            "end": 43125483,
            "strand": -1,
            "Transcript": [
                {"id": "ENST00000357654", "display_name": "BRCA1-201", "biotype": "protein_coding", "is_canonical": 1, "length": 7088},
                {"id": "ENST00000471181", "display_name": "BRCA1-202", "biotype": "processed_transcript", "is_canonical": 0, "length": 2000},
            ],
        })
        # Second call: xrefs
        xref_resp = _mock_response(200, [
            {"dbname": "Uniprot/SWISSPROT", "primary_id": "P38398", "display_id": "BRCA1_HUMAN"},
        ])
        mock_get.side_effect = [gene_resp, xref_resp]

        result = ensembl_lookup(gene="BRCA1")

        assert "summary" in result
        assert result["ensembl_id"] == "ENSG00000012048"
        assert result["biotype"] == "protein_coding"
        assert result["chromosome"] == "17"
        assert result["n_transcripts"] == 2
        assert result["strand"] == -1

    @patch("httpx.get")
    def test_ensembl_id_lookup(self, mock_get):
        from ct.tools.data_api import ensembl_lookup

        gene_resp = _mock_response(200, {
            "id": "ENSG00000012048",
            "display_name": "BRCA1",
            "description": "BRCA1 DNA repair associated",
            "biotype": "protein_coding",
            "seq_region_name": "17",
            "start": 43044295,
            "end": 43125483,
            "strand": -1,
            "Transcript": [],
        })
        xref_resp = _mock_response(200, [])
        mock_get.side_effect = [gene_resp, xref_resp]

        result = ensembl_lookup(gene="ENSG00000012048")
        assert result["ensembl_id"] == "ENSG00000012048"

    @patch("httpx.get")
    def test_gene_not_found(self, mock_get):
        from ct.tools.data_api import ensembl_lookup

        mock_get.return_value = _mock_response(400, {})

        result = ensembl_lookup(gene="FAKEGENE123")
        assert "error" in result

    @patch("httpx.get")
    def test_timeout(self, mock_get):
        import httpx
        from ct.tools.data_api import ensembl_lookup

        mock_get.side_effect = httpx.TimeoutException("timeout")

        result = ensembl_lookup(gene="BRCA1")
        assert "error" in result


# ===========================================================================
# 6. NCBI Gene
# ===========================================================================

class TestNCBIGene:
    @patch("httpx.get")
    def test_gene_search(self, mock_get):
        from ct.tools.data_api import ncbi_gene

        search_resp = _mock_response(200, {
            "esearchresult": {"count": "1", "idlist": ["672"]}
        })
        summary_resp = _mock_response(200, {
            "result": {
                "uids": ["672"],
                "672": {
                    "name": "BRCA1",
                    "description": "BRCA1 DNA repair associated",
                    "chromosome": "17",
                    "organism": {"scientificname": "Homo sapiens"},
                    "otheraliases": "BRCC1, IRIS, PNCA4, PPP1R53, PSCP, RNF53",
                    "summary": "This gene encodes a nuclear phosphoprotein...",
                    "geneticSource": "genomic",
                    "maplocation": "17q21.31",
                },
            }
        })
        mock_get.side_effect = [search_resp, summary_resp]

        result = ncbi_gene(query="BRCA1", database="gene")

        assert "summary" in result
        assert result["total_count"] == 1
        assert result["genes"][0]["symbol"] == "BRCA1"
        assert result["genes"][0]["chromosome"] == "17"

    @patch("httpx.get")
    def test_clinvar_search(self, mock_get):
        from ct.tools.data_api import ncbi_gene

        search_resp = _mock_response(200, {
            "esearchresult": {"count": "500", "idlist": ["12345", "12346"]}
        })
        summary_resp = _mock_response(200, {
            "result": {
                "uids": ["12345", "12346"],
                "12345": {
                    "title": "NM_007294.4(BRCA1):c.68_69del (p.Glu23Valfs)",
                    "clinical_significance": {"description": "Pathogenic"},
                    "gene_sort": "BRCA1",
                    "variation_set": [],
                    "obj_type": "single nucleotide variant",
                },
                "12346": {
                    "title": "NM_007294.4(BRCA1):c.5123C>A",
                    "clinical_significance": {"description": "Uncertain significance"},
                    "gene_sort": "BRCA1",
                    "variation_set": [],
                    "obj_type": "single nucleotide variant",
                },
            }
        })
        mock_get.side_effect = [search_resp, summary_resp]

        result = ncbi_gene(query="BRCA1", database="clinvar")

        assert result["database"] == "clinvar"
        assert result["total_count"] == 500
        assert len(result["variants"]) == 2

    @patch("httpx.get")
    def test_no_results(self, mock_get):
        from ct.tools.data_api import ncbi_gene

        mock_get.return_value = _mock_response(200, {
            "esearchresult": {"count": "0", "idlist": []}
        })

        result = ncbi_gene(query="XYZNOTREAL")
        assert result["total_count"] == 0

    def test_invalid_database(self):
        from ct.tools.data_api import ncbi_gene

        result = ncbi_gene(query="BRCA1", database="invalid")
        assert "error" in result

    @patch("httpx.get")
    def test_timeout(self, mock_get):
        import httpx
        from ct.tools.data_api import ncbi_gene

        mock_get.side_effect = httpx.TimeoutException("timeout")

        result = ncbi_gene(query="BRCA1")
        assert "error" in result


# ===========================================================================
# 7. ChEMBL advanced
# ===========================================================================

class TestChEMBLAdvanced:
    @patch("httpx.get")
    def test_compound_search(self, mock_get):
        from ct.tools.data_api import chembl_advanced

        mock_get.return_value = _mock_response(200, {
            "molecules": [{
                "molecule_chembl_id": "CHEMBL941",
                "pref_name": "IMATINIB",
                "molecule_type": "Small molecule",
                "max_phase": 4,
                "oral": True,
                "parenteral": False,
                "topical": False,
                "natural_product": -1,
                "molecule_structures": {
                    "canonical_smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
                    "standard_inchi_key": "KTUFNOKKBVMGRW-UHFFFAOYSA-N",
                },
                "molecule_properties": {
                    "full_mwt": "493.60",
                    "alogp": "3.50",
                    "hba": 7,
                    "hbd": 2,
                    "psa": "86.28",
                    "rtb": 7,
                    "num_ro5_violations": 0,
                    "aromatic_rings": 4,
                    "heavy_atoms": 37,
                    "qed_weighted": "0.70",
                },
            }]
        })

        result = chembl_advanced(query="imatinib", search_type="compound")

        assert "summary" in result
        assert result["compounds"][0]["chembl_id"] == "CHEMBL941"
        assert result["compounds"][0]["molecular_weight"] == "493.60"

    @patch("httpx.get")
    def test_target_activities(self, mock_get):
        from ct.tools.data_api import chembl_advanced

        # First call: target search
        target_resp = _mock_response(200, {
            "targets": [{
                "target_chembl_id": "CHEMBL203",
                "pref_name": "Epidermal growth factor receptor erbB1",
                "organism": "Homo sapiens",
                "target_type": "SINGLE PROTEIN",
            }]
        })
        # Second call: activities
        activity_resp = _mock_response(200, {
            "activities": [
                {"molecule_chembl_id": "CHEMBL1", "standard_type": "IC50", "standard_value": "50", "standard_units": "nM", "pchembl_value": "7.3"},
                {"molecule_chembl_id": "CHEMBL2", "standard_type": "IC50", "standard_value": "120", "standard_units": "nM", "pchembl_value": "6.9"},
                {"molecule_chembl_id": "CHEMBL3", "standard_type": "Ki", "standard_value": "30", "standard_units": "nM", "pchembl_value": "7.5"},
            ]
        })
        mock_get.side_effect = [target_resp, activity_resp]

        result = chembl_advanced(query="EGFR", search_type="target_activities")

        assert "summary" in result
        assert result["target_chembl_id"] == "CHEMBL203"
        assert result["n_unique_compounds"] == 3
        assert "IC50" in result["activity_statistics"]
        assert result["activity_statistics"]["IC50"]["median_nM"] == 85.0  # median of 50, 120

    @patch("httpx.get")
    def test_mechanism_search(self, mock_get):
        from ct.tools.data_api import chembl_advanced

        mock_get.return_value = _mock_response(200, {
            "mechanisms": [{
                "mechanism_of_action": "BCR-ABL protein kinase inhibitor",
                "action_type": "INHIBITOR",
                "target_chembl_id": "CHEMBL1862",
                "molecule_chembl_id": "CHEMBL941",
                "max_phase": 4,
                "direct_interaction": True,
            }]
        })

        result = chembl_advanced(query="CHEMBL941", search_type="mechanism")

        assert "summary" in result
        assert result["n_mechanisms"] == 1
        assert "BCR-ABL" in result["mechanisms"][0]["mechanism"]

    @patch("httpx.get")
    def test_drug_indication(self, mock_get):
        from ct.tools.data_api import chembl_advanced

        # First call: molecule search (resolve name to ID)
        mol_resp = _mock_response(200, {
            "molecules": [{"molecule_chembl_id": "CHEMBL941"}]
        })
        # Second call: drug indications
        ind_resp = _mock_response(200, {
            "drug_indications": [
                {"mesh_heading": "Chronic Myeloid Leukemia", "mesh_id": "D015464", "efo_id": "EFO1", "max_phase_for_ind": 4, "molecule_chembl_id": "CHEMBL941"},
                {"mesh_heading": "GIST", "mesh_id": "D046152", "efo_id": "EFO2", "max_phase_for_ind": 4, "molecule_chembl_id": "CHEMBL941"},
            ]
        })
        mock_get.side_effect = [mol_resp, ind_resp]

        result = chembl_advanced(query="imatinib", search_type="drug_indication")

        assert "summary" in result
        assert result["n_indications"] == 2
        assert result["n_approved"] == 2

    def test_invalid_search_type(self):
        from ct.tools.data_api import chembl_advanced

        result = chembl_advanced(query="imatinib", search_type="bogus")
        assert "error" in result


# ===========================================================================
# 8. DrugBank lookup
# ===========================================================================

class TestDrugInfo:
    @patch("httpx.get")
    def test_drug_lookup(self, mock_get):
        from ct.tools.data_api import drug_info

        # CID lookup
        cid_resp = _mock_response(200, {
            "IdentifierList": {"CID": [5291]}
        })
        # Properties
        props_resp = _mock_response(200, {
            "PropertyTable": {"Properties": [{
                "CID": 5291,
                "MolecularFormula": "C29H31N7O",
                "MolecularWeight": 493.6,
                "CanonicalSMILES": "CC1=C...",
                "IsomericSMILES": "CC1=C...",
                "XLogP": 3.5,
                "TPSA": 86.3,
                "HBondDonorCount": 2,
                "HBondAcceptorCount": 7,
                "RotatableBondCount": 7,
                "InChIKey": "KTUFNOKKBVMGRW-UHFFFAOYSA-N",
            }]}
        })
        # PUG View
        view_resp = _mock_response(200, {
            "Record": {"Section": [{
                "TOCHeading": "Drug and Medication Information",
                "Section": [
                    {
                        "TOCHeading": "Mechanism of Action",
                        "Information": [{"Value": {"StringWithMarkup": [
                            {"String": "BCR-ABL tyrosine kinase inhibitor"}
                        ]}}],
                    },
                    {
                        "TOCHeading": "Drug Indication",
                        "Information": [{"Value": {"StringWithMarkup": [
                            {"String": "Chronic myeloid leukemia"}
                        ]}}],
                    },
                    {
                        "TOCHeading": "Drug-Drug Interactions",
                        "Information": [
                            {"Value": {"StringWithMarkup": [{"String": "Interaction with CYP3A4 inhibitors"}]}},
                            {"Value": {"StringWithMarkup": [{"String": "Interaction with warfarin"}]}},
                        ],
                    },
                ],
            }]}
        })
        # Synonyms
        syn_resp = _mock_response(200, {
            "InformationList": {"Information": [{
                "Synonym": ["Imatinib", "Gleevec", "STI-571", "DB00619", "CGP 57148"]
            }]}
        })

        mock_get.side_effect = [cid_resp, props_resp, view_resp, syn_resp]

        result = drug_info(query="imatinib")

        assert "summary" in result
        assert result["cid"] == 5291
        assert result["drugbank_id"] == "DB00619"
        assert result["properties"]["molecular_weight"] == 493.6
        assert result["pharmacology"]["mechanism_of_action"] == "BCR-ABL tyrosine kinase inhibitor"
        assert result["n_interactions"] == 2

    @patch("httpx.get")
    def test_drug_not_found(self, mock_get):
        from ct.tools.data_api import drug_info

        mock_get.return_value = _mock_response(404, {})

        result = drug_info(query="xyznotadrug")
        assert "error" in result

    @patch("httpx.get")
    def test_timeout(self, mock_get):
        import httpx
        from ct.tools.data_api import drug_info

        mock_get.side_effect = httpx.TimeoutException("timeout")

        result = drug_info(query="imatinib")
        assert "error" in result

    @patch("httpx.get")
    def test_no_cids(self, mock_get):
        from ct.tools.data_api import drug_info

        mock_get.return_value = _mock_response(200, {
            "IdentifierList": {"CID": []}
        })

        result = drug_info(query="unknowndrug")
        assert "error" in result

    @patch("httpx.get")
    def test_combined_synonym_query_resolves_with_fallback_candidate(self, mock_get):
        from ct.tools.data_api import drug_info

        first_404 = _mock_response(404, {})
        cid_resp = _mock_response(200, {"IdentifierList": {"CID": [12345]}})
        props_resp = _mock_response(200, {
            "PropertyTable": {"Properties": [{
                "CID": 12345,
                "MolecularFormula": "C20H20N2O2",
                "MolecularWeight": 320.0,
                "CanonicalSMILES": "CCN",
                "InChIKey": "AAAA-BBBB-CCCC",
            }]}
        })
        view_resp = _mock_response(200, {"Record": {"Section": []}})
        syn_resp = _mock_response(200, {"InformationList": {"Information": [{"Synonym": ["Silmitasertib"]}]}})

        mock_get.side_effect = [first_404, cid_resp, props_resp, view_resp, syn_resp]

        result = drug_info(query="silmitasertib CX-4945")
        assert "error" not in result
        assert result["resolved_query"] == "silmitasertib"
        assert result["cid"] == 12345


# ===========================================================================
# Registration tests
# ===========================================================================

class TestDataAPIRegistration:
    """Verify data_api tools are registered in the correct category."""

    def test_all_tools_registered(self):
        from ct.tools import registry, ensure_loaded
        ensure_loaded()

        expected_tools = [
            "data_api.depmap_search",
            "data_api.opentargets_search",
            "data_api.uniprot_lookup",
            "data_api.pdb_search",
            "data_api.ensembl_lookup",
            "data_api.ncbi_gene",
            "data_api.chembl_advanced",
            "data_api.drug_info",
            "data_api.mygene_lookup",
            "data_api.mydisease_lookup",
            "data_api.myvariant_lookup",
            "data_api.mytaxon_lookup",
            "data_api.mychem_lookup",
            "data_api.pdbe_search",
            "data_api.reactome_pathway_search",
        ]
        for tool_name in expected_tools:
            tool = registry.get_tool(tool_name)
            assert tool is not None, f"Tool {tool_name} not registered"
            assert tool.category == "data_api"
            assert "summary" in tool.description.lower() or tool.description  # Has description

    def test_data_api_category_exists(self):
        from ct.tools import registry, ensure_loaded
        ensure_loaded()

        categories = registry.categories()
        assert "data_api" in categories

    def test_tool_count(self):
        from ct.tools import registry, ensure_loaded
        ensure_loaded()

        tools = registry.list_tools(category="data_api")
        assert len(tools) >= 15
