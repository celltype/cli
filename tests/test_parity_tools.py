"""Tests for clean-room parity API/tool additions."""

from __future__ import annotations

from unittest.mock import patch

from ct.tools.parity import (
    mygene_lookup,
    mydisease_lookup,
    myvariant_lookup,
    mytaxon_lookup,
    mychem_lookup,
    pdbe_search,
    reactome_pathway_search,
    preprint_search,
    sa_score,
)


def test_mygene_lookup_success():
    payload = {
        "hits": [
            {
                "symbol": "TP53",
                "name": "tumor protein p53",
                "entrezgene": 7157,
                "ensembl": {"gene": "ENSG00000141510"},
                "taxid": 9606,
                "type_of_gene": "protein-coding",
                "_score": 99.0,
            }
        ]
    }
    with patch("ct.tools.parity.request_json", return_value=(payload, None)):
        result = mygene_lookup("TP53")

    assert result["count"] == 1
    assert result["hits"][0]["symbol"] == "TP53"


def test_mygene_lookup_species_phrase_normalization():
    payload = {"hits": []}

    def _fake_request_json(method, url, params=None, **kwargs):
        del method, url, kwargs
        # Schistosoma mansoni should normalize to taxid 6183.
        assert params["species"] == "6183"
        return payload, None

    with patch("ct.tools.parity.request_json", side_effect=_fake_request_json):
        result = mygene_lookup("SCP/TAPS", species="Schistosoma mansoni genes")

    assert result["species"] == "6183"


def test_mydisease_lookup_missing_query():
    result = mydisease_lookup("")
    assert result["error"] == "missing_query"


def test_myvariant_lookup_api_error():
    with patch("ct.tools.parity.request_json", return_value=(None, "boom")):
        result = myvariant_lookup("rs1")
    assert result["error"] == "api_error"


def test_mytaxon_lookup_success():
    payload = {"hits": [{"_id": 9606, "scientific_name": "Homo sapiens", "rank": "species", "_score": 1.2}]}
    with patch("ct.tools.parity.request_json", return_value=(payload, None)):
        result = mytaxon_lookup("human")
    assert result["count"] == 1
    assert result["hits"][0]["taxid"] == 9606


def test_mychem_lookup_success():
    payload = {
        "hits": [
            {
                "_id": "CHEMBL25",
                "name": "Aspirin",
                "chembl": {"molecule_chembl_id": "CHEMBL25"},
                "drugbank": {"id": "DB00945"},
                "inchi_key": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "_score": 12.3,
            }
        ]
    }
    with patch("ct.tools.parity.request_json", return_value=(payload, None)):
        result = mychem_lookup("aspirin")
    assert result["count"] == 1
    assert result["hits"][0]["chembl_id"] == "CHEMBL25"


def test_pdbe_search_success():
    payload = {
        "response": {
            "docs": [
                {
                    "pdb_id": "1TUP",
                    "title": "Tumor suppressor p53",
                    "experimental_method": "X-ray diffraction",
                    "resolution": 2.1,
                    "organism_scientific_name": "Homo sapiens",
                }
            ]
        }
    }
    with patch("ct.tools.parity.request_json", return_value=(payload, None)):
        result = pdbe_search("p53")
    assert result["count"] == 1
    assert result["entries"][0]["pdb_id"] == "1TUP"


def test_reactome_pathway_search_success():
    payload = [{"stId": "R-HSA-123", "name": "DNA Repair", "species": "Homo sapiens", "type": "Pathway"}]
    with patch("ct.tools.parity.request_json", return_value=(payload, None)):
        result = reactome_pathway_search("DNA repair")
    assert result["count"] == 1
    assert result["pathways"][0]["st_id"] == "R-HSA-123"


def test_preprint_search_both_sources():
    epmc_payload = {
        "resultList": {
            "result": [
                {
                    "id": "PPR123",
                    "title": "AML preprint",
                    "authorString": "Doe J",
                    "journalTitle": "bioRxiv",
                    "pubYear": "2025",
                    "doi": "10.1101/2025.01.01.123456",
                }
            ]
        }
    }

    class _Resp:
        status_code = 200
        text = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>Arxiv preprint</title>
    <published>2025-01-02T00:00:00Z</published>
    <author><name>Alice</name></author>
  </entry>
</feed>"""

    with patch("ct.tools.parity.request_json", return_value=(epmc_payload, None)), patch(
        "ct.tools.parity.request", return_value=(_Resp(), None)
    ):
        result = preprint_search("AML", source="both", max_results=5)

    assert result["count"] >= 2


def test_preprint_search_invalid_source():
    result = preprint_search("x", source="bad")
    assert result["error"] == "invalid_source"


def test_sa_score_returns_result_or_dependency_error():
    result = sa_score("CCO")
    assert "summary" in result
    if "error" in result:
        assert result["error"] == "missing_dependency"
    else:
        assert "sa_score" in result
