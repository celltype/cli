"""Regression smoke tests for a Parkinson target-finding toolchain.

This test intentionally patches `httpx.get` with a strict signature check to
ensure GET requests never pass unsupported kwargs like `json`/`data`.
"""

from __future__ import annotations

from unittest.mock import patch

from ct.tools.clinical import trial_search
from ct.tools.genomics import gwas_lookup
from ct.tools.literature import pubmed_search
from ct.tools.target import druggability, expression_profile


class _Resp:
    def __init__(self, status_code: int = 200, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def _strict_httpx_get(url: str, **kwargs):
    # Core regression check: GET must never receive json/data kwargs.
    if "json" in kwargs:
        raise TypeError("get() got an unexpected keyword argument 'json'")
    if "data" in kwargs:
        raise TypeError("get() got an unexpected keyword argument 'data'")

    params = kwargs.get("params") or {}

    # genomics.gwas_lookup
    if "gwas/rest/api/singleNucleotidePolymorphisms/search/findByGene" in url:
        return _Resp(
            200,
            {
                "_embedded": {
                    "singleNucleotidePolymorphisms": [
                        {"rsId": "rs123"},
                        {"rsId": "rs456"},
                    ]
                }
            },
        )
    if "/gwas/rest/api/singleNucleotidePolymorphisms/" in url and url.endswith("/associations"):
        return _Resp(
            200,
            {
                "_embedded": {
                    "associations": [
                        {
                            "pvalueMantissa": 1.2,
                            "pvalueExponent": -9,
                            "loci": [{"strongestRiskAlleles": [{"riskAlleleName": "rs123-A"}]}],
                            "efoTraits": [{"trait": "Parkinson disease"}],
                            "orPerCopyNum": 1.15,
                            "betaNum": 0.21,
                            "betaUnit": "SD",
                            "betaDirection": "increase",
                        }
                    ]
                }
            },
        )

    # literature.pubmed_search
    if "eutils.ncbi.nlm.nih.gov" in url and url.endswith("/esearch.fcgi"):
        return _Resp(200, {"esearchresult": {"count": "1", "idlist": ["12345678"]}})
    if "eutils.ncbi.nlm.nih.gov" in url and url.endswith("/esummary.fcgi"):
        return _Resp(
            200,
            {
                "result": {
                    "uids": ["12345678"],
                    "12345678": {
                        "title": "Parkinson target discovery",
                        "authors": [{"name": "Doe J"}],
                        "source": "Nature",
                        "pubdate": "2025",
                        "articleids": [{"idtype": "doi", "value": "10.1000/test"}],
                    },
                }
            },
        )

    # clinical.trial_search
    if "clinicaltrials.gov/api/v2/studies" in url:
        return _Resp(
            200,
            {
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {
                                "nctId": "NCT00000001",
                                "briefTitle": "LRRK2 inhibitor in Parkinson disease",
                            },
                            "statusModule": {
                                "overallStatus": "RECRUITING",
                                "startDateStruct": {"date": "2025-01-01"},
                            },
                            "designModule": {
                                "phases": ["PHASE2"],
                                "enrollmentInfo": {"count": 120},
                            },
                            "conditionsModule": {"conditions": ["Parkinson Disease"]},
                            "armsInterventionsModule": {
                                "interventions": [{"type": "DRUG", "name": "DNL151"}]
                            },
                            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "BioCo"}},
                        }
                    }
                ]
            },
        )

    # target.druggability
    if "rest.uniprot.org/uniprotkb/search" in url:
        gene_query = str(params.get("query", ""))
        if "LRRK2" in gene_query:
            gene = "LRRK2"
        else:
            gene = "GENE"
        return _Resp(
            200,
            {
                "results": [
                    {
                        "primaryAccession": "Q5S007",
                        "proteinDescription": {
                            "recommendedName": {"fullName": {"value": f"{gene} protein"}}
                        },
                        "features": [{"type": "Domain", "description": "Protein kinase domain"}],
                        "keywords": [{"name": "Kinase"}],
                        "comments": [
                            {
                                "commentType": "SUBCELLULAR LOCATION",
                                "subcellularLocations": [{"location": {"value": "Cell membrane"}}],
                            }
                        ],
                        "uniProtKBCrossReferences": [
                            {"database": "ChEMBL", "id": "CHEMBL123"},
                            {"database": "PDB", "id": "7LI3"},
                        ],
                    }
                ]
            },
        )

    # target.expression_profile
    if "gtexportal.org/api/v2/reference/gene" in url:
        return _Resp(200, {"data": [{"gencodeId": "ENSG00000145335.1", "geneSymbol": "SNCA"}]})
    if "gtexportal.org/api/v2/expression/medianGeneExpression" in url:
        return _Resp(
            200,
            {
                "data": [
                    {"tissueSiteDetailId": "Brain_Cortex", "median": 15.1},
                    {"tissueSiteDetailId": "Substantia_nigra", "median": 20.2},
                    {"tissueSiteDetailId": "Blood", "median": 1.0},
                ]
            },
        )
    if "proteinatlas.org" in url and url.endswith(".json"):
        return _Resp(
            200,
            {
                "RNATissue": {
                    "summary": "Tissue enhanced (brain)",
                    "data": [{"Tissue": "brain", "TPM": 18.0, "nTPM": 16.0}],
                },
                "ProteinTissue": {"data": [{"Tissue": "brain", "Level": "High", "CellType": "Neuron"}]},
                "RNACancer": {"data": [{"Cancer": "Glioma", "TPM": 8.0, "nTPM": 7.0}]},
                "RNASingleCell": {"data": [{"CellType": "Neuron", "nTPM": 22.0}]},
            },
        )

    return _Resp(404, {"error": f"Unhandled URL in test router: {url}"}, text="not found")


def test_parkinsons_toolchain_smoke_under_strict_http_get_contract():
    with patch("httpx.get", side_effect=_strict_httpx_get):
        gwas = gwas_lookup(gene="SNCA", trait="Parkinson disease")
        assert "unexpected keyword argument" not in str(gwas)
        assert gwas.get("n_associations", 0) >= 1

        pubs = pubmed_search(query="Parkinson disease target discovery", max_results=5)
        assert "unexpected keyword argument" not in str(pubs)
        assert pubs.get("total_count", 0) >= 1

        trials = trial_search(query="Parkinson disease")
        assert "unexpected keyword argument" not in str(trials)
        assert trials.get("total_count", 0) >= 1

        drug = druggability(gene="LRRK2")
        assert "unexpected keyword argument" not in str(drug)
        assert "summary" in drug

        expr = expression_profile(gene="SNCA", top_n=3)
        assert "unexpected keyword argument" not in str(expr)
        assert "summary" in expr
