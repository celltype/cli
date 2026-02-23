"""Tests for omics data discovery, download, inspection, and analysis tools."""

import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from ct.tools.omics import (
    _downloads_dir,
    _max_download_mb,
    _stream_download,
    geo_search,
    geo_fetch,
    cellxgene_search,
    cellxgene_fetch,
    tcga_search,
    tcga_fetch,
    dataset_info,
    methylation_diff,
    methylation_profile,
    proteomics_diff,
    proteomics_enrich,
    atac_peak_annotate,
    chromatin_accessibility,
    chipseq_enrich,
    spatial_cluster,
    spatial_autocorrelation,
    cytof_cluster,
    hic_compartments,
    deseq2,
    multiomics_integrate,
    methylation_cluster,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_downloads(tmp_path):
    """Patch _downloads_dir to use a temp directory."""
    with patch("ct.tools.omics._downloads_dir", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def mock_config():
    """Patch Config.load to return a mock config."""
    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda key, default=None: {
        "data.downloads_dir": None,
        "data.max_download_mb": 500,
    }.get(key, default)
    with patch("ct.agent.config.Config.load", return_value=mock_cfg):
        yield mock_cfg


# ---------------------------------------------------------------------------
# Shared helper tests
# ---------------------------------------------------------------------------


class TestDownloadsDir:
    def test_creates_directory(self, mock_config, tmp_path):
        mock_config.get.side_effect = lambda key, default=None: {
            "data.downloads_dir": str(tmp_path / "custom_downloads"),
        }.get(key, default)
        d = _downloads_dir()
        assert d.exists()
        assert d == tmp_path / "custom_downloads"

    def test_default_path(self, mock_config):
        d = _downloads_dir()
        assert d == Path.home() / ".ct" / "downloads"


class TestMaxDownloadMb:
    def test_reads_config(self, mock_config):
        assert _max_download_mb() == 500

    def test_custom_value(self, mock_config):
        mock_config.get.side_effect = lambda key, default=None: {
            "data.max_download_mb": 1000,
        }.get(key, default)
        assert _max_download_mb() == 1000


class TestStreamDownload:
    @patch("ct.tools.omics._max_download_mb", return_value=1)
    def test_size_cap_from_content_length(self, mock_max, tmp_path):
        """Content-Length header exceeding limit returns error before download."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(10 * 1024 * 1024)}  # 10 MB
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes = MagicMock(return_value=iter([]))

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)

            dest = tmp_path / "test.gz"
            path, error = _stream_download("https://example.com/file.gz", dest, max_mb=1)
            assert path is None
            assert "exceeds limit" in error

    @patch("ct.tools.omics._max_download_mb", return_value=500)
    def test_successful_download(self, mock_max, tmp_path):
        """Successful download writes file and renames."""
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "100"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_bytes = MagicMock(return_value=iter([b"test data"]))

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)

            dest = tmp_path / "test.txt"
            path, error = _stream_download("https://example.com/file.txt", dest, max_mb=500)
            assert error is None
            assert path == dest
            assert dest.read_bytes() == b"test data"


# ---------------------------------------------------------------------------
# geo_search tests
# ---------------------------------------------------------------------------


class TestGeoSearch:
    def test_empty_query(self):
        result = geo_search("")
        assert "error" in result
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_valid_search(self, mock_rj):
        mock_rj.side_effect = [
            # esearch response
            ({"esearchresult": {"idlist": ["200012345"]}}, None),
            # esummary response
            (
                {
                    "result": {
                        "200012345": {
                            "accession": "GSE12345",
                            "title": "Test Study",
                            "summary": "A test dataset",
                            "taxon": "Homo sapiens",
                            "gpl": "GPL570",
                            "n_samples": 10,
                            "gdstype": "Expression profiling by array",
                            "pdat": "2024/01/01",
                        }
                    }
                },
                None,
            ),
        ]
        result = geo_search("TP53 AML")
        assert result["count"] == 1
        assert result["datasets"][0]["accession"] == "GSE12345"
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_organism_filter(self, mock_rj):
        mock_rj.side_effect = [
            ({"esearchresult": {"idlist": []}}, None),
        ]
        result = geo_search("TP53", organism="Mus musculus")
        assert result["count"] == 0
        # Check that organism was included in the query
        call_args = mock_rj.call_args_list[0]
        params = call_args[1].get("params") or call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("params", {})
        assert "Mus musculus" in str(params)

    @patch("ct.tools.omics.request_json")
    def test_study_type_filter(self, mock_rj):
        mock_rj.side_effect = [
            ({"esearchresult": {"idlist": []}}, None),
        ]
        result = geo_search("TP53", study_type="scRNA-seq")
        assert result["count"] == 0
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_api_error(self, mock_rj):
        mock_rj.return_value = (None, "Connection timeout")
        result = geo_search("TP53")
        assert "error" in result
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_empty_results(self, mock_rj):
        mock_rj.return_value = ({"esearchresult": {"idlist": []}}, None)
        result = geo_search("xyznonexistent")
        assert result["count"] == 0
        assert "datasets" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# geo_fetch tests
# ---------------------------------------------------------------------------


class TestGeoFetch:
    def test_invalid_accession(self):
        result = geo_fetch("INVALID123")
        assert "error" in result
        assert "summary" in result

    def test_empty_accession(self):
        result = geo_fetch("")
        assert "error" in result

    @patch("ct.tools.omics._stream_download")
    def test_valid_matrix_download(self, mock_dl, tmp_downloads):
        dest = tmp_downloads / "geo" / "GSE12345" / "GSE12345_series_matrix.txt.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake data")
        mock_dl.return_value = (dest, None)

        result = geo_fetch("GSE12345", file_type="matrix")
        assert "path" in result
        assert result["accession"] == "GSE12345"
        assert "summary" in result

    @patch("ct.tools.omics._stream_download")
    def test_download_failure(self, mock_dl, tmp_downloads):
        mock_dl.return_value = (None, "HTTP 404")
        result = geo_fetch("GSE99999", file_type="matrix")
        assert "error" in result
        assert "summary" in result

    @patch("ct.tools.omics.request")
    def test_supplementary_listing(self, mock_req, tmp_downloads):
        mock_resp = MagicMock()
        mock_resp.text = '<a href="GSE12345_data.csv.gz">GSE12345_data.csv.gz</a>'
        mock_req.return_value = (mock_resp, None)

        with patch("ct.tools.omics._stream_download") as mock_dl:
            dest = tmp_downloads / "geo" / "GSE12345" / "GSE12345_data.csv.gz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"data")
            mock_dl.return_value = (dest, None)

            result = geo_fetch("GSE12345", file_type="supplementary")
            assert "path" in result
            assert "summary" in result

    def test_invalid_file_type(self, tmp_downloads):
        result = geo_fetch("GSE12345", file_type="invalid")
        assert "error" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# cellxgene_search tests
# ---------------------------------------------------------------------------


class TestCellxgeneSearch:
    def test_empty_query(self):
        result = cellxgene_search("")
        assert "error" in result

    @patch("ct.tools.omics.request_json")
    def test_valid_search(self, mock_rj):
        mock_rj.return_value = (
            [
                {
                    "collection_id": "col-1",
                    "name": "AML Single-Cell Atlas",
                    "description": "Single-cell analysis of acute myeloid leukemia",
                    "datasets": [
                        {
                            "dataset_id": "ds-1",
                            "title": "AML scRNA-seq",
                            "cell_count": 50000,
                            "organism": [{"label": "Homo sapiens"}],
                            "tissue": [{"label": "bone marrow"}],
                            "disease": [{"label": "acute myeloid leukemia"}],
                            "assay": [{"label": "10x 3' v3"}],
                        }
                    ],
                }
            ],
            None,
        )
        result = cellxgene_search("AML")
        assert result["count"] == 1
        assert result["datasets"][0]["dataset_id"] == "ds-1"
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_tissue_filter(self, mock_rj):
        mock_rj.return_value = (
            [
                {
                    "collection_id": "col-1",
                    "name": "Multi-tissue Atlas",
                    "description": "Atlas of multiple tissues",
                    "datasets": [
                        {
                            "dataset_id": "ds-1",
                            "title": "Brain cells",
                            "cell_count": 10000,
                            "organism": [{"label": "Homo sapiens"}],
                            "tissue": [{"label": "brain"}],
                            "disease": [{"label": "normal"}],
                            "assay": [{"label": "10x"}],
                        },
                        {
                            "dataset_id": "ds-2",
                            "title": "Lung cells",
                            "cell_count": 20000,
                            "organism": [{"label": "Homo sapiens"}],
                            "tissue": [{"label": "lung"}],
                            "disease": [{"label": "normal"}],
                            "assay": [{"label": "10x"}],
                        },
                    ],
                }
            ],
            None,
        )
        result = cellxgene_search("atlas", tissue="lung")
        assert result["count"] == 1
        assert result["datasets"][0]["dataset_id"] == "ds-2"

    @patch("ct.tools.omics.request_json")
    def test_empty_results(self, mock_rj):
        mock_rj.return_value = ([], None)
        result = cellxgene_search("nonexistent12345")
        assert result["count"] == 0
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_api_error(self, mock_rj):
        mock_rj.return_value = (None, "Connection error")
        result = cellxgene_search("AML")
        assert "error" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# cellxgene_fetch tests
# ---------------------------------------------------------------------------


class TestCellxgeneFetch:
    def test_empty_dataset_id(self):
        result = cellxgene_fetch("")
        assert "error" in result

    @patch("ct.tools.omics._stream_download")
    @patch("ct.tools.omics.request_json")
    def test_valid_fetch(self, mock_rj, mock_dl, tmp_downloads):
        mock_rj.return_value = (
            [
                {
                    "filetype": "H5AD",
                    "filename": "dataset.h5ad",
                    "presigned_url": "https://example.com/dataset.h5ad",
                }
            ],
            None,
        )
        dest = tmp_downloads / "cellxgene" / "ds-1" / "dataset.h5ad"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake h5ad")
        mock_dl.return_value = (dest, None)

        result = cellxgene_fetch("ds-1")
        assert "path" in result
        assert result["dataset_id"] == "ds-1"
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_no_assets(self, mock_rj):
        mock_rj.return_value = ([], None)
        result = cellxgene_fetch("ds-nonexistent")
        assert "error" in result
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_asset_lookup_error(self, mock_rj):
        mock_rj.return_value = (None, "HTTP 404")
        result = cellxgene_fetch("ds-bad")
        assert "error" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# tcga_search tests
# ---------------------------------------------------------------------------


class TestTcgaSearch:
    def test_empty_query(self):
        result = tcga_search("")
        assert "error" in result

    def test_invalid_data_type(self):
        result = tcga_search("BRCA", data_type="invalid")
        assert "error" in result
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_valid_search(self, mock_rj):
        mock_rj.return_value = (
            {
                "data": {
                    "hits": [
                        {
                            "project_id": "TCGA-BRCA",
                            "name": "Breast Invasive Carcinoma",
                            "disease_type": "Breast Cancer",
                            "primary_site": "Breast",
                            "summary": {
                                "case_count": 1098,
                                "file_count": 45000,
                                "data_categories": [
                                    {"data_category": "Transcriptome Profiling", "file_count": 1200},
                                    {"data_category": "DNA Methylation", "file_count": 300},
                                ],
                            },
                        }
                    ]
                }
            },
            None,
        )
        result = tcga_search("breast cancer")
        assert result["count"] == 1
        assert result["projects"][0]["project_id"] == "TCGA-BRCA"
        assert result["projects"][0]["data_type_file_count"] == 1200
        assert result["projects"][0]["matching_data_category"] == "Transcriptome Profiling"
        assert result["projects"][0]["count_method"] == "project_summary_data_category"
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_api_error(self, mock_rj):
        mock_rj.return_value = (None, "Timeout")
        result = tcga_search("BRCA")
        assert "error" in result
        assert "summary" in result

    @patch("ct.tools.omics.request_json")
    def test_no_results(self, mock_rj):
        mock_rj.return_value = ({"data": {"hits": []}}, None)
        result = tcga_search("nonexistent_cancer")
        assert result["count"] == 0
        assert "summary" in result


# ---------------------------------------------------------------------------
# tcga_fetch tests
# ---------------------------------------------------------------------------


class TestTcgaFetch:
    def test_no_ids(self):
        result = tcga_fetch()
        assert "error" in result

    @patch("ct.tools.omics._stream_download")
    def test_fetch_by_file_id(self, mock_dl, tmp_downloads):
        dest = tmp_downloads / "tcga" / "abc123" / "abc123.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")
        mock_dl.return_value = (dest, None)

        result = tcga_fetch(file_id="abc123")
        assert "path" in result
        assert "summary" in result

    @patch("ct.tools.omics._stream_download")
    @patch("ct.tools.omics.request_json")
    def test_fetch_by_project_id(self, mock_rj, mock_dl, tmp_downloads):
        mock_rj.return_value = (
            {
                "data": {
                    "hits": [
                        {
                            "file_id": "file-uuid-1",
                            "file_name": "expression.tsv.gz",
                            "file_size": 1000000,
                            "data_type": "Gene Expression Quantification",
                        }
                    ]
                }
            },
            None,
        )
        dest = tmp_downloads / "tcga" / "TCGA-BRCA" / "expression.tsv.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"data")
        mock_dl.return_value = (dest, None)

        result = tcga_fetch(project_id="TCGA-BRCA")
        assert "path" in result
        assert result["file_id"] == "file-uuid-1"
        assert "summary" in result


# ---------------------------------------------------------------------------
# dataset_info tests
# ---------------------------------------------------------------------------


class TestDatasetInfo:
    def test_empty_path(self):
        result = dataset_info("")
        assert "error" in result

    def test_file_not_found(self):
        result = dataset_info("/nonexistent/path/file.h5ad")
        assert "error" in result
        assert "summary" in result

    def test_csv_inspection(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("gene,sample1,sample2\nTP53,1.2,3.4\nBRCA1,5.6,7.8\n")
        result = dataset_info(str(csv_file))
        assert result["file_type"] == "csv"
        assert result["shape"][0] == 2
        assert "summary" in result

    def test_tsv_inspection(self, tmp_path):
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("gene\tsample1\tsample2\nTP53\t1.2\t3.4\nBRCA1\t5.6\t7.8\n")
        result = dataset_info(str(tsv_file))
        assert result["file_type"] == "tsv"
        assert "summary" in result

    def test_h5ad_without_scanpy(self, tmp_path):
        h5ad_file = tmp_path / "test.h5ad"
        h5ad_file.write_bytes(b"fake h5ad")
        with patch("ct.tools.omics._check_scanpy", return_value=None):
            result = dataset_info(str(h5ad_file))
            assert "scanpy" in result.get("error", "") or "scanpy" in result.get("summary", "")

    def test_h5ad_with_scanpy(self, tmp_path):
        h5ad_file = tmp_path / "test.h5ad"
        h5ad_file.write_bytes(b"fake")

        # Mock scanpy and AnnData
        mock_adata = MagicMock()
        mock_adata.n_obs = 5000
        mock_adata.n_vars = 20000
        mock_adata.obs.columns = ["cell_type", "batch", "donor"]
        mock_adata.var.columns = ["gene_name", "highly_variable"]
        mock_adata.layers = {"counts": None, "normalized": None}
        # Mock obs column access for preview
        mock_col = MagicMock()
        mock_col.unique.return_value = ["T cell", "B cell", "Monocyte"]
        mock_adata.obs.__getitem__ = MagicMock(return_value=mock_col)

        mock_sc = MagicMock()
        mock_sc.read_h5ad.return_value = mock_adata

        with patch("ct.tools.omics._check_scanpy", return_value=mock_sc):
            result = dataset_info(str(h5ad_file))
            assert result["n_cells"] == 5000
            assert result["n_genes"] == 20000
            assert "summary" in result

    def test_matrix_gz_inspection(self, tmp_path):
        gz_file = tmp_path / "test_series_matrix.txt.gz"
        content = (
            '!Series_title\t"Test Study"\n'
            '!Series_organism\t"Homo sapiens"\n'
            '"ID_REF"\t"GSM001"\t"GSM002"\n'
            '10000\t1.5\t2.3\n'
            '10001\t3.1\t4.2\n'
        )
        with gzip.open(gz_file, "wt") as f:
            f.write(content)

        result = dataset_info(str(gz_file))
        assert result["file_type"] == "matrix.txt.gz"
        assert result["n_samples"] == 2
        assert result["n_probes_or_genes"] == 2
        assert "summary" in result

    def test_unknown_extension(self, tmp_path):
        weird_file = tmp_path / "test.xyz"
        weird_file.write_bytes(b"data")
        result = dataset_info(str(weird_file))
        assert "summary" in result
        assert "not directly inspectable" in result["summary"]


# ===========================================================================
# Analysis tool tests
# ===========================================================================


def _make_beta_matrix(tmp_path, n_sites=100, n_samples=6):
    """Create a fake methylation beta-value matrix."""
    np.random.seed(42)
    genes = [f"cg{i:08d}" for i in range(n_sites)]
    samples = [f"sample_{i}" for i in range(n_samples)]
    data = np.random.beta(2, 5, size=(n_sites, n_samples))
    # Make group2 samples more methylated at first 10 sites
    data[:10, n_samples // 2 :] += 0.3
    data = np.clip(data, 0, 1)
    df = pd.DataFrame(data, index=genes, columns=samples)
    path = tmp_path / "methylation.csv"
    df.to_csv(path)
    return path, samples


def _make_protein_matrix(tmp_path, n_proteins=50, n_samples=6):
    """Create a fake protein abundance matrix."""
    np.random.seed(42)
    proteins = [f"PROT{i}" for i in range(n_proteins)]
    samples = [f"sample_{i}" for i in range(n_samples)]
    data = np.random.randn(n_proteins, n_samples) * 2 + 10
    # Make first 5 proteins upregulated in group2
    data[:5, n_samples // 2 :] += 3
    df = pd.DataFrame(data, index=proteins, columns=samples)
    path = tmp_path / "proteomics.csv"
    df.to_csv(path)
    return path, samples


# ---------------------------------------------------------------------------
# methylation_diff tests
# ---------------------------------------------------------------------------


class TestMethylationDiff:
    def test_file_not_found(self):
        result = methylation_diff("/nonexistent.csv")
        assert "error" in result

    def test_auto_split_groups(self, tmp_path):
        path, samples = _make_beta_matrix(tmp_path)
        result = methylation_diff(str(path), auto_grouping=True)
        assert "n_sites_tested" in result
        assert result["n_sites_tested"] == 100
        assert result["auto_grouping_used"] is True
        assert "summary" in result

    def test_no_groups_requires_explicit(self, tmp_path):
        path, _ = _make_beta_matrix(tmp_path)
        result = methylation_diff(str(path))
        assert "error" in result
        assert "Explicit sample groups" in result["error"]

    def test_explicit_groups(self, tmp_path):
        path, samples = _make_beta_matrix(tmp_path)
        g1 = ",".join(samples[:3])
        g2 = ",".join(samples[3:])
        result = methylation_diff(str(path), group1=g1, group2=g2)
        assert result["n_sites_tested"] == 100
        assert "n_hypermethylated" in result
        assert "n_hypomethylated" in result
        assert "summary" in result

    def test_missing_samples(self, tmp_path):
        path, _ = _make_beta_matrix(tmp_path)
        result = methylation_diff(str(path), group1="FAKE1,FAKE2", group2="FAKE3,FAKE4")
        assert "error" in result


# ---------------------------------------------------------------------------
# methylation_profile tests
# ---------------------------------------------------------------------------


class TestMethylationProfile:
    def test_file_not_found(self):
        result = methylation_profile("/nonexistent.csv")
        assert "error" in result

    def test_valid_profile(self, tmp_path):
        path, _ = _make_beta_matrix(tmp_path)
        result = methylation_profile(str(path))
        assert result["n_sites"] == 100
        assert result["n_samples"] == 6
        assert "global_mean_beta" in result
        assert "fraction_unmethylated" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# proteomics_diff tests
# ---------------------------------------------------------------------------


class TestProteomicsDiff:
    def test_file_not_found(self):
        result = proteomics_diff("/nonexistent.csv")
        assert "error" in result

    def test_auto_split(self, tmp_path):
        path, _ = _make_protein_matrix(tmp_path)
        result = proteomics_diff(str(path), auto_grouping=True)
        assert result["n_proteins_tested"] == 50
        assert "n_upregulated" in result
        assert result["auto_grouping_used"] is True
        assert "summary" in result

    def test_no_groups_requires_explicit(self, tmp_path):
        path, _ = _make_protein_matrix(tmp_path)
        result = proteomics_diff(str(path))
        assert "error" in result
        assert "Explicit sample groups" in result["error"]

    def test_explicit_groups(self, tmp_path):
        path, samples = _make_protein_matrix(tmp_path)
        g1 = ",".join(samples[:3])
        g2 = ",".join(samples[3:])
        result = proteomics_diff(str(path), group1=g1, group2=g2)
        assert result["n_proteins_tested"] == 50
        assert "summary" in result


# ---------------------------------------------------------------------------
# proteomics_enrich tests
# ---------------------------------------------------------------------------


class TestProteomicsEnrich:
    def test_empty_list(self):
        result = proteomics_enrich(proteins="")
        assert "error" in result

    @patch("httpx.get")
    @patch("httpx.post")
    def test_enrichr_submission(self, mock_post, mock_get):
        # Mock Enrichr response
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"userListId": 12345}
        mock_post_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_post_resp

        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = {
            "KEGG_2021_Human": [
                [0, "Apoptosis", 0.001, 2.5, 0, ["TP53", "BRCA1"], 0.01],
            ],
            "Reactome_2022": [],
            "GO_Biological_Process_2023": [],
        }
        mock_get_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_get_resp

        result = proteomics_enrich(proteins="TP53,BRCA1,MDM2")
        assert result["n_proteins_submitted"] == 3
        assert result["organism"] == "Homo sapiens"
        assert "KEGG_2021_Human" in result["libraries"]
        assert "summary" in result

    def test_invalid_organism(self):
        result = proteomics_enrich(proteins="TP53,BRCA1", organism="Danio rerio")
        assert "error" in result
        assert "Unsupported organism" in result["error"]

    @patch("httpx.get")
    @patch("httpx.post")
    def test_background_filtering(self, mock_post, mock_get, tmp_path):
        bg = tmp_path / "background.txt"
        bg.write_text("TP53\nBRCA1\nEGFR\n")

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"userListId": 42}
        mock_post_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_post_resp

        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = {
            "KEGG_2021_Human": [],
            "Reactome_2022": [],
            "GO_Biological_Process_2023": [],
        }
        mock_get_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_get_resp

        result = proteomics_enrich(
            proteins="TP53,BRCA1,NOT_IN_BG",
            background_path=str(bg),
        )
        assert result["n_proteins_submitted"] == 2
        assert result["background_gene_count"] == 3
        assert result["n_proteins_before_background_filter"] == 3
        assert result["n_proteins_after_background_filter"] == 2
        assert result["background_mode"] == "input_filter_only"
        assert "summary" in result


# ---------------------------------------------------------------------------
# atac_peak_annotate tests
# ---------------------------------------------------------------------------


class TestAtacPeakAnnotate:
    def test_file_not_found(self):
        result = atac_peak_annotate("/nonexistent.bed")
        assert "error" in result

    def test_bed_format(self, tmp_path):
        bed_file = tmp_path / "peaks.tsv"
        lines = "chr\tstart\tend\tname\n"
        for i in range(50):
            width = 200 + i * 50
            lines += f"chr1\t{i*10000}\t{i*10000 + width}\tpeak_{i}\n"
        bed_file.write_text(lines)

        result = atac_peak_annotate(str(bed_file))
        assert result["n_peaks"] == 50
        assert "chromosome_distribution" in result
        assert "peak_width_stats" in result
        assert "summary" in result

    def test_count_matrix_format(self, tmp_path):
        csv_file = tmp_path / "peak_counts.csv"
        data = np.random.randint(0, 100, size=(200, 4))
        df = pd.DataFrame(data, columns=["sample1", "sample2", "sample3", "sample4"],
                          index=[f"peak_{i}" for i in range(200)])
        df.to_csv(csv_file)

        result = atac_peak_annotate(str(csv_file))
        assert "n_peaks" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# chromatin_accessibility tests
# ---------------------------------------------------------------------------


class TestChromatinAccessibility:
    def test_file_not_found(self):
        result = chromatin_accessibility("/nonexistent.csv")
        assert "error" in result

    def test_diff_accessibility(self, tmp_path):
        np.random.seed(42)
        regions = [f"peak_{i}" for i in range(80)]
        samples = [f"s{i}" for i in range(6)]
        data = np.random.poisson(50, size=(80, 6)).astype(float)
        data[:10, 3:] += 30  # first 10 regions more open in group2
        df = pd.DataFrame(data, index=regions, columns=samples)
        path = tmp_path / "atac_counts.csv"
        df.to_csv(path)

        result = chromatin_accessibility(str(path), group1="s0,s1,s2", group2="s3,s4,s5")
        assert result["n_regions_tested"] == 80
        assert "n_more_accessible" in result
        assert "summary" in result

    def test_no_groups_requires_explicit(self, tmp_path):
        np.random.seed(42)
        regions = [f"peak_{i}" for i in range(20)]
        samples = [f"s{i}" for i in range(6)]
        data = np.random.poisson(50, size=(20, 6)).astype(float)
        df = pd.DataFrame(data, index=regions, columns=samples)
        path = tmp_path / "atac_counts.csv"
        df.to_csv(path)

        result = chromatin_accessibility(str(path))
        assert "error" in result
        assert "Explicit sample groups" in result["error"]


# ---------------------------------------------------------------------------
# chipseq_enrich tests
# ---------------------------------------------------------------------------


class TestChipseqEnrich:
    def test_file_not_found(self):
        result = chipseq_enrich("/nonexistent.tsv")
        assert "error" in result

    def test_no_gene_column(self, tmp_path):
        f = tmp_path / "peaks.csv"
        f.write_text("chr,start,end\nchr1,1000,2000\n")
        result = chipseq_enrich(str(f))
        assert "error" in result
        assert "gene column" in result["error"].lower() or "gene column" in result["summary"].lower()

    @patch("httpx.get")
    @patch("httpx.post")
    def test_with_gene_column(self, mock_post, mock_get, tmp_path):
        f = tmp_path / "annotated_peaks.csv"
        f.write_text("chr,start,end,gene\nchr1,1000,2000,TP53\nchr1,3000,4000,BRCA1\n")

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"userListId": 99}
        mock_post_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_post_resp

        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = {
            "KEGG_2021_Human": [],
            "Reactome_2022": [],
            "GO_Biological_Process_2023": [],
        }
        mock_get_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_get_resp

        result = chipseq_enrich(str(f))
        assert "summary" in result


# ---------------------------------------------------------------------------
# spatial_cluster tests
# ---------------------------------------------------------------------------


class TestSpatialCluster:
    def test_file_not_found(self):
        result = spatial_cluster("/nonexistent.h5ad")
        assert "error" in result

    def test_no_scanpy(self, tmp_path):
        f = tmp_path / "test.h5ad"
        f.write_bytes(b"fake")
        with patch("ct.tools.omics._check_scanpy", return_value=None):
            result = spatial_cluster(str(f))
            assert "error" in result
            assert "scanpy" in result["error"].lower()


# ---------------------------------------------------------------------------
# spatial_autocorrelation tests
# ---------------------------------------------------------------------------


class TestSpatialAutocorrelation:
    def test_file_not_found(self):
        result = spatial_autocorrelation("/nonexistent.h5ad")
        assert "error" in result

    def test_no_scanpy(self, tmp_path):
        f = tmp_path / "test.h5ad"
        f.write_bytes(b"fake")
        with patch("ct.tools.omics._check_scanpy", return_value=None):
            result = spatial_autocorrelation(str(f))
            assert "error" in result


# ---------------------------------------------------------------------------
# cytof_cluster tests
# ---------------------------------------------------------------------------


class TestCytofCluster:
    def test_file_not_found(self):
        result = cytof_cluster("/nonexistent.csv")
        assert "error" in result

    def test_valid_clustering(self, tmp_path):
        np.random.seed(42)
        n_cells = 200
        markers = ["CD3", "CD4", "CD8", "CD19", "CD56"]
        data = np.random.randn(n_cells, len(markers))
        # Create two distinct populations
        data[:100, :2] += 3  # T cells high in CD3, CD4
        data[100:, 3:] += 3  # NK/B cells high in CD19, CD56
        df = pd.DataFrame(data, columns=markers, index=[f"cell_{i}" for i in range(n_cells)])
        path = tmp_path / "cytof.csv"
        df.to_csv(path)

        result = cytof_cluster(str(path), n_clusters=3)
        assert result["n_cells"] == 200
        assert result["n_markers"] == 5
        assert result["n_clusters"] == 3
        assert "defining_markers" in result
        assert "summary" in result

    def test_too_few_cells(self, tmp_path):
        df = pd.DataFrame({"CD3": [1.0], "CD4": [2.0]})
        path = tmp_path / "tiny.csv"
        df.to_csv(path, index=False)
        result = cytof_cluster(str(path))
        assert "error" in result


# ---------------------------------------------------------------------------
# hic_compartments tests
# ---------------------------------------------------------------------------


class TestHicCompartments:
    def test_file_not_found(self):
        result = hic_compartments("/nonexistent.csv")
        assert "error" in result

    def test_non_square_matrix(self, tmp_path):
        df = pd.DataFrame(np.ones((3, 5)), columns=[f"c{i}" for i in range(5)])
        path = tmp_path / "nonsquare.csv"
        df.to_csv(path)
        result = hic_compartments(str(path))
        assert "error" in result

    def test_valid_compartments(self, tmp_path):
        np.random.seed(42)
        n = 20
        # Create a block-diagonal-ish contact matrix (two compartments)
        matrix = np.random.poisson(5, (n, n)).astype(float)
        matrix[:10, :10] += 20  # A compartment interactions
        matrix[10:, 10:] += 20  # B compartment interactions
        matrix = (matrix + matrix.T) / 2  # symmetrize

        bins = [f"bin_{i}" for i in range(n)]
        df = pd.DataFrame(matrix, index=bins, columns=bins)
        path = tmp_path / "hic.csv"
        df.to_csv(path)

        result = hic_compartments(str(path), resolution="50kb")
        assert result["n_bins"] == 20
        assert "n_compartment_A" in result
        assert "n_compartment_B" in result
        assert result["n_compartment_A"] + result["n_compartment_B"] == 20
        assert "summary" in result

    def test_too_few_bins(self, tmp_path):
        df = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["b0", "b1"], columns=["b0", "b1"])
        path = tmp_path / "tiny_hic.csv"
        df.to_csv(path)
        result = hic_compartments(str(path))
        assert "error" in result


# ===========================================================================
# Phase 3: Library integration tool tests
# ===========================================================================


def _make_count_matrix(tmp_path, n_genes=100, n_samples=8):
    """Create a fake raw count matrix for DESeq2 testing."""
    np.random.seed(42)
    genes = [f"GENE{i}" for i in range(n_genes)]
    samples = [f"sample_{i}" for i in range(n_samples)]
    # Poisson counts with different means per group
    data = np.random.poisson(50, size=(n_genes, n_samples))
    # Make first 10 genes upregulated in group2
    data[:10, n_samples // 2 :] = np.random.poisson(200, size=(10, n_samples - n_samples // 2))
    df = pd.DataFrame(data, index=genes, columns=samples)
    path = tmp_path / "counts.csv"
    df.to_csv(path)
    return path, samples


def _make_metadata(tmp_path, samples, condition_col="condition"):
    """Create sample metadata for DESeq2."""
    mid = len(samples) // 2
    metadata = pd.DataFrame(
        {condition_col: ["control"] * mid + ["treatment"] * (len(samples) - mid)},
        index=samples,
    )
    path = tmp_path / "metadata.csv"
    metadata.to_csv(path)
    return path


# ---------------------------------------------------------------------------
# deseq2 tests
# ---------------------------------------------------------------------------


class TestDeseq2:
    def test_file_not_found(self):
        result = deseq2("/nonexistent.csv")
        assert "error" in result
        assert "summary" in result

    def test_metadata_not_found(self, tmp_path):
        path, _ = _make_count_matrix(tmp_path)
        result = deseq2(str(path), metadata_path="/nonexistent_meta.csv")
        assert "error" in result
        assert "summary" in result

    def test_auto_metadata_too_few_samples(self, tmp_path):
        """With <4 samples and no metadata, should error."""
        genes = ["G1", "G2"]
        samples = ["s0", "s1"]
        df = pd.DataFrame([[10, 20], [30, 40]], index=genes, columns=samples)
        path = tmp_path / "tiny.csv"
        df.to_csv(path)
        result = deseq2(str(path))
        assert "error" in result

    def test_missing_condition_col(self, tmp_path):
        path, samples = _make_count_matrix(tmp_path)
        meta_path = _make_metadata(tmp_path, samples, condition_col="group")
        result = deseq2(str(path), metadata_path=str(meta_path), condition_col="condition")
        assert "error" in result
        assert "condition" in result["error"]

    def test_fallback_mann_whitney(self, tmp_path):
        """When pyDESeq2 is not available, falls back to Mann-Whitney."""
        path, samples = _make_count_matrix(tmp_path)
        meta_path = _make_metadata(tmp_path, samples)

        with patch("ct.tools.omics._check_pydeseq2", return_value=False):
            result = deseq2(str(path), metadata_path=str(meta_path))
            assert "method" in result
            assert "Mann-Whitney" in result["method"]
            assert result["n_genes_tested"] == 100
            assert "n_upregulated" in result
            assert "n_downregulated" in result
            assert result["contrast"] == "treatment vs control"
            assert "summary" in result

    def test_auto_metadata_generation(self, tmp_path):
        """Without metadata, auto-generates control/treatment split."""
        path, samples = _make_count_matrix(tmp_path)

        with patch("ct.tools.omics._check_pydeseq2", return_value=False):
            result = deseq2(str(path), infer_metadata=True)
            assert result["n_genes_tested"] == 100
            assert result["contrast"] == "treatment vs control"
            assert result["metadata_inferred"] is True
            assert "summary" in result

    def test_metadata_required_without_infer(self, tmp_path):
        path, _ = _make_count_matrix(tmp_path)
        result = deseq2(str(path))
        assert "error" in result
        assert "metadata_path is required" in result["error"]

    def test_single_condition_level(self, tmp_path):
        """If metadata has only one level, should error."""
        path, samples = _make_count_matrix(tmp_path)
        meta = pd.DataFrame({"condition": ["control"] * len(samples)}, index=samples)
        meta_path = tmp_path / "bad_meta.csv"
        meta.to_csv(meta_path)
        result = deseq2(str(path), metadata_path=str(meta_path))
        assert "error" in result

    def test_few_shared_samples(self, tmp_path):
        """If counts and metadata share <4 samples, should error."""
        path, samples = _make_count_matrix(tmp_path)
        # Metadata with completely different sample names
        meta = pd.DataFrame(
            {"condition": ["control", "treatment"]},
            index=["other_0", "other_1"],
        )
        meta_path = tmp_path / "mismatch_meta.csv"
        meta.to_csv(meta_path)
        result = deseq2(str(path), metadata_path=str(meta_path))
        assert "error" in result


# ---------------------------------------------------------------------------
# multiomics_integrate tests
# ---------------------------------------------------------------------------


class TestMultiomicsIntegrate:
    def test_no_muon(self):
        with patch("ct.tools.omics._check_muon", return_value=None):
            result = multiomics_integrate(paths="a.h5ad,b.h5ad", modality_names="rna,atac")
            assert "error" in result
            assert "muon" in result["error"].lower()

    def test_no_scanpy(self):
        mock_mu = MagicMock()
        with patch("ct.tools.omics._check_muon", return_value=mock_mu):
            with patch("ct.tools.omics._check_scanpy", return_value=None):
                result = multiomics_integrate(paths="a.h5ad,b.h5ad", modality_names="rna,atac")
                assert "error" in result
                assert "scanpy" in result["error"].lower()

    def test_too_few_paths(self):
        mock_mu = MagicMock()
        mock_sc = MagicMock()
        with patch("ct.tools.omics._check_muon", return_value=mock_mu):
            with patch("ct.tools.omics._check_scanpy", return_value=mock_sc):
                result = multiomics_integrate(paths="only_one.h5ad")
                assert "error" in result
                assert "2" in result["error"]

    def test_mismatched_names_and_paths(self):
        mock_mu = MagicMock()
        mock_sc = MagicMock()
        with patch("ct.tools.omics._check_muon", return_value=mock_mu):
            with patch("ct.tools.omics._check_scanpy", return_value=mock_sc):
                result = multiomics_integrate(
                    paths="a.h5ad,b.h5ad",
                    modality_names="rna,atac,protein",
                )
                assert "error" in result

    def test_file_not_found(self, tmp_path):
        mock_mu = MagicMock()
        mock_sc = MagicMock()
        with patch("ct.tools.omics._check_muon", return_value=mock_mu):
            with patch("ct.tools.omics._check_scanpy", return_value=mock_sc):
                result = multiomics_integrate(
                    paths="/nonexistent/a.h5ad,/nonexistent/b.h5ad",
                    modality_names="rna,atac",
                )
                assert "error" in result

    def test_auto_names(self):
        """If no modality_names given, auto-generates modality_0, modality_1."""
        mock_mu = MagicMock()
        mock_sc = MagicMock()
        with patch("ct.tools.omics._check_muon", return_value=mock_mu):
            with patch("ct.tools.omics._check_scanpy", return_value=mock_sc):
                # Still fails on missing files, but tests the auto-naming path
                result = multiomics_integrate(paths="/no/a.h5ad,/no/b.h5ad")
                assert "error" in result  # file not found
                assert "Missing file" in result.get("summary", "")


# ---------------------------------------------------------------------------
# methylation_cluster tests
# ---------------------------------------------------------------------------


class TestMethylationCluster:
    def test_file_not_found(self):
        result = methylation_cluster("/nonexistent.csv")
        assert "error" in result
        assert "summary" in result

    def test_sklearn_fallback(self, tmp_path):
        """Without scanpy or episcanpy, should fall back to sklearn KMeans."""
        path, _ = _make_beta_matrix(tmp_path, n_sites=50, n_samples=20)

        with patch("ct.tools.omics._check_episcanpy", return_value=None):
            with patch("ct.tools.omics._check_scanpy", return_value=None):
                result = methylation_cluster(str(path))
                assert "method" in result
                assert "sklearn" in result["method"]
                assert result["n_samples"] == 20
                assert "n_clusters" in result
                assert result["n_clusters"] > 0
                assert "summary" in result

    def test_no_reader_for_h5ad(self, tmp_path):
        """h5ad without scanpy or episcanpy should error."""
        f = tmp_path / "test.h5ad"
        f.write_bytes(b"fake")
        with patch("ct.tools.omics._check_episcanpy", return_value=None):
            with patch("ct.tools.omics._check_scanpy", return_value=None):
                result = methylation_cluster(str(f))
                assert "error" in result
                assert "scanpy" in result["error"].lower() or "episcanpy" in result["error"].lower()

    def test_csv_input_shape(self, tmp_path):
        """CSV input: sites as rows, samples as cols — should transpose for clustering."""
        path, _ = _make_beta_matrix(tmp_path, n_sites=30, n_samples=10)

        with patch("ct.tools.omics._check_episcanpy", return_value=None):
            with patch("ct.tools.omics._check_scanpy", return_value=None):
                result = methylation_cluster(str(path))
                # After transpose: 10 samples (obs) x 30 features (var)
                assert result["n_samples"] == 10
                assert result["n_features_input"] == 30
                assert "summary" in result
