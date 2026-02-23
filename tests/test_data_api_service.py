"""End-to-end tests for the ct Data API service.

All tests run against real data at CT_DATA_ROOT=/mnt2/bronze.
No mocks — every test hits the real DuckDB engine and real files.
"""

import os

import pytest

try:
    import fastapi
    import duckdb
    HAS_API_DEPS = True
except ImportError:
    HAS_API_DEPS = False

DATA_ROOT = "/mnt2/bronze"

pytestmark = [
    pytest.mark.skipif(not HAS_API_DEPS, reason="fastapi/duckdb not installed"),
    pytest.mark.skipif(
        not os.path.isdir(os.path.join(DATA_ROOT, "perturbatlas")),
        reason=f"Real data not found at {DATA_ROOT}",
    ),
]


@pytest.fixture(autouse=True)
def _set_data_root(monkeypatch):
    """Point the API at real data for every test."""
    monkeypatch.setenv("CT_DATA_ROOT", DATA_ROOT)
    # Patch the module-level DATA_ROOT that was already imported
    from pathlib import Path
    import ct.api.config as config_mod
    import ct.api.engine as engine_mod
    monkeypatch.setattr(config_mod, "DATA_ROOT", Path(DATA_ROOT))
    # Re-create the engine with the correct root
    from ct.api import app as app_mod
    app_mod.engine = engine_mod.QueryEngine(data_root=Path(DATA_ROOT))


@pytest.fixture
def client():
    """FastAPI TestClient backed by real data."""
    from fastapi.testclient import TestClient
    from ct.api.app import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_lists_perturbatlas(self, client):
        resp = client.get("/health")
        assert "perturbatlas" in resp.json()["datasets"]

    def test_health_reports_dataset_count(self, client):
        resp = client.get("/health")
        assert resp.json()["n_datasets"] >= 1


class TestSchemaHealthEndpoint:
    def test_schema_returns_200(self, client):
        resp = client.get("/health/schema")
        assert resp.status_code == 200

    def test_perturbatlas_has_no_missing_required(self, client):
        resp = client.get("/health/schema")
        pa = resp.json()["datasets"]["perturbatlas"]
        assert pa["missing_required"] == []

    def test_perturbatlas_schema_valid_or_warning(self, client):
        resp = client.get("/health/schema")
        pa = resp.json()["datasets"]["perturbatlas"]
        assert pa["status"] in ("valid", "warning")

    def test_perturbatlas_actual_columns_include_required(self, client):
        resp = client.get("/health/schema")
        pa = resp.json()["datasets"]["perturbatlas"]
        actual = set(pa["actual_columns"])
        for col in ["gene", "log2FoldChange", "pvalue", "padj"]:
            assert col in actual, f"Required column {col!r} missing from actual columns"


class TestDatasetsEndpoint:
    def test_list_datasets_returns_list(self, client):
        resp = client.get("/datasets")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_perturbatlas_in_datasets(self, client):
        resp = client.get("/datasets")
        names = [d["name"] for d in resp.json()]
        assert "perturbatlas" in names

    def test_perturbatlas_metadata(self, client):
        resp = client.get("/datasets")
        ds = next(d for d in resp.json() if d["name"] == "perturbatlas")
        assert ds["format"] == "csv.gz"
        assert ds["n_files"] > 2000
        assert "gene" in ds["filterable_columns"]
        assert "gene" in ds["required_columns"]
        assert ds["total_size_mb"] > 0


class TestQueryEndpoint:
    def test_query_gene_returns_data(self, client):
        """Query TP53 (ENSG00000141510) — should exist across many experiments."""
        resp = client.post("/query", json={
            "dataset": "perturbatlas",
            "gene": "ENSG00000141510",
            "limit": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["dataset"] == "perturbatlas"
        assert data["total_rows"] == 5
        for row in data["data"]:
            assert row["gene"] == "ENSG00000141510"

    def test_query_returns_expected_columns(self, client):
        resp = client.post("/query", json={
            "dataset": "perturbatlas",
            "gene": "ENSG00000141510",
            "limit": 1,
        })
        row = resp.json()["data"][0]
        for col in ["gene", "log2FoldChange", "pvalue", "padj", "baseMean", "perturb_id"]:
            assert col in row, f"Expected column {col!r} in query result"

    def test_query_respects_limit(self, client):
        resp = client.post("/query", json={
            "dataset": "perturbatlas",
            "gene": "ENSG00000141510",
            "limit": 3,
        })
        assert resp.json()["total_rows"] == 3

    def test_query_with_perturb_id_filter(self, client):
        resp = client.post("/query", json={
            "dataset": "perturbatlas",
            "filters": {"perturb_id": "Perturb_1"},
            "limit": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] > 0
        for row in data["data"]:
            assert row["perturb_id"] == "Perturb_1"

    def test_query_select_columns(self, client):
        resp = client.post("/query", json={
            "dataset": "perturbatlas",
            "gene": "ENSG00000141510",
            "columns": ["gene", "log2FoldChange", "padj"],
            "limit": 2,
        })
        assert resp.status_code == 200
        row = resp.json()["data"][0]
        assert set(row.keys()) == {"gene", "log2FoldChange", "padj"}

    def test_query_order_by(self, client):
        resp = client.post("/query", json={
            "dataset": "perturbatlas",
            "gene": "ENSG00000141510",
            "order_by": "log2FoldChange",
            "limit": 10,
        })
        assert resp.status_code == 200
        lfcs = [r["log2FoldChange"] for r in resp.json()["data"]]
        assert lfcs == sorted(lfcs)

    def test_query_unknown_dataset_returns_404(self, client):
        resp = client.post("/query", json={"dataset": "nonexistent"})
        assert resp.status_code == 404

    def test_query_nonexistent_gene_returns_empty(self, client):
        resp = client.post("/query", json={
            "dataset": "perturbatlas",
            "gene": "FAKE_GENE_DOES_NOT_EXIST",
            "limit": 5,
        })
        assert resp.status_code == 200
        assert resp.json()["total_rows"] == 0


class TestGeneEndpoint:
    def test_gene_summary_tp53(self, client):
        resp = client.get("/datasets/perturbatlas/gene/ENSG00000141510")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gene"] == "ENSG00000141510"
        assert data["dataset"] == "perturbatlas"
        assert data["n_perturbations"] > 0
        assert isinstance(data["mean_effect"], float)
        assert data["n_significant"] >= 0
        assert isinstance(data["sample_data"], list)
        assert len(data["sample_data"]) > 0

    def test_gene_summary_unknown_dataset_returns_404(self, client):
        resp = client.get("/datasets/nonexistent/gene/ENSG00000141510")
        assert resp.status_code == 404

    def test_gene_summary_unknown_gene_returns_404(self, client):
        resp = client.get("/datasets/perturbatlas/gene/FAKE_GENE_DOES_NOT_EXIST")
        assert resp.status_code == 404


class TestCompoundEndpoint:
    def test_compound_unknown_dataset_returns_404(self, client):
        resp = client.get("/datasets/nonexistent/compound/imatinib")
        assert resp.status_code == 404

    def test_compound_not_found_returns_404(self, client):
        """PerturbAtlas has no compound column, so any compound query should 404."""
        resp = client.get("/datasets/perturbatlas/compound/imatinib")
        assert resp.status_code in (404, 400, 503)


class TestEngineDirectly:
    """Test the DuckDB QueryEngine directly against real files."""

    @pytest.fixture
    def engine(self):
        from pathlib import Path
        from ct.api.engine import QueryEngine
        return QueryEngine(data_root=Path(DATA_ROOT))

    def test_sample_columns(self, engine):
        cols = engine.sample_columns("perturbatlas/Homo sapiens/*/degs.csv.gz")
        assert "gene" in cols
        assert "log2FoldChange" in cols
        assert "pvalue" in cols
        assert "padj" in cols

    def test_query_parquet_returns_records(self, engine):
        data = engine.query_parquet(
            file_pattern="perturbatlas/Homo sapiens/*/degs.csv.gz",
            filters={"gene": "ENSG00000141510"},
            limit=3,
        )
        assert len(data) == 3
        assert all(r["gene"] == "ENSG00000141510" for r in data)

    def test_count(self, engine):
        n = engine.count(
            file_pattern="perturbatlas/Homo sapiens/*/degs.csv.gz",
            filters={"gene": "ENSG00000141510"},
        )
        assert n > 100  # TP53 appears in most experiments

    def test_query_with_no_filters_returns_data(self, engine):
        data = engine.query_parquet(
            file_pattern="perturbatlas/Homo sapiens/*/degs.csv.gz",
            limit=5,
        )
        assert len(data) == 5

    def test_query_nonexistent_pattern_raises(self, engine):
        with pytest.raises(FileNotFoundError):
            engine.query_parquet(
                file_pattern="does_not_exist/*.csv.gz",
                limit=1,
            )
