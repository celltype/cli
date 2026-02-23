"""Tests for the DuckDB query engine."""

import pytest
import pandas as pd
from pathlib import Path

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

pytestmark = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")


class TestQueryEngine:
    def test_query_parquet_file(self, tmp_path):
        from ct.api.engine import QueryEngine

        # Write a small parquet file
        df = pd.DataFrame({
            "gene": ["TP53", "BRCA1", "EGFR", "MYC"],
            "log2FoldChange": [-1.5, -0.8, 0.5, 1.2],
            "padj": [0.001, 0.01, 0.1, 0.5],
        })
        pq_path = tmp_path / "test.parquet"
        df.to_parquet(pq_path)

        engine = QueryEngine(data_root=tmp_path)
        result = engine.query_parquet("test.parquet")
        assert len(result) == 4
        assert result[0]["gene"] in ["TP53", "BRCA1", "EGFR", "MYC"]

    def test_query_csv_gz_file(self, tmp_path):
        import gzip

        from ct.api.engine import QueryEngine

        # Write a CSV.gz file
        csv_content = "gene,lfc,pvalue\nTP53,-1.5,0.001\nBRCA1,-0.8,0.01\n"
        gz_path = tmp_path / "test.csv.gz"
        with gzip.open(gz_path, "wt") as f:
            f.write(csv_content)

        engine = QueryEngine(data_root=tmp_path)
        result = engine.query_parquet("test.csv.gz")
        assert len(result) == 2
        assert result[0]["gene"] == "TP53"

    def test_query_with_filters(self, tmp_path):
        from ct.api.engine import QueryEngine

        df = pd.DataFrame({
            "gene": ["TP53", "BRCA1", "EGFR", "TP53"],
            "log2FoldChange": [-1.5, -0.8, 0.5, -2.0],
            "experiment": ["exp1", "exp1", "exp2", "exp2"],
        })
        df.to_parquet(tmp_path / "test.parquet")

        engine = QueryEngine(data_root=tmp_path)
        result = engine.query_parquet("test.parquet", filters={"gene": "TP53"})
        assert len(result) == 2
        assert all(r["gene"] == "TP53" for r in result)

    def test_query_limit(self, tmp_path):
        from ct.api.engine import QueryEngine

        df = pd.DataFrame({
            "gene": [f"GENE{i}" for i in range(100)],
            "value": list(range(100)),
        })
        df.to_parquet(tmp_path / "test.parquet")

        engine = QueryEngine(data_root=tmp_path)
        result = engine.query_parquet("test.parquet", limit=5)
        assert len(result) == 5

    def test_query_column_selection(self, tmp_path):
        from ct.api.engine import QueryEngine

        df = pd.DataFrame({
            "gene": ["TP53", "BRCA1"],
            "lfc": [-1.5, -0.8],
            "pvalue": [0.001, 0.01],
            "extra_col": ["a", "b"],
        })
        df.to_parquet(tmp_path / "test.parquet")

        engine = QueryEngine(data_root=tmp_path)
        result = engine.query_parquet("test.parquet", columns=["gene", "lfc"])
        assert len(result) == 2
        assert set(result[0].keys()) == {"gene", "lfc"}

    def test_nonexistent_file_raises(self, tmp_path):
        from ct.api.engine import QueryEngine

        engine = QueryEngine(data_root=tmp_path)
        with pytest.raises(FileNotFoundError):
            engine.query_parquet("nonexistent.parquet")

    def test_count(self, tmp_path):
        from ct.api.engine import QueryEngine

        df = pd.DataFrame({
            "gene": ["TP53", "BRCA1", "TP53"],
            "value": [1, 2, 3],
        })
        df.to_parquet(tmp_path / "test.parquet")

        engine = QueryEngine(data_root=tmp_path)
        assert engine.count("test.parquet") == 3
        assert engine.count("test.parquet", filters={"gene": "TP53"}) == 2

    def test_query_with_list_filter(self, tmp_path):
        from ct.api.engine import QueryEngine

        df = pd.DataFrame({
            "gene": ["TP53", "BRCA1", "EGFR", "MYC"],
            "value": [1, 2, 3, 4],
        })
        df.to_parquet(tmp_path / "test.parquet")

        engine = QueryEngine(data_root=tmp_path)
        result = engine.query_parquet(
            "test.parquet",
            filters={"gene": ["TP53", "EGFR"]},
        )
        assert len(result) == 2
        genes = {r["gene"] for r in result}
        assert genes == {"TP53", "EGFR"}
