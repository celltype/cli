"""
DuckDB query engine for the ct Data API.

Queries Parquet and CSV.gz files directly without loading into memory.
Supports predicate pushdown for efficient filtering of large datasets.
"""

from pathlib import Path

import duckdb

from ct.api.config import DATA_ROOT, MAX_QUERY_ROWS, DEFAULT_QUERY_LIMIT


class QueryEngine:
    """DuckDB-based query engine for tabular data files."""

    def __init__(self, data_root: Path = None):
        self.data_root = data_root or DATA_ROOT
        self.conn = duckdb.connect(":memory:")

    def query_parquet(self, file_pattern: str, filters: dict = None,
                      columns: list = None, limit: int = None,
                      order_by: str = None) -> list[dict]:
        """Query Parquet or CSV.gz files with optional filtering.

        Args:
            file_pattern: Glob pattern relative to data_root (e.g. "chembl/36/*.parquet")
            filters: Dict of column_name → value for WHERE clauses
            columns: List of columns to SELECT (default: all)
            limit: Max rows to return
            order_by: Column to ORDER BY

        Returns:
            List of dicts, one per row.
        """
        full_pattern = str(self.data_root / file_pattern)
        limit = min(limit or DEFAULT_QUERY_LIMIT, MAX_QUERY_ROWS)

        # Build SELECT clause
        select_cols = ", ".join(columns) if columns else "*"

        # Determine read function based on file extension
        if ".parquet" in file_pattern:
            source = f"read_parquet('{full_pattern}', union_by_name=true)"
        elif ".csv.gz" in file_pattern or ".csv" in file_pattern:
            source = f"read_csv_auto('{full_pattern}', union_by_name=true, ignore_errors=true)"
        else:
            source = f"read_parquet('{full_pattern}', union_by_name=true)"

        # Build WHERE clause
        where_parts = []
        params = []
        if filters:
            for col, val in filters.items():
                if isinstance(val, list):
                    placeholders = ", ".join(["?" for _ in val])
                    where_parts.append(f'"{col}" IN ({placeholders})')
                    params.extend(val)
                elif isinstance(val, str) and "%" in val:
                    where_parts.append(f'"{col}" LIKE ?')
                    params.append(val)
                else:
                    where_parts.append(f'"{col}" = ?')
                    params.append(val)

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        order_clause = f' ORDER BY "{order_by}"' if order_by else ""
        limit_clause = f" LIMIT {limit}"

        query = f"SELECT {select_cols} FROM {source}{where_clause}{order_clause}{limit_clause}"

        try:
            result = self.conn.execute(query, params).fetchdf()
            return result.to_dict("records")
        except duckdb.IOException as e:
            raise FileNotFoundError(f"Data files not found: {full_pattern}") from e
        except (duckdb.CatalogException, duckdb.BinderException) as e:
            raise ValueError(f"Query error (bad column?): {e}") from e

    def sample_columns(self, file_pattern: str) -> list[str]:
        """Read column names from first matching file without loading data."""
        full_pattern = str(self.data_root / file_pattern)

        if ".parquet" in file_pattern:
            source = f"read_parquet('{full_pattern}', union_by_name=true)"
        elif ".csv.gz" in file_pattern or ".csv" in file_pattern:
            source = f"read_csv_auto('{full_pattern}', union_by_name=true, ignore_errors=true)"
        else:
            source = f"read_parquet('{full_pattern}', union_by_name=true)"

        query = f"SELECT * FROM {source} LIMIT 0"
        result = self.conn.execute(query)
        return [desc[0] for desc in result.description]

    def count(self, file_pattern: str, filters: dict = None) -> int:
        """Count rows matching filters."""
        full_pattern = str(self.data_root / file_pattern)

        if ".parquet" in file_pattern:
            source = f"read_parquet('{full_pattern}', union_by_name=true)"
        else:
            source = f"read_csv_auto('{full_pattern}', union_by_name=true)"

        where_parts = []
        params = []
        if filters:
            for col, val in filters.items():
                if isinstance(val, list):
                    placeholders = ", ".join(["?" for _ in val])
                    where_parts.append(f'"{col}" IN ({placeholders})')
                    params.extend(val)
                else:
                    where_parts.append(f'"{col}" = ?')
                    params.append(val)

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        query = f"SELECT COUNT(*) as cnt FROM {source}{where_clause}"

        try:
            result = self.conn.execute(query, params).fetchone()
            return result[0] if result else 0
        except duckdb.IOException:
            return 0
