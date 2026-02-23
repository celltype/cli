"""
Configuration for the ct Data Query API.

Maps dataset names to file paths and schemas. The data root is configurable
via CT_DATA_ROOT environment variable or defaults to /data (for Docker).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.environ.get("CT_DATA_ROOT", "/data"))

# Dataset registry: name → config
DATASET_REGISTRY = {
    "perturbatlas": {
        "description": "PerturbAtlas: differential expression from 2,066 perturbation experiments",
        "path_pattern": "perturbatlas/Homo sapiens/*/degs.csv.gz",
        "format": "csv.gz",
        "columns": ["column0", "perturb_id", "gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"],
        "required_columns": ["gene", "log2FoldChange", "pvalue", "padj"],
        "filterable": ["gene", "perturb_id"],
    },
    "chembl": {
        "description": "ChEMBL v36: bioactivity data for drug-like compounds",
        "path_pattern": "chembl/chembl/36/*.parquet",
        "format": "parquet",
        "columns": [],
        "required_columns": ["molecule_chembl_id", "target_chembl_id"],
        "filterable": ["molecule_chembl_id", "target_chembl_id", "pchembl_value"],
    },
}

# Maximum rows returned per query
MAX_QUERY_ROWS = 10000
DEFAULT_QUERY_LIMIT = 100


def discover_datasets() -> dict:
    """Discover which datasets are actually available on disk."""
    available = {}
    for name, config in DATASET_REGISTRY.items():
        pattern = config["path_pattern"]
        # Check if any files match the pattern
        parts = pattern.split("*")
        base_dir = DATA_ROOT / parts[0].rstrip("/")
        if base_dir.exists():
            # Glob for matching files
            files = list(DATA_ROOT.glob(pattern))
            if files:
                available[name] = {
                    **config,
                    "n_files": len(files),
                    "total_size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1),
                }
    return available


def validate_schema(engine=None) -> dict[str, dict]:
    """Validate that on-disk files match expected schemas.

    Samples one file per dataset, reads column names via DuckDB,
    and compares against DATASET_REGISTRY declarations.
    Returns {dataset_name: {status, expected, actual, missing, extra}}.
    """
    if engine is None:
        from ct.api.engine import QueryEngine
        engine = QueryEngine()

    results = {}
    for name, config in DATASET_REGISTRY.items():
        pattern = config["path_pattern"]
        files = list(DATA_ROOT.glob(pattern))

        if not files:
            results[name] = {
                "status": "unavailable",
                "message": "No files found on disk",
            }
            continue

        try:
            actual_cols = engine.sample_columns(pattern)
        except Exception as e:
            results[name] = {
                "status": "error",
                "message": f"Failed to read columns: {e}",
            }
            continue

        expected = set(config.get("columns", []))
        required = set(config.get("required_columns", []))
        actual = set(actual_cols)

        missing_required = required - actual
        missing_declared = expected - actual if expected else set()
        extra = actual - expected if expected else set()

        if missing_required:
            status = "invalid"
        elif missing_declared:
            status = "warning"
        else:
            status = "valid"

        results[name] = {
            "status": status,
            "actual_columns": sorted(actual),
            "expected_columns": sorted(expected) if expected else None,
            "required_columns": sorted(required),
            "missing_required": sorted(missing_required) if missing_required else [],
            "missing_declared": sorted(missing_declared) if missing_declared else [],
            "extra_columns": sorted(extra) if extra else [],
        }

    return results
