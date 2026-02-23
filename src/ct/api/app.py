"""
FastAPI application for the ct Data Query API.

Serves filtered queries against large datasets (PerturbAtlas, ChEMBL, etc.)
using DuckDB for efficient Parquet/CSV.gz querying.

Run with: uvicorn ct.api.app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ct.api.config import discover_datasets, validate_schema, DATASET_REGISTRY, DEFAULT_QUERY_LIMIT
from ct.api.engine import QueryEngine

app = FastAPI(
    title="ct Data API",
    description="Query large drug discovery datasets via DuckDB",
    version="0.1.0",
)

engine = QueryEngine()


class QueryRequest(BaseModel):
    dataset: str = Field(..., description="Dataset name (e.g. 'perturbatlas', 'chembl')")
    gene: str | None = Field(None, description="Filter by gene symbol")
    compound: str | None = Field(None, description="Filter by compound name/ID")
    filters: dict | None = Field(None, description="Additional column filters")
    columns: list[str] | None = Field(None, description="Columns to return")
    limit: int = Field(DEFAULT_QUERY_LIMIT, ge=1, le=10000, description="Max rows")
    order_by: str | None = Field(None, description="Column to sort by")


class QueryResponse(BaseModel):
    dataset: str
    total_rows: int
    data: list[dict]


@app.get("/health")
def health():
    """Health check — also reports available datasets."""
    available = discover_datasets()
    return {
        "status": "ok",
        "datasets": list(available.keys()),
        "n_datasets": len(available),
    }


@app.get("/health/schema")
def schema_health():
    """Deep health check — validates file schemas match expectations."""
    results = validate_schema(engine)
    has_invalid = any(r["status"] == "invalid" for r in results.values())

    status_code = 503 if has_invalid else 200
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "unhealthy" if has_invalid else "ok",
            "datasets": results,
        },
    )


@app.get("/datasets")
def list_datasets():
    """List available datasets with metadata."""
    available = discover_datasets()
    result = []
    for name, info in available.items():
        registry_entry = DATASET_REGISTRY.get(name, {})
        result.append({
            "name": name,
            "description": info["description"],
            "format": info["format"],
            "n_files": info["n_files"],
            "total_size_mb": info["total_size_mb"],
            "filterable_columns": info.get("filterable", []),
            "columns": registry_entry.get("columns", []),
            "required_columns": registry_entry.get("required_columns", []),
        })
    return result


@app.post("/query", response_model=QueryResponse)
def query_dataset(req: QueryRequest):
    """Run a filtered query against a dataset."""
    if req.dataset not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {req.dataset}")

    ds_config = DATASET_REGISTRY[req.dataset]

    # Build filters
    filters = dict(req.filters) if req.filters else {}
    if req.gene:
        filters["gene"] = req.gene
    if req.compound:
        # Try common column names for compound
        for col in ["compound", "molecule_chembl_id", "pert_name", "compound_name"]:
            if col in ds_config.get("filterable", []):
                filters[col] = req.compound
                break
        else:
            filters["compound"] = req.compound

    try:
        data = engine.query_parquet(
            file_pattern=ds_config["path_pattern"],
            filters=filters if filters else None,
            columns=req.columns,
            limit=req.limit,
            order_by=req.order_by,
        )
        return QueryResponse(
            dataset=req.dataset,
            total_rows=len(data),
            data=data,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=f"Dataset '{req.dataset}' files not found on disk",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/datasets/{name}/gene/{gene}")
def gene_summary(name: str, gene: str):
    """Get a gene-level summary from a dataset."""
    if name not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {name}")

    ds_config = DATASET_REGISTRY[name]

    try:
        data = engine.query_parquet(
            file_pattern=ds_config["path_pattern"],
            filters={"gene": gene},
            limit=1000,
        )

        if not data:
            raise HTTPException(status_code=404, detail=f"Gene {gene} not found in {name}")

        # Compute summary stats
        import statistics
        effects = [r.get("log2FoldChange", r.get("lfc", 0)) for r in data if r.get("log2FoldChange") or r.get("lfc")]
        mean_effect = statistics.mean(effects) if effects else 0
        n_perturbations = len(data)

        return {
            "gene": gene,
            "dataset": name,
            "n_perturbations": n_perturbations,
            "mean_effect": round(mean_effect, 4),
            "n_significant": sum(
                1 for r in data
                if (r.get("padj") or r.get("pvalue", 1)) < 0.05
            ),
            "sample_data": data[:10],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"Dataset '{name}' files not found")


@app.get("/datasets/{name}/compound/{compound}")
def compound_summary(name: str, compound: str):
    """Get a compound-level summary from a dataset."""
    if name not in DATASET_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {name}")

    ds_config = DATASET_REGISTRY[name]

    # Try common compound column names — use filterable from dataset config
    filters = {}
    filterable = ds_config.get("filterable", [])
    for col in ["compound", "molecule_chembl_id", "pert_name", "compound_name"]:
        if col in filterable:
            filters[col] = compound
            break
    else:
        filters["compound"] = compound

    try:
        data = engine.query_parquet(
            file_pattern=ds_config["path_pattern"],
            filters=filters,
            limit=1000,
        )

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"Compound {compound} not found in {name}",
            )

        return {
            "compound": compound,
            "dataset": name,
            "n_records": len(data),
            "sample_data": data[:10],
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"Dataset '{name}' files not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
