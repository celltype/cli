"""
Remote data query tool — queries the hosted ct Data API.

Connects to a running ct Data API instance to query large datasets
(PerturbAtlas, ChEMBL, scPerturb, etc.) that are too large for local download.

Configure the endpoint with: ct config set api.data_endpoint http://host:8000
Or set CT_DATA_ENDPOINT environment variable.
"""

import os
import httpx
from ct.tools import registry
from ct.agent.config import Config


def _get_endpoint() -> str | None:
    """Get the configured data API endpoint."""
    endpoint = os.environ.get("CT_DATA_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/")
    cfg = Config.load()
    val = cfg.get("api.data_endpoint")
    return val.rstrip("/") if val else None


def _no_endpoint_error() -> dict:
    return {
        "error": "Data API endpoint not configured.",
        "summary": (
            "No data API endpoint configured. Set with: "
            "ct config set api.data_endpoint http://your-host:8000 "
            "or set CT_DATA_ENDPOINT environment variable."
        ),
    }


@registry.register(
    name="remote_data.query",
    description="Query a dataset on the hosted ct Data API (PerturbAtlas, ChEMBL, etc.)",
    category="remote_data",
    parameters={
        "dataset": "Dataset name (e.g. 'perturbatlas', 'chembl')",
        "gene": "Filter by gene symbol (optional)",
        "compound": "Filter by compound name/ID (optional)",
        "filters": "Additional column filters as dict (optional)",
        "limit": "Max rows to return (default 100, max 10000)",
    },
    usage_guide=(
        "You need to query a large dataset that's hosted on the data API (PerturbAtlas, ChEMBL, "
        "scPerturb). Use when the dataset is too large for local download. "
        "Requires api.data_endpoint to be configured."
    ),
)
def query(dataset: str, gene: str = None, compound: str = None,
          filters: dict = None, limit: int = 100, **kwargs) -> dict:
    """Query a dataset on the hosted Data API."""
    endpoint = _get_endpoint()
    if not endpoint:
        return _no_endpoint_error()

    payload = {"dataset": dataset, "limit": limit}
    if gene:
        payload["gene"] = gene
    if compound:
        payload["compound"] = compound
    if filters:
        payload["filters"] = filters

    try:
        resp = httpx.post(
            f"{endpoint}/query",
            json=payload,
            timeout=30,
        )

        if resp.status_code == 404:
            return {
                "error": f"Dataset '{dataset}' not found on the data API.",
                "summary": f"Dataset '{dataset}' is not available. Use remote_data.list_datasets to see what's available.",
            }
        if resp.status_code == 503:
            return {
                "error": f"Dataset '{dataset}' files not found on disk.",
                "summary": f"Dataset '{dataset}' is registered but files are missing on the server.",
            }
        if resp.status_code != 200:
            return {
                "error": f"Data API error: HTTP {resp.status_code}",
                "summary": f"Data API returned HTTP {resp.status_code}: {resp.text[:200]}",
            }

        data = resp.json()
        rows = data.get("data", [])
        total = data.get("total_rows", len(rows))

        # Build summary
        filter_desc = []
        if gene:
            filter_desc.append(f"gene={gene}")
        if compound:
            filter_desc.append(f"compound={compound}")
        filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""

        summary = f"Query {dataset}{filter_str}: {total} rows returned"
        if rows:
            cols = list(rows[0].keys())
            summary += f". Columns: {', '.join(cols[:8])}"

        return {
            "summary": summary,
            "dataset": dataset,
            "total_rows": total,
            "columns": list(rows[0].keys()) if rows else [],
            "data": rows,
        }

    except httpx.ConnectError:
        return {
            "error": f"Cannot connect to Data API at {endpoint}",
            "summary": f"Data API unreachable at {endpoint}. Check the server is running.",
        }
    except httpx.HTTPError as e:
        return {
            "error": f"Data API request failed: {e}",
            "summary": f"Failed to query Data API: {e}",
        }


@registry.register(
    name="remote_data.list_datasets",
    description="List datasets available on the hosted ct Data API",
    category="remote_data",
    parameters={},
    usage_guide=(
        "You want to see what datasets are available on the configured data API. "
        "Run this first to discover available data before querying."
    ),
)
def list_datasets(**kwargs) -> dict:
    """List datasets available on the Data API."""
    endpoint = _get_endpoint()
    if not endpoint:
        return _no_endpoint_error()

    try:
        resp = httpx.get(f"{endpoint}/datasets", timeout=10)

        if resp.status_code != 200:
            return {
                "error": f"Data API error: HTTP {resp.status_code}",
                "summary": f"Failed to list datasets: HTTP {resp.status_code}",
            }

        datasets = resp.json()
        if not datasets:
            return {
                "summary": "No datasets available on the Data API.",
                "datasets": [],
            }

        names = [d["name"] for d in datasets]
        summary = f"Data API has {len(datasets)} datasets: {', '.join(names)}"

        return {
            "summary": summary,
            "datasets": datasets,
        }

    except httpx.ConnectError:
        return {
            "error": f"Cannot connect to Data API at {endpoint}",
            "summary": f"Data API unreachable at {endpoint}. Check the server is running.",
        }
    except httpx.HTTPError as e:
        return {
            "error": f"Data API request failed: {e}",
            "summary": f"Failed to list datasets: {e}",
        }
