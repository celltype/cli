"""
Omics data discovery, download, and inspection tools.

Provides search and fetch capabilities for major public omics repositories:
- NCBI GEO (Gene Expression Omnibus)
- CELLxGENE Discover (Chan Zuckerberg Initiative)
- TCGA/GDC (The Cancer Genome Atlas via Genomic Data Commons)

Also provides local dataset inspection for downloaded files.
"""

import gzip
import logging
import re
import shutil
import tempfile
from pathlib import Path

from ct.tools import registry
from ct.tools.http_client import request, request_json

logger = logging.getLogger("ct.tools.omics")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _downloads_dir() -> Path:
    """Return (and create) the downloads directory."""
    from ct.agent.config import Config

    config = Config.load()
    base = config.get("data.downloads_dir", None)
    if base:
        d = Path(base).expanduser()
    else:
        d = Path.home() / ".ct" / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _max_download_mb() -> int:
    """Return the configured max download size in MB."""
    from ct.agent.config import Config

    config = Config.load()
    return int(config.get("data.max_download_mb", 500))


def _stream_download(url: str, dest_path: Path, max_mb: int | None = None) -> tuple[Path | None, str | None]:
    """Stream-download a file with size cap.

    Returns (path, None) on success or (None, error_string) on failure.
    """
    import httpx

    if max_mb is None:
        max_mb = _max_download_mb()

    max_bytes = max_mb * 1024 * 1024
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()

            # Check Content-Length if available
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                return None, (
                    f"File size ({int(content_length) // (1024*1024)} MB) "
                    f"exceeds limit ({max_mb} MB). "
                    f"Increase with: ct config set data.max_download_mb <value>"
                )

            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        tmp_path.unlink(missing_ok=True)
                        return None, (
                            f"Download exceeded size limit ({max_mb} MB). "
                            f"Increase with: ct config set data.max_download_mb <value>"
                        )
                    f.write(chunk)

        # Atomic rename
        shutil.move(str(tmp_path), str(dest_path))
        return dest_path, None

    except httpx.HTTPStatusError as exc:
        tmp_path.unlink(missing_ok=True)
        return None, f"HTTP {exc.response.status_code}: {str(exc)[:200]}"
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return None, f"Download failed: {str(exc)[:200]}"


def _check_scanpy():
    """Check if scanpy is available."""
    try:
        import scanpy as sc

        return sc
    except Exception as exc:
        logger.debug("scanpy unavailable or failed to import: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 1. omics.geo_search
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.geo_search",
    description="Search NCBI GEO for datasets by keyword, organism, and study type",
    category="omics",
    parameters={
        "query": "Search terms (gene, disease, compound, etc.)",
        "organism": "Organism filter (default 'Homo sapiens')",
        "study_type": "Filter: 'scRNA-seq', 'bulk RNA-seq', 'methylation', 'ATAC-seq', 'ChIP-seq', or 'all'",
        "max_results": "Maximum results to return (default 10)",
    },
    usage_guide=(
        "Search NCBI GEO for public omics datasets. Use before omics.geo_fetch "
        "to find relevant accessions. Supports filtering by organism and study type."
    ),
)
def geo_search(
    query: str,
    organism: str = "Homo sapiens",
    study_type: str = "all",
    max_results: int = 10,
    **kwargs,
) -> dict:
    """Search NCBI GEO for datasets."""
    if not query or not query.strip():
        return {"error": "Query is required", "summary": "No query provided"}

    # Build search term
    terms = [query.strip()]
    if organism and organism.lower() != "all":
        terms.append(f'"{organism}"[Organism]')

    study_type_keywords = {
        "scrna-seq": "single cell RNA-seq",
        "bulk rna-seq": "RNA-seq",
        "methylation": "methylation profiling",
        "atac-seq": "ATAC-seq",
        "chip-seq": "ChIP-seq",
    }
    st = study_type.lower().strip()
    if st != "all" and st in study_type_keywords:
        terms.append(study_type_keywords[st])

    search_term = " AND ".join(terms)

    # Step 1: esearch
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    data, error = request_json(
        "GET",
        esearch_url,
        params={
            "db": "gds",
            "term": search_term,
            "retmax": str(min(max_results, 50)),
            "retmode": "json",
        },
        timeout=15,
    )
    if error:
        return {"error": f"GEO search failed: {error}", "summary": f"GEO search error: {error}"}

    esearch_result = data.get("esearchresult", {})
    id_list = esearch_result.get("idlist", [])
    if not id_list:
        return {
            "datasets": [],
            "query": search_term,
            "count": 0,
            "summary": f"No GEO datasets found for '{query}' (organism={organism}, type={study_type})",
        }

    # Step 2: esummary
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    summary_data, error = request_json(
        "GET",
        esummary_url,
        params={
            "db": "gds",
            "id": ",".join(id_list),
            "retmode": "json",
        },
        timeout=15,
    )
    if error:
        return {"error": f"GEO summary fetch failed: {error}", "summary": f"GEO summary error: {error}"}

    result_block = summary_data.get("result", {})
    datasets = []
    for uid in id_list:
        entry = result_block.get(uid, {})
        if not entry or isinstance(entry, str):
            continue
        accession = entry.get("accession", "")
        # GDS entries may not have GSE accession directly; extract from related
        if not accession.startswith("GSE"):
            gse = entry.get("gse", "")
            if gse:
                accession = f"GSE{gse}"
        datasets.append({
            "accession": accession,
            "title": entry.get("title", ""),
            "summary": (entry.get("summary", "") or "")[:300],
            "organism": entry.get("taxon", ""),
            "platform": entry.get("gpl", ""),
            "sample_count": entry.get("n_samples", 0),
            "study_type": entry.get("gdstype", study_type),
            "date": entry.get("pdat", ""),
        })

    return {
        "datasets": datasets,
        "query": search_term,
        "count": len(datasets),
        "summary": (
            f"Found {len(datasets)} GEO dataset(s) for '{query}'. "
            + "; ".join(
                f"{d['accession']}: {d['title'][:60]}" for d in datasets[:3]
            )
        ),
    }


# ---------------------------------------------------------------------------
# 2. omics.geo_fetch
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.geo_fetch",
    description="Download a GEO dataset (expression matrix or supplementary files)",
    category="omics",
    parameters={
        "accession": "GEO accession (e.g., 'GSE12345')",
        "file_type": "Type to download: 'matrix', 'h5ad', 'supplementary' (default 'matrix')",
    },
    usage_guide=(
        "Download data from NCBI GEO after finding accessions with omics.geo_search. "
        "Use 'matrix' for series matrix files, 'supplementary' for raw/processed supplements."
    ),
)
def geo_fetch(accession: str, file_type: str = "matrix", **kwargs) -> dict:
    """Download a GEO dataset."""
    # Validate accession
    if not accession or not re.match(r"^GSE\d+$", accession.strip()):
        return {
            "error": f"Invalid GEO accession '{accession}'. Expected format: GSE12345",
            "summary": f"Invalid accession format: {accession}",
        }

    accession = accession.strip().upper()
    # GEO FTP path uses first 3+nnn digits: GSE12345 → GSE12nnn
    prefix = accession[:len(accession) - 3] + "nnn"

    dest_dir = _downloads_dir() / "geo" / accession
    dest_dir.mkdir(parents=True, exist_ok=True)

    if file_type == "matrix":
        filename = f"{accession}_series_matrix.txt.gz"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/matrix/{filename}"
        dest = dest_dir / filename

        if dest.exists():
            size_mb = round(dest.stat().st_size / (1024 * 1024), 2)
            return {
                "path": str(dest),
                "accession": accession,
                "file_type": file_type,
                "size_mb": size_mb,
                "summary": f"Already downloaded: {dest.name} ({size_mb} MB)",
            }

        path, error = _stream_download(url, dest)
        if error:
            return {"error": error, "accession": accession, "summary": f"Download failed for {accession}: {error}"}

        size_mb = round(path.stat().st_size / (1024 * 1024), 2)
        return {
            "path": str(path),
            "accession": accession,
            "file_type": file_type,
            "size_mb": size_mb,
            "summary": f"Downloaded {accession} series matrix ({size_mb} MB) to {path}",
        }

    elif file_type in ("h5ad", "supplementary"):
        # List supplementary files page
        suppl_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/suppl/"
        resp, error = request("GET", suppl_url, timeout=15, raise_for_status=False)
        if error:
            return {"error": f"Could not list supplementary files: {error}", "summary": f"Supplementary listing failed for {accession}"}

        # Parse HTML directory listing for file links
        text = resp.text if hasattr(resp, "text") else str(resp)
        links = re.findall(r'href="([^"]+)"', text)
        data_files = [l for l in links if not l.startswith("?") and not l.startswith("/") and l != "../"]

        if not data_files:
            return {
                "error": f"No supplementary files found for {accession}",
                "summary": f"No supplementary files available for {accession}",
            }

        # For h5ad, prefer .h5ad files; otherwise take first data file
        target = None
        if file_type == "h5ad":
            h5ad_files = [f for f in data_files if f.endswith(".h5ad") or f.endswith(".h5ad.gz")]
            if h5ad_files:
                target = h5ad_files[0]
            else:
                return {
                    "error": f"No h5ad files found in {accession} supplementary files",
                    "files_available": data_files[:10],
                    "summary": f"No h5ad files in {accession}. Available: {', '.join(data_files[:5])}",
                }
        else:
            target = data_files[0]

        file_url = f"{suppl_url}{target}"
        dest = dest_dir / target

        if dest.exists():
            size_mb = round(dest.stat().st_size / (1024 * 1024), 2)
            return {
                "path": str(dest),
                "accession": accession,
                "file_type": file_type,
                "filename": target,
                "size_mb": size_mb,
                "summary": f"Already downloaded: {target} ({size_mb} MB)",
            }

        path, error = _stream_download(file_url, dest)
        if error:
            return {"error": error, "accession": accession, "summary": f"Download failed: {error}"}

        size_mb = round(path.stat().st_size / (1024 * 1024), 2)
        return {
            "path": str(path),
            "accession": accession,
            "file_type": file_type,
            "filename": target,
            "size_mb": size_mb,
            "summary": f"Downloaded {target} ({size_mb} MB) from {accession}",
        }

    else:
        return {
            "error": f"Invalid file_type '{file_type}'. Choose: matrix, h5ad, supplementary",
            "summary": f"Invalid file_type: {file_type}",
        }


# ---------------------------------------------------------------------------
# 3. omics.cellxgene_search
# ---------------------------------------------------------------------------

_CELLXGENE_API = "https://api.cellxgene.cziscience.com/curation/v1"


@registry.register(
    name="omics.cellxgene_search",
    description="Search CELLxGENE Discover for curated single-cell datasets",
    category="omics",
    parameters={
        "query": "Search terms (gene, disease, tissue, etc.)",
        "tissue": "Filter by tissue (optional)",
        "disease": "Filter by disease (optional)",
        "organism": "Filter by organism (default 'Homo sapiens')",
        "max_results": "Maximum results to return (default 10)",
    },
    usage_guide=(
        "Search the CZI CELLxGENE Discover portal for curated, analysis-ready "
        "single-cell datasets. Use before omics.cellxgene_fetch to get dataset IDs."
    ),
)
def cellxgene_search(
    query: str,
    tissue: str = "",
    disease: str = "",
    organism: str = "Homo sapiens",
    max_results: int = 10,
    **kwargs,
) -> dict:
    """Search CELLxGENE Discover for single-cell datasets."""
    if not query or not query.strip():
        return {"error": "Query is required", "summary": "No query provided"}

    # Fetch collections
    url = f"{_CELLXGENE_API}/collections"
    data, error = request_json("GET", url, timeout=20)
    if error:
        return {"error": f"CELLxGENE search failed: {error}", "summary": f"CELLxGENE error: {error}"}

    if not isinstance(data, list):
        return {"error": "Unexpected CELLxGENE response format", "summary": "CELLxGENE returned unexpected format"}

    query_lower = query.lower().strip()
    query_terms = query_lower.split()
    results = []

    for collection in data:
        # Check collection-level match
        col_title = (collection.get("name") or "").lower()
        col_desc = (collection.get("description") or "").lower()
        col_text = col_title + " " + col_desc

        col_matches = any(term in col_text for term in query_terms)

        for dataset in collection.get("datasets", []):
            ds_title = (dataset.get("title") or dataset.get("name") or "").lower()
            ds_text = ds_title + " " + col_text

            # Match query
            if not col_matches and not any(term in ds_text for term in query_terms):
                continue

            # Filter organism
            ds_organisms = [
                o.get("label", "").lower()
                for o in (dataset.get("organism", []) if isinstance(dataset.get("organism"), list) else [])
            ]
            if organism and organism.lower() not in " ".join(ds_organisms) and ds_organisms:
                continue

            # Filter tissue
            ds_tissues = [
                t.get("label", "").lower()
                for t in (dataset.get("tissue", []) if isinstance(dataset.get("tissue"), list) else [])
            ]
            if tissue and tissue.lower() not in " ".join(ds_tissues):
                continue

            # Filter disease
            ds_diseases = [
                d.get("label", "").lower()
                for d in (dataset.get("disease", []) if isinstance(dataset.get("disease"), list) else [])
            ]
            if disease and disease.lower() not in " ".join(ds_diseases):
                continue

            # Extract assay info
            ds_assays = [
                a.get("label", "")
                for a in (dataset.get("assay", []) if isinstance(dataset.get("assay"), list) else [])
            ]

            results.append({
                "dataset_id": dataset.get("dataset_id", ""),
                "collection_id": collection.get("collection_id", ""),
                "title": dataset.get("title") or dataset.get("name") or col_title,
                "description": (col_desc[:200] if col_desc else ""),
                "tissue": ", ".join(t.get("label", "") for t in (dataset.get("tissue", []) if isinstance(dataset.get("tissue"), list) else [])),
                "disease": ", ".join(d.get("label", "") for d in (dataset.get("disease", []) if isinstance(dataset.get("disease"), list) else [])),
                "cell_count": dataset.get("cell_count", 0),
                "organism": ", ".join(o.get("label", "") for o in (dataset.get("organism", []) if isinstance(dataset.get("organism"), list) else [])),
                "assay": ", ".join(ds_assays),
            })

            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    return {
        "datasets": results,
        "query": query,
        "count": len(results),
        "summary": (
            f"Found {len(results)} CELLxGENE dataset(s) for '{query}'. "
            + ("; ".join(f"{d['title'][:50]} ({d['cell_count']} cells)" for d in results[:3]) if results else "Try broader search terms.")
        ),
    }


# ---------------------------------------------------------------------------
# 4. omics.cellxgene_fetch
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.cellxgene_fetch",
    description="Download an h5ad dataset from CELLxGENE Discover",
    category="omics",
    parameters={
        "dataset_id": "CELLxGENE dataset ID (from omics.cellxgene_search results)",
    },
    usage_guide=(
        "Download a single-cell dataset from CELLxGENE. Requires a dataset_id "
        "from omics.cellxgene_search results. Downloads as h5ad format."
    ),
)
def cellxgene_fetch(dataset_id: str, **kwargs) -> dict:
    """Download an h5ad dataset from CELLxGENE."""
    if not dataset_id or not dataset_id.strip():
        return {"error": "dataset_id is required", "summary": "No dataset_id provided"}

    dataset_id = dataset_id.strip()

    # Get asset list
    assets_url = f"{_CELLXGENE_API}/datasets/{dataset_id}/assets"
    assets, error = request_json("GET", assets_url, timeout=15)
    if error:
        return {"error": f"Failed to get assets: {error}", "summary": f"CELLxGENE asset lookup failed: {error}"}

    if not isinstance(assets, list) or not assets:
        return {
            "error": f"No downloadable assets found for dataset {dataset_id}",
            "summary": f"No assets for dataset {dataset_id}",
        }

    # Find h5ad asset
    h5ad_asset = None
    for asset in assets:
        filetype = (asset.get("filetype") or asset.get("file_type") or "").lower()
        filename = (asset.get("filename") or "").lower()
        if "h5ad" in filetype or filename.endswith(".h5ad"):
            h5ad_asset = asset
            break

    if not h5ad_asset:
        # Fall back to first asset
        h5ad_asset = assets[0]

    download_url = h5ad_asset.get("presigned_url") or h5ad_asset.get("url", "")
    if not download_url:
        return {
            "error": "No download URL in asset metadata",
            "summary": "CELLxGENE asset has no download URL",
        }

    filename = h5ad_asset.get("filename", f"{dataset_id}.h5ad")
    dest_dir = _downloads_dir() / "cellxgene" / dataset_id
    dest = dest_dir / filename

    if dest.exists():
        size_mb = round(dest.stat().st_size / (1024 * 1024), 2)
        return {
            "path": str(dest),
            "dataset_id": dataset_id,
            "size_mb": size_mb,
            "summary": f"Already downloaded: {filename} ({size_mb} MB)",
        }

    path, error = _stream_download(download_url, dest)
    if error:
        return {"error": error, "dataset_id": dataset_id, "summary": f"Download failed: {error}"}

    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    return {
        "path": str(path),
        "dataset_id": dataset_id,
        "filename": filename,
        "size_mb": size_mb,
        "summary": f"Downloaded CELLxGENE dataset {dataset_id} ({size_mb} MB) to {path}",
    }


# ---------------------------------------------------------------------------
# 5. omics.tcga_search
# ---------------------------------------------------------------------------

_GDC_API = "https://api.gdc.cancer.gov"


@registry.register(
    name="omics.tcga_search",
    description="Search TCGA/GDC for cancer genomics projects and data files",
    category="omics",
    parameters={
        "query": "Search terms (cancer type, gene, etc.)",
        "data_type": "Filter: 'gene_expression', 'methylation', 'mutation', 'clinical' (default 'gene_expression')",
        "max_results": "Maximum results to return (default 10)",
    },
    usage_guide=(
        "Search the NCI Genomic Data Commons (GDC) for TCGA and other cancer "
        "genomics projects. Use before omics.tcga_fetch to find file UUIDs."
    ),
)
def tcga_search(
    query: str,
    data_type: str = "gene_expression",
    max_results: int = 10,
    **kwargs,
) -> dict:
    """Search TCGA/GDC for projects and data files."""
    if not query or not query.strip():
        return {"error": "Query is required", "summary": "No query provided"}

    valid_types = {"gene_expression", "methylation", "mutation", "clinical"}
    if data_type not in valid_types:
        return {
            "error": f"Invalid data_type '{data_type}'. Choose from: {', '.join(valid_types)}",
            "summary": f"Invalid data_type: {data_type}",
        }

    # Map requested analysis type to GDC project summary data categories.
    # Project summaries expose category-level counts, not file-level data_type counts.
    gdc_data_category_map = {
        "gene_expression": "Transcriptome Profiling",
        "methylation": "DNA Methylation",
        "mutation": "Simple Nucleotide Variation",
        "clinical": "Clinical",
    }

    # Search projects first
    projects_url = f"{_GDC_API}/projects"
    filters = {
        "op": "or",
        "content": [
            {"op": "in", "content": {"field": "project_id", "value": [query.upper()]}},
            {"op": "like", "content": {"field": "name", "value": f"*{query}*"}},
            {"op": "like", "content": {"field": "disease_type", "value": f"*{query}*"}},
            {"op": "like", "content": {"field": "primary_site", "value": f"*{query}*"}},
        ],
    }

    import json

    params = {
        "filters": json.dumps(filters),
        "fields": "project_id,name,disease_type,primary_site,summary.case_count,summary.file_count,summary.data_categories.data_category,summary.data_categories.file_count",
        "size": str(min(max_results, 50)),
        "format": "json",
    }

    data, error = request_json("GET", projects_url, params=params, timeout=15)
    if error:
        return {"error": f"GDC search failed: {error}", "summary": f"GDC error: {error}"}

    hits = data.get("data", {}).get("hits", [])
    projects = []
    for hit in hits:
        summary = hit.get("summary", {})
        # Count files in the category most relevant to requested data_type.
        data_cats = summary.get("data_categories", [])
        requested_category = gdc_data_category_map.get(data_type, "")
        category_file_count = 0
        available_categories = []
        for cat in data_cats:
            cat_name = cat.get("data_category", "")
            if cat_name:
                available_categories.append(cat_name)
            if cat_name.lower() == requested_category.lower():
                category_file_count = int(cat.get("file_count", 0) or 0)

        projects.append({
            "project_id": hit.get("project_id", ""),
            "name": hit.get("name", ""),
            "disease_type": hit.get("disease_type", ""),
            "primary_site": hit.get("primary_site", ""),
            "case_count": summary.get("case_count", 0),
            "file_count": summary.get("file_count", 0),
            "data_type": data_type,
            "matching_data_category": requested_category,
            "data_type_file_count": category_file_count,
            "available_data_categories": available_categories[:20],
            "count_method": "project_summary_data_category",
        })

    if not projects:
        return {
            "projects": [],
            "query": query,
            "count": 0,
            "summary": f"No TCGA/GDC projects found for '{query}'",
        }

    return {
        "projects": projects,
        "query": query,
        "data_type": data_type,
        "count": len(projects),
        "summary": (
            f"Found {len(projects)} GDC project(s) for '{query}'. "
            + "; ".join(f"{p['project_id']}: {p['name'][:40]} ({p['case_count']} cases)" for p in projects[:3])
        ),
    }


# ---------------------------------------------------------------------------
# 6. omics.tcga_fetch
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.tcga_fetch",
    description="Download a data file from TCGA/GDC",
    category="omics",
    parameters={
        "file_id": "GDC file UUID to download",
        "project_id": "GDC project ID (optional, used to search for files if file_id not provided)",
    },
    usage_guide=(
        "Download a specific file from GDC by UUID. If only project_id is given, "
        "searches for the most relevant gene expression file and downloads it."
    ),
)
def tcga_fetch(file_id: str = "", project_id: str = "", **kwargs) -> dict:
    """Download a data file from TCGA/GDC."""
    import json

    if not file_id and not project_id:
        return {
            "error": "Either file_id or project_id is required",
            "summary": "No file_id or project_id provided",
        }

    # If no file_id, search for one from the project
    if not file_id:
        files_url = f"{_GDC_API}/files"
        filters = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
                {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ],
        }
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,data_type",
            "size": "1",
            "format": "json",
        }
        data, error = request_json("GET", files_url, params=params, timeout=15)
        if error:
            return {"error": f"File search failed: {error}", "summary": f"GDC file search error: {error}"}

        hits = data.get("data", {}).get("hits", [])
        if not hits:
            return {
                "error": f"No open-access files found for project {project_id}",
                "summary": f"No downloadable files for {project_id}",
            }
        file_id = hits[0].get("file_id", "")
        file_name = hits[0].get("file_name", f"{file_id}.gz")
    else:
        file_name = f"{file_id}.gz"

    # Download the file
    download_url = f"{_GDC_API}/data/{file_id}"
    label = project_id or file_id[:12]
    dest_dir = _downloads_dir() / "tcga" / label
    dest = dest_dir / file_name

    if dest.exists():
        size_mb = round(dest.stat().st_size / (1024 * 1024), 2)
        return {
            "path": str(dest),
            "file_id": file_id,
            "project_id": project_id,
            "size_mb": size_mb,
            "summary": f"Already downloaded: {file_name} ({size_mb} MB)",
        }

    path, error = _stream_download(download_url, dest)
    if error:
        return {"error": error, "file_id": file_id, "summary": f"Download failed: {error}"}

    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    return {
        "path": str(path),
        "file_id": file_id,
        "project_id": project_id,
        "filename": file_name,
        "size_mb": size_mb,
        "summary": f"Downloaded GDC file {file_name} ({size_mb} MB) to {path}",
    }


# ---------------------------------------------------------------------------
# 7. omics.dataset_info
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.dataset_info",
    description="Inspect a downloaded dataset file and return metadata summary",
    category="omics",
    parameters={
        "path": "Path to the downloaded dataset file (h5ad, CSV, TSV, or matrix.txt.gz)",
    },
    usage_guide=(
        "Inspect a downloaded omics file before analysis. Returns shape, columns, "
        "metadata. Use after omics.*_fetch to understand the data before running "
        "singlecell.* or code.execute on it."
    ),
)
def dataset_info(path: str, **kwargs) -> dict:
    """Inspect a downloaded dataset file and return metadata."""
    if not path:
        return {"error": "Path is required", "summary": "No path provided"}

    filepath = Path(path).expanduser()
    if not filepath.exists():
        return {"error": f"File not found: {path}", "summary": f"File not found: {path}"}

    size_mb = round(filepath.stat().st_size / (1024 * 1024), 2)
    suffix = filepath.suffix.lower()

    # Handle .gz suffix
    if suffix == ".gz":
        inner_suffix = Path(filepath.stem).suffix.lower()
        suffix = inner_suffix + suffix  # e.g. ".txt.gz"

    try:
        if suffix == ".h5ad":
            return _inspect_h5ad(filepath, size_mb)
        elif suffix in (".csv", ".tsv", ".txt"):
            return _inspect_tabular(filepath, size_mb, sep="," if suffix == ".csv" else "\t")
        elif suffix in (".txt.gz",):
            return _inspect_matrix_gz(filepath, size_mb)
        else:
            return {
                "path": str(filepath),
                "file_type": suffix,
                "size_mb": size_mb,
                "summary": f"File type '{suffix}' not directly inspectable. Size: {size_mb} MB. Try loading with code.execute.",
            }
    except Exception as exc:
        return {
            "error": f"Inspection failed: {str(exc)[:200]}",
            "path": str(filepath),
            "size_mb": size_mb,
            "summary": f"Could not inspect {filepath.name}: {str(exc)[:100]}",
        }


def _inspect_h5ad(filepath: Path, size_mb: float) -> dict:
    """Inspect an h5ad file using scanpy."""
    sc = _check_scanpy()
    if sc is None:
        return {
            "path": str(filepath),
            "file_type": "h5ad",
            "size_mb": size_mb,
            "error": "scanpy not installed. Install with: pip install scanpy",
            "summary": f"h5ad file ({size_mb} MB) — install scanpy to inspect: pip install scanpy",
        }

    adata = sc.read_h5ad(filepath)
    obs_cols = list(adata.obs.columns)
    var_cols = list(adata.var.columns)
    layers = list(adata.layers.keys()) if adata.layers else []

    return {
        "path": str(filepath),
        "file_type": "h5ad",
        "size_mb": size_mb,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "obs_columns": obs_cols[:20],
        "var_columns": var_cols[:20],
        "layers": layers,
        "obs_preview": {col: list(adata.obs[col].unique()[:5]) for col in obs_cols[:5]},
        "summary": (
            f"h5ad: {adata.n_obs:,} cells x {adata.n_vars:,} genes ({size_mb} MB). "
            f"Obs columns: {', '.join(obs_cols[:8])}. "
            f"Layers: {', '.join(layers) if layers else 'X only'}."
        ),
    }


def _inspect_tabular(filepath: Path, size_mb: float, sep: str = ",") -> dict:
    """Inspect a CSV/TSV file."""
    import pandas as pd

    # Read just the first rows to get shape info without loading everything
    df_head = pd.read_csv(filepath, sep=sep, nrows=5, index_col=0)
    # Get full shape by counting lines
    with open(filepath) as f:
        n_lines = sum(1 for _ in f) - 1  # subtract header

    columns = list(df_head.columns)
    dtypes = {col: str(dtype) for col, dtype in df_head.dtypes.items()}

    return {
        "path": str(filepath),
        "file_type": "csv" if sep == "," else "tsv",
        "size_mb": size_mb,
        "shape": [n_lines, len(columns)],
        "columns": columns[:30],
        "dtypes": {k: v for k, v in list(dtypes.items())[:15]},
        "head_preview": df_head.head(3).to_dict(),
        "summary": (
            f"Tabular: {n_lines:,} rows x {len(columns)} columns ({size_mb} MB). "
            f"Columns: {', '.join(columns[:8])}"
        ),
    }


def _inspect_matrix_gz(filepath: Path, size_mb: float) -> dict:
    """Inspect a GEO series matrix .txt.gz file."""
    metadata = {}
    n_rows = 0
    columns = []

    with gzip.open(filepath, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!"):
                # Parse metadata lines
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    key = parts[0].lstrip("!").strip()
                    val = parts[1].strip().strip('"')
                    if key not in metadata:
                        metadata[key] = val
                    elif isinstance(metadata[key], list):
                        metadata[key].append(val)
                    else:
                        metadata[key] = [metadata[key], val]
            elif line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                columns = [c.strip('"') for c in line.strip().split("\t")]
            elif not line.startswith("!") and line.strip():
                n_rows += 1

    # Extract key metadata fields
    title = metadata.get("Series_title", "")
    organism = metadata.get("Series_organism", "")
    n_samples = len(columns) - 1 if columns else 0

    return {
        "path": str(filepath),
        "file_type": "matrix.txt.gz",
        "size_mb": size_mb,
        "title": title,
        "organism": organism,
        "n_probes_or_genes": n_rows,
        "n_samples": n_samples,
        "sample_ids": columns[1:11] if columns else [],
        "metadata_keys": list(metadata.keys())[:15],
        "summary": (
            f"GEO matrix: {n_rows:,} probes/genes x {n_samples} samples ({size_mb} MB). "
            f"Title: {title[:80]}. Organism: {organism}."
        ),
    }


# ===========================================================================
# Analysis tools — modality-specific processing of downloaded data
# ===========================================================================


def _load_tabular(path: str, **read_kwargs) -> "tuple[pd.DataFrame | None, str | None]":
    """Load a tabular file, returning (df, error)."""
    import pandas as pd

    filepath = Path(path).expanduser()
    if not filepath.exists():
        return None, f"File not found: {path}"
    suffix = filepath.suffix.lower()
    kwargs = dict(read_kwargs)
    try:
        # Keep prior behavior by defaulting to first column as index,
        # while allowing callers to override.
        kwargs.setdefault("index_col", 0)
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(filepath, **kwargs)
            return df, None
        if suffix == ".csv":
            df = pd.read_csv(filepath, sep=",", **kwargs)
            return df, None
        if suffix in {".tsv", ".tab"}:
            df = pd.read_csv(filepath, sep="\t", **kwargs)
            return df, None
        if suffix == ".txt":
            # Many omics count matrices are whitespace-delimited.
            try:
                df = pd.read_csv(filepath, sep=r"\s+", engine="python", **kwargs)
                return df, None
            except Exception:
                df = pd.read_csv(filepath, sep="\t", **kwargs)
                return df, None
        # Generic fallback: delimiter sniffing for unknown text-like files.
        df = pd.read_csv(filepath, sep=None, engine="python", **kwargs)
        return df, None
    except Exception as exc:
        return None, f"Failed to read {filepath.name}: {str(exc)[:200]}"


def _parse_sample_groups(
    df,
    group1: str = "",
    group2: str = "",
    *,
    auto_grouping: bool = False,
    min_group_size: int = 2,
    group_names: tuple[str, str] = ("group1", "group2"),
) -> tuple[list[str], list[str], dict | None]:
    """Resolve and validate group sample assignments for two-group comparisons."""
    all_samples = [str(c) for c in df.columns]
    g1_label, g2_label = group_names
    g1_samples = [s.strip() for s in group1.split(",") if s.strip()] if group1 else []
    g2_samples = [s.strip() for s in group2.split(",") if s.strip()] if group2 else []

    # Require explicit groups unless user opts in to auto-splitting.
    if not g1_samples and not g2_samples:
        if not auto_grouping:
            return [], [], {
                "error": (
                    f"Explicit sample groups are required. Provide {g1_label} and {g2_label} "
                    "as comma-separated sample names. "
                    "Set auto_grouping=True only for quick exploratory analysis."
                ),
                "available_samples": all_samples[:30],
                "n_samples": len(all_samples),
                "summary": (
                    f"No groups provided. Define {g1_label}/{g2_label} using sample names "
                    f"(found {len(all_samples)} samples)."
                ),
            }

        if len(all_samples) < (min_group_size * 2):
            return [], [], {
                "error": (
                    f"Need at least {min_group_size * 2} samples for auto_grouping "
                    f"({min_group_size} per group), found {len(all_samples)}."
                ),
                "available_samples": all_samples[:30],
                "summary": f"Too few samples for auto_grouping: {len(all_samples)}",
            }

        mid = len(all_samples) // 2
        g1_samples = all_samples[:mid]
        g2_samples = all_samples[mid:]

    elif (g1_samples and not g2_samples) or (g2_samples and not g1_samples):
        return [], [], {
            "error": f"Both {g1_label} and {g2_label} must be provided together.",
            "available_samples": all_samples[:30],
            "summary": f"Incomplete group definition: need both {g1_label} and {g2_label}",
        }

    missing = [s for s in (g1_samples + g2_samples) if s not in all_samples]
    if missing:
        return [], [], {
            "error": f"Samples not found: {missing}",
            "available_samples": all_samples[:30],
            "summary": f"Sample names not found in matrix. Available: {', '.join(all_samples[:10])}",
        }

    overlap = sorted(set(g1_samples).intersection(g2_samples))
    if overlap:
        return [], [], {
            "error": f"Samples cannot appear in both groups: {overlap}",
            "summary": "Group overlap detected",
        }

    if len(g1_samples) < min_group_size or len(g2_samples) < min_group_size:
        return [], [], {
            "error": (
                f"Each group needs at least {min_group_size} samples. "
                f"Got {g1_label}={len(g1_samples)}, {g2_label}={len(g2_samples)}."
            ),
            "summary": "Insufficient replicates per group",
        }

    return g1_samples, g2_samples, None


def _fdr_correct(pvalues):
    """Benjamini-Hochberg FDR correction. Returns array of q-values."""
    import numpy as np

    pvals = np.asarray(pvalues, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    ranked = pvals.argsort().argsort() + 1  # 1-based rank
    qvals = pvals * n / ranked
    # Enforce monotonicity (from largest p-value down)
    order = pvals.argsort()[::-1]
    qvals_sorted = qvals[order]
    for i in range(1, len(qvals_sorted)):
        if qvals_sorted[i] > qvals_sorted[i - 1]:
            qvals_sorted[i] = qvals_sorted[i - 1]
    qvals[order] = qvals_sorted
    return np.clip(qvals, 0, 1)


# ---------------------------------------------------------------------------
# 8. omics.methylation_diff
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.methylation_diff",
    description="Differential methylation analysis between two sample groups",
    category="omics",
    parameters={
        "path": "Path to methylation beta-value matrix (rows=CpG sites, cols=samples)",
        "group1": "Comma-separated sample names for group 1",
        "group2": "Comma-separated sample names for group 2",
        "auto_grouping": "If true, splits samples by column order for exploratory use (default false)",
        "delta_beta_cutoff": "Minimum absolute delta-beta to call DMR (default 0.2)",
        "fdr_cutoff": "FDR significance threshold (default 0.05)",
    },
    usage_guide=(
        "Analyze differential methylation from beta-value matrices (e.g., Illumina 450K/EPIC). "
        "Requires a matrix with CpG sites as rows and samples as columns. "
        "Use after omics.geo_fetch or omics.tcga_fetch to download methylation data. "
        "For reliable analysis, provide explicit group1/group2 sample lists."
    ),
)
def methylation_diff(
    path: str,
    group1: str = "",
    group2: str = "",
    auto_grouping: bool = False,
    delta_beta_cutoff: float = 0.2,
    fdr_cutoff: float = 0.05,
    **kwargs,
) -> dict:
    """Differential methylation analysis between two groups."""
    import numpy as np
    from scipy import stats

    df, error = _load_tabular(path)
    if error:
        return {"error": error, "summary": f"Could not load methylation data: {error}"}

    g1_samples, g2_samples, group_error = _parse_sample_groups(
        df,
        group1=group1,
        group2=group2,
        auto_grouping=auto_grouping,
        min_group_size=2,
    )
    if group_error:
        return group_error

    g1 = df[g1_samples].dropna(how="all")
    g2 = df[g2_samples].dropna(how="all")
    common_sites = g1.index.intersection(g2.index)
    g1 = g1.loc[common_sites]
    g2 = g2.loc[common_sites]

    # Calculate delta-beta and p-values
    mean1 = g1.mean(axis=1)
    mean2 = g2.mean(axis=1)
    delta_beta = mean2 - mean1

    pvals = []
    for site in common_sites:
        v1 = g1.loc[site].dropna().values
        v2 = g2.loc[site].dropna().values
        if len(v1) >= 2 and len(v2) >= 2:
            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            pvals.append(p)
        else:
            pvals.append(1.0)

    pvals = np.array(pvals)
    qvals = _fdr_correct(pvals)

    # Identify DMRs
    sig_mask = (qvals < fdr_cutoff) & (np.abs(delta_beta.values) >= delta_beta_cutoff)
    n_sig = int(sig_mask.sum())
    hyper = int(((qvals < fdr_cutoff) & (delta_beta.values >= delta_beta_cutoff)).sum())
    hypo = int(((qvals < fdr_cutoff) & (delta_beta.values <= -delta_beta_cutoff)).sum())

    # Top hits
    import pandas as pd

    results_df = pd.DataFrame({
        "mean_group1": mean1,
        "mean_group2": mean2,
        "delta_beta": delta_beta,
        "pvalue": pvals,
        "fdr": qvals,
    }, index=common_sites)
    results_df = results_df.sort_values("fdr")
    top_hits = results_df.head(20).to_dict("index")

    return {
        "n_sites_tested": len(common_sites),
        "n_significant": n_sig,
        "n_hypermethylated": hyper,
        "n_hypomethylated": hypo,
        "group1_samples": g1_samples,
        "group2_samples": g2_samples,
        "auto_grouping_used": bool(auto_grouping and not group1 and not group2),
        "delta_beta_cutoff": delta_beta_cutoff,
        "fdr_cutoff": fdr_cutoff,
        "top_hits": top_hits,
        "summary": (
            f"Tested {len(common_sites):,} CpG sites: {n_sig} significant (FDR<{fdr_cutoff}, "
            f"|Δβ|≥{delta_beta_cutoff}). {hyper} hypermethylated, {hypo} hypomethylated."
        ),
    }


# ---------------------------------------------------------------------------
# 9. omics.methylation_profile
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.methylation_profile",
    description="Summarize methylation landscape: distribution, variability, and global patterns",
    category="omics",
    parameters={
        "path": "Path to methylation beta-value matrix",
    },
    usage_guide=(
        "Get an overview of a methylation dataset: global methylation levels, "
        "bimodal distribution (typical of 450K/EPIC), most variable CpGs. "
        "Use as a first step before methylation_diff."
    ),
)
def methylation_profile(path: str, **kwargs) -> dict:
    """Summarize methylation dataset landscape."""
    import numpy as np

    df, error = _load_tabular(path)
    if error:
        return {"error": error, "summary": f"Could not load: {error}"}

    n_sites, n_samples = df.shape
    all_vals = df.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]

    # Global statistics
    global_mean = float(np.mean(all_vals))
    global_median = float(np.median(all_vals))
    frac_low = float(np.mean(all_vals < 0.2))  # unmethylated
    frac_mid = float(np.mean((all_vals >= 0.2) & (all_vals <= 0.8)))  # intermediate
    frac_high = float(np.mean(all_vals > 0.8))  # methylated

    # Most variable sites
    site_var = df.var(axis=1).dropna().sort_values(ascending=False)
    top_variable = list(site_var.head(20).index)

    # Per-sample mean methylation
    sample_means = df.mean(axis=0).to_dict()

    return {
        "n_sites": n_sites,
        "n_samples": n_samples,
        "global_mean_beta": round(global_mean, 4),
        "global_median_beta": round(global_median, 4),
        "fraction_unmethylated": round(frac_low, 3),
        "fraction_intermediate": round(frac_mid, 3),
        "fraction_methylated": round(frac_high, 3),
        "top_variable_sites": top_variable,
        "sample_mean_betas": {k: round(v, 4) for k, v in list(sample_means.items())[:20]},
        "summary": (
            f"Methylation profile: {n_sites:,} sites x {n_samples} samples. "
            f"Global mean β={global_mean:.3f}. "
            f"Distribution: {frac_low:.0%} low (<0.2), {frac_mid:.0%} intermediate, {frac_high:.0%} high (>0.8)."
        ),
    }


# ---------------------------------------------------------------------------
# 10. omics.proteomics_diff
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.proteomics_diff",
    description="Differential protein abundance analysis between two groups",
    category="omics",
    parameters={
        "path": "Path to protein abundance matrix (rows=proteins, cols=samples)",
        "group1": "Comma-separated sample names for group 1",
        "group2": "Comma-separated sample names for group 2",
        "auto_grouping": "If true, splits samples by column order for exploratory use (default false)",
        "fc_cutoff": "Minimum absolute log2 fold-change (default 1.0)",
        "fdr_cutoff": "FDR significance threshold (default 0.05)",
    },
    usage_guide=(
        "Differential protein abundance from proteomics data (e.g., TMT, LFQ). "
        "Input is a protein x sample matrix of log2 abundances or intensities. "
        "Provide explicit group1/group2 sample lists for production analyses."
    ),
)
def proteomics_diff(
    path: str,
    group1: str = "",
    group2: str = "",
    auto_grouping: bool = False,
    fc_cutoff: float = 1.0,
    fdr_cutoff: float = 0.05,
    **kwargs,
) -> dict:
    """Differential protein abundance analysis."""
    import numpy as np
    from scipy import stats

    df, error = _load_tabular(path)
    if error:
        return {"error": error, "summary": f"Could not load proteomics data: {error}"}

    g1_samples, g2_samples, group_error = _parse_sample_groups(
        df,
        group1=group1,
        group2=group2,
        auto_grouping=auto_grouping,
        min_group_size=2,
    )
    if group_error:
        return group_error

    g1 = df[g1_samples]
    g2 = df[g2_samples]

    mean1 = g1.mean(axis=1)
    mean2 = g2.mean(axis=1)
    log2fc = mean2 - mean1  # already log2 if input is log2

    pvals = []
    for prot in df.index:
        v1 = g1.loc[prot].dropna().values
        v2 = g2.loc[prot].dropna().values
        if len(v1) >= 2 and len(v2) >= 2:
            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            pvals.append(p)
        else:
            pvals.append(1.0)

    pvals = np.array(pvals)
    qvals = _fdr_correct(pvals)

    sig_mask = (qvals < fdr_cutoff) & (np.abs(log2fc.values) >= fc_cutoff)
    n_sig = int(sig_mask.sum())
    n_up = int(((qvals < fdr_cutoff) & (log2fc.values >= fc_cutoff)).sum())
    n_down = int(((qvals < fdr_cutoff) & (log2fc.values <= -fc_cutoff)).sum())

    import pandas as pd

    results_df = pd.DataFrame({
        "mean_group1": mean1,
        "mean_group2": mean2,
        "log2fc": log2fc,
        "pvalue": pvals,
        "fdr": qvals,
    }, index=df.index)
    results_df = results_df.sort_values("fdr")

    return {
        "n_proteins_tested": len(df.index),
        "n_significant": n_sig,
        "n_upregulated": n_up,
        "n_downregulated": n_down,
        "group1_samples": g1_samples,
        "group2_samples": g2_samples,
        "auto_grouping_used": bool(auto_grouping and not group1 and not group2),
        "top_hits": results_df.head(20).to_dict("index"),
        "summary": (
            f"Tested {len(df.index):,} proteins: {n_sig} significant "
            f"(FDR<{fdr_cutoff}, |log2FC|≥{fc_cutoff}). {n_up} up, {n_down} down."
        ),
    }


# ---------------------------------------------------------------------------
# 11. omics.proteomics_enrich
# ---------------------------------------------------------------------------


def _parse_gene_list_file(path: str) -> tuple[set[str], str | None]:
    """Load a gene list from text/CSV/TSV file and return an uppercase gene set."""
    import pandas as pd

    fp = Path(path).expanduser()
    if not fp.exists():
        return set(), f"Background file not found: {path}"

    suffix = fp.suffix.lower()
    try:
        if suffix == ".txt":
            genes = {
                line.strip().split("\t")[0].split(",")[0].strip().upper()
                for line in fp.read_text(errors="replace").splitlines()
                if line.strip()
            }
            return {g for g in genes if g}, None

        if suffix in {".csv", ".tsv"}:
            sep = "," if suffix == ".csv" else "\t"
            df = pd.read_csv(fp, sep=sep)
            if df.empty:
                return set(), f"Background file is empty: {path}"
            first_col = df.columns[0]
            genes = {
                str(v).strip().upper()
                for v in df[first_col].dropna().tolist()
                if str(v).strip()
            }
            return genes, None

        # Fallback: treat as newline-delimited text.
        genes = {
            line.strip().split("\t")[0].split(",")[0].strip().upper()
            for line in fp.read_text(errors="replace").splitlines()
            if line.strip()
        }
        return {g for g in genes if g}, None
    except Exception as exc:
        return set(), f"Failed to parse background file: {str(exc)[:200]}"


def _enrichr_libraries_for_organism(organism: str) -> tuple[list[str] | None, str | None]:
    """Map organism names to Enrichr libraries."""
    org = (organism or "Homo sapiens").strip().lower()
    human_aliases = {"human", "homo sapiens", "hs", "h. sapiens"}
    mouse_aliases = {"mouse", "mus musculus", "mm", "m. musculus"}

    if org in human_aliases:
        return ["KEGG_2021_Human", "Reactome_2022", "GO_Biological_Process_2023"], None
    if org in mouse_aliases:
        return ["KEGG_2021_Mouse", "WikiPathway_2021_Mouse", "GO_Biological_Process_2023"], None
    return None, (
        f"Unsupported organism '{organism}'. "
        "Supported: Homo sapiens, Mus musculus."
    )


@registry.register(
    name="omics.proteomics_enrich",
    description="Pathway enrichment analysis from a list of differentially abundant proteins",
    category="omics",
    parameters={
        "proteins": "Comma-separated list of protein/gene symbols",
        "background_path": "Path to full protein list (optional, for background set)",
        "organism": "Organism for gene set lookup (default 'Homo sapiens')",
    },
    usage_guide=(
        "Run over-representation analysis on a set of differentially expressed proteins. "
        "Uses Enrichr API for pathway databases (KEGG, Reactome, GO)."
    ),
)
def proteomics_enrich(
    proteins: str = "",
    background_path: str = "",
    organism: str = "Homo sapiens",
    **kwargs,
) -> dict:
    """Pathway enrichment for a protein list via Enrichr."""
    seen = set()
    gene_list = []
    for gene in (g.strip() for g in proteins.split(",") if g.strip()):
        key = gene.upper()
        if key not in seen:
            seen.add(key)
            gene_list.append(gene)
    if not gene_list:
        return {"error": "No proteins provided", "summary": "Empty protein list"}

    libraries, org_error = _enrichr_libraries_for_organism(organism)
    if org_error:
        return {"error": org_error, "summary": org_error}

    background_info = {}
    if background_path:
        background_genes, bg_error = _parse_gene_list_file(background_path)
        if bg_error:
            return {"error": bg_error, "summary": bg_error}
        if not background_genes:
            return {
                "error": "Background file contains no genes after parsing",
                "summary": f"Empty background set: {background_path}",
            }

        original_n = len(gene_list)
        gene_list = [g for g in gene_list if g.upper() in background_genes]
        if not gene_list:
            return {
                "error": "None of the input genes were found in the provided background set",
                "background_gene_count": len(background_genes),
                "summary": "No overlap between input list and background set",
            }

        background_info = {
            "background_path": str(Path(background_path).expanduser()),
            "background_gene_count": len(background_genes),
            "n_proteins_before_background_filter": original_n,
            "n_proteins_after_background_filter": len(gene_list),
            # Enrichr endpoint has no custom-universe parameter; we apply background as input filter.
            "background_mode": "input_filter_only",
        }

    # Submit to Enrichr
    add_url = "https://maayanlab.cloud/Enrichr/addList"
    payload = {"list": (None, "\n".join(gene_list)), "description": (None, "ct proteomics enrichment")}

    import httpx

    try:
        resp = httpx.post(add_url, files=payload, timeout=15)
        resp.raise_for_status()
        user_list_id = resp.json().get("userListId")
    except Exception as exc:
        return {"error": f"Enrichr submission failed: {str(exc)[:200]}", "summary": f"Enrichr error: {str(exc)[:100]}"}

    if not user_list_id:
        return {"error": "Enrichr did not return a list ID", "summary": "Enrichr submission failed"}

    # Query key libraries
    all_results = {}
    library_errors = {}

    for lib in libraries:
        enrich_url = f"https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType={lib}"
        try:
            resp = httpx.get(enrich_url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            terms = data.get(lib, [])
            top_terms = []
            for term in terms[:10]:
                top_terms.append({
                    "term": term[1],
                    "pvalue": term[2],
                    "adj_pvalue": term[6],
                    "odds_ratio": term[3],
                    "genes": term[5],
                })
            all_results[lib] = top_terms
        except Exception as exc:
            all_results[lib] = []
            library_errors[lib] = str(exc)[:200]

    # Flatten top hits
    top_summary = []
    for lib, terms in all_results.items():
        for t in terms[:3]:
            if t["adj_pvalue"] < 0.05:
                top_summary.append(f"{t['term']} (q={t['adj_pvalue']:.2e})")

    return {
        "n_proteins_submitted": len(gene_list),
        "organism": organism,
        "libraries": libraries,
        "enrichment_results": all_results,
        "library_errors": library_errors,
        **background_info,
        "summary": (
            f"Enrichment of {len(gene_list)} proteins. "
            + (f"Top enriched: {'; '.join(top_summary[:5])}" if top_summary else "No significant enrichments (FDR<0.05).")
            + (" Background set applied as input filter." if background_path else "")
        ),
    }


# ---------------------------------------------------------------------------
# 12. omics.atac_peak_annotate
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.atac_peak_annotate",
    description="Annotate ATAC-seq peaks by genomic features and summarize accessibility landscape",
    category="omics",
    parameters={
        "path": "Path to peak file (BED-like CSV/TSV with chr, start, end columns or peak count matrix)",
    },
    usage_guide=(
        "Summarize ATAC-seq peak data: genomic distribution, peak sizes, "
        "chromosome distribution. Works on BED-like files or peak count matrices. "
        "Use after omics.geo_fetch to download ATAC-seq data."
    ),
)
def atac_peak_annotate(path: str, **kwargs) -> dict:
    """Annotate and summarize ATAC-seq peaks."""
    import numpy as np
    import pandas as pd

    filepath = Path(path).expanduser()
    if not filepath.exists():
        return {"error": f"File not found: {path}", "summary": f"File not found: {path}"}

    suffix = filepath.suffix.lower()
    sep = "," if suffix == ".csv" else "\t"

    try:
        df = pd.read_csv(filepath, sep=sep, comment="#")
    except Exception as exc:
        return {"error": f"Failed to read: {str(exc)[:200]}", "summary": f"Parse error: {str(exc)[:100]}"}

    # Detect BED-like format: look for chr/start/end columns
    col_lower = {c.lower(): c for c in df.columns}
    chr_col = col_lower.get("chr") or col_lower.get("chrom") or col_lower.get("chromosome")
    start_col = col_lower.get("start") or col_lower.get("chromstart")
    end_col = col_lower.get("end") or col_lower.get("chromend")

    # Also try positional (first 3 columns as chr, start, end)
    if not chr_col and len(df.columns) >= 3:
        first_col_vals = df.iloc[:, 0].astype(str)
        if first_col_vals.str.startswith("chr").mean() > 0.5:
            chr_col = df.columns[0]
            start_col = df.columns[1]
            end_col = df.columns[2]

    if chr_col and start_col and end_col:
        # BED-like format
        peaks = df[[chr_col, start_col, end_col]].copy()
        peaks.columns = ["chr", "start", "end"]
        peaks["start"] = pd.to_numeric(peaks["start"], errors="coerce")
        peaks["end"] = pd.to_numeric(peaks["end"], errors="coerce")
        peaks = peaks.dropna()
        peaks["width"] = peaks["end"] - peaks["start"]

        n_peaks = len(peaks)
        chr_counts = peaks["chr"].value_counts().head(24).to_dict()
        width_stats = {
            "mean": round(float(peaks["width"].mean()), 0),
            "median": round(float(peaks["width"].median()), 0),
            "min": int(peaks["width"].min()),
            "max": int(peaks["width"].max()),
        }

        # Estimate genomic feature distribution by peak width heuristic
        promoter_like = int((peaks["width"] < 500).sum())
        enhancer_like = int(((peaks["width"] >= 500) & (peaks["width"] < 2000)).sum())
        broad_peaks = int((peaks["width"] >= 2000).sum())

        return {
            "n_peaks": n_peaks,
            "chromosome_distribution": chr_counts,
            "peak_width_stats": width_stats,
            "promoter_like_peaks": promoter_like,
            "enhancer_like_peaks": enhancer_like,
            "broad_peaks": broad_peaks,
            "summary": (
                f"ATAC-seq: {n_peaks:,} peaks. Median width: {width_stats['median']:.0f} bp. "
                f"Estimated: {promoter_like:,} promoter-like (<500bp), "
                f"{enhancer_like:,} enhancer-like (500-2000bp), {broad_peaks:,} broad (>2000bp). "
                f"Top chromosomes: {', '.join(f'{k}:{v}' for k, v in list(chr_counts.items())[:5])}"
            ),
        }
    else:
        # Peak count matrix (peaks x samples)
        n_peaks, n_samples = df.shape
        return {
            "n_peaks": n_peaks,
            "n_samples": n_samples,
            "columns": list(df.columns[:20]),
            "summary": (
                f"ATAC-seq count matrix: {n_peaks:,} peaks x {n_samples} samples. "
                f"Use omics.chromatin_accessibility for differential analysis."
            ),
        }


# ---------------------------------------------------------------------------
# 13. omics.chromatin_accessibility
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.chromatin_accessibility",
    description="Differential chromatin accessibility analysis between two sample groups",
    category="omics",
    parameters={
        "path": "Path to peak count matrix (rows=peaks/genes, cols=samples)",
        "group1": "Comma-separated sample names for group 1",
        "group2": "Comma-separated sample names for group 2",
        "auto_grouping": "If true, splits samples by column order for exploratory use (default false)",
        "fdr_cutoff": "FDR threshold (default 0.05)",
    },
    usage_guide=(
        "Compare chromatin accessibility between groups from ATAC-seq count matrices. "
        "Works on peak-level or gene-level accessibility scores. "
        "Provide explicit group1/group2 sample lists for robust comparisons."
    ),
)
def chromatin_accessibility(
    path: str,
    group1: str = "",
    group2: str = "",
    auto_grouping: bool = False,
    fdr_cutoff: float = 0.05,
    **kwargs,
) -> dict:
    """Differential chromatin accessibility analysis."""
    import numpy as np
    from scipy import stats

    df, error = _load_tabular(path)
    if error:
        return {"error": error, "summary": f"Could not load: {error}"}

    g1_samples, g2_samples, group_error = _parse_sample_groups(
        df,
        group1=group1,
        group2=group2,
        auto_grouping=auto_grouping,
        min_group_size=2,
    )
    if group_error:
        return group_error

    g1 = df[g1_samples]
    g2 = df[g2_samples]

    mean1 = g1.mean(axis=1)
    mean2 = g2.mean(axis=1)
    # Log2 fold-change (add pseudocount to avoid log(0))
    log2fc = np.log2((mean2 + 1) / (mean1 + 1))

    pvals = []
    for region in df.index:
        v1 = g1.loc[region].dropna().values
        v2 = g2.loc[region].dropna().values
        if len(v1) >= 2 and len(v2) >= 2:
            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            pvals.append(p)
        else:
            pvals.append(1.0)

    pvals = np.array(pvals)
    qvals = _fdr_correct(pvals)

    sig_mask = qvals < fdr_cutoff
    n_sig = int(sig_mask.sum())
    n_more_open = int((sig_mask & (log2fc.values > 0)).sum())
    n_more_closed = int((sig_mask & (log2fc.values < 0)).sum())

    import pandas as pd

    results_df = pd.DataFrame({
        "mean_group1": mean1, "mean_group2": mean2,
        "log2fc": log2fc, "pvalue": pvals, "fdr": qvals,
    }, index=df.index).sort_values("fdr")

    return {
        "n_regions_tested": len(df.index),
        "n_significant": n_sig,
        "n_more_accessible": n_more_open,
        "n_less_accessible": n_more_closed,
        "group1_samples": g1_samples,
        "group2_samples": g2_samples,
        "auto_grouping_used": bool(auto_grouping and not group1 and not group2),
        "top_hits": results_df.head(20).to_dict("index"),
        "summary": (
            f"Tested {len(df.index):,} regions: {n_sig} differentially accessible "
            f"(FDR<{fdr_cutoff}). {n_more_open} gained, {n_more_closed} lost accessibility."
        ),
    }


# ---------------------------------------------------------------------------
# 14. omics.chipseq_enrich
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.chipseq_enrich",
    description="Enrichment analysis of ChIP-seq target genes",
    category="omics",
    parameters={
        "path": "Path to peak file with gene annotations (CSV/TSV with a gene column)",
        "gene_column": "Column name containing gene symbols (default auto-detect)",
    },
    usage_guide=(
        "Extract target genes from ChIP-seq peak annotations and run pathway "
        "enrichment. Works on peak files that include nearest-gene annotations."
    ),
)
def chipseq_enrich(path: str, gene_column: str = "", **kwargs) -> dict:
    """Enrichment analysis of ChIP-seq target genes."""
    import pandas as pd

    filepath = Path(path).expanduser()
    if not filepath.exists():
        return {"error": f"File not found: {path}", "summary": f"File not found: {path}"}

    suffix = filepath.suffix.lower()
    sep = "," if suffix == ".csv" else "\t"
    try:
        df = pd.read_csv(filepath, sep=sep, comment="#")
    except Exception as exc:
        return {"error": f"Failed to read: {str(exc)[:200]}", "summary": f"Parse error: {str(exc)[:100]}"}

    # Auto-detect gene column
    if gene_column and gene_column in df.columns:
        gcol = gene_column
    else:
        candidates = ["gene", "gene_name", "symbol", "gene_symbol", "nearest_gene",
                       "GENE", "Gene", "SYMBOL", "geneName"]
        gcol = None
        for c in candidates:
            if c in df.columns:
                gcol = c
                break
        if gcol is None:
            # Try case-insensitive
            col_lower = {c.lower(): c for c in df.columns}
            for c in ["gene", "gene_name", "symbol"]:
                if c in col_lower:
                    gcol = col_lower[c]
                    break

    if gcol is None:
        return {
            "error": "No gene column found. Provide gene_column parameter.",
            "available_columns": list(df.columns[:20]),
            "summary": f"Could not auto-detect gene column. Columns: {', '.join(df.columns[:10])}",
        }

    genes = df[gcol].dropna().unique().tolist()
    genes = [str(g).strip() for g in genes if str(g).strip() and str(g).strip().upper() != "NAN"]

    if not genes:
        return {"error": "No genes found in column", "summary": "Empty gene list after filtering"}

    # Delegate to Enrichr
    return proteomics_enrich(proteins=",".join(genes), **kwargs)


# ---------------------------------------------------------------------------
# 15. omics.spatial_cluster
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.spatial_cluster",
    description="Spatial-aware clustering of spatial transcriptomics data",
    category="omics",
    parameters={
        "path": "Path to h5ad file with spatial coordinates in .obsm['spatial']",
        "resolution": "Leiden clustering resolution (default 1.0)",
        "n_neighbors": "Number of spatial neighbors (default 15)",
    },
    usage_guide=(
        "Cluster spatial transcriptomics data (Visium, MERFISH, etc.) using "
        "both expression similarity and spatial proximity. Requires scanpy; "
        "squidpy is optional for enhanced spatial analysis."
    ),
)
def spatial_cluster(
    path: str,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    **kwargs,
) -> dict:
    """Spatial-aware clustering of spatial transcriptomics data."""
    filepath = Path(path).expanduser()
    if not filepath.exists():
        return {"error": f"File not found: {path}", "summary": f"File not found: {path}"}

    sc = _check_scanpy()
    if sc is None:
        return {
            "error": "scanpy required. Install with: pip install scanpy",
            "summary": "Install scanpy for spatial clustering: pip install scanpy",
        }

    try:
        adata = sc.read_h5ad(filepath)
    except Exception as exc:
        return {"error": f"Failed to load h5ad: {str(exc)[:200]}", "summary": f"Could not read file: {str(exc)[:100]}"}

    has_spatial = "spatial" in (adata.obsm or {})

    # Standard preprocessing if raw
    if adata.X.max() > 50:  # likely raw counts
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.tl.pca(adata, n_comps=min(50, adata.n_vars - 1, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    # Try squidpy for spatial neighbors
    sq = None
    try:
        import squidpy as sq_mod
        sq = sq_mod
    except ImportError:
        pass

    if has_spatial and sq:
        try:
            sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors)
            # Combine spatial + expression connectivity
            from scipy.sparse import csr_matrix
            expr_conn = adata.obsp.get("connectivities")
            spatial_conn = adata.obsp.get("spatial_connectivities")
            if expr_conn is not None and spatial_conn is not None:
                combined = 0.5 * expr_conn + 0.5 * spatial_conn
                adata.obsp["connectivities"] = csr_matrix(combined)
        except Exception:
            pass  # Fall back to expression-only neighbors

    sc.tl.leiden(adata, resolution=resolution, key_added="spatial_cluster")

    clusters = adata.obs["spatial_cluster"].value_counts().to_dict()
    n_clusters = len(clusters)

    result = {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "n_clusters": n_clusters,
        "cluster_sizes": clusters,
        "has_spatial_coords": has_spatial,
        "used_squidpy": sq is not None and has_spatial,
        "resolution": resolution,
        "summary": (
            f"Spatial clustering: {adata.n_obs:,} cells → {n_clusters} clusters "
            f"(resolution={resolution}). "
            f"{'Used spatial+expression neighbors (squidpy).' if sq and has_spatial else 'Expression-based neighbors only.'} "
            f"Largest cluster: {max(clusters.values()):,} cells."
        ),
    }

    # Try to find marker genes per cluster
    try:
        sc.tl.rank_genes_groups(adata, "spatial_cluster", method="wilcoxon", n_genes=5)
        markers = {}
        for cl in adata.obs["spatial_cluster"].unique():
            genes = list(adata.uns["rank_genes_groups"]["names"][cl][:5])
            markers[str(cl)] = genes
        result["cluster_markers"] = markers
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# 16. omics.spatial_autocorrelation
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.spatial_autocorrelation",
    description="Compute spatial autocorrelation (Moran's I) for gene expression patterns",
    category="omics",
    parameters={
        "path": "Path to h5ad file with spatial coordinates",
        "genes": "Comma-separated gene names to test (default: top variable genes)",
        "n_genes": "Number of top variable genes to test if genes not specified (default 50)",
    },
    usage_guide=(
        "Test whether gene expression shows spatial patterning using Moran's I. "
        "High Moran's I = spatially clustered expression. Requires scanpy."
    ),
)
def spatial_autocorrelation(
    path: str,
    genes: str = "",
    n_genes: int = 50,
    **kwargs,
) -> dict:
    """Compute Moran's I spatial autocorrelation for genes."""
    filepath = Path(path).expanduser()
    if not filepath.exists():
        return {"error": f"File not found: {path}", "summary": f"File not found: {path}"}

    sc = _check_scanpy()
    if sc is None:
        return {"error": "scanpy required", "summary": "Install scanpy: pip install scanpy"}

    try:
        adata = sc.read_h5ad(filepath)
    except Exception as exc:
        return {"error": f"Failed to load: {str(exc)[:200]}", "summary": f"Read error: {str(exc)[:100]}"}

    has_spatial = "spatial" in (adata.obsm or {})
    if not has_spatial:
        return {"error": "No spatial coordinates found in .obsm['spatial']", "summary": "Not spatial data: no coordinates found"}

    # Try squidpy
    try:
        import squidpy as sq
    except ImportError:
        return {"error": "squidpy required for Moran's I. Install: pip install squidpy", "summary": "Install squidpy: pip install squidpy"}

    # Preprocess if needed
    if adata.X.max() > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Build spatial graph
    sq.gr.spatial_neighbors(adata)

    # Select genes
    gene_list = [g.strip() for g in genes.split(",") if g.strip()] if genes else []
    if not gene_list:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(n_genes, adata.n_vars))
        gene_list = list(adata.var_names[adata.var["highly_variable"]])[:n_genes]

    # Filter to genes present in data
    valid_genes = [g for g in gene_list if g in adata.var_names]
    if not valid_genes:
        return {"error": "None of the specified genes found in dataset", "summary": "No matching genes in data"}

    # Compute Moran's I
    sq.gr.spatial_autocorr(adata, mode="moran", genes=valid_genes)

    moranI = adata.uns.get("moranI")
    if moranI is None:
        return {"error": "Moran's I computation failed", "summary": "Spatial autocorrelation computation failed"}

    results = moranI.sort_values("I", ascending=False)
    top_spatial = results.head(20).to_dict("index")

    highly_spatial = results[results["pval_norm"] < 0.05]
    n_spatial = len(highly_spatial)

    return {
        "n_genes_tested": len(valid_genes),
        "n_spatially_patterned": n_spatial,
        "top_spatial_genes": top_spatial,
        "summary": (
            f"Moran's I on {len(valid_genes)} genes: {n_spatial} show significant spatial "
            f"patterning (p<0.05). Top: "
            + ", ".join(f"{g} (I={row['I']:.3f})" for g, row in list(results.head(5).iterrows()))
        ),
    }


# ---------------------------------------------------------------------------
# 17. omics.cytof_cluster
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.cytof_cluster",
    description="Cluster CyTOF or flow cytometry data and characterize marker expression per cluster",
    category="omics",
    parameters={
        "path": "Path to CyTOF/flow data (CSV with markers as columns, cells as rows)",
        "n_clusters": "Number of clusters for KMeans (default 10). Use 0 for auto (Leiden).",
        "markers": "Comma-separated marker columns to use (default: all numeric columns)",
    },
    usage_guide=(
        "Cluster mass/flow cytometry data. Input is a cells x markers matrix. "
        "Identifies cell populations and characterizes each by marker expression."
    ),
)
def cytof_cluster(
    path: str,
    n_clusters: int = 10,
    markers: str = "",
    **kwargs,
) -> dict:
    """Cluster CyTOF/flow cytometry data."""
    import numpy as np

    df, error = _load_tabular(path)
    if error:
        # Try without index_col since CyTOF data often has no row names
        import pandas as pd

        filepath = Path(path).expanduser()
        if not filepath.exists():
            return {"error": error, "summary": f"Could not load: {error}"}
        suffix = filepath.suffix.lower()
        sep = "," if suffix == ".csv" else "\t"
        try:
            df = pd.read_csv(filepath, sep=sep)
        except Exception as exc2:
            return {"error": str(exc2), "summary": f"Could not load: {str(exc2)[:100]}"}

    marker_cols = [m.strip() for m in markers.split(",") if m.strip()] if markers else []
    if not marker_cols:
        marker_cols = list(df.select_dtypes(include=[np.number]).columns)

    if not marker_cols:
        return {"error": "No numeric marker columns found", "summary": "No numeric columns in data"}

    data = df[marker_cols].dropna()
    n_cells = len(data)
    if n_cells < 10:
        return {"error": f"Too few cells ({n_cells}) for clustering", "summary": f"Only {n_cells} cells — need at least 10"}

    # Standardize
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.values)

    # Cluster
    if n_clusters > 0:
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(n_clusters=min(n_clusters, n_cells), random_state=42, n_init=3)
        labels = model.fit_predict(scaled)
    else:
        # Use Leiden via scanpy on a neighbors graph
        sc = _check_scanpy()
        if sc is not None:
            import anndata

            adata = anndata.AnnData(X=scaled)
            sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.leiden(adata, resolution=1.0)
            labels = adata.obs["leiden"].astype(int).values
        else:
            from sklearn.cluster import MiniBatchKMeans

            model = MiniBatchKMeans(n_clusters=10, random_state=42, n_init=3)
            labels = model.fit_predict(scaled)

    import pandas as pd

    data = data.copy()
    data["cluster"] = labels

    cluster_sizes = data["cluster"].value_counts().sort_index().to_dict()
    n_clusters_found = len(cluster_sizes)

    # Per-cluster marker expression (median)
    cluster_medians = data.groupby("cluster")[marker_cols].median()
    cluster_profiles = cluster_medians.to_dict("index")

    # Find defining markers per cluster (highest z-score)
    defining_markers = {}
    global_means = data[marker_cols].mean()
    global_stds = data[marker_cols].std().replace(0, 1)
    for cl in sorted(cluster_sizes.keys()):
        cl_means = cluster_medians.loc[cl]
        z_scores = ((cl_means - global_means) / global_stds).sort_values(ascending=False)
        defining_markers[str(cl)] = list(z_scores.head(5).index)

    return {
        "n_cells": n_cells,
        "n_markers": len(marker_cols),
        "n_clusters": n_clusters_found,
        "cluster_sizes": {str(k): v for k, v in cluster_sizes.items()},
        "defining_markers": defining_markers,
        "cluster_profiles": {str(k): {mk: round(v, 3) for mk, v in prof.items()} for k, prof in cluster_profiles.items()},
        "summary": (
            f"CyTOF clustering: {n_cells:,} cells x {len(marker_cols)} markers → "
            f"{n_clusters_found} clusters. Largest: {max(cluster_sizes.values()):,} cells. "
            f"Top defining markers per cluster identified."
        ),
    }


# ---------------------------------------------------------------------------
# 18. omics.hic_compartments
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.hic_compartments",
    description="Identify A/B compartments from Hi-C contact matrices",
    category="omics",
    parameters={
        "path": "Path to Hi-C contact matrix (CSV/TSV, symmetric matrix with genomic bins)",
        "resolution": "Bin resolution description (for reporting, e.g. '50kb')",
    },
    usage_guide=(
        "Identify chromatin A/B compartments from Hi-C contact frequency matrices. "
        "A compartments are gene-rich/active, B compartments are gene-poor/repressed. "
        "Input should be a symmetric bin x bin contact matrix."
    ),
)
def hic_compartments(path: str, resolution: str = "unknown", **kwargs) -> dict:
    """Identify A/B compartments from Hi-C contact matrix via PCA."""
    import numpy as np

    df, error = _load_tabular(path)
    if error:
        return {"error": error, "summary": f"Could not load: {error}"}

    n_bins = df.shape[0]
    if df.shape[0] != df.shape[1]:
        return {
            "error": f"Expected symmetric matrix, got {df.shape[0]}x{df.shape[1]}",
            "summary": "Hi-C contact matrix must be square (symmetric)",
        }

    if n_bins < 3:
        return {"error": f"Too few bins ({n_bins})", "summary": "Need at least 3 bins for compartment analysis"}

    # Convert to numpy, handle NaN
    matrix = df.values.astype(float)
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Normalize: observed/expected
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    expected = np.outer(row_sums, row_sums) / row_sums.sum()
    expected[expected == 0] = 1
    oe_matrix = matrix / expected

    # Correlation matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(oe_matrix)
        corr = np.nan_to_num(corr, nan=0.0)

    # PCA — first eigenvector gives A/B compartments
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Take the eigenvector with largest eigenvalue
    pc1 = eigenvectors[:, -1]

    # A compartments = positive PC1 (convention: gene-rich is positive)
    compartments = np.where(pc1 > 0, "A", "B")
    n_A = int((compartments == "A").sum())
    n_B = int((compartments == "B").sum())
    frac_A = n_A / n_bins

    # Compartment runs (contiguous blocks)
    transitions = 0
    for i in range(1, len(compartments)):
        if compartments[i] != compartments[i - 1]:
            transitions += 1

    return {
        "n_bins": n_bins,
        "resolution": resolution,
        "n_compartment_A": n_A,
        "n_compartment_B": n_B,
        "fraction_A": round(frac_A, 3),
        "n_transitions": transitions,
        "pc1_values": pc1.tolist()[:50],
        "compartment_assignments": compartments.tolist()[:50],
        "explained_variance": round(float(eigenvalues[-1] / eigenvalues.sum()), 4),
        "summary": (
            f"Hi-C compartments ({resolution} resolution): {n_bins} bins → "
            f"{n_A} A-compartment ({frac_A:.0%}), {n_B} B-compartment ({1-frac_A:.0%}). "
            f"{transitions} A/B transitions. PC1 explains {eigenvalues[-1]/eigenvalues.sum():.1%} of variance."
        ),
    }


# ===========================================================================
# Specialized library integrations (optional deps)
# ===========================================================================


def _check_pydeseq2():
    """Check if pyDESeq2 is available."""
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        return True
    except Exception as exc:
        logger.debug("pyDESeq2 unavailable or failed to import: %s", exc)
        return False


def _check_muon():
    """Check if muon is available."""
    try:
        import muon
        import mudata

        return muon
    except Exception as exc:
        logger.debug("muon unavailable or failed to import: %s", exc)
        return None


def _check_episcanpy():
    """Check if episcanpy is available."""
    try:
        import episcanpy.api as epi

        return epi
    except Exception as exc:
        logger.debug("episcanpy unavailable or failed to import: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 19. omics.deseq2
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.deseq2",
    description="Differential expression with DESeq2 (negative binomial model for count data)",
    category="omics",
    parameters={
        "counts_path": "Path to raw count matrix (genes as rows, samples as columns)",
        "metadata_path": "Path to sample metadata table (CSV/TSV/TXT/XLSX; must have a condition column; required unless infer_metadata=true)",
        "condition_col": "Column in metadata for the contrast (default 'condition')",
        "ref_level": "Reference level for contrast (default: alphabetically first)",
        "test_level": "Test level for contrast (default: alphabetically second)",
        "covariates": "Optional comma-separated covariates to include in design (e.g., 'sex,batch')",
        "infer_metadata": "If true, infer two groups from sample column order for exploratory use only (default false)",
        "alpha": "Significance threshold for adjusted p-values (default 0.05)",
        "use_r_deseq2": "If true, prefer R DESeq2 backend via rpy2 when available (default true)",
        "prefilter_min_count": "Optional prefilter threshold: minimum count per sample (default 0 disables prefilter)",
        "prefilter_min_samples": "Optional prefilter threshold: minimum number of samples meeting prefilter_min_count (default 1)",
        "lfc_shrink": "If true, apply apeglm LFC shrinkage when possible (default false)",
        "enrichment_library": "Optional gseapy/Enrichr library name (e.g., Reactome_2022) for post-DE enrichment",
        "pathway_term": "Optional pathway term to match and extract odds ratio from enrichment results",
        "gene_map_path": "Optional gene ID -> symbol mapping table used before enrichment",
        "gene_id_col": "Optional gene ID column name in mapping table",
        "gene_symbol_col": "Optional symbol column name in mapping table",
        "min_abs_lfc": "Optional absolute log2FC threshold for enrichment gene list",
        "min_base_mean": "Optional baseMean threshold for enrichment gene list",
        "target_gene": "Optional target gene symbol/ID to report explicitly (returns log2FoldChange/baseMean/padj even if not in top hits)",
    },
    usage_guide=(
        "Proper count-based differential expression using the DESeq2 negative binomial model. "
        "Preferred over Mann-Whitney for bulk RNA-seq count data. Requires pydeseq2: "
        "pip install pydeseq2. Falls back to scipy Mann-Whitney if not installed. "
        "Supports optional covariate-adjusted design, LFC shrinkage, and optional gseapy enrichment "
        "from DE genes (including pathway-specific odds ratio extraction). "
        "Use explicit sample metadata in production; inferred metadata is exploratory only."
    ),
)
def deseq2(
    counts_path: str,
    metadata_path: str = "",
    condition_col: str = "condition",
    ref_level: str = "",
    test_level: str = "",
    covariates: str = "",
    infer_metadata: bool = False,
    alpha: float = 0.05,
    use_r_deseq2: bool = True,
    prefilter_min_count: int = 0,
    prefilter_min_samples: int = 1,
    lfc_shrink: bool = False,
    enrichment_library: str = "",
    pathway_term: str = "",
    gene_map_path: str = "",
    gene_id_col: str = "",
    gene_symbol_col: str = "",
    min_abs_lfc: float = 0.0,
    min_base_mean: float = 0.0,
    target_gene: str = "",
    **kwargs,
) -> dict:
    """Run DESeq2 differential expression on count data."""
    import pandas as pd

    # Load counts
    df, error = _load_tabular(counts_path)
    if error:
        return {"error": error, "summary": f"Could not load counts: {error}"}

    # Load or infer metadata
    if metadata_path:
        metadata, meta_error = _load_tabular(metadata_path)
        if meta_error:
            return {"error": meta_error, "summary": f"Metadata load failed: {meta_error}"}
    else:
        if not infer_metadata:
            samples = list(df.columns)
            return {
                "error": (
                    "metadata_path is required for reliable DESeq2 analysis. "
                    "Set infer_metadata=True only for quick exploratory analysis."
                ),
                "available_samples": samples[:30],
                "summary": f"No metadata provided for {len(samples)} samples; cannot define conditions.",
            }

        # Exploratory-only mode: split samples into two halves.
        samples = list(df.columns)
        mid = len(samples) // 2
        if mid < 2:
            return {"error": "Need at least 4 samples (2 per group) without metadata", "summary": "Too few samples"}
        metadata = pd.DataFrame(
            {"condition": ["control"] * mid + ["treatment"] * (len(samples) - mid)},
            index=samples,
        )

    if condition_col not in metadata.columns:
        return {
            "error": f"Column '{condition_col}' not in metadata. Available: {list(metadata.columns)}",
            "summary": f"Missing condition column: {condition_col}",
        }

    # Align samples (drop metadata-only and counts-only samples deterministically)
    common = df.columns.intersection(metadata.index)
    if len(common) < 4:
        return {"error": f"Need ≥4 shared samples, found {len(common)}", "summary": "Too few matching samples"}
    counts = df[common]
    metadata = metadata.loc[common]

    # Optional pre-filtering to stabilize dispersion fitting and reduce noise.
    if prefilter_min_count > 0:
        required = max(int(prefilter_min_samples), 1)
        keep_mask = (counts >= int(prefilter_min_count)).sum(axis=1) >= required
        counts = counts.loc[keep_mask]
        if counts.empty:
            return {
                "error": "All genes were removed by prefilter.",
                "summary": "No genes left after prefilter; relax prefilter thresholds.",
            }

    levels = sorted(metadata[condition_col].unique())
    if len(levels) < 2:
        return {"error": "Need at least 2 condition levels", "summary": "Only one condition level found"}
    ref = ref_level if ref_level else levels[0]
    test = test_level if test_level else levels[1]

    def _resolve_level_name(requested: str, available_levels: list[str]) -> str:
        """Best-effort map of user/planner shorthand labels to metadata factor levels."""
        import re

        req = str(requested or "").strip()
        if not req:
            return req
        if req in available_levels:
            return req

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        req_norm = _norm(req)
        if not req_norm:
            return req

        # 1) Exact normalized match.
        exact = [lvl for lvl in available_levels if _norm(str(lvl)) == req_norm]
        if len(exact) == 1:
            return exact[0]

        # 2) Prefix/token containment match (e.g., "CBD" -> "CBD_IC50").
        token_like = [
            lvl
            for lvl in available_levels
            if _norm(str(lvl)).startswith(req_norm) or req_norm in _norm(str(lvl))
        ]
        if len(token_like) == 1:
            return token_like[0]

        # 3) Prefer non-combined condition when shorthand maps to several levels.
        if len(token_like) > 1:
            non_combo = [
                lvl
                for lvl in token_like
                if "serum_starvation" not in str(lvl).lower()
                and "cisplatin" not in str(lvl).lower()
                and "comb" not in str(lvl).lower()
                and "plus" not in str(lvl).lower()
            ]
            if len(non_combo) == 1:
                return non_combo[0]
        return req

    ref = _resolve_level_name(ref, levels)
    test = _resolve_level_name(test, levels)
    if ref not in levels or test not in levels:
        return {
            "error": f"Requested contrast levels not found. Levels available: {levels}",
            "summary": f"Invalid contrast levels: ref={ref}, test={test}",
        }
    if ref == test:
        return {
            "error": "ref_level and test_level must be different",
            "summary": "Invalid contrast: identical levels",
        }

    n_ref = int((metadata[condition_col] == ref).sum())
    n_test = int((metadata[condition_col] == test).sum())
    if n_ref < 2 or n_test < 2:
        return {
            "error": f"Need at least 2 replicates per condition for {ref} vs {test} (found {n_ref} and {n_test})",
            "summary": "Insufficient biological replicates per condition",
        }

    # Build design formula with optional covariates.
    if isinstance(covariates, (list, tuple)):
        covars = [str(c).strip() for c in covariates if str(c).strip()]
    else:
        raw = str(covariates or "").strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        covars = [c.strip().strip("'\"") for c in raw.split(",") if c.strip().strip("'\"")]
    missing_covars = [c for c in covars if c not in metadata.columns]
    if missing_covars:
        return {
            "error": f"Covariate column(s) not in metadata: {missing_covars}",
            "summary": f"Missing covariates: {', '.join(missing_covars)}",
        }
    design_terms = covars + [condition_col]
    design_formula = "~ " + " + ".join(design_terms)

    target_gene = str(target_gene or "").strip()

    def _resolve_target_gene(results_df: "pd.DataFrame") -> "dict | None":
        """Resolve a user-requested target gene against DE results."""
        if not target_gene:
            return None

        idx_series = pd.Series(results_df.index.astype(str), index=results_df.index)
        idx_no_ver = idx_series.str.split(".").str[0]
        tgt = target_gene
        tgt_no_ver = tgt.split(".")[0]
        tgt_lower = tgt.lower()
        tgt_no_ver_lower = tgt_no_ver.lower()

        mask = (idx_series.str.lower() == tgt_lower) | (idx_no_ver.str.lower() == tgt_no_ver_lower)

        # If the target appears to be a symbol and IDs are in results, use mapping if provided/discoverable.
        if not mask.any():
            mapper = None

            def _build_mapper(gm_df: "pd.DataFrame") -> "dict[str, str]":
                nonlocal gene_id_col, gene_symbol_col
                id_col = gene_id_col or ("ENSG_ID" if "ENSG_ID" in gm_df.columns else gm_df.columns[0])
                sym_col = gene_symbol_col or (
                    "gene_name"
                    if "gene_name" in gm_df.columns
                    else ("symbol" if "symbol" in gm_df.columns else gm_df.columns[-1])
                )
                gm2 = gm_df[[id_col, sym_col]].dropna().copy()
                gm2[id_col] = gm2[id_col].astype(str).str.split(".").str[0]
                gm2[sym_col] = gm2[sym_col].astype(str)
                return {
                    symbol.lower(): gid
                    for gid, symbol in zip(gm2[id_col], gm2[sym_col])
                    if symbol
                }

            if gene_map_path:
                gm, gm_err = _load_tabular(gene_map_path, index_col=None)
                if not gm_err and gm is not None and not gm.empty:
                    mapper = _build_mapper(gm)
            else:
                # Best-effort auto-discovery for common capsule naming patterns.
                try:
                    base_dir = Path(counts_path).expanduser().resolve().parent
                    candidates = sorted(
                        [
                            p
                            for p in base_dir.iterdir()
                            if p.is_file()
                            and p.suffix.lower() in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
                            and ("gene" in p.name.lower() and ("meta" in p.name.lower() or "annot" in p.name.lower()))
                        ]
                    )
                    for cand in candidates:
                        gm, gm_err = _load_tabular(str(cand), index_col=None)
                        if gm_err or gm is None or gm.empty:
                            continue
                        mapper = _build_mapper(gm)
                        if mapper:
                            break
                except Exception:
                    mapper = None

            if mapper:
                mapped_id = mapper.get(tgt_lower)
                if mapped_id:
                    mask = idx_no_ver.str.lower() == mapped_id.lower()

        if not mask.any():
            return {
                "target_gene": target_gene,
                "found": False,
            }

        row = results_df.loc[mask].iloc[0]
        return {
            "target_gene": target_gene,
            "found": True,
            "matched_gene_id": str(results_df.loc[mask].index[0]),
            "log2FoldChange": float(row.get("log2FoldChange")) if pd.notna(row.get("log2FoldChange")) else None,
            "baseMean": float(row.get("baseMean")) if pd.notna(row.get("baseMean")) else None,
            "padj": float(row.get("padj")) if pd.notna(row.get("padj")) else None,
            "pvalue": float(row.get("pvalue")) if "pvalue" in row and pd.notna(row.get("pvalue")) else None,
        }

    # Ensure categorical encoding for design variables.
    # Coerce to string first so mixed numeric/string covariates (e.g., batch IDs)
    # convert cleanly through pandas2ri into R factors.
    metadata = metadata.copy()
    for col in design_terms:
        metadata[col] = metadata[col].astype(str).astype("category")
    if ref in metadata[condition_col].cat.categories:
        ordered_levels = [ref] + [x for x in metadata[condition_col].cat.categories if x != ref]
        metadata[condition_col] = metadata[condition_col].cat.reorder_categories(ordered_levels)

    # Try native R DESeq2 first (when requested and available), then fall back
    # to pyDESeq2 for environments without DESeq2.
    if use_r_deseq2:
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.packages import importr

            # Ensure DESeq2 is available in either user or system R library.
            ro.r(".libPaths(c('~/R/library', .libPaths()))")
            importr("DESeq2")
            if lfc_shrink:
                try:
                    importr("apeglm")
                except Exception:
                    # Shrinkage is optional; proceed without if apeglm unavailable.
                    lfc_shrink = False

            counts_r = counts.astype(int)
            meta_r = metadata.copy()
            # R expects colData rownames to match countData colnames.
            meta_r.index = counts_r.columns

            with localconverter(ro.default_converter + pandas2ri.converter):
                ro.globalenv["counts_df"] = counts_r
                ro.globalenv["meta_df"] = meta_r

            ro.globalenv["design_formula_str"] = design_formula
            ro.globalenv["condition_col_str"] = condition_col
            ro.globalenv["test_level_str"] = test
            ro.globalenv["ref_level_str"] = ref
            ro.globalenv["alpha_val"] = float(alpha)
            ro.globalenv["do_shrink"] = bool(lfc_shrink)

            r_script = """
            suppressPackageStartupMessages(library(DESeq2))
            if (isTRUE(do_shrink)) {
              suppressPackageStartupMessages(library(apeglm))
            }
            counts_mat <- as.matrix(counts_df)
            mode(counts_mat) <- "integer"
            meta <- as.data.frame(meta_df)
            cond_vals <- as.character(meta[[condition_col_str]])
            meta[[condition_col_str]] <- factor(cond_vals)
            meta[[condition_col_str]] <- relevel(meta[[condition_col_str]], ref = ref_level_str)
            dds <- DESeqDataSetFromMatrix(
              countData = counts_mat,
              colData = meta,
              design = as.formula(design_formula_str)
            )
            dds <- DESeq(dds, quiet = TRUE)
            res <- results(
              dds,
              contrast = c(condition_col_str, test_level_str, ref_level_str),
              alpha = alpha_val
            )
            shrink_coeff <- NA_character_
            if (isTRUE(do_shrink)) {
              rn <- resultsNames(dds)
              cand <- rn[grepl(paste0("^", condition_col_str, "_"), rn)]
              if (length(cand) > 0) {
                shrink_coeff <- cand[1]
                res <- lfcShrink(dds, coef = shrink_coeff, type = "apeglm")
              }
            }
            res_df <- as.data.frame(res)
            res_df$gene_id <- rownames(res_df)
            """
            ro.r(r_script)
            with localconverter(ro.default_converter + pandas2ri.converter):
                res_df = ro.conversion.rpy2py(ro.globalenv["res_df"])
            shrink_coeff = str(ro.globalenv["shrink_coeff"][0]) if "shrink_coeff" in ro.globalenv else None
            if shrink_coeff in {"NA", "NA_character_", "None"}:
                shrink_coeff = None

            # Normalize column names to match pyDESeq2-style payload.
            if "log2FoldChange" not in res_df.columns and "log2FoldChange" in [str(c) for c in res_df.columns]:
                pass
            if "baseMean" not in res_df.columns or "padj" not in res_df.columns:
                raise ValueError("R DESeq2 results missing required columns")
            res_df = res_df.set_index("gene_id")
            results = res_df.sort_values("padj")

            n_sig = int((results["padj"] < alpha).sum())
            n_up = int(((results["padj"] < alpha) & (results["log2FoldChange"] > 0)).sum())
            n_down = int(((results["padj"] < alpha) & (results["log2FoldChange"] < 0)).sum())
            target_gene_result = _resolve_target_gene(results)
            target_gene_summary = ""
            if target_gene_result:
                if target_gene_result.get("found"):
                    lfc_val = target_gene_result.get("log2FoldChange")
                    lfc_txt = f"{lfc_val:.6g}" if lfc_val is not None else "NA"
                    target_gene_summary = f" {target_gene} log2FoldChange={lfc_txt}."
                else:
                    target_gene_summary = f" {target_gene} was not found in result gene IDs."
            return {
                "method": "DESeq2 (R via rpy2)",
                "n_genes_tested": len(results),
                "n_significant": n_sig,
                "n_upregulated": n_up,
                "n_downregulated": n_down,
                "contrast": f"{test} vs {ref}",
                "design": design_formula,
                "covariates": covars,
                "n_samples_ref": n_ref,
                "n_samples_test": n_test,
                "n_shared_samples": int(len(common)),
                "prefilter": {
                    "min_count": int(prefilter_min_count),
                    "min_samples": int(prefilter_min_samples),
                    "n_genes_after": int(len(results)),
                },
                "metadata_inferred": bool(infer_metadata and not metadata_path),
                "alpha": alpha,
                "lfc_shrink": bool(lfc_shrink),
                "lfc_shrink_coeff": shrink_coeff,
                "top_hits": results.head(20).to_dict("index"),
                "target_gene_result": target_gene_result,
                "summary": (
                    f"DESeq2 (R): {len(results):,} genes tested ({test} vs {ref}) with design {design_formula}. "
                    f"{n_sig} significant (padj<{alpha}): {n_up} up, {n_down} down."
                    + (" Metadata was inferred from sample order (exploratory)." if infer_metadata and not metadata_path else "")
                    + target_gene_summary
                ),
            }
        except Exception as exc:
            logger.warning("R DESeq2 backend failed, falling back to pyDESeq2: %s", exc)

    # Try pyDESeq2
    if _check_pydeseq2():
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds import DeseqStats
            from pydeseq2.default_inference import DefaultInference

            inference = DefaultInference(n_cpus=1)
            # pyDESeq2 wants samples as rows, genes as columns
            dds = DeseqDataSet(
                counts=counts.T,
                metadata=metadata,
                design=design_formula,
                refit_cooks=True,
                inference=inference,
                quiet=True,
            )
            dds.deseq2()

            stat = DeseqStats(
                dds,
                contrast=[condition_col, test, ref],
                alpha=alpha,
                inference=inference,
                quiet=True,
            )
            stat.summary()

            # Optional apeglm shrinkage on the requested condition coefficient.
            shrink_coeff = None
            if lfc_shrink and hasattr(dds, "varm") and "LFC" in dds.varm:
                coeffs = list(dds.varm["LFC"].columns)
                preferred = [
                    c for c in coeffs
                    if condition_col in c and (test in c or test.replace("-", "_") in c)
                ]
                if preferred:
                    shrink_coeff = preferred[0]
                    try:
                        stat.lfc_shrink(coeff=shrink_coeff)
                    except Exception as exc:
                        logger.warning("LFC shrinkage failed for coeff %s: %s", shrink_coeff, exc)

            results = stat.results_df.sort_values("padj")
            n_sig = int((results["padj"] < alpha).sum())
            n_up = int(((results["padj"] < alpha) & (results["log2FoldChange"] > 0)).sum())
            n_down = int(((results["padj"] < alpha) & (results["log2FoldChange"] < 0)).sum())
            target_gene_result = _resolve_target_gene(results)
            target_gene_summary = ""
            if target_gene_result:
                if target_gene_result.get("found"):
                    lfc_val = target_gene_result.get("log2FoldChange")
                    lfc_txt = f"{lfc_val:.6g}" if lfc_val is not None else "NA"
                    target_gene_summary = f" {target_gene} log2FoldChange={lfc_txt}."
                else:
                    target_gene_summary = f" {target_gene} was not found in result gene IDs."

            result_payload = {
                "method": "DESeq2 (pydeseq2)",
                "n_genes_tested": len(results),
                "n_significant": n_sig,
                "n_upregulated": n_up,
                "n_downregulated": n_down,
                "contrast": f"{test} vs {ref}",
                "design": design_formula,
                "covariates": covars,
                "n_samples_ref": n_ref,
                "n_samples_test": n_test,
                "n_shared_samples": int(len(common)),
                "prefilter": {
                    "min_count": int(prefilter_min_count),
                    "min_samples": int(prefilter_min_samples),
                    "n_genes_after": int(len(results)),
                },
                "metadata_inferred": bool(infer_metadata and not metadata_path),
                "alpha": alpha,
                "lfc_shrink": bool(lfc_shrink),
                "lfc_shrink_coeff": shrink_coeff,
                "top_hits": results.head(20).to_dict("index"),
                "target_gene_result": target_gene_result,
                "summary": (
                    f"DESeq2: {len(results):,} genes tested ({test} vs {ref}) with design {design_formula}. "
                    f"{n_sig} significant (padj<{alpha}): {n_up} up, {n_down} down."
                    + (" Metadata was inferred from sample order (exploratory)." if infer_metadata and not metadata_path else "")
                    + target_gene_summary
                ),
            }

            # Optional enrichment over significant DEGs with effect filters.
            if enrichment_library:
                sig = results[results["padj"] < alpha].copy()
                if min_abs_lfc > 0:
                    sig = sig[sig["log2FoldChange"].abs() >= float(min_abs_lfc)]
                if min_base_mean > 0 and "baseMean" in sig.columns:
                    sig = sig[sig["baseMean"] >= float(min_base_mean)]

                genes_for_enrichment = list(sig.index.astype(str))
                mapped_gene_count = None
                if gene_map_path:
                    gm, gm_err = _load_tabular(gene_map_path, index_col=None)
                    if gm_err:
                        result_payload["enrichment_error"] = f"Gene map load failed: {gm_err}"
                    else:
                        id_col = gene_id_col or ("ENSG_ID" if "ENSG_ID" in gm.columns else gm.columns[0])
                        sym_col = gene_symbol_col or (
                            "gene_name" if "gene_name" in gm.columns else ("symbol" if "symbol" in gm.columns else gm.columns[-1])
                        )
                        gm2 = gm[[id_col, sym_col]].dropna().copy()
                        gm2[id_col] = gm2[id_col].astype(str)
                        gm2[sym_col] = gm2[sym_col].astype(str)
                        mapper = dict(zip(gm2[id_col], gm2[sym_col]))
                        mapped = []
                        for gid in genes_for_enrichment:
                            mapped_sym = mapper.get(gid, mapper.get(gid.split(".")[0]))
                            if mapped_sym:
                                mapped.append(mapped_sym)
                        genes_for_enrichment = sorted(set(mapped))
                        mapped_gene_count = len(genes_for_enrichment)

                if genes_for_enrichment:
                    try:
                        import gseapy

                        enr = gseapy.enrichr(
                            gene_list=genes_for_enrichment,
                            gene_sets=enrichment_library,
                            outdir=None,
                            no_plot=True,
                        )
                        enr_df = enr.results.copy()
                        result_payload["enrichment"] = {
                            "library": enrichment_library,
                            "n_input_genes": len(genes_for_enrichment),
                            "mapped_gene_count": mapped_gene_count,
                            "n_terms": int(len(enr_df)),
                            "top_terms": enr_df.head(20).to_dict("records"),
                        }

                        if pathway_term:
                            terms = enr_df["Term"].astype(str)
                            exact = enr_df[terms.str.lower() == pathway_term.lower()]
                            target_df = exact if not exact.empty else enr_df[terms.str.contains(pathway_term, case=False, na=False)]
                            if not target_df.empty:
                                target = target_df.iloc[0].to_dict()
                                result_payload["pathway_match"] = target
                                result_payload["pathway_odds_ratio"] = target.get("Odds Ratio")
                                result_payload["summary"] += (
                                    f" Enrichment: '{target.get('Term', pathway_term)}' odds ratio "
                                    f"{target.get('Odds Ratio')}."
                                )
                            else:
                                result_payload["pathway_match"] = None
                                result_payload["summary"] += f" Enrichment ran but pathway '{pathway_term}' was not found."
                    except Exception as exc:
                        result_payload["enrichment_error"] = str(exc)
                        result_payload["summary"] += " Enrichment step failed."

            return result_payload

        except Exception as exc:
            logger.warning("pyDESeq2 failed, falling back to Mann-Whitney: %s", exc)

    # Fallback: Mann-Whitney U
    import numpy as np
    from scipy import stats

    g1_samples = metadata.index[metadata[condition_col] == ref].tolist()
    g2_samples = metadata.index[metadata[condition_col] == test].tolist()
    g1 = counts[g1_samples]
    g2 = counts[g2_samples]

    log2fc = np.log2((g2.mean(axis=1) + 1) / (g1.mean(axis=1) + 1))
    pvals = []
    for gene in counts.index:
        v1 = g1.loc[gene].dropna().values
        v2 = g2.loc[gene].dropna().values
        if len(v1) >= 2 and len(v2) >= 2:
            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            pvals.append(p)
        else:
            pvals.append(1.0)

    pvals = np.array(pvals)
    qvals = _fdr_correct(pvals)

    n_sig = int((qvals < alpha).sum())
    n_up = int(((qvals < alpha) & (log2fc.values > 0)).sum())
    n_down = int(((qvals < alpha) & (log2fc.values < 0)).sum())

    results = pd.DataFrame({
        "log2FoldChange": log2fc, "pvalue": pvals, "padj": qvals,
    }, index=counts.index).sort_values("padj")
    target_gene_result = _resolve_target_gene(results)
    target_gene_summary = ""
    if target_gene_result:
        if target_gene_result.get("found"):
            lfc_val = target_gene_result.get("log2FoldChange")
            lfc_txt = f"{lfc_val:.6g}" if lfc_val is not None else "NA"
            target_gene_summary = f" {target_gene} log2FoldChange={lfc_txt}."
        else:
            target_gene_summary = f" {target_gene} was not found in result gene IDs."

    return {
        "method": "Mann-Whitney U (fallback — install pydeseq2 for proper DESeq2)",
        "n_genes_tested": len(results),
        "n_significant": n_sig,
        "n_upregulated": n_up,
        "n_downregulated": n_down,
        "contrast": f"{test} vs {ref}",
        "n_samples_ref": n_ref,
        "n_samples_test": n_test,
        "metadata_inferred": bool(infer_metadata and not metadata_path),
        "alpha": alpha,
        "top_hits": results.head(20).to_dict("index"),
        "target_gene_result": target_gene_result,
        "summary": (
            f"Differential expression (Mann-Whitney fallback): {len(results):,} genes ({test} vs {ref}). "
            f"{n_sig} significant (FDR<{alpha}): {n_up} up, {n_down} down. "
            f"Install pydeseq2 for proper negative binomial modeling."
            + (" Metadata was inferred from sample order (exploratory)." if infer_metadata and not metadata_path else "")
            + target_gene_summary
        ),
    }


# ---------------------------------------------------------------------------
# 20. omics.multiomics_integrate
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.multiomics_integrate",
    description="Integrate multiple omics modalities using MOFA+ (Multi-Omics Factor Analysis)",
    category="omics",
    parameters={
        "paths": "Comma-separated paths to h5ad files for each modality",
        "modality_names": "Comma-separated names for each modality (e.g., 'rna,atac,protein')",
        "n_factors": "Number of latent factors to learn (default 10)",
    },
    usage_guide=(
        "Integrate multiple omics datasets (RNA + ATAC, RNA + protein, etc.) into a shared "
        "latent space using MOFA+. Requires muon: pip install muon. Each modality should be "
        "an h5ad file with overlapping cell barcodes."
    ),
)
def multiomics_integrate(
    paths: str = "",
    modality_names: str = "",
    n_factors: int = 10,
    **kwargs,
) -> dict:
    """Integrate multiple omics modalities using MOFA+."""
    mu = _check_muon()
    if mu is None:
        return {
            "error": "muon required. Install with: pip install muon mudata",
            "summary": "Install muon for multi-omics integration: pip install muon mudata",
        }

    sc = _check_scanpy()
    if sc is None:
        return {"error": "scanpy required. Install with: pip install scanpy", "summary": "Install scanpy: pip install scanpy"}

    from mudata import MuData

    path_list = [p.strip() for p in paths.split(",") if p.strip()]
    name_list = [n.strip() for n in modality_names.split(",") if n.strip()]

    if len(path_list) < 2:
        return {"error": "Need at least 2 modality paths", "summary": "Provide ≥2 h5ad paths for integration"}

    if not name_list:
        name_list = [f"modality_{i}" for i in range(len(path_list))]
    if len(name_list) != len(path_list):
        return {"error": "Number of names must match number of paths", "summary": "Mismatched path/name count"}

    # Load modalities
    modalities = {}
    for name, fpath in zip(name_list, path_list):
        fp = Path(fpath).expanduser()
        if not fp.exists():
            return {"error": f"File not found: {fpath}", "summary": f"Missing file: {fpath}"}
        try:
            adata = sc.read_h5ad(fp)
            modalities[name] = adata
        except Exception as exc:
            return {"error": f"Failed to load {fpath}: {str(exc)[:200]}", "summary": f"Load error: {str(exc)[:100]}"}

    # Create MuData
    try:
        mdata = MuData(modalities)
    except Exception as exc:
        return {"error": f"MuData creation failed: {str(exc)[:200]}", "summary": f"Integration setup error: {str(exc)[:100]}"}

    n_shared = mdata.n_obs
    mod_shapes = {name: (ad.n_obs, ad.n_vars) for name, ad in modalities.items()}

    # Preprocess each modality
    for name in name_list:
        ad = mdata.mod[name]
        if ad.X.max() > 50:  # likely raw counts
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)
        sc.pp.highly_variable_genes(ad, min_mean=0.0125, max_mean=3, min_disp=0.5)

    # Run MOFA+
    try:
        mu.tl.mofa(mdata, n_factors=n_factors, quiet=True)
    except Exception as exc:
        return {
            "error": f"MOFA+ failed: {str(exc)[:200]}",
            "summary": f"MOFA+ integration failed: {str(exc)[:100]}",
            "n_shared_cells": n_shared,
            "modality_shapes": mod_shapes,
        }

    # Extract results
    has_mofa = "X_mofa" in mdata.obsm
    if not has_mofa:
        return {
            "error": "MOFA+ did not produce embeddings",
            "summary": "Integration ran but produced no factors",
        }

    # Downstream: neighbors + leiden on MOFA space
    sc.pp.neighbors(mdata, use_rep="X_mofa")
    sc.tl.leiden(mdata, resolution=1.0, key_added="joint_cluster")

    clusters = mdata.obs["joint_cluster"].value_counts().to_dict()
    n_clusters = len(clusters)

    return {
        "n_shared_cells": n_shared,
        "n_factors": n_factors,
        "modalities": name_list,
        "modality_shapes": mod_shapes,
        "n_joint_clusters": n_clusters,
        "joint_cluster_sizes": clusters,
        "summary": (
            f"MOFA+ integration of {len(name_list)} modalities "
            f"({', '.join(f'{n}: {s[0]}cells x {s[1]}features' for n, s in mod_shapes.items())}). "
            f"{n_shared:,} shared cells → {n_factors} factors → {n_clusters} joint clusters."
        ),
    }


# ---------------------------------------------------------------------------
# 21. omics.methylation_cluster
# ---------------------------------------------------------------------------


@registry.register(
    name="omics.methylation_cluster",
    description="Cluster samples by methylation patterns using episcanpy",
    category="omics",
    parameters={
        "path": "Path to methylation matrix (h5ad or CSV, CpG sites as rows, samples as columns)",
        "n_top_features": "Number of most variable CpGs to use (default 5000)",
        "resolution": "Leiden clustering resolution (default 1.0)",
    },
    usage_guide=(
        "Cluster cells/samples by DNA methylation profiles. Uses episcanpy for "
        "methylation-aware preprocessing if available, falls back to scanpy/sklearn. "
        "Works on Illumina 450K/EPIC beta-value matrices or single-cell methylation h5ad."
    ),
)
def methylation_cluster(
    path: str,
    n_top_features: int = 5000,
    resolution: float = 1.0,
    **kwargs,
) -> dict:
    """Cluster samples by methylation patterns."""
    import numpy as np

    filepath = Path(path).expanduser()
    if not filepath.exists():
        return {"error": f"File not found: {path}", "summary": f"File not found: {path}"}

    epi = _check_episcanpy()
    sc = _check_scanpy()

    # Load data
    adata = None
    if filepath.suffix.lower() == ".h5ad":
        if sc is None and epi is None:
            return {"error": "scanpy or episcanpy required for h5ad", "summary": "Install scanpy or episcanpy"}
        reader = epi if epi else sc
        try:
            adata = reader.read_h5ad(filepath)
        except Exception as exc:
            return {"error": f"Failed to load h5ad: {str(exc)[:200]}", "summary": f"Load error: {str(exc)[:100]}"}
    else:
        # Tabular: load as AnnData
        import pandas as pd

        df, error = _load_tabular(str(filepath))
        if error:
            return {"error": error, "summary": f"Could not load: {error}"}
        try:
            import anndata

            # Transpose so samples are obs and CpGs are var
            adata = anndata.AnnData(X=df.T.values, obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))
        except ImportError:
            return {"error": "anndata required: pip install anndata", "summary": "Install anndata"}

    n_obs, n_vars = adata.n_obs, adata.n_vars

    # Use episcanpy pipeline if available
    if epi is not None:
        try:
            # episcanpy variable feature selection
            epi.pp.filter_features(adata, min_cells=max(1, int(n_obs * 0.05)))
            epi.pp.select_var_feature(adata, nb_features=min(n_top_features, adata.n_vars))
            adata_use = adata[:, adata.var["highly_variable"]] if "highly_variable" in adata.var else adata
            epi.pp.pca(adata_use, n_comps=min(50, adata_use.n_vars - 1, adata_use.n_obs - 1))
            epi.pp.neighbors(adata_use, n_neighbors=15)
            epi.tl.leiden(adata_use, resolution=resolution)

            clusters = adata_use.obs["leiden"].value_counts().to_dict()

            result = {
                "method": "episcanpy",
                "n_samples": n_obs,
                "n_features_input": n_vars,
                "n_features_used": adata_use.n_vars,
                "n_clusters": len(clusters),
                "cluster_sizes": clusters,
                "summary": (
                    f"Methylation clustering (episcanpy): {n_obs} samples, {adata_use.n_vars} variable CpGs → "
                    f"{len(clusters)} clusters."
                ),
            }

            # Try to find marker CpGs
            try:
                epi.tl.rank_features(adata_use, groupby="leiden")
                markers = {}
                for cl in adata_use.obs["leiden"].unique():
                    markers[str(cl)] = list(adata_use.uns["rank_features_groups"]["names"][cl][:5])
                result["cluster_markers"] = markers
            except Exception:
                pass

            return result

        except Exception as exc:
            logger.warning("episcanpy pipeline failed, falling back to scanpy: %s", exc)

    # Fallback: scanpy or sklearn
    if sc is not None:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_features, adata.n_vars))
            adata_use = adata[:, adata.var["highly_variable"]]
            sc.tl.pca(adata_use, n_comps=min(50, adata_use.n_vars - 1, adata_use.n_obs - 1))
            sc.pp.neighbors(adata_use, n_neighbors=15)
            sc.tl.leiden(adata_use, resolution=resolution)

            clusters = adata_use.obs["leiden"].value_counts().to_dict()
            return {
                "method": "scanpy (episcanpy not installed)",
                "n_samples": n_obs,
                "n_features_input": n_vars,
                "n_features_used": adata_use.n_vars,
                "n_clusters": len(clusters),
                "cluster_sizes": clusters,
                "summary": (
                    f"Methylation clustering (scanpy fallback): {n_obs} samples → "
                    f"{len(clusters)} clusters. Install episcanpy for methylation-specific analysis."
                ),
            }
        except Exception as exc:
            logger.warning("scanpy fallback failed: %s", exc)

    # Last resort: sklearn KMeans
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = adata.X
    X = np.nan_to_num(X, nan=0.0)
    X = StandardScaler().fit_transform(X)
    n_comps = min(50, X.shape[0] - 1, X.shape[1] - 1)
    X_pca = PCA(n_components=n_comps).fit_transform(X)
    n_k = min(10, X.shape[0] // 2)
    labels = KMeans(n_clusters=max(n_k, 2), random_state=42, n_init=3).fit_predict(X_pca)

    import pandas as pd

    cluster_counts = pd.Series(labels).value_counts().to_dict()
    return {
        "method": "sklearn (install episcanpy or scanpy for better results)",
        "n_samples": n_obs,
        "n_features_input": n_vars,
        "n_clusters": len(cluster_counts),
        "cluster_sizes": {str(k): v for k, v in cluster_counts.items()},
        "summary": (
            f"Methylation clustering (sklearn fallback): {n_obs} samples → "
            f"{len(cluster_counts)} clusters. Install episcanpy for methylation-specific analysis."
        ),
    }


# ---------------------------------------------------------------------------
# KEGG over-representation analysis (code-gen tool)
# ---------------------------------------------------------------------------

KEGG_ORA_SYSTEM_PROMPT = """You are an expert bioinformatics data analyst performing KEGG pathway over-representation analysis.

{namespace_description}

## Available Data
{data_files_description}

## DATA EXPLORATION (DO THIS FIRST)
```python
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Head:\\n", df.head(3))
if 'Unnamed: 0' in df.columns:
    df = df.set_index('Unnamed: 0')
```

## KEGG ORA METHOD
### Step 1: Determine organism code
Common codes: 'hsa' (human), 'mmu' (mouse), 'eco' (E. coli), 'sce' (yeast).
Check https://rest.kegg.jp/list/organism for others.

### Step 2: Fetch gene-pathway mappings
- `/link/pathway/{{org}}` returns gene-to-pathway mapping (strip `path:` prefix from pathway IDs)
- `/list/pathway/{{org}}` returns pathway names (already without `path:` prefix)
- `/list/{{org}}` returns ALL genes (use as background universe — not just pathway-annotated genes)
- Pathway names include organism suffix; use substring matching when searching.

### ORA parameters
- **Background**: all genes from `/list/{{org}}` (typically much larger than the pathway-annotated subset)
- **Size filters**: skip pathways with < 5 or > 500 genes
- **Significance**: p < 0.05 and BH-adjusted p < 0.05

### Step 3: Fisher's exact test
```python
import urllib.request
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

def run_kegg_ora(gene_ids, all_kegg_genes, path2genes, path_names, min_size=5, max_size=500):
    deg_kegg = set(gene_ids) & all_kegg_genes
    N = len(all_kegg_genes)
    n = len(deg_kegg)
    if n == 0:
        return pd.DataFrame()
    results = []
    for pid, pgenes in path2genes.items():
        K = len(pgenes)
        if K < min_size or K > max_size:
            continue
        k = len(deg_kegg & pgenes)
        if k == 0:
            continue
        _, pval = fisher_exact([[k, n-k], [K-k, N-K-n+k]], alternative='greater')
        results.append({{'pathway': pid, 'name': path_names.get(pid, ''),
                        'overlap': k, 'pathway_size': K, 'pvalue': pval}})
    if not results:
        return pd.DataFrame()
    res_df = pd.DataFrame(results)
    _, res_df['padj'], _, _ = multipletests(res_df['pvalue'], method='fdr_bh')
    return res_df
```

### Step 4: Gene ID matching
KEGG uses its own gene IDs. Always print examples from both your DEG list and KEGG to
verify overlap. If overlap is low (< 10%), try stripping prefixes or case normalization.

### Directional analysis
When working with DEG results (log2FoldChange), run ORA separately on upregulated
(log2FC > threshold) and downregulated (log2FC < -threshold) genes. Combined analysis
mixes opposing signals and can produce different pathway results.

## Rules
1. Do NOT import libraries already in the namespace (pd, np, plt, sns, scipy_stats, etc.)
2. Save plots to OUTPUT_DIR: `plt.savefig(OUTPUT_DIR / "filename.png", dpi=150, bbox_inches="tight")`; `plt.close()`
3. Assign result: `result = {{"summary": "...", "answer": "PRECISE_ANSWER"}}`
4. Use print() for intermediate output to verify correctness.
5. If 0 results from a filter: print the column values and debug — do not return "N/A".

Write ONLY the Python code. No explanation, no markdown fences.
"""


@registry.register(
    name="omics.kegg_ora",
    description=(
        "KEGG pathway over-representation analysis (ORA) on differentially expressed genes "
        "using KEGG REST API + Fisher's exact test + BH correction"
    ),
    category="omics",
    parameters={"goal": "ORA analysis to perform (include organism code if known, e.g. 'hsa' for human)"},
    usage_guide=(
        "Use when the question asks about KEGG pathway enrichment via ORA (not GSEA). "
        "Handles non-human organisms via KEGG REST API. Uses Fisher's exact test with "
        "Benjamini-Hochberg FDR correction. "
        "For human gene set enrichment with gseapy, use code.execute instead."
    ),
)
def kegg_ora(goal: str, _session=None, _prior_results=None, **kwargs) -> dict:
    """Perform KEGG pathway over-representation analysis using generated code."""
    from ct.tools.code import _generate_and_execute_code

    return _generate_and_execute_code(
        goal=goal,
        system_prompt_template=KEGG_ORA_SYSTEM_PROMPT,
        session=_session,
        prior_results=_prior_results,
    )
