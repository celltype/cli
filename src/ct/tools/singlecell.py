"""
Single-cell analysis tools: clustering, trajectory inference, cell type annotation.

Requires scanpy for computation. Gracefully returns install instructions if missing.
"""

from ct.tools import registry


# Canonical marker gene panels for common cell types
MARKER_PANELS = {
    "T cells": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC"],
    "CD4+ T cells": ["CD4", "IL7R", "CCR7", "LEF1"],
    "CD8+ T cells": ["CD8A", "CD8B", "GZMK", "GZMB"],
    "Regulatory T cells": ["FOXP3", "IL2RA", "CTLA4", "TIGIT"],
    "B cells": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5"],
    "Plasma cells": ["JCHAIN", "MZB1", "SDC1", "XBP1"],
    "NK cells": ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCAM1"],
    "Monocytes": ["LYZ", "S100A8", "S100A9", "CD14", "FCGR3A"],
    "Macrophages": ["CD68", "CD163", "MRC1", "MSR1", "MARCO"],
    "Dendritic cells": ["FCER1A", "CLEC10A", "CD1C", "ITGAX"],
    "Plasmacytoid DCs": ["LILRA4", "IRF7", "TCF4", "CLEC4C"],
    "Neutrophils": ["CSF3R", "FCGR3B", "CXCR2", "S100A12"],
    "Mast cells": ["KIT", "TPSAB1", "TPSB2", "CPA3"],
    "Erythrocytes": ["HBA1", "HBA2", "HBB", "GYPA"],
    "Platelets": ["PPBP", "PF4", "GP9", "ITGA2B"],
    "Fibroblasts": ["DCN", "COL1A1", "COL1A2", "LUM", "PDGFRA"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "ERG", "FLT1"],
    "Epithelial": ["EPCAM", "KRT18", "KRT19", "CDH1"],
}


def _check_scanpy():
    """Check if scanpy is installed and return it, or None."""
    try:
        import scanpy as sc
        return sc
    except ImportError:
        return None


@registry.register(
    name="singlecell.cluster",
    description="Cluster single-cell RNA-seq data using Leiden/Louvain community detection with PCA and UMAP embedding",
    category="singlecell",
    parameters={
        "data_path": "Path to h5ad or CSV file with single-cell expression data",
        "resolution": "Clustering resolution (higher = more clusters, default 1.0)",
        "method": "Clustering method: 'leiden' or 'louvain' (default 'leiden')",
    },
    usage_guide="You have single-cell RNA-seq data and need to identify cell populations. Run this first in any single-cell analysis workflow. Produces cluster assignments and UMAP coordinates for downstream annotation.",
)
def cluster(data_path: str, resolution: float = 1.0, method: str = "leiden", **kwargs) -> dict:
    """Cluster single-cell data: load -> normalize -> PCA -> neighbors -> clustering -> UMAP.

    Supports h5ad (AnnData) and CSV input formats. Returns cluster assignments,
    top marker genes per cluster, and UMAP coordinate summary.
    """
    sc = _check_scanpy()
    if sc is None:
        return {
            "error": "scanpy is required for single-cell clustering. Install with: pip install scanpy",
            "summary": "scanpy not installed. Install with: pip install scanpy",
        }

    import numpy as np

    # Load data
    try:
        if data_path.endswith(".h5ad"):
            adata = sc.read_h5ad(data_path)
        elif data_path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(data_path, index_col=0)
            from anndata import AnnData
            adata = AnnData(df)
        else:
            return {
                "error": f"Unsupported file format: {data_path}. Use .h5ad or .csv",
                "summary": f"Cannot read {data_path} — expected .h5ad or .csv format",
            }
    except Exception as e:
        return {
            "error": f"Failed to load data: {e}",
            "summary": f"Could not read single-cell data from {data_path}",
        }

    n_cells, n_genes = adata.shape

    # Store raw counts for marker gene detection
    adata.layers["counts"] = adata.X.copy()

    # Standard preprocessing pipeline
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Identify highly variable genes
    if n_genes > 2000:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, n_genes))
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata_hvg = adata.copy()

    # Scale and PCA
    sc.pp.scale(adata_hvg, max_value=10)
    n_pcs = min(50, adata_hvg.shape[1] - 1, adata_hvg.shape[0] - 1)
    sc.tl.pca(adata_hvg, n_comps=n_pcs)

    # Transfer PCA to full adata
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    # Neighbors and clustering
    n_neighbors = min(15, n_cells - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    if method == "leiden":
        try:
            sc.tl.leiden(adata, resolution=resolution, key_added="cluster")
        except Exception:
            # Fall back to louvain if leiden not available
            sc.tl.louvain(adata, resolution=resolution, key_added="cluster")
            method = "louvain"
    else:
        sc.tl.louvain(adata, resolution=resolution, key_added="cluster")

    # UMAP
    sc.tl.umap(adata)

    # Get cluster assignments
    clusters = adata.obs["cluster"].astype(str)
    n_clusters = clusters.nunique()
    cluster_sizes = clusters.value_counts().to_dict()

    # Find marker genes per cluster
    try:
        sc.tl.rank_genes_groups(adata, groupby="cluster", method="wilcoxon", layer="counts")
        marker_genes = {}
        for cl in sorted(clusters.unique(), key=lambda x: int(x) if x.isdigit() else x):
            names = adata.uns["rank_genes_groups"]["names"][cl][:5]
            scores = adata.uns["rank_genes_groups"]["scores"][cl][:5]
            marker_genes[str(cl)] = [
                {"gene": str(n), "score": round(float(s), 3)}
                for n, s in zip(names, scores)
            ]
    except Exception:
        marker_genes = {}

    # UMAP summary statistics
    umap_coords = adata.obsm["X_umap"]
    umap_summary = {
        "min_x": round(float(np.min(umap_coords[:, 0])), 3),
        "max_x": round(float(np.max(umap_coords[:, 0])), 3),
        "min_y": round(float(np.min(umap_coords[:, 1])), 3),
        "max_y": round(float(np.max(umap_coords[:, 1])), 3),
    }

    # Build summary text
    marker_str_parts = []
    for cl in sorted(marker_genes.keys(), key=lambda x: int(x) if x.isdigit() else x)[:5]:
        genes = ", ".join(m["gene"] for m in marker_genes[cl][:3])
        marker_str_parts.append(f"cluster {cl}: {genes}")
    marker_summary = "; ".join(marker_str_parts) if marker_str_parts else "N/A"

    summary = (
        f"Clustered {n_cells} cells into {n_clusters} clusters "
        f"({method} r={resolution}). "
        f"Top markers: {marker_summary}"
    )

    return {
        "summary": summary,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_clusters": n_clusters,
        "method": method,
        "resolution": resolution,
        "cluster_sizes": cluster_sizes,
        "marker_genes": marker_genes,
        "umap_summary": umap_summary,
    }


@registry.register(
    name="singlecell.trajectory",
    description="Infer developmental trajectories and pseudotime from single-cell data using diffusion maps and PAGA",
    category="singlecell",
    parameters={
        "data_path": "Path to h5ad file (ideally pre-clustered from singlecell.cluster)",
        "root_cluster": "Cluster to use as root for pseudotime (optional, auto-detected if not set)",
        "method": "Trajectory method: 'diffmap' (default) or 'paga'",
    },
    usage_guide="You have clustered single-cell data and want to understand differentiation trajectories, lineage relationships, or developmental ordering. Run after singlecell.cluster. Computes pseudotime and identifies branch points.",
)
def trajectory(data_path: str, root_cluster: str = None, method: str = "diffmap", **kwargs) -> dict:
    """Infer trajectories using diffusion map + PAGA.

    Computes diffusion pseudotime from a root cell (selected from root_cluster
    or auto-detected as the cluster with lowest diffusion component 1).
    PAGA provides a coarse-grained graph of cluster connectivity.
    """
    sc = _check_scanpy()
    if sc is None:
        return {
            "error": "scanpy is required for trajectory analysis. Install with: pip install scanpy",
            "summary": "scanpy not installed. Install with: pip install scanpy",
        }

    import numpy as np

    # Load data
    try:
        if data_path.endswith(".h5ad"):
            adata = sc.read_h5ad(data_path)
        else:
            return {
                "error": "Trajectory analysis requires h5ad format with pre-computed neighbors",
                "summary": "Use singlecell.cluster first to generate an h5ad file",
            }
    except Exception as e:
        return {"error": f"Failed to load data: {e}", "summary": f"Could not read {data_path}"}

    n_cells = adata.shape[0]

    # Ensure neighbors are computed
    if "neighbors" not in adata.uns:
        n_neighbors = min(15, n_cells - 1)
        n_pcs = min(50, adata.shape[1] - 1, n_cells - 1)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Ensure clustering exists
    cluster_key = None
    for key in ["cluster", "leiden", "louvain"]:
        if key in adata.obs.columns:
            cluster_key = key
            break
    if cluster_key is None:
        return {
            "error": "No cluster assignments found. Run singlecell.cluster first.",
            "summary": "Pre-clustered data required for trajectory analysis",
        }

    # Compute diffusion map
    sc.tl.diffmap(adata, n_comps=15)

    # PAGA for coarse-grained connectivity
    sc.tl.paga(adata, groups=cluster_key)
    paga_connectivities = adata.uns["paga"]["connectivities"].toarray()

    # Determine root cell
    clusters = adata.obs[cluster_key].astype(str)
    if root_cluster is not None:
        root_cluster = str(root_cluster)
        if root_cluster not in clusters.values:
            return {
                "error": f"Root cluster '{root_cluster}' not found. Available: {sorted(clusters.unique())}",
                "summary": f"Invalid root cluster: {root_cluster}",
            }
        # Select root as cell in root_cluster with lowest DC1
        mask = clusters == root_cluster
        dc1_values = adata.obsm["X_diffmap"][mask, 0]
        root_idx_in_cluster = np.argmin(dc1_values)
        root_idx = np.where(mask)[0][root_idx_in_cluster]
    else:
        # Auto-detect: cell with lowest DC1 value
        root_idx = int(np.argmin(adata.obsm["X_diffmap"][:, 0]))
        root_cluster = str(clusters.iloc[root_idx])

    adata.uns["iroot"] = root_idx

    # Compute diffusion pseudotime
    sc.tl.dpt(adata)
    pseudotime = adata.obs["dpt_pseudotime"].values

    # Identify branches (clusters connected in PAGA)
    cluster_names = sorted(clusters.unique(), key=lambda x: int(x) if x.isdigit() else x)
    n_clusters = len(cluster_names)

    # Find branch points: clusters connected to 3+ other clusters in PAGA
    branch_points = []
    paga_threshold = 0.1
    for i, cl in enumerate(cluster_names):
        n_connections = np.sum(paga_connectivities[i] > paga_threshold)
        if n_connections >= 3:
            branch_points.append({
                "cluster": cl,
                "n_connections": int(n_connections),
                "connected_to": [
                    cluster_names[j]
                    for j in range(n_clusters)
                    if paga_connectivities[i, j] > paga_threshold and i != j
                ],
            })

    # Pseudotime statistics per cluster
    pseudotime_stats = {}
    for cl in cluster_names:
        mask = clusters == cl
        pt_values = pseudotime[mask]
        valid = pt_values[np.isfinite(pt_values)]
        if len(valid) > 0:
            pseudotime_stats[cl] = {
                "mean": round(float(np.mean(valid)), 4),
                "median": round(float(np.median(valid)), 4),
                "min": round(float(np.min(valid)), 4),
                "max": round(float(np.max(valid)), 4),
            }

    # Lineage ordering: sort clusters by mean pseudotime
    lineage_order = sorted(
        pseudotime_stats.keys(),
        key=lambda x: pseudotime_stats[x]["mean"],
    )

    valid_pt = pseudotime[np.isfinite(pseudotime)]
    pt_range = (round(float(np.min(valid_pt)), 4), round(float(np.max(valid_pt)), 4))

    summary = (
        f"Trajectory analysis: {len(branch_points)} branch point(s) from root "
        f"(cluster {root_cluster}), pseudotime range {pt_range[0]}-{pt_range[1]}"
    )

    return {
        "summary": summary,
        "n_cells": n_cells,
        "root_cluster": root_cluster,
        "root_cell_index": int(root_idx),
        "method": method,
        "pseudotime_range": pt_range,
        "pseudotime_per_cluster": pseudotime_stats,
        "lineage_order": lineage_order,
        "branch_points": branch_points,
        "n_branches": len(branch_points),
        "paga_connectivities": {
            cluster_names[i]: {
                cluster_names[j]: round(float(paga_connectivities[i, j]), 4)
                for j in range(n_clusters)
                if paga_connectivities[i, j] > paga_threshold and i != j
            }
            for i in range(n_clusters)
        },
    }


@registry.register(
    name="singlecell.cell_type_annotate",
    description="Annotate cell clusters with cell type labels using marker gene panels or CellTypist",
    category="singlecell",
    parameters={
        "data_path": "Path to h5ad file (should be clustered, e.g. from singlecell.cluster)",
        "reference": "Reference panel: 'immune', 'pbmc', 'tissue', or 'all' (default 'immune')",
        "method": "Annotation method: 'marker_based' (default) or 'celltypist'",
    },
    usage_guide="You have clustered single-cell data and need to assign cell type identities. Run after singlecell.cluster. Uses canonical marker genes to score each cluster against known cell type signatures.",
)
def cell_type_annotate(data_path: str, reference: str = "immune", method: str = "marker_based", **kwargs) -> dict:
    """Annotate clusters with cell type labels.

    marker_based: Score each cluster using canonical marker gene panels.
    celltypist: Use CellTypist automated annotation (requires celltypist package).
    """
    if method == "celltypist":
        try:
            import celltypist
        except ImportError:
            return {
                "error": "celltypist is required for automated annotation. Install with: pip install celltypist",
                "summary": "celltypist not installed. Use method='marker_based' or install with: pip install celltypist",
            }

    sc = _check_scanpy()
    if sc is None:
        return {
            "error": "scanpy is required for cell type annotation. Install with: pip install scanpy",
            "summary": "scanpy not installed. Install with: pip install scanpy",
        }

    import numpy as np

    # Load data
    try:
        if data_path.endswith(".h5ad"):
            adata = sc.read_h5ad(data_path)
        elif data_path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(data_path, index_col=0)
            from anndata import AnnData
            adata = AnnData(df)
        else:
            return {
                "error": f"Unsupported file format: {data_path}",
                "summary": f"Cannot read {data_path} — expected .h5ad or .csv",
            }
    except Exception as e:
        return {"error": f"Failed to load data: {e}", "summary": f"Could not read {data_path}"}

    # Find cluster key
    cluster_key = None
    for key in ["cluster", "leiden", "louvain", "cell_type"]:
        if key in adata.obs.columns:
            cluster_key = key
            break

    if cluster_key is None:
        return {
            "error": "No cluster assignments found. Run singlecell.cluster first.",
            "summary": "Pre-clustered data required for annotation",
        }

    clusters = adata.obs[cluster_key].astype(str)
    cluster_names = sorted(clusters.unique(), key=lambda x: int(x) if x.isdigit() else x)
    n_clusters = len(cluster_names)
    n_cells = adata.shape[0]

    # Select marker panels based on reference
    if reference in ("immune", "pbmc"):
        panels = {k: v for k, v in MARKER_PANELS.items()
                  if k not in ("Fibroblasts", "Endothelial", "Epithelial")}
    elif reference == "tissue":
        panels = MARKER_PANELS.copy()
    else:
        panels = MARKER_PANELS.copy()

    if method == "celltypist":
        # CellTypist annotation path
        try:
            import celltypist
            from celltypist import models as ct_models

            ct_models.download_models(force_update=False)
            model = ct_models.Model.load(model="Immune_All_Low.pkl")
            predictions = celltypist.annotate(adata, model=model, majority_voting=True)
            adata_result = predictions.to_adata()

            annotations = {}
            for cl in cluster_names:
                mask = clusters == cl
                cl_types = adata_result.obs.loc[mask, "majority_voting"].value_counts()
                top_type = cl_types.index[0] if len(cl_types) > 0 else "Unknown"
                confidence = float(cl_types.iloc[0] / cl_types.sum()) if len(cl_types) > 0 else 0.0
                annotations[cl] = {
                    "cell_type": top_type,
                    "confidence": round(confidence, 3),
                    "n_cells": int(mask.sum()),
                    "method": "celltypist",
                }

            annotation_list = list(annotations.values())
        except Exception as e:
            return {
                "error": f"CellTypist annotation failed: {e}",
                "summary": f"CellTypist error — try method='marker_based' instead",
            }
    else:
        # Marker-based annotation
        gene_names = set(adata.var_names)

        annotations = {}
        for cl in cluster_names:
            mask = clusters == cl
            n_cells_cl = int(mask.sum())

            # Get mean expression for this cluster
            if hasattr(adata.X, "toarray"):
                cl_expr = np.array(adata.X[mask].toarray().mean(axis=0)).flatten()
            else:
                cl_expr = np.array(adata.X[mask].mean(axis=0)).flatten()

            gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

            # Score each cell type panel
            scores = {}
            for cell_type, markers in panels.items():
                present_markers = [m for m in markers if m in gene_names]
                if not present_markers:
                    continue
                marker_indices = [gene_to_idx[m] for m in present_markers]
                marker_expr = cl_expr[marker_indices]
                # Score = mean expression of present markers, weighted by fraction present
                score = float(np.mean(marker_expr)) * (len(present_markers) / len(markers))
                scores[cell_type] = round(score, 4)

            if scores:
                best_type = max(scores, key=scores.get)
                best_score = scores[best_type]
                # Confidence: ratio of best score to second-best
                sorted_scores = sorted(scores.values(), reverse=True)
                if len(sorted_scores) > 1 and sorted_scores[1] > 0:
                    specificity = sorted_scores[0] / sorted_scores[1]
                else:
                    specificity = float("inf") if best_score > 0 else 0.0
                confidence = min(1.0, best_score * min(specificity, 5.0) / 5.0) if best_score > 0 else 0.0
            else:
                best_type = "Unknown"
                best_score = 0.0
                confidence = 0.0
                scores = {}

            annotations[cl] = {
                "cell_type": best_type,
                "confidence": round(confidence, 3),
                "n_cells": n_cells_cl,
                "marker_score": round(best_score, 4),
                "all_scores": dict(sorted(scores.items(), key=lambda x: -x[1])[:5]),
                "method": "marker_based",
            }

        annotation_list = list(annotations.values())

    # Compute cell type distribution
    type_counts = {}
    for ann in annotation_list:
        ct = ann["cell_type"]
        type_counts[ct] = type_counts.get(ct, 0) + ann["n_cells"]

    total_cells = sum(type_counts.values())
    type_distribution = {
        ct: f"{count / total_cells:.0%}" for ct, count in
        sorted(type_counts.items(), key=lambda x: -x[1])
    }

    # Summary
    dist_str = ", ".join(f"{ct} ({pct})" for ct, pct in list(type_distribution.items())[:5])
    summary = f"Annotated {n_clusters} clusters: {dist_str}"

    return {
        "summary": summary,
        "n_cells": n_cells,
        "n_clusters": n_clusters,
        "method": method,
        "reference": reference,
        "annotations": annotations,
        "cell_type_distribution": type_distribution,
    }
