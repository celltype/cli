"""
Expression analysis tools: L1000 signatures, pathway enrichment, TF activity, immune scoring.
"""

import pandas as pd
import numpy as np
from ct.tools import registry


@registry.register(
    name="expression.pathway_enrichment",
    description="Score compounds for pathway activation/suppression using L1000 gene expression signatures",
    category="expression",
    parameters={
        "compound_id": "Compound ID to score (or 'all' for full library)",
        "pathways": "Pathway collection: hallmark, kegg, reactome, go_bp, or custom dict",
    },
    requires_data=["l1000"],
    usage_guide="You want to understand which biological pathways a compound activates or suppresses. Use for mechanism of action investigation and to identify pathway-level effects from transcriptomic data.",
)
def pathway_enrichment(compound_id: str = "all", pathways: str = "hallmark",
                       gene_sets: dict = None, **kwargs) -> dict:
    """Score compounds for pathway enrichment using mean z-score method."""
    from ct.data.loaders import load_l1000
    from ct.tools._compound_resolver import resolve_compound

    if compound_id != "all":
        compound_id = resolve_compound(compound_id, dataset="l1000")

    l1000 = load_l1000()

    # Z-score normalize per gene (replace zero std with NaN to avoid inf)
    std = l1000.std()
    std = std.replace(0, float("nan"))
    z = (l1000 - l1000.mean()) / std

    # Load or use provided gene sets
    if gene_sets is None:
        gene_sets = _get_default_gene_sets(pathways)

    results = []
    compounds = [compound_id] if compound_id != "all" else z.index.tolist()
    compounds_not_found = []
    low_coverage_pathways = 0

    for cpd in compounds:
        if cpd not in z.index:
            compounds_not_found.append(cpd)
            continue
        row = z.loc[cpd]

        for pathway_name, genes in gene_sets.items():
            available = [g for g in genes if g in row.index]
            if len(available) < 3:
                low_coverage_pathways += 1
                continue
            score = row[available].mean()
            results.append({
                "compound": cpd,
                "pathway": pathway_name,
                "score": round(float(score), 4),
                "n_genes": len(available),
                "coverage": round(len(available) / len(genes), 2),
            })

    df = pd.DataFrame(results)

    if compound_id != "all" and len(df) > 0:
        top_activated = df.nlargest(5, "score")
        top_suppressed = df.nsmallest(5, "score")
        summary = (
            f"Pathway enrichment for {compound_id}:\n"
            f"Top activated: {', '.join(top_activated['pathway'].tolist())}\n"
            f"Top suppressed: {', '.join(top_suppressed['pathway'].tolist())}"
        )
    else:
        summary = f"Scored {len(df)} compound-pathway pairs"

    result = {"summary": summary, "results": df.to_dict("records") if len(df) < 1000 else f"{len(df)} rows"}

    # Add diagnostics for empty results
    if len(df) == 0:
        diag_parts = []
        if compounds_not_found:
            diag_parts.append(
                f"Compound(s) not found in L1000 data: {', '.join(compounds_not_found)}. "
                "L1000 uses BRD IDs (e.g., BRD-K12345678) — try resolving via chemistry.pubchem_lookup first."
            )
        if low_coverage_pathways > 0:
            diag_parts.append(
                f"{low_coverage_pathways} pathways skipped due to <3 genes overlapping with L1000 landmark genes."
            )
        if not diag_parts:
            diag_parts.append("No compound-pathway pairs scored. Check compound ID format.")
        result["summary"] = "No pathway enrichment results. " + " ".join(diag_parts)
        result["compounds_not_found"] = compounds_not_found
        result["low_coverage_pathways"] = low_coverage_pathways

    return result


@registry.register(
    name="expression.immune_score",
    description="Score compounds for immune pathway activation (IFN-gamma, antigen presentation, IO potential)",
    category="expression",
    parameters={"compound_id": "Compound ID (or 'all')"},
    requires_data=["l1000"],
    usage_guide="You want to assess a compound's immuno-oncology potential — IFN-gamma response, antigen presentation, immune checkpoint effects. Use when evaluating IO combination strategies or immunogenic cell death.",
)
def immune_score(compound_id: str = "all", **kwargs) -> dict:
    """Score compounds across 11 immune gene sets."""
    immune_sets = {
        "ifn_gamma": ["STAT1", "IRF1", "GBP1", "GBP2", "CXCL10", "CXCL9", "IDO1", "TAP1",
                      "PSMB9", "PSMB8", "B2M", "HLA-A", "HLA-B", "HLA-C", "HLA-E"],
        "antigen_presentation": ["TAP1", "TAP2", "TAPBP", "B2M", "HLA-A", "HLA-B", "HLA-C",
                                  "HLA-DRA", "HLA-DRB1", "PSMB8", "PSMB9", "CALR", "CANX"],
        "nfkb": ["NFKB1", "NFKB2", "RELA", "RELB", "REL", "NFKBIA", "NFKBIB", "TNFAIP3",
                 "BCL2", "BCL2L1", "XIAP", "BIRC3", "CFLAR", "TRAF1", "TRAF2"],
        "t_cell_cytotoxicity": ["GZMA", "GZMB", "GZMK", "PRF1", "IFNG", "TNF", "FASLG",
                                 "CD8A", "CD8B"],
        "immune_checkpoints": ["CD274", "PDCD1LG2", "CTLA4", "HAVCR2", "LAG3", "TIGIT",
                                "CD47", "SIRPA", "CD80", "CD86"],
        "icd": ["CALR", "HMGB1", "ATP", "ANXA1", "HSP90AA1", "HSPA1A", "HSPA1B"],
    }

    result = pathway_enrichment(compound_id=compound_id, gene_sets=immune_sets)

    # Compute composite IO score
    if isinstance(result["results"], list):
        df = pd.DataFrame(result["results"])
        if len(df) > 0 and compound_id != "all":
            io_pathways = ["ifn_gamma", "antigen_presentation", "icd"]
            io_scores = df[df["pathway"].isin(io_pathways)]["score"]
            io_potential = io_scores.mean() if len(io_scores) > 0 else 0

            hot_pathways = ["ifn_gamma", "t_cell_cytotoxicity"]
            hot_scores = df[df["pathway"].isin(hot_pathways)]["score"]
            hot_tumor = hot_scores.mean() if len(hot_scores) > 0 else 0

            result["io_potential"] = round(float(io_potential), 4)
            result["hot_tumor_signature"] = round(float(hot_tumor), 4)
            result["immune_classification"] = (
                "immune_hot" if io_potential > 0.3 else
                "immune_cold" if io_potential < -0.3 else
                "neutral"
            )
            result["summary"] += (
                f"\nIO potential: {io_potential:.3f} ({result['immune_classification']})"
                f"\nHot tumor signature: {hot_tumor:.3f}"
            )

    return result


@registry.register(
    name="expression.l1000_similarity",
    description="Find compounds with similar or opposite L1000 transcriptomic signatures",
    category="expression",
    parameters={"compound_id": "Query compound", "mode": "'similar' or 'opposite'", "top_n": "Number of hits"},
    requires_data=["l1000"],
    usage_guide="You want to find compounds with similar mechanisms (mode='similar') or complementary/opposing effects (mode='opposite'). Use for drug repurposing or finding synergy partners.",
)
def l1000_similarity(compound_id: str, mode: str = "similar", top_n: int = 20, **kwargs) -> dict:
    """Find transcriptionally similar or anti-correlated compounds."""
    from ct.data.loaders import load_l1000
    from sklearn.metrics.pairwise import cosine_similarity
    from ct.tools._compound_resolver import resolve_compound

    compound_id = resolve_compound(compound_id, dataset="l1000")

    l1000 = load_l1000()

    if compound_id not in l1000.index:
        return {"error": f"Compound {compound_id} not found in L1000 data", "summary": f"Compound {compound_id} not found in L1000 data"}
    query = l1000.loc[compound_id].values.reshape(1, -1)
    sims = cosine_similarity(query, l1000.values)[0]
    sim_df = pd.DataFrame({"compound": l1000.index, "cosine_similarity": sims})
    sim_df = sim_df[sim_df["compound"] != compound_id]

    if mode == "similar":
        hits = sim_df.nlargest(top_n, "cosine_similarity")
    elif mode == "opposite":
        hits = sim_df.nsmallest(top_n, "cosine_similarity")
    else:
        return {"error": f"Unknown mode: {mode}. Use 'similar' or 'opposite'", "summary": f"Unknown mode: {mode}. Use 'similar' or 'opposite'"}
    return {
        "summary": f"Top {top_n} {mode} compounds to {compound_id}",
        "query": compound_id,
        "mode": mode,
        "hits": hits.to_dict("records"),
    }


def _get_default_gene_sets(collection: str) -> dict:
    """Get default gene sets for pathway enrichment."""
    # Hallmark-lite: key pathways for drug discovery
    if collection == "hallmark":
        return {
            "androgen_response": ["KLK3", "KLK2", "FKBP5", "TMPRSS2", "NKX3-1", "PMEPA1"],
            "ifn_alpha": ["ISG15", "MX1", "MX2", "IFIT1", "IFIT2", "IFIT3", "OAS1", "OAS2"],
            "apoptosis": ["BCL2", "BAX", "BAK1", "BID", "CASP3", "CASP8", "CASP9", "CYCS"],
            "p53_pathway": ["CDKN1A", "MDM2", "BAX", "GADD45A", "SFN", "DDB2", "SESN1"],
            "mtorc1_signaling": ["SLC7A5", "SLC3A2", "DDIT4", "VEGFA", "HK2", "PKM", "LDHA"],
            "unfolded_protein_response": ["HSPA5", "DDIT3", "ATF4", "XBP1", "HERPUD1", "DNAJB9"],
            "nfkb_signaling": ["NFKB1", "NFKB2", "RELA", "NFKBIA", "BCL2", "TNFAIP3"],
            "oxidative_phosphorylation": ["NDUFA1", "SDHA", "UQCRC1", "COX5A", "ATP5F1A"],
            "glycolysis": ["HK2", "PFKM", "ALDOA", "GAPDH", "PKM", "LDHA", "ENO1"],
            "dna_repair": ["BRCA1", "BRCA2", "RAD51", "ATM", "ATR", "CHEK1", "CHEK2"],
        }

    # Return empty if collection not recognized
    return {}


def _resolve_groups_by_lineage(
    group_a: list, group_b: list, expr: "pd.DataFrame"
) -> tuple:
    """Resolve descriptive group labels to L1000 compound IDs.

    When group labels (e.g. 'multiple_myeloma', 'solid_tumor') don't match
    L1000 index entries, try to map them via DepMap Model.csv lineage info.
    As a fallback, split available compounds into two halves so the analysis
    can still proceed.
    """
    all_ids = list(expr.index)

    # Try DepMap lineage mapping
    try:
        from ct.data.loaders import load_depmap_model
        models = load_depmap_model()

        # Build lineage -> set of cell line IDs mapping
        lineage_col = None
        for col in ["OncotreeLineage", "lineage", "Lineage", "primary_disease",
                     "PrimaryDisease", "disease"]:
            if col in models.columns:
                lineage_col = col
                break

        if lineage_col is not None:
            # Normalise lineage values for fuzzy matching
            def _norm(s):
                return str(s).lower().replace(" ", "_").replace("-", "_")

            lineage_map = {}
            for _, row in models.iterrows():
                lin = _norm(row[lineage_col])
                # Use ModelID or DepMap_ID as identifier
                mid = None
                for id_col in ["ModelID", "DepMap_ID", "stripped_cell_line_name",
                                "StrippedCellLineName"]:
                    if id_col in models.columns and pd.notna(row.get(id_col)):
                        mid = str(row[id_col])
                        break
                if mid and mid in expr.index:
                    lineage_map.setdefault(lin, []).append(mid)

            if lineage_map:
                norm_a = [_norm(label) for label in group_a]
                norm_b = [_norm(label) for label in group_b]

                matched_a = []
                for label in norm_a:
                    for lin, ids in lineage_map.items():
                        if label in lin or lin in label:
                            matched_a.extend(ids)
                matched_b = []
                for label in norm_b:
                    for lin, ids in lineage_map.items():
                        if label in lin or lin in label:
                            matched_b.extend(ids)

                if len(matched_a) >= 2 and len(matched_b) >= 2:
                    return list(set(matched_a)), list(set(matched_b))
    except Exception:
        pass

    return [], []


# ---- Marker gene sets for immune cell deconvolution ----
IMMUNE_MARKERS = {
    "T cells": ["CD3D", "CD3E", "CD8A", "CD4"],
    "B cells": ["CD19", "MS4A1", "CD79A"],
    "NK cells": ["NKG7", "GNLY", "KLRD1"],
    "Monocytes": ["CD14", "LYZ", "FCGR3A"],
    "Macrophages": ["CD68", "CD163", "CSF1R"],
    "Dendritic cells": ["ITGAX", "CLEC4C", "CD1C"],
    "Neutrophils": ["FCGR3B", "CSF3R", "CXCR2"],
    "Tregs": ["FOXP3", "IL2RA", "CTLA4"],
}

# ---- Curated TF regulons (TF -> target genes) ----
TF_REGULONS = {
    "TP53": ["CDKN1A", "MDM2", "BAX", "BBC3", "PUMA"],
    "MYC": ["ODC1", "LDHA", "CDK4", "NCL"],
    "NFkB": ["NFKBIA", "TNF", "IL6", "CXCL8"],
    "HIF1A": ["VEGFA", "SLC2A1", "LDHA", "PDK1"],
    "STAT3": ["BCL2L1", "MMP9", "VEGFA", "MYC"],
    "E2F": ["CCNE1", "MCM2", "PCNA", "RRM2"],
    "AP1": ["FOS", "JUN", "MMP1", "IL8"],
}


@registry.register(
    name="expression.deconvolution",
    description="Estimate immune cell type composition from bulk gene expression using marker gene-based deconvolution",
    category="expression",
    parameters={
        "gene_expression": "Dict of gene:value pairs (expression levels), OR omit and provide compound_id",
        "compound_id": "Compound ID to pull L1000 signature for (optional, used if gene_expression not provided)",
    },
    requires_data=[],
    usage_guide="You want to estimate the immune cell type composition implied by a gene expression profile. "
                "Useful for understanding immune microenvironment effects of compounds or patient samples. "
                "Provide a gene expression dict directly, or a compound_id to pull from L1000 data.",
)
def deconvolution(gene_expression: dict = None, compound_id: str = None, **kwargs) -> dict:
    """Estimate immune cell type proportions from bulk gene expression using marker genes.

    Uses a simple marker gene averaging approach: for each immune cell type, compute
    the mean expression of its marker genes, then normalize to proportions. This is
    a lightweight alternative to CIBERSORT that requires no license.
    """
    if gene_expression is None and compound_id is None:
        return {"error": "Provide either gene_expression (dict) or compound_id", "summary": "Provide either gene_expression (dict) or compound_id"}
    # If compound_id provided, pull expression from L1000
    if gene_expression is None:
        from ct.data.loaders import load_l1000
        from ct.tools._compound_resolver import resolve_compound
        compound_id = resolve_compound(compound_id, dataset="l1000")
        l1000 = load_l1000()
        if compound_id not in l1000.index:
            return {"error": f"Compound {compound_id} not found in L1000 data", "summary": f"Compound {compound_id} not found in L1000 data"}
        row = l1000.loc[compound_id]
        gene_expression = row.to_dict()

    # Score each cell type by mean expression of its markers
    cell_scores = {}
    marker_details = {}
    for cell_type, markers in IMMUNE_MARKERS.items():
        available = [g for g in markers if g in gene_expression]
        if not available:
            cell_scores[cell_type] = 0.0
            marker_details[cell_type] = {"n_markers": 0, "found": [], "mean_expr": 0.0}
            continue
        values = [gene_expression[g] for g in available]
        mean_val = float(np.mean(values))
        cell_scores[cell_type] = max(mean_val, 0.0)  # clamp negatives to 0
        marker_details[cell_type] = {
            "n_markers": len(available),
            "found": available,
            "mean_expr": round(mean_val, 4),
        }

    # Normalize to proportions
    total = sum(cell_scores.values())
    if total > 0:
        proportions = {ct: round(v / total, 4) for ct, v in cell_scores.items()}
    else:
        proportions = {ct: round(1.0 / len(cell_scores), 4) for ct in cell_scores}

    # Sort by proportion (descending)
    sorted_props = dict(sorted(proportions.items(), key=lambda x: x[1], reverse=True))
    dominant = next(iter(sorted_props))
    dominant_pct = sorted_props[dominant]

    # Compute aggregate immune score (sum of raw marker means, higher = more immune)
    immune_score = round(sum(max(v, 0) for v in cell_scores.values()), 4)

    # Format proportions for summary
    top3 = list(sorted_props.items())[:3]
    top3_str = ", ".join(f"{ct} {pct:.1%}" for ct, pct in top3)

    source = f"compound {compound_id}" if compound_id else "provided expression"
    summary = (
        f"Immune deconvolution ({source}):\n"
        f"Dominant: {dominant} ({dominant_pct:.1%})\n"
        f"Top 3: {top3_str}\n"
        f"Immune score: {immune_score:.2f}"
    )

    return {
        "summary": summary,
        "proportions": sorted_props,
        "dominant_cell_type": dominant,
        "immune_score": immune_score,
        "marker_details": marker_details,
    }


@registry.register(
    name="expression.tf_activity",
    description="Infer transcription factor activity from gene expression signatures using curated regulons",
    category="expression",
    parameters={
        "gene_expression": "Dict of gene:value pairs (expression changes), OR omit and provide compound_id",
        "compound_id": "Compound ID to pull L1000 signature for (optional)",
    },
    requires_data=[],
    usage_guide="You want to infer which transcription factors are activated or suppressed by a compound or "
                "in a condition. Uses curated regulons (TF -> target gene sets) to score TF activity from "
                "expression data. Provide a gene expression dict or compound_id for L1000 lookup.",
)
def tf_activity(gene_expression: dict = None, compound_id: str = None, **kwargs) -> dict:
    """Infer transcription factor activity from expression signatures.

    For each TF, scores activity as the mean expression change of its known target
    genes (regulon). Positive score = TF activated, negative = TF suppressed.
    """
    if gene_expression is None and compound_id is None:
        return {"error": "Provide either gene_expression (dict) or compound_id", "summary": "Provide either gene_expression (dict) or compound_id"}
    # If compound_id provided, pull expression from L1000
    if gene_expression is None:
        from ct.data.loaders import load_l1000
        from ct.tools._compound_resolver import resolve_compound
        compound_id = resolve_compound(compound_id, dataset="l1000")
        l1000 = load_l1000()
        if compound_id not in l1000.index:
            return {"error": f"Compound {compound_id} not found in L1000 data", "summary": f"Compound {compound_id} not found in L1000 data"}
        row = l1000.loc[compound_id]
        gene_expression = row.to_dict()

    # Score each TF by mean expression of its targets
    tf_scores = {}
    tf_details = {}
    for tf_name, targets in TF_REGULONS.items():
        available = [g for g in targets if g in gene_expression]
        if not available:
            tf_details[tf_name] = {"n_targets": 0, "found": [], "score": None}
            continue
        values = [gene_expression[g] for g in available]
        score = float(np.mean(values))
        tf_scores[tf_name] = score
        tf_details[tf_name] = {
            "n_targets": len(available),
            "found": available,
            "score": round(score, 4),
            "target_values": {g: round(gene_expression[g], 4) for g in available},
        }

    if not tf_scores:
        return {
            "summary": "No TF regulon targets found in expression data",
            "tf_scores": {},
            "activated": [],
            "suppressed": [],
        }

    # Rank by absolute activity
    sorted_tfs = sorted(tf_scores.items(), key=lambda x: abs(x[1]), reverse=True)

    # Classify as activated (> threshold) or suppressed (< threshold)
    activation_threshold = 0.3
    activated = [(tf, round(s, 4)) for tf, s in sorted_tfs if s > activation_threshold]
    suppressed = [(tf, round(s, 4)) for tf, s in sorted_tfs if s < -activation_threshold]
    neutral = [(tf, round(s, 4)) for tf, s in sorted_tfs
               if -activation_threshold <= s <= activation_threshold]

    # Build summary
    source = f"compound {compound_id}" if compound_id else "provided expression"
    act_str = ", ".join(f"{tf}(+{s:.2f})" for tf, s in activated) if activated else "none"
    sup_str = ", ".join(f"{tf}({s:.2f})" for tf, s in suppressed) if suppressed else "none"

    summary = (
        f"TF activity analysis ({source}):\n"
        f"Activated: {act_str}\n"
        f"Suppressed: {sup_str}\n"
        f"TFs scored: {len(tf_scores)}/{len(TF_REGULONS)}"
    )

    return {
        "summary": summary,
        "tf_scores": dict(sorted_tfs),
        "activated": [{"tf": tf, "score": s} for tf, s in activated],
        "suppressed": [{"tf": tf, "score": s} for tf, s in suppressed],
        "neutral": [{"tf": tf, "score": s} for tf, s in neutral],
        "details": tf_details,
    }


@registry.register(
    name="expression.diff_expression",
    description="Differential expression analysis between two groups of samples using L1000 data",
    category="expression",
    parameters={
        "gene": "Gene symbol to test, or 'all' to test all landmark genes",
        "group_a": "List of compound IDs or cell line names for group A",
        "group_b": "List of compound IDs or cell line names for group B",
        "dataset": "Expression dataset to use (default 'l1000')",
    },
    requires_data=["l1000"],
    usage_guide="You want to compare gene expression between two conditions — e.g. treated vs control, "
                "or two compound classes. Use Mann-Whitney U for robust rank-based testing. "
                "Set gene='all' for genome-wide differential expression with FDR correction.",
)
def diff_expression(
    gene: str = "all",
    group_a: list = None,
    group_b: list = None,
    dataset: str = "l1000",
    **kwargs,
) -> dict:
    """Differential expression between two groups of samples.

    Uses Mann-Whitney U test (rank-based, non-parametric) to compare expression
    of one or all landmark genes between two sample groups. Computes fold change,
    p-value, effect size (Cohen's d), and Benjamini-Hochberg FDR correction when
    testing multiple genes.
    """
    from scipy import stats as scipy_stats

    if group_a is None or group_b is None:
        return {"error": "Both group_a and group_b must be provided as lists of sample IDs", "summary": "Both group_a and group_b must be provided as lists of sample IDs"}
    if not group_a or not group_b:
        return {"error": "Both group_a and group_b must be non-empty lists", "summary": "Both group_a and group_b must be non-empty lists"}
    # Load expression data
    from ct.data.loaders import load_l1000
    expr = load_l1000()

    # Normalise inputs to lists
    if isinstance(group_a, str):
        group_a = [group_a]
    if isinstance(group_b, str):
        group_b = [group_b]

    if not group_a or not group_b:
        return {"error": "Both group_a and group_b must be non-empty lists"}

    # Identify which group samples are in the data (rows = samples/compounds)
    available_a = [s for s in group_a if s in expr.index]
    available_b = [s for s in group_b if s in expr.index]

    # If either group has too few direct matches, try resolving via DepMap lineage
    if len(available_a) < 2 or len(available_b) < 2:
        resolved_a, resolved_b = _resolve_groups_by_lineage(
            group_a, group_b, expr
        )
        if len(resolved_a) >= 2 and len(resolved_b) >= 2:
            available_a, available_b = resolved_a, resolved_b

    sample_hint = ", ".join(list(expr.index[:5])) + ", ..."
    if len(available_a) < 2:
        return {"error": f"Group A: only {len(available_a)} of {len(group_a)} labels found in data (need >=2). "
                         f"Provide compound IDs matching the L1000 index. Examples: {sample_hint}",
                 "summary": f"Group A: only {len(available_a)} of {len(group_a)} samples found in data (need >=2)"}
    if len(available_b) < 2:
        return {"error": f"Group B: only {len(available_b)} of {len(group_b)} labels found in data (need >=2). "
                         f"Provide compound IDs matching the L1000 index. Examples: {sample_hint}",
                 "summary": f"Group B: only {len(available_b)} of {len(group_b)} samples found in data (need >=2)"}
    data_a = expr.loc[available_a]
    data_b = expr.loc[available_b]

    # Determine which genes to test
    if gene == "all":
        genes_to_test = list(expr.columns)
    else:
        if gene not in expr.columns:
            return {"error": f"Gene '{gene}' not found in {dataset} expression data", "summary": f"Gene '{gene}' not found in {dataset} expression data"}
        genes_to_test = [gene]

    results = []
    for g in genes_to_test:
        vals_a = data_a[g].dropna()
        vals_b = data_b[g].dropna()

        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        mean_a = float(vals_a.mean())
        mean_b = float(vals_b.mean())

        # Fold change (group_a vs group_b): positive means higher in A
        fold_change = mean_a - mean_b  # log-scale data, so difference = log2 FC

        # Mann-Whitney U test
        try:
            stat, pval = scipy_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        except ValueError:
            continue

        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(vals_a) - 1) * vals_a.std() ** 2 + (len(vals_b) - 1) * vals_b.std() ** 2)
            / (len(vals_a) + len(vals_b) - 2)
        )
        cohens_d = fold_change / pooled_std if pooled_std > 0 else 0.0

        direction = "up_in_A" if fold_change > 0 else "up_in_B" if fold_change < 0 else "unchanged"

        results.append({
            "gene": g,
            "mean_a": round(mean_a, 4),
            "mean_b": round(mean_b, 4),
            "log2_fold_change": round(fold_change, 4),
            "direction": direction,
            "p_value": float(pval),
            "cohens_d": round(float(cohens_d), 4),
            "n_a": len(vals_a),
            "n_b": len(vals_b),
        })

    if not results:
        return {
            "summary": f"No testable genes found between groups (group_a={len(available_a)}, group_b={len(available_b)} samples)",
            "results": [],
        }

    df = pd.DataFrame(results).sort_values("p_value")

    # Benjamini-Hochberg FDR correction
    if len(df) > 1:
        n_tests = len(df)
        ranks = df["p_value"].rank(method="first")
        df["fdr"] = (df["p_value"] * n_tests / ranks).clip(upper=1.0)
        # Ensure monotonicity: work backward from largest rank
        fdr_vals = np.array(df.sort_values("p_value", ascending=False)["fdr"], dtype=float)
        for i in range(1, len(fdr_vals)):
            fdr_vals[i] = min(fdr_vals[i], fdr_vals[i - 1])
        df.loc[df.sort_values("p_value", ascending=False).index, "fdr"] = fdr_vals
        df = df.sort_values("p_value")
    else:
        df["fdr"] = df["p_value"]

    n_sig = int((df["p_value"] < 0.05).sum())
    n_fdr_sig = int((df["fdr"] < 0.05).sum())

    if gene != "all":
        row = df.iloc[0]
        summary = (
            f"Differential expression of {gene}: "
            f"log2FC={row['log2_fold_change']:.3f} ({row['direction']}), "
            f"p={row['p_value']:.2e}, Cohen's d={row['cohens_d']:.3f}"
        )
    else:
        top_genes = df.head(5)
        top_str = ", ".join(
            f"{r['gene']}(FC={r['log2_fold_change']:.2f}, p={r['p_value']:.2e})"
            for _, r in top_genes.iterrows()
        )
        summary = (
            f"Differential expression: {len(results)} genes tested, "
            f"{n_sig} nominally significant (p<0.05), {n_fdr_sig} FDR-significant. "
            f"Top: {top_str}"
        )

    return {
        "summary": summary,
        "n_tested": len(results),
        "n_significant_nominal": n_sig,
        "n_significant_fdr": n_fdr_sig,
        "group_a_size": len(available_a),
        "group_b_size": len(available_b),
        "results": df.to_dict("records"),
    }
