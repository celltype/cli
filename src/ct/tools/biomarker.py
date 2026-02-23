"""
Biomarker tools: mutation sensitivity, resistance profiling, dependency validation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from ct.tools import registry


@registry.register(
    name="biomarker.mutation_sensitivity",
    description="Test whether specific mutations sensitize or confer resistance to a compound",
    category="biomarker",
    parameters={"compound_id": "Compound YU ID", "gene": "Gene to test (or 'all' for genome-wide)"},
    requires_data=["prism", "depmap_mutations", "depmap_model"],
    usage_guide="You want to find predictive biomarkers — which mutations make cells more or less sensitive to a compound. Use for patient stratification and clinical trial design.",
)
def mutation_sensitivity(compound_id: str, gene: str = "all", **kwargs) -> dict:
    """Test mutation-sensitivity associations."""
    from ct.data.loaders import load_prism, load_mutations, load_model_metadata
    from ct.tools._compound_resolver import resolve_compound

    compound_id = resolve_compound(compound_id, dataset="prism")

    prism = load_prism()
    mutations = load_mutations()
    model = load_model_metadata()

    # Map PRISM cell lines to DepMap ModelIDs
    ccle_to_model = {}
    for _, row in model.iterrows():
        ccle = row.get("CCLEName", "")
        mid = row.get("ModelID", "")
        if pd.notna(ccle) and pd.notna(mid):
            ccle_to_model[ccle] = mid

    # Get compound sensitivity at highest dose
    cpd = prism[prism["pert_name"] == compound_id]
    if len(cpd) == 0:
        return {"error": f"Compound {compound_id} not in PRISM", "summary": f"Compound {compound_id} not found in PRISM data"}

    max_dose = cpd["pert_dose"].max()
    cpd_hd = cpd[cpd["pert_dose"] == max_dose].groupby("ccle_name")["LFC"].mean()

    # Map to ModelIDs
    sensitivity = {}
    for ccle, lfc in cpd_hd.items():
        mid = ccle_to_model.get(ccle)
        if mid and mid in mutations.index:
            sensitivity[mid] = lfc

    if len(sensitivity) < 20:
        return {"error": f"Insufficient overlap: only {len(sensitivity)} cell lines mapped", "summary": f"Insufficient cell line overlap ({len(sensitivity)} < 20 required)"}

    # Test genes
    genes_to_test = [gene] if gene != "all" else [g for g in mutations.columns if mutations[g].sum() >= 3]
    model_ids = list(sensitivity.keys())
    lfc_values = pd.Series(sensitivity)

    results = []
    for g in genes_to_test:
        if g not in mutations.columns:
            continue

        mut_status = mutations.loc[model_ids, g].reindex(model_ids).fillna(0)
        mutant_ids = [m for m in model_ids if mut_status.loc[m] > 0]
        wt_ids = [m for m in model_ids if mut_status.loc[m] == 0]

        if len(mutant_ids) < 3 or len(wt_ids) < 3:
            continue

        mut_lfc = lfc_values[mutant_ids]
        wt_lfc = lfc_values[wt_ids]
        stat, pval = stats.mannwhitneyu(mut_lfc, wt_lfc, alternative="two-sided")

        results.append({
            "gene": g,
            "mut_mean_lfc": round(float(mut_lfc.mean()), 3),
            "wt_mean_lfc": round(float(wt_lfc.mean()), 3),
            "delta": round(float(mut_lfc.mean() - wt_lfc.mean()), 3),
            "direction": "sensitizing" if mut_lfc.mean() < wt_lfc.mean() else "resistance",
            "pval": float(pval),
            "n_mutant": len(mutant_ids),
            "n_wt": len(wt_ids),
        })

    if not results:
        return {
            "summary": (
                f"Mutation sensitivity for {compound_id}: {len(genes_to_test)} genes tested, "
                f"0 significant (p<0.05). No genes met minimum sample size (3 mutant, 3 WT)."
            ),
            "compound": compound_id,
            "significant_mutations": [],
            "n_tested": 0,
        }

    df = pd.DataFrame(results).sort_values("pval")
    if len(df) > 0:
        # Benjamini-Hochberg FDR: p * m / rank (monotonicity enforced)
        ranks = df["pval"].rank(method="first")
        df["fdr"] = (df["pval"] * len(df) / ranks).clip(upper=1.0)
        # Enforce monotonicity: walk backwards to ensure FDR is non-decreasing with p-value
        fdr_arr = df["fdr"].values.copy()
        for i in range(len(fdr_arr) - 2, -1, -1):
            fdr_arr[i] = min(fdr_arr[i], fdr_arr[i + 1])
        df["fdr"] = fdr_arr

    sig = df[df["pval"] < 0.05]

    return {
        "summary": (
            f"Mutation sensitivity for {compound_id}: {len(genes_to_test)} genes tested, "
            f"{len(sig)} significant (p<0.05)"
        ),
        "compound": compound_id,
        "significant_mutations": sig.head(20).to_dict("records") if len(sig) > 0 else [],
        "n_tested": len(results),
    }


@registry.register(
    name="biomarker.resistance_profile",
    description="Profile resistance mechanisms for a compound (lineage, mutation, dependency enrichment)",
    category="biomarker",
    parameters={"compound_id": "Compound YU ID"},
    requires_data=["prism", "depmap_crispr", "depmap_mutations", "depmap_model"],
    usage_guide="You want to understand why some cell lines resist a compound — lineage effects, specific mutations, or dependency patterns. Use to anticipate clinical resistance mechanisms.",
)
def resistance_profile(compound_id: str, **kwargs) -> dict:
    """Comprehensive resistance profiling for a compound."""
    from ct.data.loaders import load_prism, load_model_metadata
    from ct.tools._compound_resolver import resolve_compound

    compound_id = resolve_compound(compound_id, dataset="prism")

    prism = load_prism()
    model = load_model_metadata()

    cpd = prism[prism["pert_name"] == compound_id]
    if len(cpd) == 0:
        return {"error": f"Compound {compound_id} not in PRISM", "summary": f"Compound {compound_id} not found in PRISM data"}

    max_dose = cpd["pert_dose"].max()
    cpd_hd = cpd[cpd["pert_dose"] == max_dose].groupby("ccle_name")["LFC"].mean()

    n_sensitive = (cpd_hd < -0.5).sum()
    n_resistant = (cpd_hd > -0.1).sum()
    n_intermediate = len(cpd_hd) - n_sensitive - n_resistant

    # Lineage enrichment
    ccle_to_lineage = {}
    for _, row in model.iterrows():
        ccle = row.get("CCLEName", "")
        lin = row.get("OncotreeLineage", "Unknown")
        if pd.notna(ccle) and pd.notna(lin):
            ccle_to_lineage[ccle] = lin

    sens_lineages = [ccle_to_lineage.get(c, "Unknown") for c in cpd_hd[cpd_hd < -0.5].index]
    res_lineages = [ccle_to_lineage.get(c, "Unknown") for c in cpd_hd[cpd_hd > -0.1].index]

    lineage_counts = {}
    for lin in set(sens_lineages + res_lineages):
        if lin == "Unknown":
            continue
        s = sens_lineages.count(lin)
        r = res_lineages.count(lin)
        if s + r >= 3:
            lineage_counts[lin] = {"sensitive": s, "resistant": r, "total": s + r}

    return {
        "summary": (
            f"Resistance profile for {compound_id}:\n"
            f"  Sensitive: {n_sensitive}, Intermediate: {n_intermediate}, Resistant: {n_resistant}\n"
            f"  {len(lineage_counts)} lineages profiled"
        ),
        "compound": compound_id,
        "n_sensitive": int(n_sensitive),
        "n_resistant": int(n_resistant),
        "n_intermediate": int(n_intermediate),
        "lineage_profiles": lineage_counts,
    }


@registry.register(
    name="biomarker.panel_select",
    description="ML-based biomarker panel selection: identify top predictive mutations for compound sensitivity",
    category="biomarker",
    parameters={
        "compound_id": "Compound ID (PRISM pert_name)",
        "n_features": "Number of top biomarker features to return (default 10)",
        "method": "Feature selection method: 'mutual_info', 'lasso', or 'random_forest' (default 'mutual_info')",
    },
    requires_data=["prism", "depmap_mutations", "depmap_model"],
    usage_guide="You want to select the best biomarker panel for predicting compound response — "
                "which mutations best predict sensitivity. Use for patient stratification, companion "
                "diagnostic design, and clinical trial enrichment. Methods: mutual_info (fast, nonlinear), "
                "lasso (sparse linear), random_forest (handles interactions).",
)
def panel_select(
    compound_id: str,
    n_features: int = 10,
    method: str = "mutual_info",
    **kwargs,
) -> dict:
    """ML-based biomarker panel selection using sklearn.

    Uses PRISM sensitivity as target (LFC < -0.5 = sensitive) and the DepMap
    mutation matrix as features. Supports three methods:
    - mutual_info: mutual information classification (fast, captures nonlinear)
    - lasso: LassoCV with L1 regularization (sparse, linear)
    - random_forest: random forest feature importances (handles interactions)

    Returns ranked list of biomarker genes with importance scores and
    cross-validation AUC.
    """
    from ct.data.loaders import load_prism, load_mutations, load_model_metadata
    from ct.tools._compound_resolver import resolve_compound

    compound_id = resolve_compound(compound_id, dataset="prism")

    valid_methods = ("mutual_info", "lasso", "random_forest")
    if method not in valid_methods:
        return {"error": f"Unknown method '{method}'. Choose from: {', '.join(valid_methods)}", "summary": f"Unknown method '{method}'. Choose from: {', '.join(valid_methods)}"}
    prism = load_prism()
    mutations = load_mutations()
    model = load_model_metadata()

    # --- Map PRISM cell lines to DepMap ModelIDs ---
    ccle_to_model = {}
    for _, row in model.iterrows():
        ccle = row.get("CCLEName", "")
        mid = row.get("ModelID", "")
        if pd.notna(ccle) and pd.notna(mid):
            ccle_to_model[ccle] = mid

    # --- Get compound sensitivity at highest dose ---
    cpd = prism[prism["pert_name"] == compound_id]
    if len(cpd) == 0:
        return {"error": f"Compound {compound_id} not found in PRISM data", "summary": f"Compound {compound_id} not found in PRISM data"}
    max_dose = cpd["pert_dose"].max()
    cpd_hd = cpd[cpd["pert_dose"] == max_dose].groupby("ccle_name")["LFC"].mean()

    # Map to ModelIDs and build target vector
    sensitivity = {}
    for ccle, lfc in cpd_hd.items():
        mid = ccle_to_model.get(ccle)
        if mid and mid in mutations.index:
            sensitivity[mid] = lfc

    if len(sensitivity) < 20:
        return {
            "error": f"Insufficient overlap: only {len(sensitivity)} cell lines mapped between PRISM and mutation data (need >=20)",
        }

    common_ids = list(sensitivity.keys())
    y_lfc = np.array([sensitivity[mid] for mid in common_ids])
    y_binary = (y_lfc < -0.5).astype(int)  # 1 = sensitive

    n_sensitive = int(y_binary.sum())
    n_resistant = int(len(y_binary) - n_sensitive)

    if n_sensitive < 3 or n_resistant < 3:
        return {
            "error": f"Insufficient class balance: {n_sensitive} sensitive, {n_resistant} resistant (need >=3 each)",
        }

    # --- Build feature matrix (mutation status) ---
    # Filter to genes with at least 3 mutated samples
    X_full = mutations.loc[common_ids].fillna(0)
    gene_counts = (X_full > 0).sum()
    usable_genes = gene_counts[(gene_counts >= 3) & (gene_counts <= len(common_ids) - 3)].index.tolist()

    if len(usable_genes) < 3:
        return {
            "error": f"Only {len(usable_genes)} genes with sufficient mutation frequency for feature selection",
        }

    X = X_full[usable_genes].values
    feature_names = usable_genes

    # --- Feature selection ---
    importances = np.zeros(len(feature_names))

    if method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        importances = mutual_info_classif(X, y_binary, random_state=42)

    elif method == "lasso":
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lasso = LassoCV(cv=min(5, n_sensitive, n_resistant), random_state=42, max_iter=5000)
        lasso.fit(X_scaled, y_lfc)  # Use continuous LFC for lasso
        importances = np.abs(lasso.coef_)

    elif method == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        )
        rf.fit(X, y_binary)
        importances = rf.feature_importances_

    # --- Rank features ---
    ranked_idx = np.argsort(importances)[::-1]
    top_idx = ranked_idx[:n_features]

    biomarkers = []
    for i in top_idx:
        gene_name = feature_names[i]
        imp = float(importances[i])
        if imp <= 0 and method != "lasso":
            continue  # Skip zero-importance features
        n_mut = int((X[:, i] > 0).sum())
        biomarkers.append({
            "gene": gene_name,
            "importance": round(imp, 6),
            "n_mutated": n_mut,
            "mutation_frequency": round(n_mut / len(common_ids), 4),
        })

    # --- Cross-validation AUC using top features ---
    cv_auc = None
    if biomarkers:
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier as RFC

        top_gene_idx = [feature_names.index(b["gene"]) for b in biomarkers if b["gene"] in feature_names]
        if len(top_gene_idx) >= 1:
            X_top = X[:, top_gene_idx]
            cv_folds = min(5, n_sensitive, n_resistant)
            if cv_folds >= 2:
                clf = RFC(n_estimators=50, max_depth=3, random_state=42, class_weight="balanced")
                try:
                    scores = cross_val_score(clf, X_top, y_binary, cv=cv_folds, scoring="roc_auc")
                    cv_auc = round(float(scores.mean()), 4)
                except ValueError:
                    cv_auc = None

    # --- Summary ---
    top_genes_str = ", ".join(
        f"{b['gene']}({b['importance']:.4f})" for b in biomarkers[:5]
    )
    auc_str = f", CV-AUC={cv_auc:.3f}" if cv_auc is not None else ""
    summary = (
        f"Biomarker panel for {compound_id} ({method}): "
        f"{len(biomarkers)} features selected from {len(usable_genes)} candidates. "
        f"Top: {top_genes_str}{auc_str}"
    )

    return {
        "summary": summary,
        "compound": compound_id,
        "method": method,
        "n_cell_lines": len(common_ids),
        "n_sensitive": n_sensitive,
        "n_resistant": n_resistant,
        "n_features_tested": len(usable_genes),
        "biomarkers": biomarkers,
        "cv_auc": cv_auc,
    }
