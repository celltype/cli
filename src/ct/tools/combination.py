"""
Combination therapy tools: synergy prediction, synthetic lethality, metabolic vulnerability.

References crews-glue-discovery/scripts/synergy_prediction.py and metabolic_vulnerability.py
for scoring logic.
"""

import pandas as pd
import numpy as np
from ct.tools import registry


# Metabolic pathway gene sets (from metabolic_vulnerability.py)
METABOLIC_PATHWAYS = {
    "glycolysis": ["HK1", "HK2", "GPI", "PFKM", "PFKL", "ALDOA", "TPI1",
                   "GAPDH", "PGK1", "PGAM1", "ENO1", "ENO2", "PKM", "LDHA", "LDHB"],
    "oxidative_phosphorylation": ["NDUFA1", "NDUFA2", "NDUFB1", "NDUFS1", "SDHA", "SDHB",
                                  "UQCRC1", "UQCRC2", "COX5A", "COX5B", "ATP5F1A", "ATP5F1B"],
    "fatty_acid_synthesis": ["FASN", "ACACA", "ACLY", "SCD", "ELOVL1", "ELOVL5", "ELOVL6"],
    "fatty_acid_oxidation": ["CPT1A", "CPT1B", "CPT2", "ACADM", "ACADL", "ACADVL",
                             "HADHA", "HADHB", "ECHS1"],
    "glutamine_metabolism": ["GLS", "GLS2", "GLUD1", "SLC1A5", "SLC7A5", "GOT1", "GOT2"],
    "one_carbon_metabolism": ["MTHFR", "MTHFD1", "MTHFD2", "SHMT1", "SHMT2", "DHFR",
                              "TYMS", "MTR", "MAT2A"],
    "nucleotide_synthesis": ["CAD", "DHODH", "UMPS", "CTPS1", "CTPS2", "IMPDH1", "IMPDH2",
                             "PAICS", "ATIC", "GART"],
    "pentose_phosphate": ["G6PD", "PGLS", "PGD", "TKT", "TALDO1", "RPIA", "RPE"],
    "tca_cycle": ["CS", "ACO1", "ACO2", "IDH1", "IDH2", "IDH3A", "OGDH", "SUCLA2",
                  "SDHA", "FH", "MDH1", "MDH2"],
}

# Known metabolic inhibitors for combination suggestions
METABOLIC_INHIBITORS = {
    "glycolysis": ["2-DG (2-deoxyglucose)", "3-bromopyruvate", "lonidamine"],
    "oxidative_phosphorylation": ["metformin", "IACS-010759", "oligomycin A"],
    "fatty_acid_synthesis": ["TVB-2640 (denifanstat)", "orlistat", "TOFA"],
    "fatty_acid_oxidation": ["etomoxir", "ranolazine", "perhexiline"],
    "glutamine_metabolism": ["CB-839 (telaglenastat)", "BPTES", "DON"],
    "one_carbon_metabolism": ["methotrexate", "pemetrexed", "AG-270 (MAT2A inhibitor)"],
    "nucleotide_synthesis": ["brequinar (DHODH)", "mycophenolate (IMPDH)", "leflunomide"],
    "pentose_phosphate": ["6-AN (G6PD)", "DHEA (G6PD)", "oxythiamine (TKT)"],
    "tca_cycle": ["ivosidenib (IDH1)", "enasidenib (IDH2)", "CPI-613 (devimistat)"],
}


@registry.register(
    name="combination.synergy_predict",
    description="Predict synergistic compound pairs from anti-correlated L1000 transcriptomic signatures",
    category="combination",
    parameters={
        "compound_id": "Query compound (or 'all' for full pairwise)",
        "top_n": "Number of top synergy candidates to return",
    },
    requires_data=["l1000", "prism", "depmap_model"],
    usage_guide="You want to find compounds that work well together — anti-correlated transcriptomic profiles suggest complementary mechanisms. Use for rational combination therapy design.",
)
def synergy_predict(compound_id: str = "all", top_n: int = 20, **kwargs) -> dict:
    """Find synergistic compound pairs based on anti-correlated L1000 signatures.

    Synergy score = |anti-correlation| x tissue_complementarity x potency_bonus
    """
    from ct.data.loaders import load_l1000, load_prism, load_model_metadata
    from sklearn.metrics.pairwise import cosine_similarity
    from ct.tools._compound_resolver import resolve_compound

    if compound_id != "all":
        compound_id = resolve_compound(compound_id, dataset="l1000")

    l1000 = load_l1000()

    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(l1000.values)
    compounds = l1000.index.tolist()

    # Build PRISM tissue profiles for complementarity
    prism = load_prism()
    model = load_model_metadata()
    ccle_to_lineage = {}
    for _, row in model.iterrows():
        ccle = row.get("CCLEName", "")
        lin = row.get("OncotreeLineage", "Unknown")
        if pd.notna(ccle) and pd.notna(lin):
            ccle_to_lineage[ccle] = lin

    # Compute per-compound tissue profiles (mean LFC per lineage at max dose)
    tissue_profiles = {}
    for cpd in prism["pert_name"].unique():
        cpd_data = prism[prism["pert_name"] == cpd]
        max_dose = cpd_data["pert_dose"].max()
        cpd_hd = cpd_data[cpd_data["pert_dose"] == max_dose].copy()
        cpd_hd["lineage"] = cpd_hd["ccle_name"].map(ccle_to_lineage)
        tissue_mean = cpd_hd.groupby("lineage")["LFC"].mean()
        tissue_profiles[cpd] = tissue_mean

    # Find anti-correlated pairs
    if compound_id != "all" and compound_id in compounds:
        query_idx = compounds.index(compound_id)
        query_compounds = [compound_id]
    else:
        query_compounds = compounds
        query_idx = None

    ANTICORR_THRESHOLD = -0.3
    results = []

    for i, cpd1 in enumerate(query_compounds):
        idx1 = compounds.index(cpd1) if query_idx is None else query_idx
        for j in range(len(compounds)):
            if j <= idx1 and query_idx is None:
                continue
            cpd2 = compounds[j]
            if cpd1 == cpd2:
                continue

            cosine = sim_matrix[idx1, j]
            if cosine >= ANTICORR_THRESHOLD:
                continue

            # Tissue complementarity
            tissue_comp = 0.5  # default
            if cpd1 in tissue_profiles and cpd2 in tissue_profiles:
                t1 = tissue_profiles[cpd1]
                t2 = tissue_profiles[cpd2]
                common = t1.index.intersection(t2.index)
                if len(common) >= 3:
                    kills_1 = t1[common] < -0.3
                    kills_2 = t2[common] < -0.3
                    comp_tissues = (kills_1 & ~kills_2).sum() + (kills_2 & ~kills_1).sum()
                    overlap_tissues = (kills_1 & kills_2).sum()
                    tissue_comp = comp_tissues / (comp_tissues + overlap_tissues + 0.001)

            # Synergy score
            anticorr_strength = abs(cosine)
            score = anticorr_strength * (0.4 + 0.6 * tissue_comp)

            # Potency bonus
            if cpd1 in tissue_profiles and cpd2 in tissue_profiles:
                pot1 = abs(tissue_profiles[cpd1].mean()) if len(tissue_profiles[cpd1]) > 0 else 0
                pot2 = abs(tissue_profiles[cpd2].mean()) if len(tissue_profiles[cpd2]) > 0 else 0
                avg_potency = (pot1 + pot2) / 2.0
                score *= (1.0 + min(avg_potency, 2.0) / 4.0)

            results.append({
                "compound_1": cpd1,
                "compound_2": cpd2,
                "cosine_similarity": round(float(cosine), 4),
                "anticorrelation_strength": round(float(anticorr_strength), 4),
                "tissue_complementarity": round(float(tissue_comp), 4),
                "synergy_score": round(float(score), 4),
            })

    if not results:
        return {
            "summary": f"Synergy prediction: 0 anti-correlated pairs (cosine < {ANTICORR_THRESHOLD}). No synergistic candidates found.",
            "n_pairs": 0,
            "top_candidates": [],
        }

    df = pd.DataFrame(results).sort_values("synergy_score", ascending=False)
    top_hits = df.head(top_n)

    return {
        "summary": (
            f"Synergy prediction: {len(df)} anti-correlated pairs (cosine < {ANTICORR_THRESHOLD})\n"
            f"Top synergy score: {top_hits.iloc[0]['synergy_score']:.4f}" if len(top_hits) > 0 else "No pairs found"
        ),
        "n_pairs": len(df),
        "top_candidates": top_hits.to_dict("records"),
    }


@registry.register(
    name="combination.synthetic_lethality",
    description="Mine DepMap CRISPR data for synthetic lethal gene pairs with a target",
    category="combination",
    parameters={
        "gene": "Target gene to find synthetic lethal partners for",
        "top_n": "Number of top partners to return",
    },
    requires_data=["depmap_crispr"],
    usage_guide="You want to find genes whose loss is lethal only when your target gene is also disrupted. Use for identifying combination targets and understanding genetic dependencies.",
)
def synthetic_lethality(gene: str, top_n: int = 20, **kwargs) -> dict:
    """Find synthetic lethal partners via anti-correlated CRISPR dependencies.

    Genes with strong negative correlation in DepMap CRISPR effect = when one is
    essential, the other is dispensable -> synthetic lethality.
    """
    from ct.data.loaders import load_crispr
    from scipy import stats

    crispr = load_crispr()

    if gene not in crispr.columns:
        return {"error": f"Gene {gene} not found in DepMap CRISPR data", "summary": f"Gene {gene} not found in DepMap CRISPR data"}
    target_vals = crispr[gene].dropna()

    # Compute anti-correlations (negative r = synthetic lethal)
    results = []
    for other_gene in crispr.columns:
        if other_gene == gene:
            continue
        other_vals = crispr[other_gene].dropna()
        common = target_vals.index.intersection(other_vals.index)
        if len(common) < 50:
            continue

        r, p = stats.pearsonr(target_vals[common], other_vals[common])
        if r < -0.1:  # only anti-correlated
            results.append({
                "gene": other_gene,
                "correlation": round(float(r), 4),
                "p_value": float(p),
                "n_cell_lines": len(common),
            })

    if not results:
        return {
            "summary": f"Synthetic lethality screen for {gene}: no anti-correlated genes found",
            "target_gene": gene,
            "n_candidates": 0,
            "top_partners": [],
        }

    df = pd.DataFrame(results).sort_values("correlation")

    # Classify synthetic lethal strength
    for i, row in df.iterrows():
        if row["correlation"] < -0.3:
            df.at[i, "strength"] = "strong"
        elif row["correlation"] < -0.2:
            df.at[i, "strength"] = "moderate"
        else:
            df.at[i, "strength"] = "weak"

    top_sl = df.head(top_n)

    return {
        "summary": (
            f"Synthetic lethality screen for {gene}: {len(df)} anti-correlated genes\n"
            f"Strong (r < -0.3): {(df['correlation'] < -0.3).sum()}, "
            f"Moderate (r < -0.2): {((df['correlation'] >= -0.3) & (df['correlation'] < -0.2)).sum()}"
        ),
        "target_gene": gene,
        "n_candidates": len(df),
        "top_partners": top_sl.to_dict("records"),
    }


@registry.register(
    name="combination.metabolic_vulnerability",
    description="Map metabolic vulnerabilities: compounds that suppress metabolic pathways where dependent cells are more sensitive",
    category="combination",
    parameters={
        "compound_id": "Compound to profile (or 'all')",
        "pathway": "Specific metabolic pathway (or 'all')",
    },
    requires_data=["l1000", "depmap_crispr", "prism", "depmap_model"],
    usage_guide="You want to exploit metabolic dependencies — find pathways a compound suppresses where dependent cells are more sensitive. Use to identify metabolic inhibitor combinations (e.g., add metformin to exploit OxPhos dependency).",
)
def metabolic_vulnerability(compound_id: str = "all", pathway: str = "all", **kwargs) -> dict:
    """Identify exploitable metabolic vulnerabilities (the vulnerability triangle).

    Triangle: compound suppresses pathway (L1000) + pathway-dependent cells more
    sensitive (PRISM) = exploitable vulnerability.
    """
    from ct.data.loaders import load_l1000, load_crispr, load_prism, load_model_metadata
    from ct.tools._compound_resolver import resolve_compound
    from scipy import stats

    if compound_id != "all":
        compound_id = resolve_compound(compound_id, dataset="l1000")

    l1000 = load_l1000()
    crispr = load_crispr()
    prism = load_prism()
    model = load_model_metadata()

    # Step 1: Score L1000 metabolic pathways
    l1000_genes = set(l1000.columns)
    pathway_scores = {}
    pathways_to_test = {pathway: METABOLIC_PATHWAYS[pathway]} if pathway != "all" else METABOLIC_PATHWAYS

    for pw_name, genes in pathways_to_test.items():
        found = [g for g in genes if g in l1000_genes]
        if len(found) < 2:
            continue
        sub = l1000[found]
        zscored = (sub - sub.mean()) / sub.std()
        pathway_scores[pw_name] = zscored.mean(axis=1)

    if not pathway_scores:
        return {"error": "No metabolic pathways have sufficient gene coverage in L1000", "summary": "No metabolic pathways have sufficient gene coverage in L1000"}
    pw_score_df = pd.DataFrame(pathway_scores)

    # Step 2: DepMap metabolic dependency
    crispr_genes = set(crispr.columns)
    dep_binary = {}
    for pw_name, genes in pathways_to_test.items():
        found = [g for g in genes if g in crispr_genes]
        if not found:
            continue
        dep_binary[pw_name] = (crispr[found].min(axis=1) < -0.5)

    # Step 3: Map PRISM to DepMap and test vulnerability triangle
    ccle_to_model_id = {}
    for _, row in model.iterrows():
        ccle = row.get("CCLEName", "")
        mid = row.get("ModelID", "")
        if pd.notna(ccle) and pd.notna(mid):
            ccle_to_model_id[ccle] = mid

    prism_10 = prism[prism["pert_dose"] == prism["pert_dose"].max()]
    prism_wide = prism_10.pivot_table(index="ccle_name", columns="pert_name", values="LFC", aggfunc="mean")
    prism_wide["ModelID"] = prism_wide.index.map(ccle_to_model_id)
    prism_mapped = prism_wide.dropna(subset=["ModelID"])

    overlap = set(crispr.index) & set(prism_mapped["ModelID"])
    if len(overlap) < 20:
        return {"error": f"Insufficient overlap: only {len(overlap)} cell lines in both PRISM and DepMap", "summary": f"Insufficient overlap: only {len(overlap)} cell lines in both PRISM and DepMap"}
    overlap_list = sorted(overlap)
    compounds_to_test = [compound_id] if compound_id != "all" else [
        c for c in pw_score_df.index if c in prism_wide.columns
    ]

    vulnerabilities = []
    for pw_name in dep_binary:
        if pw_name not in pw_score_df.columns:
            continue
        dep_mask = dep_binary[pw_name].reindex(overlap_list).fillna(False)
        n_dep = dep_mask.sum()
        n_indep = (~dep_mask).sum()
        if n_dep < 5 or n_indep < 5:
            continue

        for cpd in compounds_to_test:
            if cpd not in pw_score_df.index or cpd not in prism_wide.columns:
                continue

            l1000_z = pw_score_df.loc[cpd, pw_name]
            if abs(l1000_z) < 1.0:
                continue

            # Get PRISM LFC for this compound in overlapping cell lines
            prism_by_model = prism_mapped.set_index("ModelID")
            if cpd not in prism_by_model.columns:
                continue
            lfc_vals = prism_by_model.loc[overlap_list, cpd].values
            valid = ~np.isnan(lfc_vals)

            dep_lfc = lfc_vals[valid & dep_mask.values[:len(valid)]]
            indep_lfc = lfc_vals[valid & ~dep_mask.values[:len(valid)]]

            if len(dep_lfc) < 5 or len(indep_lfc) < 5:
                continue

            t_stat, p_val = stats.ttest_ind(dep_lfc, indep_lfc, equal_var=False)
            delta_lfc = float(np.mean(dep_lfc) - np.mean(indep_lfc))

            # Classify
            if l1000_z < -1.0 and delta_lfc < 0:
                vuln_type = "EXPLOIT"
            elif l1000_z > 1.0 and delta_lfc < 0:
                vuln_type = "ACTIVATION_SENSITIZES"
            elif l1000_z < -1.0 and delta_lfc > 0:
                vuln_type = "PARADOXICAL_RESISTANCE"
            else:
                vuln_type = "WEAK_SIGNAL"

            # Combination suggestion
            combo_drugs = METABOLIC_INHIBITORS.get(pw_name, [])

            vulnerabilities.append({
                "compound": cpd,
                "pathway": pw_name,
                "l1000_zscore": round(float(l1000_z), 3),
                "delta_lfc": round(delta_lfc, 3),
                "p_value": round(float(p_val), 4),
                "n_dependent": int(dep_mask.sum()),
                "n_independent": int((~dep_mask).sum()),
                "vulnerability_type": vuln_type,
                "suggested_combinations": combo_drugs[:3] if vuln_type == "EXPLOIT" else [],
            })

    df = pd.DataFrame(vulnerabilities)
    exploits = df[df["vulnerability_type"] == "EXPLOIT"] if len(df) > 0 else df

    return {
        "summary": (
            f"Metabolic vulnerability analysis: {len(df)} compound-pathway pairs tested\n"
            f"Exploitable vulnerabilities: {len(exploits)}\n"
            f"Pathways screened: {', '.join(pathways_to_test.keys())}"
        ),
        "n_total": len(df),
        "n_exploitable": len(exploits),
        "vulnerabilities": df.to_dict("records") if len(df) < 200 else exploits.to_dict("records"),
    }
