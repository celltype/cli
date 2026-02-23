"""Statistical analysis tools: survival, dose-response, biomarker panels."""

import numpy as np
from ct.tools import registry


@registry.register(
    name="statistics.dose_response_fit",
    description="Fit a 4-parameter logistic (Hill equation) dose-response curve and compute IC50",
    category="statistics",
    parameters={
        "doses": "List of dose concentrations (floats)",
        "responses": "List of response values (floats, e.g. viability or inhibition %)",
        "compound_name": "Optional compound name for labeling",
    },
    requires_data=[],
    usage_guide="You have dose-response data and want to fit a curve to compute IC50, Hill slope, "
                "and assess curve quality. Provide matched lists of doses and responses. "
                "Works with any dose-response data (viability, inhibition, binding, etc.).",
)
def dose_response_fit(doses: list = None, responses: list = None,
                      compound_name: str = "unknown", **kwargs) -> dict:
    """Fit a 4-parameter logistic (Hill equation) dose-response curve.

    Hill equation: f(x) = bottom + (top - bottom) / (1 + (IC50/x)^slope)

    Parameters
    ----------
    doses : list of float
        Concentration values (must be positive).
    responses : list of float
        Response values (e.g. % viability, % inhibition).
    compound_name : str
        Label for the compound.

    Returns
    -------
    dict with fitted parameters, IC50, R-squared, quality assessment.
    """
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    if doses is None or responses is None:
        return {"error": "Both 'doses' and 'responses' lists are required", "summary": "Both 'doses' and 'responses' lists are required"}
    doses = [float(d) for d in doses]
    responses = [float(r) for r in responses]

    if len(doses) != len(responses):
        return {"error": f"Length mismatch: {len(doses)} doses vs {len(responses)} responses", "summary": f"Length mismatch: {len(doses)} doses vs {len(responses)} responses"}
    if len(doses) < 4:
        return {"error": f"Need at least 4 data points for 4PL fit, got {len(doses)}", "summary": f"Need at least 4 data points for 4PL fit, got {len(doses)}"}
    # Filter out non-positive doses (log-space fitting)
    valid = [(d, r) for d, r in zip(doses, responses) if d > 0]
    if len(valid) < 4:
        return {"error": "Need at least 4 positive dose values", "summary": "Need at least 4 positive dose values"}
    doses_arr = np.array([v[0] for v in valid])
    resp_arr = np.array([v[1] for v in valid])

    # 4-parameter logistic (Hill equation)
    def hill(x, bottom, top, ic50, slope):
        return bottom + (top - bottom) / (1.0 + (ic50 / x) ** slope)

    # Initial parameter guesses
    bottom_guess = float(np.min(resp_arr))
    top_guess = float(np.max(resp_arr))
    ic50_guess = float(np.median(doses_arr))
    slope_guess = 1.0

    try:
        popt, pcov = curve_fit(
            hill, doses_arr, resp_arr,
            p0=[bottom_guess, top_guess, ic50_guess, slope_guess],
            bounds=(
                [-np.inf, -np.inf, 1e-15, 0.01],   # lower bounds (IC50 > 0, slope > 0)
                [np.inf, np.inf, np.inf, 100.0]      # upper bounds
            ),
            maxfev=10000,
        )
        bottom, top, ic50, slope = popt
        perr = np.sqrt(np.diag(pcov))

        # Compute R-squared
        predicted = hill(doses_arr, *popt)
        ss_res = np.sum((resp_arr - predicted) ** 2)
        ss_tot = np.sum((resp_arr - np.mean(resp_arr)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Quality assessment
        if r_squared > 0.95 and perr[2] / abs(ic50) < 0.5:
            quality = "HIGH"
            quality_detail = "Excellent fit with tight IC50 confidence"
        elif r_squared > 0.8:
            quality = "MEDIUM"
            quality_detail = "Good fit, IC50 estimate reliable"
        elif r_squared > 0.5:
            quality = "LOW"
            quality_detail = "Marginal fit, IC50 estimate approximate"
        else:
            quality = "POOR"
            quality_detail = "Poor fit, IC50 unreliable"

        # Max effect (dynamic range)
        max_effect = abs(top - bottom)

        summary = (
            f"Dose-response fit for {compound_name}:\n"
            f"IC50 = {ic50:.4g}, Hill slope = {slope:.2f}\n"
            f"R² = {r_squared:.4f}, Quality: {quality}\n"
            f"Bottom = {bottom:.2f}, Top = {top:.2f}, Max effect = {max_effect:.2f}"
        )

        return {
            "summary": summary,
            "compound": compound_name,
            "ic50": round(float(ic50), 6),
            "hill_slope": round(float(slope), 4),
            "bottom": round(float(bottom), 4),
            "top": round(float(top), 4),
            "r_squared": round(float(r_squared), 4),
            "max_effect": round(float(max_effect), 4),
            "parameter_errors": {
                "bottom_se": round(float(perr[0]), 4),
                "top_se": round(float(perr[1]), 4),
                "ic50_se": round(float(perr[2]), 6),
                "slope_se": round(float(perr[3]), 4),
            },
            "quality": quality,
            "quality_detail": quality_detail,
            "n_points": len(doses_arr),
        }

    except RuntimeError as e:
        return {
            "summary": f"Dose-response fit FAILED for {compound_name}: curve fitting did not converge",
            "error": f"Convergence failure: {str(e)}",
            "compound": compound_name,
            "n_points": len(doses_arr),
            "dose_range": [float(np.min(doses_arr)), float(np.max(doses_arr))],
            "response_range": [float(np.min(resp_arr)), float(np.max(resp_arr))],
        }
    except Exception as e:
        return {
            "summary": f"Dose-response fit FAILED for {compound_name}: {str(e)}",
            "error": str(e),
            "compound": compound_name,
        }


@registry.register(
    name="statistics.survival_analysis",
    description="Perform Kaplan-Meier survival analysis with optional log-rank test for group comparison",
    category="statistics",
    parameters={
        "times": "List of survival/follow-up times",
        "events": "List of event indicators (1=event occurred, 0=censored)",
        "groups": "Optional list of group labels for comparing survival between groups",
    },
    requires_data=[],
    usage_guide="You have time-to-event data and want to estimate survival curves, median survival, "
                "and compare groups. Provide times, event indicators, and optionally group labels. "
                "Use for clinical trial analysis, patient stratification, or biomarker validation.",
)
def survival_analysis(times: list = None, events: list = None,
                      groups: list = None, **kwargs) -> dict:
    """Perform Kaplan-Meier survival analysis with log-rank test.

    Implements the Kaplan-Meier estimator from scratch:
    S(t) = product of (1 - d_i/n_i) for all event times t_i <= t

    If groups are provided, computes separate KM curves and a log-rank test.

    Parameters
    ----------
    times : list of float
        Survival or follow-up times.
    events : list of int
        Event indicators: 1 = event (death/progression), 0 = censored.
    groups : list, optional
        Group labels for stratified analysis (e.g., ["high", "low", "high", ...]).
    """
    from scipy import stats as sp_stats

    if times is None or events is None:
        return {"error": "Both 'times' and 'events' lists are required", "summary": "Both 'times' and 'events' lists are required"}
    times = [float(t) for t in times]
    events = [int(e) for e in events]

    if len(times) != len(events):
        return {"error": f"Length mismatch: {len(times)} times vs {len(events)} events", "summary": f"Length mismatch: {len(times)} times vs {len(events)} events"}
    if len(times) < 3:
        return {"error": f"Need at least 3 observations, got {len(times)}", "summary": f"Need at least 3 observations, got {len(times)}"}
    def _kaplan_meier(t_arr, e_arr):
        """Compute KM survival curve from times and events arrays."""
        # Sort by time
        order = np.argsort(t_arr)
        t_sorted = t_arr[order]
        e_sorted = e_arr[order]

        # Get unique event times (only where event=1)
        event_times = np.unique(t_sorted[e_sorted == 1])

        km_times = [0.0]
        km_survival = [1.0]
        n_at_risk = len(t_sorted)
        current_s = 1.0

        for et in event_times:
            # Number who have been censored or had event before this time
            # n_at_risk at time et: those with time >= et
            n_at_risk = int(np.sum(t_sorted >= et))
            # Deaths at this time
            d = int(np.sum((t_sorted == et) & (e_sorted == 1)))

            if n_at_risk > 0:
                current_s *= (1.0 - d / n_at_risk)

            km_times.append(float(et))
            km_survival.append(round(current_s, 6))

        # Median survival: first time S(t) <= 0.5
        median_surv = None
        for t_val, s_val in zip(km_times, km_survival):
            if s_val <= 0.5:
                median_surv = t_val
                break

        return {
            "times": km_times,
            "survival": km_survival,
            "median_survival": median_surv,
            "n_events": int(np.sum(e_sorted)),
            "n_censored": int(np.sum(e_sorted == 0)),
            "n_total": len(t_sorted),
        }

    t_arr = np.array(times)
    e_arr = np.array(events)

    # Single-group analysis
    if groups is None:
        km = _kaplan_meier(t_arr, e_arr)
        median_str = f"{km['median_survival']:.1f}" if km['median_survival'] is not None else "not reached"
        summary = (
            f"Kaplan-Meier survival analysis (n={km['n_total']}):\n"
            f"Events: {km['n_events']}, Censored: {km['n_censored']}\n"
            f"Median survival: {median_str}"
        )
        return {
            "summary": summary,
            "kaplan_meier": km,
        }

    # Multi-group analysis
    groups = list(groups)
    if len(groups) != len(times):
        return {"error": f"Length mismatch: {len(times)} times vs {len(groups)} groups", "summary": f"Length mismatch: {len(times)} times vs {len(groups)} groups"}
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return {"error": f"Need at least 2 groups for comparison, got {len(unique_groups)}", "summary": f"Need at least 2 groups for comparison, got {len(unique_groups)}"}
    group_arr = np.array(groups)
    group_results = {}

    for g in unique_groups:
        mask = group_arr == g
        group_results[str(g)] = _kaplan_meier(t_arr[mask], e_arr[mask])

    # Log-rank test (for 2 groups, generalizable)
    # Implementation: compare observed vs expected events in each group
    # at each unique event time across all groups
    all_event_times = np.unique(t_arr[e_arr == 1])

    # Compute chi-squared statistic for log-rank
    observed_minus_expected = {str(g): 0.0 for g in unique_groups}
    variance_sum = 0.0

    for et in all_event_times:
        # Total at risk and total events at this time
        at_risk_total = int(np.sum(t_arr >= et))
        events_total = int(np.sum((t_arr == et) & (e_arr == 1)))

        if at_risk_total == 0:
            continue

        for g in unique_groups:
            mask = group_arr == g
            at_risk_g = int(np.sum(t_arr[mask] >= et))
            events_g = int(np.sum((t_arr[mask] == et) & (e_arr[mask] == 1)))

            # Expected events under null
            expected_g = at_risk_g * events_total / at_risk_total if at_risk_total > 0 else 0
            observed_minus_expected[str(g)] += (events_g - expected_g)

        # Variance contribution (hypergeometric variance)
        if at_risk_total > 1:
            for g in unique_groups:
                mask = group_arr == g
                n_g = int(np.sum(t_arr[mask] >= et))
                frac = n_g / at_risk_total
                censored_total = at_risk_total - events_total
                var_contrib = (events_total * censored_total * frac * (1 - frac)) / (at_risk_total - 1)
                # Only accumulate for the first group (2-group test)
                if g == unique_groups[0]:
                    variance_sum += var_contrib

    # Chi-squared statistic (1 df for 2 groups)
    if variance_sum > 0:
        chi2 = (observed_minus_expected[str(unique_groups[0])] ** 2) / variance_sum
        p_value = float(1.0 - sp_stats.chi2.cdf(chi2, df=len(unique_groups) - 1))
    else:
        chi2 = 0.0
        p_value = 1.0

    # Event rate ratio (simplified — not a proper Cox hazard ratio).
    # Computed as (events_1 / total_time_1) / (events_2 / total_time_2).
    hr = None
    hr_str = "N/A"
    if len(unique_groups) == 2:
        g1, g2 = str(unique_groups[0]), str(unique_groups[1])
        r1 = group_results[g1]
        r2 = group_results[g2]
        rate1 = r1["n_events"] / max(np.sum(t_arr[group_arr == unique_groups[0]]), 1e-10)
        rate2 = r2["n_events"] / max(np.sum(t_arr[group_arr == unique_groups[1]]), 1e-10)
        if rate2 > 0:
            hr = round(float(rate1 / rate2), 4)
            hr_str = f"{hr:.3f}"

    # Build summary
    median_parts = []
    for g in unique_groups:
        med = group_results[str(g)]["median_survival"]
        med_str = f"{med:.1f}" if med is not None else "NR"
        n = group_results[str(g)]["n_total"]
        median_parts.append(f"{g}(n={n}): {med_str}")

    significance = "significant" if p_value < 0.05 else "not significant"

    summary = (
        f"Kaplan-Meier survival analysis ({len(unique_groups)} groups, n={len(times)}):\n"
        f"Median survival: {', '.join(median_parts)}\n"
        f"Log-rank p = {p_value:.4g} ({significance})\n"
        f"Event rate ratio (simplified HR): {hr_str}"
    )

    return {
        "summary": summary,
        "groups": group_results,
        "log_rank": {
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < 0.05,
        },
        "hazard_ratio": hr,  # Note: simplified event rate ratio, not Cox regression HR
    }


@registry.register(
    name="statistics.enrichment_test",
    description="Gene set over-representation analysis using hypergeometric test with FDR correction",
    category="statistics",
    parameters={
        "gene_list": "List of gene symbols (your query genes)",
        "gene_set": "Dict of set_name:gene_list, or 'hallmark' for built-in MSigDB hallmark sets",
        "background_size": "Total background gene count (default 20000)",
    },
    requires_data=[],
    usage_guide="You have a list of genes (e.g. differentially expressed, mutated, degraded) and want "
                "to know which pathways or gene sets they are enriched in. Provide your gene list and "
                "optionally a custom gene set dict, or use built-in hallmark sets. Returns FDR-corrected p-values.",
)
def enrichment_test(gene_list: list = None, gene_set: dict | str = "hallmark",
                    background_size: int = 20000, **kwargs) -> dict:
    """Gene set over-representation analysis (ORA) with hypergeometric test.

    For each gene set, computes:
    - Hypergeometric p-value (Fisher's exact one-tailed)
    - Fold enrichment = (overlap/query) / (set_size/background)
    - Benjamini-Hochberg FDR correction

    Parameters
    ----------
    gene_list : list of str
        Query genes (e.g. upregulated genes, hit list).
    gene_set : dict or str
        Dict mapping set names to gene lists, or "hallmark" for built-in.
    background_size : int
        Total number of genes in the background (default 20000).
    """
    from scipy.stats import hypergeom

    if gene_list is None or len(gene_list) == 0:
        return {"error": "Provide a non-empty gene_list", "summary": "Provide a non-empty gene_list"}
    gene_list = [str(g).upper() for g in gene_list]
    query_set = set(gene_list)
    n_query = len(query_set)

    # Load or use provided gene sets
    if isinstance(gene_set, str):
        if gene_set == "hallmark":
            gene_set = _get_hallmark_sets()
        else:
            return {"error": f"Unknown gene set collection: {gene_set}. Provide a dict or 'hallmark'", "summary": f"Unknown gene set collection: {gene_set}. Provide a dict or 'hallmark'"}
    if not isinstance(gene_set, dict) or len(gene_set) == 0:
        return {"error": "gene_set must be a non-empty dict of set_name: gene_list", "summary": "gene_set must be a non-empty dict of set_name: gene_list"}
    N = int(background_size)  # total background genes
    results = []

    for set_name, set_genes in gene_set.items():
        set_genes_upper = [str(g).upper() for g in set_genes]
        set_size = len(set_genes_upper)
        gene_set_set = set(set_genes_upper)

        # Overlap
        overlap = query_set & gene_set_set
        k = len(overlap)

        if k == 0:
            continue

        # Hypergeometric test
        # P(X >= k) where X ~ Hypergeometric(N, K, n)
        # N = background, K = set_size, n = query_size
        p_value = float(hypergeom.sf(k - 1, N, set_size, n_query))

        # Fold enrichment
        expected = (set_size / N) * n_query if N > 0 else 0
        fold_enrichment = k / expected if expected > 0 else float('inf')

        results.append({
            "gene_set": set_name,
            "overlap_count": k,
            "overlap_genes": sorted(overlap),
            "set_size": set_size,
            "p_value": p_value,
            "fold_enrichment": round(float(fold_enrichment), 2),
        })

    if not results:
        return {
            "summary": f"No enrichment found: {n_query} query genes had no overlap with {len(gene_set)} gene sets",
            "n_query_genes": n_query,
            "n_gene_sets_tested": len(gene_set),
            "enriched": [],
        }

    # Sort by p-value
    results.sort(key=lambda x: x["p_value"])

    # Benjamini-Hochberg FDR correction
    n_tests = len(results)
    for i, r in enumerate(results):
        rank = i + 1
        r["fdr"] = round(min(r["p_value"] * n_tests / rank, 1.0), 6)

    # Enforce monotonicity (FDR should be non-decreasing from bottom)
    for i in range(n_tests - 2, -1, -1):
        results[i]["fdr"] = min(results[i]["fdr"], results[i + 1]["fdr"])

    # Round p-values for output
    for r in results:
        r["p_value"] = round(r["p_value"], 8)

    significant = [r for r in results if r["fdr"] < 0.05]

    # Top 10 for summary
    top = results[:10]
    top_str = "\n".join(
        f"  {r['gene_set']}: {r['overlap_count']}/{r['set_size']} genes, "
        f"FE={r['fold_enrichment']:.1f}x, FDR={r['fdr']:.2g}"
        for r in top
    )

    summary = (
        f"Gene set enrichment ({n_query} query genes, {len(gene_set)} sets tested):\n"
        f"Significant (FDR<0.05): {len(significant)}/{n_tests}\n"
        f"Top enriched:\n{top_str}"
    )

    return {
        "summary": summary,
        "n_query_genes": n_query,
        "n_gene_sets_tested": len(gene_set),
        "n_significant": len(significant),
        "enriched": results,
    }


def _get_hallmark_sets() -> dict:
    """Get built-in hallmark-lite gene sets for enrichment analysis."""
    # Try to load full MSigDB hallmark sets first
    try:
        from ct.data.loaders import load_msigdb
        msigdb = load_msigdb("h")
        # MSigDB JSON: {set_name: {geneSymbols: [...]}}
        if isinstance(msigdb, dict):
            parsed = {}
            for name, data in msigdb.items():
                if isinstance(data, dict) and "geneSymbols" in data:
                    parsed[name] = data["geneSymbols"]
                elif isinstance(data, list):
                    parsed[name] = data
            if parsed:
                return parsed
    except (FileNotFoundError, ImportError):
        pass

    # Fallback: curated hallmark-lite sets
    return {
        "HALLMARK_P53_PATHWAY": ["CDKN1A", "MDM2", "BAX", "GADD45A", "SFN", "DDB2",
                                  "SESN1", "TP53I3", "PMAIP1", "BBC3"],
        "HALLMARK_APOPTOSIS": ["BCL2", "BAX", "BAK1", "BID", "CASP3", "CASP8",
                                "CASP9", "CYCS", "APAF1", "FADD"],
        "HALLMARK_MTORC1_SIGNALING": ["SLC7A5", "SLC3A2", "DDIT4", "VEGFA", "HK2",
                                       "PKM", "LDHA", "SLC2A1", "RPS6", "EIF4E"],
        "HALLMARK_MYC_TARGETS_V1": ["ODC1", "LDHA", "CDK4", "NCL", "NPM1", "NOP56",
                                     "BOP1", "MRTO4", "RRP12", "WDR12"],
        "HALLMARK_E2F_TARGETS": ["CCNE1", "MCM2", "PCNA", "RRM2", "MCM3", "MCM4",
                                  "MCM5", "MCM6", "CDC6", "ORC1"],
        "HALLMARK_G2M_CHECKPOINT": ["CDK1", "CCNB1", "CCNB2", "BUB1", "BUB1B",
                                     "AURKA", "AURKB", "PLK1", "TOP2A", "BIRC5"],
        "HALLMARK_DNA_REPAIR": ["BRCA1", "BRCA2", "RAD51", "ATM", "ATR", "CHEK1",
                                 "CHEK2", "MSH2", "MSH6", "MLH1"],
        "HALLMARK_INFLAMMATORY_RESPONSE": ["TNF", "IL6", "IL1B", "CXCL8", "CCL2",
                                            "ICAM1", "VCAM1", "SELE", "PTGS2", "MMP9"],
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB": ["NFKBIA", "TNF", "IL6", "CXCL8", "CCL2",
                                               "TNFAIP3", "BIRC3", "TRAF1", "RELB", "BCL3"],
        "HALLMARK_INTERFERON_GAMMA_RESPONSE": ["STAT1", "IRF1", "GBP1", "GBP2", "CXCL10",
                                                "CXCL9", "IDO1", "TAP1", "PSMB9", "B2M"],
        "HALLMARK_INTERFERON_ALPHA_RESPONSE": ["ISG15", "MX1", "MX2", "IFIT1", "IFIT2",
                                                "IFIT3", "OAS1", "OAS2", "RSAD2", "IFI44L"],
        "HALLMARK_HYPOXIA": ["VEGFA", "SLC2A1", "LDHA", "PDK1", "BNIP3", "DDIT4",
                              "ENO1", "PGK1", "ALDOA", "HK2"],
        "HALLMARK_GLYCOLYSIS": ["HK2", "PFKM", "ALDOA", "GAPDH", "PKM", "LDHA",
                                 "ENO1", "TPI1", "PGK1", "GPI"],
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION": ["NDUFA1", "SDHA", "UQCRC1", "COX5A",
                                                "ATP5F1A", "NDUFS1", "SDHB", "COX7A2",
                                                "UQCRB", "ATP5F1B"],
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": ["HSPA5", "DDIT3", "ATF4", "XBP1",
                                                "HERPUD1", "DNAJB9", "PDIA4", "ERN1",
                                                "ATF6", "EDEM1"],
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING": ["CTNNB1", "LEF1", "TCF7", "MYC",
                                                  "CCND1", "AXIN2", "DKK1", "WNT3A",
                                                  "FZD1", "LRP6"],
        "HALLMARK_NOTCH_SIGNALING": ["NOTCH1", "HES1", "HEY1", "JAG1", "DLL1",
                                      "RBPJ", "MAML1", "NRARP", "DTX1", "HEYL"],
        "HALLMARK_HEDGEHOG_SIGNALING": ["SHH", "SMO", "PTCH1", "GLI1", "GLI2",
                                         "GLI3", "SUFU", "HHIP", "GAS1", "BOC"],
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": ["VIM", "CDH2", "FN1", "SNAI1",
                                                        "SNAI2", "TWIST1", "ZEB1", "ZEB2",
                                                        "MMP2", "MMP9"],
        "HALLMARK_ANGIOGENESIS": ["VEGFA", "KDR", "FLT1", "PECAM1", "ANGPT1",
                                   "ANGPT2", "TEK", "NRP1", "ENG", "HIF1A"],
    }
