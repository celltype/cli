"""
Clinical translation tools: indication mapping, patient population sizing, TCGA stratification.

References crews-glue-discovery/scripts/patient_population_sizing.py and tcga_stratification.py
for data sources and scoring logic.
"""

import pandas as pd
import numpy as np
import re
from ct.tools import registry
from ct.tools.http_client import request, request_json


# US annual incidence by cancer type (SEER/Globocan estimates)
US_INCIDENCE = {
    "Lung": 238000,
    "Breast": 310000,
    "Colorectal": 153000,
    "Prostate": 288000,
    "Lymphoma (NHL)": 80000,
    "AML": 20000,
    "ALL": 6000,
    "Multiple Myeloma": 35000,
    "Kidney": 82000,
    "Liver": 42000,
    "Ovarian": 20000,
    "Pancreatic": 64000,
    "Melanoma": 100000,
    "Bladder": 83000,
    "Thyroid": 44000,
    "Glioma/Brain": 25000,
    "Cervical": 14000,
    "Endometrial/Uterine": 66000,
    "Head & Neck": 66000,
    "Gastric/Esophageal": 49000,
    "Sarcoma": 14000,
    "Neuroblastoma": 800,
    "Mesothelioma": 3000,
}

# PRISM lineage to standard cancer type mapping
LINEAGE_TO_CANCER = {
    "Lung": {"incidence_key": "Lung", "fraction": 0.85, "five_yr_survival": 0.25,
             "unmet_need": True, "name": "Non-Small Cell Lung Cancer"},
    "CNS/Brain": {"incidence_key": "Glioma/Brain", "fraction": 0.60, "five_yr_survival": 0.05,
                  "unmet_need": True, "name": "Diffuse Glioma"},
    "Skin": {"incidence_key": "Melanoma", "fraction": 1.0, "five_yr_survival": 0.93,
             "unmet_need": False, "name": "Melanoma"},
    "Lymphoid": {"incidence_key": "Lymphoma (NHL)", "fraction": 0.85, "five_yr_survival": 0.73,
                 "unmet_need": False, "name": "B-Cell Lymphoma"},
    "Head and Neck": {"incidence_key": "Head & Neck", "fraction": 0.90, "five_yr_survival": 0.66,
                      "unmet_need": False, "name": "Head & Neck SCC"},
    "Bowel": {"incidence_key": "Colorectal", "fraction": 0.95, "five_yr_survival": 0.65,
              "unmet_need": False, "name": "Colorectal Cancer"},
    "Ovary/Fallopian Tube": {"incidence_key": "Ovarian", "fraction": 0.90, "five_yr_survival": 0.50,
                             "unmet_need": True, "name": "Ovarian Cancer"},
    "Pancreas": {"incidence_key": "Pancreatic", "fraction": 0.85, "five_yr_survival": 0.12,
                 "unmet_need": True, "name": "Pancreatic Cancer"},
    "Breast": {"incidence_key": "Breast", "fraction": 0.95, "five_yr_survival": 0.90,
               "unmet_need": False, "name": "Breast Cancer"},
    "Prostate": {"incidence_key": "Prostate", "fraction": 0.95, "five_yr_survival": 0.97,
                 "unmet_need": False, "name": "Prostate Cancer"},
    "Myeloid": {"incidence_key": "AML", "fraction": 1.0, "five_yr_survival": 0.30,
                "unmet_need": True, "name": "Acute Myeloid Leukemia"},
    "Liver": {"incidence_key": "Liver", "fraction": 0.80, "five_yr_survival": 0.20,
              "unmet_need": True, "name": "Hepatocellular Carcinoma"},
    "Kidney": {"incidence_key": "Kidney", "fraction": 0.85, "five_yr_survival": 0.77,
               "unmet_need": False, "name": "Renal Cell Carcinoma"},
    "Bladder/Urinary Tract": {"incidence_key": "Bladder", "fraction": 0.90, "five_yr_survival": 0.77,
                              "unmet_need": False, "name": "Bladder Cancer"},
    "Stomach": {"incidence_key": "Gastric/Esophageal", "fraction": 0.65, "five_yr_survival": 0.22,
                "unmet_need": True, "name": "Gastric/Esophageal Cancer"},
    "Uterus": {"incidence_key": "Endometrial/Uterine", "fraction": 0.90, "five_yr_survival": 0.81,
               "unmet_need": False, "name": "Endometrial Cancer"},
    "Cervix": {"incidence_key": "Cervical", "fraction": 0.70, "five_yr_survival": 0.66,
               "unmet_need": True, "name": "Cervical Cancer"},
    "Bone": {"incidence_key": "Sarcoma", "fraction": 0.11, "five_yr_survival": 0.60,
             "unmet_need": True, "name": "Bone Sarcoma"},
    "Soft Tissue": {"incidence_key": "Sarcoma", "fraction": 0.05, "five_yr_survival": 0.63,
                    "unmet_need": True, "name": "Soft Tissue Sarcoma"},
    "Pleura": {"incidence_key": "Mesothelioma", "fraction": 0.80, "five_yr_survival": 0.12,
               "unmet_need": True, "name": "Mesothelioma"},
}


@registry.register(
    name="clinical.indication_map",
    description="Map compound sensitivity profiles to cancer indications with response rates",
    category="clinical",
    parameters={
        "compound_id": "Compound ID to map (or 'all')",
        "min_response_rate": "Minimum response rate to include (default 0.1)",
    },
    requires_data=["prism", "depmap_model"],
    usage_guide="You want to know which cancer types a compound is active against. Maps PRISM cell line sensitivity to clinical cancer indications. Use for indication selection and clinical positioning.",
)
def indication_map(compound_id: str = "all", min_response_rate: float = 0.1, **kwargs) -> dict:
    """Map compound PRISM sensitivity to cancer indications.

    Uses cell line lineage annotations to group by cancer type and compute
    per-indication response rates (fraction of cell lines with LFC < -0.5).
    """
    from ct.data.loaders import load_prism, load_model_metadata

    prism = load_prism()
    model = load_model_metadata()

    # Map cell lines to lineages
    ccle_to_lineage = {}
    for _, row in model.iterrows():
        ccle = row.get("CCLEName", "")
        lin = row.get("OncotreeLineage", "Unknown")
        if pd.notna(ccle) and pd.notna(lin):
            ccle_to_lineage[ccle] = lin

    compounds = [compound_id] if compound_id != "all" else prism["pert_name"].unique().tolist()
    results = []

    for cpd in compounds:
        cpd_data = prism[prism["pert_name"] == cpd]
        if len(cpd_data) == 0:
            continue

        max_dose = cpd_data["pert_dose"].max()
        cpd_hd = cpd_data[cpd_data["pert_dose"] == max_dose].copy()
        cpd_hd["lineage"] = cpd_hd["ccle_name"].map(ccle_to_lineage)

        for lineage, group in cpd_hd.groupby("lineage"):
            if lineage == "Unknown" or len(group) < 3:
                continue

            n_cells = len(group)
            n_sensitive = (group["LFC"] < -0.5).sum()
            response_rate = n_sensitive / n_cells
            mean_lfc = float(group["LFC"].mean())

            if response_rate < min_response_rate:
                continue

            # Map to clinical indication
            cancer_info = LINEAGE_TO_CANCER.get(lineage, {})
            cancer_name = cancer_info.get("name", lineage)

            results.append({
                "compound": cpd,
                "lineage": lineage,
                "cancer_type": cancer_name,
                "n_cell_lines": n_cells,
                "n_sensitive": int(n_sensitive),
                "response_rate": round(response_rate, 3),
                "mean_lfc": round(mean_lfc, 3),
                "unmet_need": cancer_info.get("unmet_need"),
                "five_yr_survival": cancer_info.get("five_yr_survival"),
            })

    if not results:
        return {
            "summary": f"No indications found for {compound_id} (compound may not be in PRISM data or no lineages met criteria)",
            "n_indications": 0,
            "indications": [],
        }

    df = pd.DataFrame(results).sort_values("response_rate", ascending=False)

    if compound_id != "all":
        top = df.head(5)
        top_names = ", ".join(top["cancer_type"].tolist()) if len(top) > 0 else "none"
        summary = f"Indication mapping for {compound_id}: {len(df)} indications (top: {top_names})"
    else:
        summary = f"Mapped {len(compounds)} compounds across {df['cancer_type'].nunique()} indications"

    return {
        "summary": summary,
        "n_indications": len(df),
        "indications": df.to_dict("records"),
    }


@registry.register(
    name="clinical.population_size",
    description="Estimate addressable patient population per compound and indication using SEER incidence data",
    category="clinical",
    parameters={
        "compound_id": "Compound ID to size (or 'all')",
        "clinical_adjustment": "Clinical reality factor (default 0.10 = 10% of cell-line estimate)",
    },
    requires_data=["prism", "depmap_model"],
    usage_guide="You want to estimate how many patients could benefit from a compound — combines PRISM response rates with US cancer incidence data. Use for market sizing and clinical development prioritization.",
)
def population_size(compound_id: str = "all", clinical_adjustment: float = 0.10, **kwargs) -> dict:
    """Estimate addressable patient populations.

    addressable = annual_incidence x subtype_fraction x cell_line_response_rate
    clinical_adjusted = addressable x clinical_adjustment_factor
    """
    # Get indication mapping first
    ind_result = indication_map(compound_id=compound_id, min_response_rate=0.05)
    if "error" in ind_result:
        return ind_result

    indications = ind_result["indications"]
    results = []

    for ind in indications:
        lineage = ind["lineage"]
        cancer_info = LINEAGE_TO_CANCER.get(lineage)
        if not cancer_info:
            continue

        incidence_key = cancer_info["incidence_key"]
        if incidence_key not in US_INCIDENCE:
            continue

        annual_incidence = US_INCIDENCE[incidence_key]
        subtype_fraction = cancer_info["fraction"]
        base_population = int(annual_incidence * subtype_fraction)
        addressable = int(base_population * ind["response_rate"])
        clinical_est = int(addressable * clinical_adjustment)

        results.append({
            "compound": ind["compound"],
            "cancer_type": ind["cancer_type"],
            "annual_us_incidence": annual_incidence,
            "subtype_fraction": subtype_fraction,
            "base_population": base_population,
            "response_rate": ind["response_rate"],
            "addressable_patients": addressable,
            "clinical_adjusted": clinical_est,
            "mean_lfc": ind["mean_lfc"],
            "n_cell_lines": ind["n_cell_lines"],
            "unmet_need": ind.get("unmet_need"),
            "five_yr_survival": ind.get("five_yr_survival"),
        })

    if not results:
        return {
            "summary": f"No addressable populations identified for {compound_id} (compound may not be in PRISM data)",
            "clinical_adjustment": clinical_adjustment,
            "per_indication": [],
            "per_compound": {},
        }

    df = pd.DataFrame(results).sort_values("addressable_patients", ascending=False)

    # Per-compound totals
    if len(df) > 0:
        cpd_totals = df.groupby("compound").agg(
            total_addressable=("addressable_patients", "sum"),
            total_clinical=("clinical_adjusted", "sum"),
            n_indications=("cancer_type", "nunique"),
        ).sort_values("total_addressable", ascending=False)

        top_cpd = cpd_totals.index[0] if len(cpd_totals) > 0 else "N/A"
        total = int(cpd_totals.iloc[0]["total_addressable"]) if len(cpd_totals) > 0 else 0

        summary = (
            f"Patient population sizing ({clinical_adjustment:.0%} clinical adjustment):\n"
            f"Top compound: {top_cpd} ({total:,} addressable, "
            f"{int(total * clinical_adjustment):,} clinical estimate)"
        )
    else:
        summary = "No addressable populations identified"
        cpd_totals = pd.DataFrame()

    return {
        "summary": summary,
        "clinical_adjustment": clinical_adjustment,
        "per_indication": df.to_dict("records"),
        "per_compound": cpd_totals.to_dict("index") if len(cpd_totals) > 0 else {},
    }


@registry.register(
    name="clinical.tcga_stratify",
    description="Stratify patients by target expression using TCGA data from Human Protein Atlas",
    category="clinical",
    parameters={
        "gene": "Gene symbol to query (e.g. CDC25C, GATA2)",
    },
    usage_guide="You want to check if a target gene is expressed in patient tumors — queries TCGA expression data from Human Protein Atlas. Use for clinical biomarker validation and patient stratification strategy.",
)
def tcga_stratify(gene: str, **kwargs) -> dict:
    """Query Human Protein Atlas for TCGA expression data.

    Returns expression levels across cancer types and prognostic associations.
    Convergence = log2(median_FPKM + 1) x |compound_LFC| (when PRISM data available).
    """
    import math
    import re

    try:
        import httpx
    except ImportError:
        return {"error": "httpx required for TCGA queries (pip install httpx)", "summary": "httpx required for TCGA queries (pip install httpx)"}
    # Fast-path cache for common targets (avoids API call)
    _GENE_ENSEMBL_CACHE = {
        "CDC25C": "ENSG00000158402", "GATA2": "ENSG00000179348",
        "RBCK1": "ENSG00000125826", "ZNF687": "ENSG00000143373",
        "BCOR": "ENSG00000183337", "CEP57": "ENSG00000166037",
        "BTBD1": "ENSG00000084693", "FLCN": "ENSG00000154803",
        "LYZ": "ENSG00000090382", "CRBN": "ENSG00000113851",
        "PDCD2": "ENSG00000126249",
    }

    ensembl_id = _GENE_ENSEMBL_CACHE.get(gene.upper())
    if not ensembl_id:
        # Look up via Ensembl REST API — works for any human gene symbol
        try:
            xref_url = f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene}"
            xref_data, xref_error = request_json(
                "GET",
                xref_url,
                timeout=15,
                retries=2,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ct-celltype/0.1",
                },
            )
            if not xref_error and isinstance(xref_data, list):
                for xref in xref_data:
                    if xref.get("type") == "gene" and xref.get("id", "").startswith("ENSG"):
                        ensembl_id = xref["id"]
                        break
        except Exception:
            pass

    if not ensembl_id:
        return {"error": f"Could not resolve Ensembl ID for gene '{gene}'. Check the gene symbol is correct.", "summary": f"Could not resolve Ensembl ID for gene '{gene}'. Check the gene symbol is correct."}
    # Fetch gene data from HPA JSON API
    url = f"https://www.proteinatlas.org/{ensembl_id}.json"
    resp, error = request(
        "GET",
        url,
        timeout=30,
        retries=2,
        headers={"User-Agent": "ct-celltype/0.1"},
        raise_for_status=False,
    )
    if error:
        return {"error": f"Failed to fetch HPA data: {error}", "summary": f"Failed to fetch HPA data: {error}"}
    if resp.status_code != 200:
        return {"error": f"HPA API returned status {resp.status_code} for {gene}", "summary": f"HPA API error for {gene}"}
    content_type = ""
    try:
        ct_raw = resp.headers.get("content-type", "")
        if isinstance(ct_raw, str):
            content_type = ct_raw.lower()
    except Exception:
        pass
    if content_type and "json" not in content_type:
        return {"error": f"HPA API returned {content_type} instead of JSON for {gene}", "summary": f"HPA returned non-JSON for {gene}"}
    try:
        gene_json = resp.json()
    except Exception:
        return {"error": f"HPA API returned invalid JSON for {gene}", "summary": f"HPA invalid JSON for {gene}"}

    # Extract prognostic data
    prognostics = []
    for key, val in gene_json.items():
        if key.startswith("Cancer prognostics -") and val is not None:
            m = re.match(r"Cancer prognostics - (.+?) \((TCGA|validation)\)", key)
            if m and val.get("is_prognostic"):
                prognostics.append({
                    "cancer_type": m.group(1),
                    "dataset": m.group(2),
                    "direction": val.get("prognostic type", ""),
                    "status": val.get("prognostic", ""),
                    "p_value": val.get("p_val", ""),
                })

    gene_info = {
        "cancer_specificity": gene_json.get("RNA cancer specificity", ""),
        "cancer_distribution": gene_json.get("RNA cancer distribution", ""),
        "tissue_specificity": gene_json.get("RNA tissue specificity", ""),
    }

    # Extract RNA expression by cancer type
    rna_cancer = gene_json.get("RNA cancer sample", {})
    cancer_expression = []
    if isinstance(rna_cancer, dict):
        for cancer_type, data in rna_cancer.items():
            if isinstance(data, dict):
                fpkm = data.get("value", 0)
                cancer_expression.append({
                    "cancer_type": cancer_type,
                    "fpkm": float(fpkm) if fpkm else 0,
                    "expr_score": round(math.log2(float(fpkm) + 1), 3) if fpkm else 0,
                })

    cancer_expression.sort(key=lambda x: x["fpkm"], reverse=True)

    # Classify expression levels
    for entry in cancer_expression:
        fpkm = entry["fpkm"]
        if fpkm >= 10:
            entry["level"] = "HIGH"
        elif fpkm >= 3:
            entry["level"] = "MEDIUM"
        elif fpkm >= 1:
            entry["level"] = "LOW"
        else:
            entry["level"] = "VERY_LOW"

    high_expr = [e for e in cancer_expression if e["level"] in ("HIGH", "MEDIUM")]

    return {
        "summary": (
            f"TCGA stratification for {gene}:\n"
            f"Expressed (FPKM>=3) in {len(high_expr)}/{len(cancer_expression)} cancer types\n"
            f"Prognostic in {len(prognostics)} cancer types\n"
            f"Cancer specificity: {gene_info['cancer_specificity']}"
        ),
        "gene": gene,
        "gene_info": gene_info,
        "cancer_expression": cancer_expression,
        "prognostics": prognostics,
    }


@registry.register(
    name="clinical.trial_search",
    description="Search ClinicalTrials.gov for relevant clinical trials by gene, drug, or indication",
    category="clinical",
    parameters={
        "query": "Search term (gene name, drug name, indication, or free text)",
        "status": "Optional trial status filter: RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING, etc.",
    },
    usage_guide="You want to find clinical trials for a target, compound, or disease. Use to assess clinical precedent, competitive landscape, and development activity.",
)
def trial_search(query: str, status: str = "", **kwargs) -> dict:
    """Search ClinicalTrials.gov API v2 for clinical trials.

    Returns trial metadata including NCT ID, phase, status, conditions,
    interventions, sponsor, and enrollment.
    """
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": query,
        "pageSize": 20,
    }
    if status:
        params["filter.overallStatus"] = status

    data, error = request_json(
        "GET",
        url,
        params=params,
        timeout=15,
        retries=2,
    )
    if error:
        return {"error": f"ClinicalTrials.gov search failed: {error}", "summary": f"ClinicalTrials.gov search failed: {error}"}
    studies = data.get("studies", [])
    total_count = len(studies)
    has_more = data.get("nextPageToken") is not None

    trials = []
    phase_counts = {}
    status_counts = {}

    for study in studies:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        desc = proto.get("descriptionModule", {})
        contacts = proto.get("contactsLocationsModule", {})
        arms = proto.get("armsInterventionsModule", {})
        cond_mod = proto.get("conditionsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

        nct_id = ident.get("nctId", "")
        title = ident.get("briefTitle", "")
        overall_status = status_mod.get("overallStatus", "")
        start_date = status_mod.get("startDateStruct", {}).get("date", "")

        # Phase
        phases = design.get("phases", [])
        phase = ", ".join(phases) if phases else "N/A"

        # Conditions
        conditions = cond_mod.get("conditions", [])

        # Interventions
        interventions_raw = arms.get("interventions", [])
        interventions = []
        for iv in interventions_raw:
            interventions.append({
                "type": iv.get("type", ""),
                "name": iv.get("name", ""),
            })

        # Sponsor
        lead_sponsor = sponsor_mod.get("leadSponsor", {})
        sponsor_name = lead_sponsor.get("name", "")

        # Enrollment
        enrollment_info = design.get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count", "")

        trial = {
            "nct_id": nct_id,
            "title": title,
            "status": overall_status,
            "phase": phase,
            "conditions": conditions[:5],  # Cap to keep output manageable
            "interventions": interventions[:5],
            "sponsor": sponsor_name,
            "enrollment": enrollment,
            "start_date": start_date,
        }
        trials.append(trial)

        # Aggregate counts
        for p in phases:
            phase_counts[p] = phase_counts.get(p, 0) + 1
        status_counts[overall_status] = status_counts.get(overall_status, 0) + 1

    # Build summary
    if trials:
        top_phases = ", ".join(f"{k}: {v}" for k, v in sorted(phase_counts.items()))
        top_statuses = ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))
        summary = (
            f"ClinicalTrials.gov search '{query}': {total_count}{'+ (more pages)' if has_more else ''} results\n"
            f"Phase distribution: {top_phases}\n"
            f"Status distribution: {top_statuses}"
        )
    else:
        summary = f"No clinical trials found for '{query}'"

    return {
        "summary": summary,
        "query": query,
        "total_count": total_count,
        "has_more": has_more,
        "trials": trials,
        "phase_distribution": phase_counts,
        "status_distribution": status_counts,
    }


def _normalize_phase_token(phase_value: str) -> str:
    """Normalize trial phase labels for robust filtering."""
    return re.sub(r"[^A-Z0-9]", "", str(phase_value or "").upper())


@registry.register(
    name="clinical.trial_design_benchmark",
    description="Benchmark clinical trial design patterns for a query (endpoints, enrollment, randomization, biomarker criteria)",
    category="clinical",
    parameters={
        "query": "Search term for indication/target/drug",
        "phase": "Optional phase filter (e.g., 'PHASE2', 'PHASE3', 'EARLY_PHASE1')",
        "status": "Optional trial status filter (e.g., RECRUITING, COMPLETED)",
        "max_results": "Max studies to include from ClinicalTrials.gov API v2 (default 20, max 100)",
    },
    usage_guide=(
        "Use to benchmark protocol design against the current landscape. Summarizes common "
        "endpoints, intervention patterns, enrollment benchmarks, and key eligibility traits."
    ),
)
def trial_design_benchmark(
    query: str,
    phase: str = "",
    status: str = "",
    max_results: int = 20,
    **kwargs,
) -> dict:
    """Benchmark trial design characteristics from ClinicalTrials.gov API v2."""
    if not query or not query.strip():
        return {"error": "query is required", "summary": "No query provided"}

    max_results = max(1, min(int(max_results or 20), 100))
    params = {
        "query.term": query.strip(),
        "pageSize": str(max_results),
    }
    if status:
        params["filter.overallStatus"] = status

    data, error = request_json(
        "GET",
        "https://clinicaltrials.gov/api/v2/studies",
        params=params,
        timeout=20,
        retries=2,
    )
    if error:
        return {
            "error": f"ClinicalTrials.gov benchmark failed: {error}",
            "summary": f"Clinical trial design benchmark failed: {error}",
        }

    studies = data.get("studies", [])
    has_more = data.get("nextPageToken") is not None
    phase_filter = phase.strip()
    phase_filter_norm = _normalize_phase_token(phase_filter) if phase_filter else ""

    trials = []
    phase_counts = {}
    status_counts = {}
    endpoint_counts = {}
    intervention_counts = {}
    enrollment_values = []

    design_patterns = {
        "randomized_trials": 0,
        "blinded_trials": 0,
        "placebo_control_trials": 0,
        "biomarker_criteria_trials": 0,
        "ecog_criteria_trials": 0,
    }

    for study in studies:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        outcomes_mod = proto.get("outcomesModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        cond_mod = proto.get("conditionsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

        phases = design_mod.get("phases", []) or []
        if phase_filter_norm:
            phase_tokens = {_normalize_phase_token(p) for p in phases}
            if phase_filter_norm not in phase_tokens:
                continue

        overall_status = status_mod.get("overallStatus", "") or "UNKNOWN"
        phase_label = ", ".join(phases) if phases else "N/A"
        phase_counts[phase_label] = phase_counts.get(phase_label, 0) + 1
        status_counts[overall_status] = status_counts.get(overall_status, 0) + 1

        design_info = design_mod.get("designInfo", {})
        allocation = design_info.get("allocation", "")
        intervention_model = design_info.get("interventionModel", "")
        masking = design_info.get("maskingInfo", {}).get("masking", "")

        interventions = []
        for iv in arms_mod.get("interventions", []) or []:
            iv_name = (iv.get("name", "") or "").strip()
            if iv_name:
                interventions.append(iv_name)
                intervention_counts[iv_name] = intervention_counts.get(iv_name, 0) + 1

        primary_endpoints = []
        for out in outcomes_mod.get("primaryOutcomes", []) or []:
            measure = (out.get("measure", "") or "").strip()
            if measure:
                primary_endpoints.append(measure)
                endpoint_counts[measure] = endpoint_counts.get(measure, 0) + 1

        enrollment_raw = design_mod.get("enrollmentInfo", {}).get("count")
        enrollment = None
        try:
            enrollment = int(enrollment_raw)
            enrollment_values.append(enrollment)
        except Exception:
            enrollment = enrollment_raw

        eligibility_text = (elig_mod.get("eligibilityCriteria", "") or "").lower()
        biomarker_criteria = any(
            term in eligibility_text
            for term in ("biomarker", "mutation", "genotype", "expression", "pd-l1", "her2", "egfr", "alk")
        )
        ecog_criteria = "ecog" in eligibility_text

        allocation_norm = str(allocation).strip().upper().replace("-", "_")
        if allocation_norm == "RANDOMIZED" or (
            "RANDOMIZED" in allocation_norm and "NON_RANDOMIZED" not in allocation_norm
        ):
            design_patterns["randomized_trials"] += 1
        if masking and str(masking).upper() not in {"NONE", "OPEN_LABEL"}:
            design_patterns["blinded_trials"] += 1
        if any("placebo" in iv.lower() for iv in interventions):
            design_patterns["placebo_control_trials"] += 1
        if biomarker_criteria:
            design_patterns["biomarker_criteria_trials"] += 1
        if ecog_criteria:
            design_patterns["ecog_criteria_trials"] += 1

        trials.append({
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "phase": phase_label,
            "status": overall_status,
            "study_type": design_mod.get("studyType", ""),
            "allocation": allocation,
            "intervention_model": intervention_model,
            "masking": masking,
            "enrollment": enrollment,
            "conditions": (cond_mod.get("conditions", []) or [])[:5],
            "interventions": interventions[:8],
            "primary_endpoints": primary_endpoints[:8],
            "sponsor": (sponsor_mod.get("leadSponsor", {}) or {}).get("name", ""),
            "start_date": (status_mod.get("startDateStruct", {}) or {}).get("date", ""),
            "biomarker_criteria": biomarker_criteria,
            "ecog_criteria": ecog_criteria,
        })

    if not trials:
        phase_text = f", phase={phase_filter}" if phase_filter else ""
        status_text = f", status={status}" if status else ""
        return {
            "query": query,
            "phase_filter": phase_filter,
            "status_filter": status,
            "trials": [],
            "summary": f"No trials found for '{query}' with current filters{phase_text}{status_text}.",
        }

    median_enrollment = float(np.median(enrollment_values)) if enrollment_values else None

    endpoint_top = sorted(endpoint_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    intervention_top = sorted(intervention_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

    top_endpoint_text = ", ".join(f"{name} ({count})" for name, count in endpoint_top[:3]) or "none"
    summary = (
        f"Trial design benchmark for '{query}': {len(trials)} trial(s)"
        f"{' (+ more pages)' if has_more else ''}. "
        f"Median enrollment: {int(median_enrollment) if median_enrollment is not None else 'NA'}. "
        f"Top primary endpoints: {top_endpoint_text}."
    )

    return {
        "summary": summary,
        "query": query,
        "phase_filter": phase_filter,
        "status_filter": status,
        "has_more": has_more,
        "n_trials": len(trials),
        "median_enrollment": median_enrollment,
        "phase_distribution": phase_counts,
        "status_distribution": status_counts,
        "design_patterns": design_patterns,
        "top_primary_endpoints": [{"endpoint": k, "count": v} for k, v in endpoint_top],
        "top_interventions": [{"intervention": k, "count": v} for k, v in intervention_top],
        "trials": trials,
    }


@registry.register(
    name="clinical.endpoint_benchmark",
    description="Benchmark endpoint usage patterns and enrollment norms for an indication/target query",
    category="clinical",
    parameters={
        "query": "Search term for indication/target/drug",
        "phase": "Optional phase filter",
        "status": "Optional status filter",
        "max_results": "Maximum studies to include (default 30, max 100)",
    },
    usage_guide=(
        "Use during protocol planning to benchmark what endpoints and enrollment levels are commonly used "
        "by competitors in similar trials."
    ),
)
def endpoint_benchmark(
    query: str,
    phase: str = "",
    status: str = "",
    max_results: int = 30,
    **kwargs,
) -> dict:
    """Summarize endpoint conventions from ClinicalTrials.gov records."""
    del kwargs
    base = trial_design_benchmark(
        query=query,
        phase=phase,
        status=status,
        max_results=max_results,
    )
    if "error" in base:
        return {
            "error": base["error"],
            "summary": base["summary"],
        }

    trials = base.get("trials", []) or []
    if not trials:
        return {
            "summary": f"No trials available for endpoint benchmark on '{query}'.",
            "query": query,
            "trials": [],
        }

    endpoint_family_counts = {
        "overall_survival": 0,
        "progression_free_survival": 0,
        "response_rate": 0,
        "safety_tolerability": 0,
        "quality_of_life": 0,
        "biomarker_driven": 0,
        "other": 0,
    }

    endpoint_examples = {k: [] for k in endpoint_family_counts}
    for trial in trials:
        endpoints = trial.get("primary_endpoints", []) or []
        if not endpoints:
            endpoint_family_counts["other"] += 1
            continue
        classified = False
        for endpoint in endpoints:
            text = str(endpoint).lower()
            if "overall survival" in text or text.strip() == "os":
                key = "overall_survival"
            elif "progression-free survival" in text or "pfs" in text:
                key = "progression_free_survival"
            elif "objective response rate" in text or "orr" in text or "response rate" in text:
                key = "response_rate"
            elif "adverse event" in text or "safety" in text or "tolerability" in text:
                key = "safety_tolerability"
            elif "quality of life" in text or "qol" in text or "patient-reported" in text:
                key = "quality_of_life"
            elif any(k in text for k in ("biomarker", "mutation", "pd-l1", "ctdna", "mrd")):
                key = "biomarker_driven"
            else:
                key = "other"

            endpoint_family_counts[key] += 1
            if len(endpoint_examples[key]) < 5:
                endpoint_examples[key].append(endpoint)
            classified = True
        if not classified:
            endpoint_family_counts["other"] += 1

    # Enrollment statistics
    enrollments = []
    for trial in trials:
        value = trial.get("enrollment")
        if isinstance(value, int):
            enrollments.append(value)
    enrollment_median = float(np.median(enrollments)) if enrollments else None
    enrollment_p75 = float(np.percentile(enrollments, 75)) if len(enrollments) >= 2 else None

    ranked_families = sorted(
        endpoint_family_counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_families = [{"family": k, "count": v} for k, v in ranked_families if v > 0][:6]
    top_family_text = ", ".join(f"{x['family']} ({x['count']})" for x in top_families[:3]) or "none"

    summary = (
        f"Endpoint benchmark for '{query}': {len(trials)} trial(s). "
        f"Top endpoint families: {top_family_text}. "
        f"Median enrollment: {int(enrollment_median) if enrollment_median is not None else 'NA'}."
    )

    return {
        "summary": summary,
        "query": query,
        "phase_filter": phase,
        "status_filter": status,
        "n_trials": len(trials),
        "endpoint_families": endpoint_family_counts,
        "top_endpoint_families": top_families,
        "endpoint_examples": endpoint_examples,
        "median_enrollment": enrollment_median,
        "p75_enrollment": enrollment_p75,
        "phase_distribution": base.get("phase_distribution", {}),
        "status_distribution": base.get("status_distribution", {}),
        "trials": trials,
    }


@registry.register(
    name="clinical.competitive_landscape",
    description="Aggregate competitive intelligence for a target or indication from trials, ChEMBL, and Open Targets",
    category="clinical",
    parameters={
        "gene": "Target gene symbol (e.g. CRBN, BRAF, EGFR)",
        "indication": "Optional indication to focus the search (e.g. 'melanoma', 'lung cancer')",
    },
    usage_guide="You want a comprehensive view of the competitive landscape around a drug target — combines ClinicalTrials.gov, ChEMBL, and Open Targets to show active programs, phase distribution, and mechanism diversity. Use for strategic positioning and differentiation.",
)
def competitive_landscape(gene: str, indication: str = "", **kwargs) -> dict:
    """Aggregate competitive intelligence from multiple sources.

    Combines:
    1. ClinicalTrials.gov: active clinical programs
    2. ChEMBL: known compounds and bioactivities against the target
    3. Open Targets: known drugs and mechanisms via GraphQL
    """
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    results = {
        "gene": gene,
        "indication": indication or "all",
    }

    # --- Source 1: ClinicalTrials.gov ---
    ct_query = f"{gene} {indication}".strip() if indication else gene
    trial_data = trial_search(query=ct_query)

    if "error" not in trial_data:
        results["trials"] = {
            "total_count": trial_data.get("total_count", 0),
            "phase_distribution": trial_data.get("phase_distribution", {}),
            "status_distribution": trial_data.get("status_distribution", {}),
            "top_trials": trial_data.get("trials", [])[:10],
        }
    else:
        results["trials"] = {"error": trial_data["error"], "total_count": 0}

    # --- Source 2: ChEMBL target search + activities ---
    chembl_compounds = []
    chembl_base = "https://www.ebi.ac.uk/chembl/api/data"
    headers = {"Accept": "application/json"}

    try:
        # Find target in ChEMBL
        tgt_data, error = request_json(
            "GET",
            f"{chembl_base}/target/search.json",
            params={"q": gene, "limit": 5},
            headers=headers,
            timeout=10,
            retries=2,
        )
        if error:
            raise RuntimeError(error)

        targets = tgt_data.get("targets", [])
        chembl_target_id = None
        for tgt in targets:
            # Prefer human SINGLE PROTEIN targets
            if (tgt.get("organism", "") == "Homo sapiens" and
                    tgt.get("target_type", "") == "SINGLE PROTEIN"):
                chembl_target_id = tgt.get("target_chembl_id")
                break

        if not chembl_target_id and targets:
            chembl_target_id = targets[0].get("target_chembl_id")

        if chembl_target_id:
            # Get activities for the target
            act_data, error = request_json(
                "GET",
                f"{chembl_base}/activity.json",
                params={
                    "target_chembl_id": chembl_target_id,
                    "limit": 50,
                    "standard_type__in": "IC50,Ki,Kd,EC50",
                },
                headers=headers,
                timeout=10,
                retries=2,
            )
            if error:
                raise RuntimeError(error)

            # Deduplicate by molecule
            seen_mols = set()
            moa_types = set()
            for act in act_data.get("activities", []):
                mol_id = act.get("molecule_chembl_id", "")
                if mol_id and mol_id not in seen_mols:
                    seen_mols.add(mol_id)
                    chembl_compounds.append({
                        "chembl_id": mol_id,
                        "name": act.get("molecule_pref_name", "") or mol_id,
                        "activity_type": act.get("standard_type", ""),
                        "activity_value": act.get("standard_value"),
                        "activity_units": act.get("standard_units", ""),
                        "pchembl": act.get("pchembl_value"),
                    })
                    assay_desc = act.get("assay_description", "")
                    if assay_desc:
                        # Extract broad MoA categories from assay descriptions
                        desc_lower = assay_desc.lower()
                        if "inhibit" in desc_lower:
                            moa_types.add("Inhibitor")
                        if "degrad" in desc_lower:
                            moa_types.add("Degrader")
                        if "agonist" in desc_lower:
                            moa_types.add("Agonist")
                        if "antagonist" in desc_lower:
                            moa_types.add("Antagonist")
                        if "allosteric" in desc_lower:
                            moa_types.add("Allosteric modulator")
                        if "antibod" in desc_lower:
                            moa_types.add("Antibody")
                        if "covalent" in desc_lower:
                            moa_types.add("Covalent binder")

            results["chembl"] = {
                "target_chembl_id": chembl_target_id,
                "unique_compounds": len(chembl_compounds),
                "moa_types": sorted(moa_types),
                "top_compounds": chembl_compounds[:15],
            }
        else:
            results["chembl"] = {"error": f"No ChEMBL target found for {gene}", "unique_compounds": 0}

    except Exception as e:
        results["chembl"] = {"error": f"ChEMBL query failed: {e}", "unique_compounds": 0}

    # --- Source 3: Open Targets known drugs (GraphQL) ---
    ot_drugs = []
    ot_url = "https://api.platform.opentargets.org/api/v4/graphql"
    graphql_query = """
    query knownDrugs($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        knownDrugs(size: 30) {
          uniqueDrugs
          uniqueTargets
          rows {
            drugId
            prefName
            drugType
            mechanismOfAction
            phase
            status
            disease {
              id
              name
            }
          }
        }
      }
    }
    """

    # Map gene symbol to Ensembl ID via Open Targets search
    try:
        search_data, error = request_json(
            "POST",
            ot_url,
            json={
                "query": """
                query searchTarget($q: String!) {
                  search(queryString: $q, entityNames: ["target"], page: {size: 5, index: 0}) {
                    hits {
                      id
                      name
                      entity
                    }
                  }
                }
                """,
                "variables": {"q": gene},
            },
            timeout=10,
            retries=2,
        )
        if error:
            raise RuntimeError(error)

        hits = search_data.get("data", {}).get("search", {}).get("hits", [])
        ensembl_id = None
        for hit in hits:
            if hit.get("entity") == "target":
                ensembl_id = hit.get("id")
                break

        if ensembl_id:
            drugs_data, error = request_json(
                "POST",
                ot_url,
                json={
                    "query": graphql_query,
                    "variables": {"ensemblId": ensembl_id},
                },
                timeout=10,
                retries=2,
            )
            if error:
                raise RuntimeError(error)

            known_drugs = drugs_data.get("data", {}).get("target", {}).get("knownDrugs", {})
            if known_drugs:
                unique_drugs = known_drugs.get("uniqueDrugs", 0)
                phase_dist_ot = {}
                moa_set = set()

                for row in known_drugs.get("rows", []):
                    drug_name = row.get("prefName", "") or row.get("drugId", "")
                    phase = row.get("phase", 0)
                    moa = row.get("mechanismOfAction", "")
                    disease = row.get("disease", {})
                    disease_name = disease.get("name", "") if disease else ""

                    # Filter by indication if specified
                    if indication and disease_name:
                        if indication.lower() not in disease_name.lower():
                            continue

                    ot_drugs.append({
                        "drug": drug_name,
                        "drug_type": row.get("drugType", ""),
                        "mechanism": moa,
                        "phase": phase,
                        "status": row.get("status", ""),
                        "disease": disease_name,
                    })

                    phase_key = f"Phase {phase}" if phase else "Unknown"
                    phase_dist_ot[phase_key] = phase_dist_ot.get(phase_key, 0) + 1
                    if moa:
                        moa_set.add(moa)

                results["open_targets"] = {
                    "ensembl_id": ensembl_id,
                    "unique_drugs": unique_drugs,
                    "phase_distribution": phase_dist_ot,
                    "mechanisms": sorted(moa_set),
                    "drugs": ot_drugs[:20],
                }
            else:
                results["open_targets"] = {"error": "No known drugs found", "unique_drugs": 0}
        else:
            results["open_targets"] = {"error": f"Could not resolve Ensembl ID for {gene}", "unique_drugs": 0}

    except Exception as e:
        results["open_targets"] = {"error": f"Open Targets query failed: {e}", "unique_drugs": 0}

    # --- Aggregate summary ---
    total_trials = results.get("trials", {}).get("total_count", 0)
    chembl_count = results.get("chembl", {}).get("unique_compounds", 0)
    ot_count = results.get("open_targets", {}).get("unique_drugs", 0)

    trial_phases = results.get("trials", {}).get("phase_distribution", {})
    chembl_moas = results.get("chembl", {}).get("moa_types", [])
    ot_moas = results.get("open_targets", {}).get("mechanisms", [])
    all_moas = sorted(set(chembl_moas + ot_moas))

    phase_str = ", ".join(f"{k}: {v}" for k, v in sorted(trial_phases.items())) if trial_phases else "none"
    moa_str = ", ".join(all_moas[:5]) if all_moas else "not characterized"

    ind_label = f" in {indication}" if indication else ""
    summary = (
        f"Competitive landscape for {gene}{ind_label}:\n"
        f"Clinical trials: {total_trials} ({phase_str})\n"
        f"ChEMBL compounds: {chembl_count}\n"
        f"Open Targets known drugs: {ot_count}\n"
        f"Mechanism diversity: {moa_str}"
    )

    results["summary"] = summary
    return results
