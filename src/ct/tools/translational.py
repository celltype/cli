"""
Translational strategy tools bridging biomarkers to development readiness.
"""

from __future__ import annotations

from ct.tools import registry


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@registry.register(
    name="translational.biomarker_readiness",
    description="Assess translational readiness of a biomarker in a disease setting",
    category="translational",
    parameters={
        "biomarker": "Biomarker gene/protein or signature label (e.g., PD-L1, IL23R, KRAS G12C)",
        "indication": "Disease/indication context",
        "max_evidence": "Maximum literature/trial records to include per source (default 10)",
    },
    usage_guide=(
        "Use before clinical design to evaluate whether a biomarker is deployable for patient selection: "
        "evidence depth, trial usage, and practical stratification signal."
    ),
)
def biomarker_readiness(
    biomarker: str,
    indication: str,
    max_evidence: int = 10,
    **kwargs,
) -> dict:
    """Estimate biomarker readiness from trial and literature evidence."""
    del kwargs
    biomarker = (biomarker or "").strip()
    indication = (indication or "").strip()
    if not biomarker:
        return {"summary": "biomarker is required.", "error": "missing_biomarker"}
    if not indication:
        return {"summary": "indication is required.", "error": "missing_indication"}

    from ct.tools.clinical import trial_search
    from ct.tools.literature import openalex_search, pubmed_search

    max_evidence = max(1, min(int(max_evidence or 10), 25))
    query = f"{biomarker} {indication}".strip()

    trial_result = trial_search(query=query)
    pubmed_result = pubmed_search(query=f"{query} predictive biomarker", max_results=max_evidence)
    openalex_result = openalex_search(query=f"{query} biomarker stratification", max_results=max_evidence)

    if "error" in trial_result and "error" in pubmed_result and "error" in openalex_result:
        return {
            "summary": f"Biomarker readiness failed for '{query}': data sources unavailable.",
            "error": "all_sources_failed",
            "sources": {
                "trial_error": trial_result.get("error"),
                "pubmed_error": pubmed_result.get("error"),
                "openalex_error": openalex_result.get("error"),
            },
        }

    trial_total = _safe_int(trial_result.get("total_count", 0))
    status_dist = trial_result.get("status_distribution", {}) or {}
    recruiting = _safe_int(status_dist.get("RECRUITING", 0))

    pubmed_total = _safe_int(pubmed_result.get("total_count", 0))
    openalex_total = _safe_int(openalex_result.get("total_count", 0))

    score = 0
    score += min(35, trial_total)
    score += min(20, recruiting * 3)
    score += min(30, (pubmed_total // 5) * 5)
    score += min(15, (openalex_total // 10) * 5)
    score = min(100, score)

    if score >= 70:
        readiness = "high"
    elif score >= 40:
        readiness = "moderate"
    else:
        readiness = "early"

    risks = []
    if trial_total == 0:
        risks.append("No direct trial usage signal in the current query window.")
    if recruiting == 0 and trial_total > 0:
        risks.append("No recruiting trials currently detected; may indicate development pause.")
    if pubmed_total < 5:
        risks.append("Limited predictive biomarker publication depth.")

    summary = (
        f"Biomarker readiness for {biomarker} in {indication}: {readiness} ({score}/100). "
        f"Trials={trial_total}, recruiting={recruiting}, literature={pubmed_total + openalex_total}."
    )

    return {
        "summary": summary,
        "biomarker": biomarker,
        "indication": indication,
        "readiness_level": readiness,
        "readiness_score": score,
        "risks": risks,
        "trials": {
            "total_count": trial_total,
            "status_distribution": status_dist,
            "phase_distribution": trial_result.get("phase_distribution", {}),
            "records": (trial_result.get("trials") or [])[:max_evidence],
            "error": trial_result.get("error"),
        },
        "literature": {
            "pubmed_total": pubmed_total,
            "openalex_total": openalex_total,
            "pubmed_records": (pubmed_result.get("articles") or [])[:max_evidence],
            "openalex_records": (openalex_result.get("articles") or [])[:max_evidence],
            "pubmed_error": pubmed_result.get("error"),
            "openalex_error": openalex_result.get("error"),
        },
    }
