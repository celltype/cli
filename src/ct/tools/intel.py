"""
Competitive and pipeline intelligence tools for pharma R&D.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ct.tools import registry


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@registry.register(
    name="intel.pipeline_watch",
    description="Track pipeline activity for a target/indication across trials and literature",
    category="intel",
    parameters={
        "query": "Target, drug class, or mechanism to monitor",
        "indication": "Optional disease/indication filter",
        "max_trials": "Maximum trial records to retain (default 20)",
        "max_papers": "Maximum papers per source to retain (default 10)",
    },
    usage_guide=(
        "Use for ongoing landscape monitoring. Aggregates clinical trial momentum and publication "
        "velocity into a concise watchlist snapshot for strategy discussions."
    ),
)
def pipeline_watch(
    query: str,
    indication: str = "",
    max_trials: int = 20,
    max_papers: int = 10,
    **kwargs,
) -> dict:
    """Create a compact pipeline watch snapshot from public sources."""
    del kwargs
    if not query or not query.strip():
        return {"summary": "query is required.", "error": "missing_query"}

    from ct.tools.clinical import trial_search
    from ct.tools.literature import openalex_search, pubmed_search

    max_trials = max(1, min(int(max_trials or 20), 100))
    max_papers = max(1, min(int(max_papers or 10), 50))

    search_query = f"{query} {indication}".strip()
    trial_result = trial_search(query=search_query)
    pubmed_result = pubmed_search(query=search_query, max_results=max_papers)
    openalex_result = openalex_search(query=search_query, max_results=max_papers)

    if "error" in trial_result and "error" in pubmed_result and "error" in openalex_result:
        return {
            "summary": (
                f"Pipeline watch failed for '{search_query}': all upstream sources returned errors."
            ),
            "error": "all_sources_failed",
            "sources": {
                "trials_error": trial_result.get("error"),
                "pubmed_error": pubmed_result.get("error"),
                "openalex_error": openalex_result.get("error"),
            },
        }

    trials = (trial_result.get("trials") or [])[:max_trials]
    phase_dist = trial_result.get("phase_distribution", {}) or {}
    status_dist = trial_result.get("status_distribution", {}) or {}
    recruiting = _to_int(status_dist.get("RECRUITING", 0), 0)
    phase3 = _to_int(phase_dist.get("PHASE3", 0), 0)

    pubmed_articles = pubmed_result.get("articles", []) if isinstance(pubmed_result, dict) else []
    openalex_articles = openalex_result.get("articles", []) if isinstance(openalex_result, dict) else []

    current_year = datetime.utcnow().year
    recent_pubmed = 0
    for item in pubmed_articles:
        pub_date = str(item.get("pub_date", ""))
        if str(current_year) in pub_date or str(current_year - 1) in pub_date:
            recent_pubmed += 1

    recent_openalex = 0
    for item in openalex_articles:
        year = _to_int(item.get("publication_year"), 0)
        if year >= current_year - 1:
            recent_openalex += 1

    momentum_score = 0
    momentum_score += min(40, _to_int(trial_result.get("total_count", 0), 0))
    momentum_score += min(25, recruiting * 3)
    momentum_score += min(20, phase3 * 5)
    momentum_score += min(15, recent_pubmed + recent_openalex)
    momentum_score = min(100, momentum_score)

    if momentum_score >= 70:
        momentum = "high"
    elif momentum_score >= 40:
        momentum = "moderate"
    else:
        momentum = "early"

    summary = (
        f"Pipeline watch for '{search_query}': momentum={momentum} ({momentum_score}/100). "
        f"Trials={trial_result.get('total_count', 0)}, recruiting={recruiting}, phase3={phase3}, "
        f"recent publications={recent_pubmed + recent_openalex}."
    )

    return {
        "summary": summary,
        "query": query,
        "indication": indication or None,
        "momentum": momentum,
        "momentum_score": momentum_score,
        "trials": {
            "total_count": trial_result.get("total_count", 0),
            "phase_distribution": phase_dist,
            "status_distribution": status_dist,
            "records": trials,
            "error": trial_result.get("error"),
        },
        "literature": {
            "pubmed_total": pubmed_result.get("total_count", 0),
            "pubmed_recent_last_2y": recent_pubmed,
            "openalex_total": openalex_result.get("total_count", 0),
            "openalex_recent_last_2y": recent_openalex,
            "pubmed_top": pubmed_articles[:max_papers],
            "openalex_top": openalex_articles[:max_papers],
            "pubmed_error": pubmed_result.get("error"),
            "openalex_error": openalex_result.get("error"),
        },
    }


@registry.register(
    name="intel.competitor_snapshot",
    description="Generate a one-shot competitor snapshot for a target and indication",
    category="intel",
    parameters={
        "gene": "Target gene symbol (e.g., LRRK2, IL23R)",
        "indication": "Optional indication filter",
        "max_programs": "Maximum trial/program records to include (default 15)",
    },
    usage_guide=(
        "Use for decision meetings and external positioning. Summarizes active sponsors, phases, "
        "mechanism diversity, and top benchmark endpoints around a target."
    ),
)
def competitor_snapshot(
    gene: str,
    indication: str = "",
    max_programs: int = 15,
    **kwargs,
) -> dict:
    """Build a compact competitor snapshot using clinical and target landscape tools."""
    del kwargs
    if not gene or not gene.strip():
        return {"summary": "gene is required.", "error": "missing_gene"}

    from ct.tools.clinical import competitive_landscape, trial_design_benchmark

    max_programs = max(1, min(int(max_programs or 15), 50))
    landscape = competitive_landscape(gene=gene.strip(), indication=indication.strip())
    benchmark = trial_design_benchmark(
        query=f"{gene} {indication}".strip(),
        max_results=min(100, max_programs * 2),
    )

    if "error" in landscape and "error" in benchmark:
        return {
            "summary": f"Competitor snapshot failed for {gene}: upstream sources unavailable.",
            "error": "snapshot_failed",
            "sources": {
                "landscape_error": landscape.get("error"),
                "benchmark_error": benchmark.get("error"),
            },
        }

    trial_records = ((landscape.get("trials") or {}).get("top_trials") or [])[:max_programs]
    sponsors = {}
    for trial in trial_records:
        sponsor = str(trial.get("sponsor", "")).strip()
        if sponsor:
            sponsors[sponsor] = sponsors.get(sponsor, 0) + 1
    top_sponsors = sorted(sponsors.items(), key=lambda kv: kv[1], reverse=True)[:10]

    phase_dist = ((landscape.get("trials") or {}).get("phase_distribution") or {})
    chembl = (landscape.get("chembl") or {})
    ot = (landscape.get("open_targets") or {})
    top_endpoints = (benchmark.get("top_primary_endpoints") or [])[:5]

    differentiation_flags = []
    if _to_int(phase_dist.get("PHASE3", 0), 0) == 0:
        differentiation_flags.append("No Phase 3 pressure detected in returned trial window.")
    if _to_int(chembl.get("unique_compounds", 0), 0) < 10:
        differentiation_flags.append("Limited small-molecule density; potential white space.")
    if _to_int(ot.get("n_known_drugs", 0), 0) == 0:
        differentiation_flags.append("No known drugs in Open Targets snapshot for this target.")

    summary = (
        f"Competitor snapshot for {gene}{f' in {indication}' if indication else ''}: "
        f"{_to_int((landscape.get('trials') or {}).get('total_count', 0), 0)} trial records, "
        f"{_to_int(chembl.get('unique_compounds', 0), 0)} ChEMBL compounds, "
        f"{_to_int(ot.get('n_known_drugs', 0), 0)} known drugs."
    )

    return {
        "summary": summary,
        "gene": gene,
        "indication": indication or None,
        "top_sponsors": [{"sponsor": name, "trial_count": count} for name, count in top_sponsors],
        "phase_distribution": phase_dist,
        "top_primary_endpoints": top_endpoints,
        "mechanism_classes": sorted(chembl.get("moa_types", []) or []),
        "differentiation_flags": differentiation_flags,
        "programs": trial_records,
        "landscape": landscape,
        "benchmark": benchmark,
    }
