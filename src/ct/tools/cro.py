"""
CRO (Contract Research Organization) tools: search, match, compare, and contact CROs.

PLACEHOLDER IMPLEMENTATION: All CRO data comes from a static JSON file bundled with ct.
This is not a live database — CRO listings, pricing, and turnaround times may be
outdated or incomplete. A real implementation would integrate with CRO directories
or vendor APIs. Treat results as illustrative, not authoritative.
"""

import json
from pathlib import Path
from ct.tools import registry


# Module-level cache for CRO database
_cro_db_cache = None


def _load_cro_db() -> list[dict]:
    """Load and cache the CRO database from JSON."""
    global _cro_db_cache
    if _cro_db_cache is not None:
        return _cro_db_cache
    db_path = Path(__file__).parent.parent / "data" / "cro_database.json"
    with open(db_path) as f:
        _cro_db_cache = json.load(f)
    return _cro_db_cache


def _score_cro(cro: dict, assay_type: str, target: str | None,
               compound: str | None, species: str, scale: str) -> float:
    """Score a CRO for a specific experiment.

    Weights:
      - service match:       0.4
      - capability match:    0.3
      - therapeutic area:    0.2
      - specialty match:     0.1
    """
    score = 0.0

    # Service category match (0.4)
    service_cats = [s["category"] for s in cro.get("services", [])]
    if assay_type in service_cats:
        score += 0.4
    else:
        # Partial match: check if assay_type words appear in any category
        assay_words = set(assay_type.lower().replace("_", " ").split())
        for cat in service_cats:
            cat_words = set(cat.lower().replace("_", " ").split())
            if assay_words & cat_words:
                score += 0.2
                break

    # Capability match (0.3)
    capabilities_text = " ".join(cro.get("capabilities", [])).lower()
    keywords = []
    if assay_type:
        keywords.extend(assay_type.lower().replace("_", " ").split())
    if target:
        keywords.append(target.lower())
    if compound:
        keywords.append(compound.lower())
    # Add TPD-relevant keywords
    tpd_keywords = ["degradation", "protac", "molecular glue", "tpd",
                     "ubiquitin", "e3 ligase", "ternary", "neo-substrate"]
    for kw in keywords:
        if kw in capabilities_text:
            score += 0.15
            break
    for kw in tpd_keywords:
        if kw in capabilities_text:
            score += 0.15
            break

    # Therapeutic area match (0.2) — oncology is default for TPD
    therapeutic_areas = [t.lower() for t in cro.get("therapeutic_areas", [])]
    if "oncology" in therapeutic_areas:
        score += 0.2

    # Specialty match (0.1)
    specialties_text = " ".join(cro.get("specialties", [])).lower()
    specialty_keywords = keywords + ["degradation", "tpd", "protac", "molecular glue"]
    for kw in specialty_keywords:
        if kw in specialties_text:
            score += 0.1
            break

    # Scale bonus: large CROs score slightly higher for large-scale work
    if scale == "large" and cro.get("size") == "large":
        score += 0.05
    elif scale == "small" and cro.get("size") in ("small", "medium"):
        score += 0.05

    return round(min(score, 1.0), 3)


@registry.register(
    name="cro.search",
    description="Search built-in CRO directory by keyword, service type, or therapeutic area",
    category="cro",
    parameters={
        "query": "Free-text search across CRO names, capabilities, and specialties",
        "service_type": "Filter by service category (e.g. cell_based_assay, structural_biology)",
        "therapeutic_area": "Filter by therapeutic area (e.g. oncology, neuroscience)",
    },
    usage_guide="You need to find CROs that offer a specific service or work in a therapeutic area. Use for initial CRO discovery before match_experiment.",
)
def search(query: str, service_type: str = None,
           therapeutic_area: str = None, **kwargs) -> dict:
    """Full-text search across CRO database with optional filtering."""
    db = _load_cro_db()
    query_lower = query.lower()
    results = []

    for cro in db:
        # Build searchable text
        searchable = " ".join([
            cro["name"],
            " ".join(cro.get("capabilities", [])),
            " ".join(cro.get("specialties", [])),
            " ".join(cro.get("therapeutic_areas", [])),
        ]).lower()

        if query_lower not in searchable:
            continue

        # Filter by service_type
        if service_type:
            service_cats = [s["category"] for s in cro.get("services", [])]
            if service_type not in service_cats:
                continue

        # Filter by therapeutic_area
        if therapeutic_area:
            areas = [t.lower() for t in cro.get("therapeutic_areas", [])]
            if therapeutic_area.lower() not in areas:
                continue

        results.append({
            "id": cro["id"],
            "name": cro["name"],
            "headquarters": cro["headquarters"],
            "size": cro["size"],
            "capabilities": cro.get("capabilities", []),
            "specialties": cro.get("specialties", []),
            "services": [s["category"] for s in cro.get("services", [])],
        })

    return {
        "summary": f"[PLACEHOLDER] CRO search for '{query}': {len(results)} matches from static directory (not live data)",
        "placeholder": True,
        "query": query,
        "filters": {"service_type": service_type, "therapeutic_area": therapeutic_area},
        "results": results,
    }


@registry.register(
    name="cro.match_experiment",
    description="Rank CROs from built-in directory by fit for a specific experiment type, target, and compound",
    category="cro",
    parameters={
        "assay_type": "Experiment/assay type (e.g. cell_based_assay, structural_biology, in_vivo_efficacy)",
        "target": "Target protein or gene (optional)",
        "compound": "Compound name or class (optional)",
        "species": "Species for the experiment (default: human)",
        "scale": "Scale: small, medium, large (default: small)",
    },
    usage_guide="You have a specific experiment in mind and need to find the best-fit CRO. Run after experiment.design_assay to match the assay to capable vendors.",
)
def match_experiment(assay_type: str, target: str = None,
                     compound: str = None, species: str = "human",
                     scale: str = "small", **kwargs) -> dict:
    """Score and rank all CROs for a specific experiment."""
    db = _load_cro_db()
    scored = []

    for cro in db:
        score = _score_cro(cro, assay_type, target, compound, species, scale)
        if score > 0:
            # Find matching service for pricing/turnaround
            matching_service = None
            for svc in cro.get("services", []):
                if svc["category"] == assay_type:
                    matching_service = svc
                    break

            entry = {
                "id": cro["id"],
                "name": cro["name"],
                "score": score,
                "headquarters": cro["headquarters"],
                "size": cro["size"],
                "relevant_capabilities": [
                    c for c in cro.get("capabilities", [])
                    if any(kw in c.lower() for kw in [
                        assay_type.lower().replace("_", " "),
                        "degradation", "protac", "tpd", "glue",
                    ] + ([target.lower()] if target else []))
                ],
            }
            if matching_service:
                entry["turnaround_days"] = matching_service["turnaround_days"]
                entry["price_range"] = matching_service["price_range"]

            scored.append(entry)

    scored.sort(key=lambda x: x["score"], reverse=True)

    top_names = ", ".join(s["name"] for s in scored[:3]) if scored else "none"
    return {
        "summary": (
            f"[PLACEHOLDER] CRO matching for {assay_type}"
            + (f" (target={target})" if target else "")
            + (f" (compound={compound})" if compound else "")
            + f": {len(scored)} CROs scored from static directory (not live data), top matches: {top_names}"
        ),
        "placeholder": True,
        "assay_type": assay_type,
        "target": target,
        "compound": compound,
        "species": species,
        "scale": scale,
        "ranked_cros": scored,
    }


@registry.register(
    name="cro.compare",
    description="Side-by-side comparison of selected CROs from built-in directory on services, pricing, and capabilities",
    category="cro",
    parameters={
        "cro_ids": "List of CRO IDs to compare (e.g. ['reaction-biology', 'promega'])",
    },
    usage_guide="You have shortlisted CROs and need to compare them on services, pricing, and capabilities before making a decision.",
)
def compare(cro_ids: list[str], **kwargs) -> dict:
    """Compare selected CROs side-by-side."""
    db = _load_cro_db()
    id_to_cro = {cro["id"]: cro for cro in db}

    comparisons = []
    not_found = []

    for cro_id in cro_ids:
        cro = id_to_cro.get(cro_id)
        if not cro:
            not_found.append(cro_id)
            continue

        services_summary = {
            s["category"]: {
                "turnaround_days": s["turnaround_days"],
                "price_range": s["price_range"],
            }
            for s in cro.get("services", [])
        }

        comparisons.append({
            "id": cro["id"],
            "name": cro["name"],
            "headquarters": cro["headquarters"],
            "size": cro["size"],
            "website": cro["website"],
            "services": services_summary,
            "capabilities": cro.get("capabilities", []),
            "specialties": cro.get("specialties", []),
            "therapeutic_areas": cro.get("therapeutic_areas", []),
        })

    # Build comparison matrix for common service categories
    all_categories = set()
    for comp in comparisons:
        all_categories.update(comp["services"].keys())

    names = [c["name"] for c in comparisons]
    summary = f"[PLACEHOLDER] Comparison of {len(comparisons)} CROs from static directory (not live data): {', '.join(names)}"
    if not_found:
        summary += f" (not found: {', '.join(not_found)})"

    return {
        "summary": summary,
        "placeholder": True,
        "comparisons": comparisons,
        "service_categories": sorted(all_categories),
        "not_found": not_found,
    }


@registry.register(
    name="cro.draft_inquiry",
    description="Draft a professional inquiry email to a CRO (from built-in directory) for a specific experiment",
    category="cro",
    parameters={
        "cro_id": "CRO identifier (e.g. 'reaction-biology')",
        "experiment_description": "Description of the experiment / assay needed",
        "compound": "Compound name or identifier (optional)",
        "target": "Target protein or gene (optional)",
        "timeline": "Desired timeline (optional, e.g. '3 months')",
    },
    usage_guide="You've selected a CRO and need to draft a professional inquiry email. Run after cro.match_experiment to contact the top-ranked CRO.",
)
def draft_inquiry(cro_id: str, experiment_description: str,
                  compound: str = None, target: str = None,
                  timeline: str = None, **kwargs) -> dict:
    """Generate a professional inquiry email to a CRO."""
    db = _load_cro_db()
    id_to_cro = {cro["id"]: cro for cro in db}

    cro = id_to_cro.get(cro_id)
    if not cro:
        return {"error": f"CRO '{cro_id}' not found in database", "summary": f"CRO '{cro_id}' not found in database"}
    subject = f"Inquiry: {experiment_description[:60]}"
    if compound:
        subject = f"Inquiry: {experiment_description[:40]} — {compound}"

    body_parts = [
        f"Dear {cro['name']} Team,",
        "",
        f"I am writing to inquire about your services for the following study:",
        "",
        f"**Experiment:** {experiment_description}",
    ]
    if target:
        body_parts.append(f"**Target:** {target}")
    if compound:
        body_parts.append(f"**Compound:** {compound}")
    if timeline:
        body_parts.append(f"**Desired Timeline:** {timeline}")

    body_parts.extend([
        "",
        "We are interested in understanding:",
        "1. Your capacity and availability for this type of study",
        "2. Estimated timeline and pricing",
        "3. Any relevant experience with similar projects (particularly in targeted protein degradation)",
        "4. Required compound quantity and format",
        "",
        "Could you please provide a preliminary quote and study outline? We would also welcome a call to discuss the details.",
        "",
        "Thank you for your time.",
        "",
        "Best regards",
    ])

    body = "\n".join(body_parts)

    return {
        "summary": f"[PLACEHOLDER] Drafted inquiry to {cro['name']} ({cro['contact_email']}) re: {experiment_description[:50]} — verify CRO contact details before sending",
        "placeholder": True,
        "cro_id": cro_id,
        "cro_name": cro["name"],
        "to_email": cro["contact_email"],
        "subject": subject,
        "body": body,
    }


@registry.register(
    name="cro.send_inquiry",
    description="Send a drafted inquiry email to a CRO (dry_run by default)",
    category="cro",
    parameters={
        "cro_id": "CRO identifier",
        "subject": "Email subject line",
        "body": "Email body text",
        "dry_run": "If True (default), simulate sending without actually delivering",
    },
    usage_guide="You have a finalized inquiry email and want to send it to the CRO. Always runs in dry_run mode unless explicitly overridden.",
)
def send_inquiry(cro_id: str, subject: str, body: str,
                 dry_run: bool = True, **kwargs) -> dict:
    """Send an inquiry email to a CRO."""
    db = _load_cro_db()
    id_to_cro = {cro["id"]: cro for cro in db}

    cro = id_to_cro.get(cro_id)
    if not cro:
        return {"error": f"CRO '{cro_id}' not found in database", "summary": f"CRO '{cro_id}' not found in database"}
    to_email = cro["contact_email"]

    if dry_run:
        return {
            "summary": f"[DRY RUN] Would send email to {cro['name']} ({to_email}): {subject}",
            "dry_run": True,
            "to_email": to_email,
            "cro_name": cro["name"],
            "subject": subject,
            "body": body,
        }

    # Attempt to send via notification module
    try:
        from ct.tools.notification import send_email
        result = send_email(to=to_email, subject=subject, body=body)
        return {
            "summary": f"Sent inquiry to {cro['name']} ({to_email}): {subject}",
            "dry_run": False,
            "to_email": to_email,
            "cro_name": cro["name"],
            "subject": subject,
            "send_result": result,
        }
    except ImportError:
        return {
            "summary": f"[FAILED] notification module not available — email not sent to {to_email}",
            "dry_run": False,
            "error": "notification.send_email not available. Install notification module or use dry_run=True.",
            "to_email": to_email,
            "subject": subject,
            "body": body,
        }
