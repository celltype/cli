"""
Literature and database tools: PubMed, OpenAlex, ChEMBL API queries.

These are REST API wrappers -- no local data required.
"""

import re as _re

from ct.tools import registry
from ct.tools.http_client import request, request_json


def _normalize_pubmed_query(query: str) -> str:
    """Normalize a PubMed query for NCBI E-utilities.

    - Uppercase standalone boolean operators (and→AND, or→OR, not→NOT)
    - Preserve text inside quoted phrases
    - Normalize whitespace
    """
    # Split on quoted phrases to preserve them
    parts = _re.split(r'(".*?")', query)
    normalized = []
    for i, part in enumerate(parts):
        if part.startswith('"'):
            # Quoted phrase — keep as-is
            normalized.append(part)
        else:
            # Uppercase standalone boolean operators
            part = _re.sub(r'\b(and)\b', 'AND', part, flags=_re.IGNORECASE)
            part = _re.sub(r'\b(or)\b', 'OR', part, flags=_re.IGNORECASE)
            part = _re.sub(r'\b(not)\b', 'NOT', part, flags=_re.IGNORECASE)
            normalized.append(part)
    result = "".join(normalized)
    # Normalize whitespace
    return " ".join(result.split())


def _simplify_query(query: str) -> list[str]:
    """Generate progressively simpler queries by dropping terms.

    PubMed ANDs all terms by default, so long queries (8+ terms) often return
    zero results. We try shorter versions as fallbacks.
    """
    # Remove parenthesized groups and quoted phrases for counting
    clean = _re.sub(r'\([^)]*\)', '', query)
    clean = _re.sub(r'"[^"]*"', '', clean)
    # Split on whitespace, ignoring boolean operators
    words = [w for w in query.split() if w.upper() not in ("AND", "OR", "NOT")]

    if len(words) <= 4:
        return []  # Already short enough

    # Try keeping just the most distinctive terms (drop common qualifiers)
    # Strategy: take first N words from the original query
    shorter = []
    if len(words) > 6:
        shorter.append(" ".join(words[:5]))
    if len(words) > 4:
        shorter.append(" ".join(words[:3]))
    return shorter


@registry.register(
    name="literature.pubmed_search",
    description="Search PubMed for publications via NCBI E-utilities API",
    category="literature",
    parameters={
        "query": "Search query (e.g. 'molecular glue degrader CRBN')",
        "max_results": "Maximum number of results (default 20)",
    },
    usage_guide="You need recent publications on a target, compound, or mechanism. Use to support or challenge computational findings with published evidence.",
)
def pubmed_search(query: str, max_results: int = 20, **kwargs) -> dict:
    """Search PubMed using NCBI E-utilities (ESearch + ESummary)."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Step 1: ESearch to get PMIDs
    search_url = f"{base}/esearch.fcgi"
    normalized = _normalize_pubmed_query(query)
    params = {
        "db": "pubmed",
        "term": normalized,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }

    search_data, error = request_json(
        "GET",
        search_url,
        params=params,
        timeout=30,
        retries=2,
    )
    if error:
        return {"error": f"PubMed search failed: {error}", "summary": f"PubMed search failed: {error}"}
    result = search_data.get("esearchresult", {})
    pmids = result.get("idlist", [])
    total_count = int(result.get("count", 0))

    # If no results with a long query, retry with progressively simpler versions
    used_query = query
    if not pmids:
        for simpler in _simplify_query(query):
            params["term"] = _normalize_pubmed_query(simpler)
            search_data, fallback_error = request_json(
                "GET",
                search_url,
                params=params,
                timeout=30,
                retries=2,
            )
            if fallback_error:
                continue
            result = search_data.get("esearchresult", {})
            pmids = result.get("idlist", [])
            total_count = int(result.get("count", 0))
            if pmids:
                used_query = simpler
                break

    if not pmids:
        return {"summary": f"No results for '{query}'", "total_count": 0, "articles": []}

    # Step 2: ESummary for article details
    summary_url = f"{base}/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }

    summary_data, error = request_json(
        "GET",
        summary_url,
        params=params,
        timeout=30,
        retries=2,
    )
    if error:
        return {"error": f"PubMed summary failed: {error}", "summary": f"PubMed summary failed: {error}"}
    articles = []
    for pmid in pmids:
        info = summary_data.get("result", {}).get(pmid, {})
        if not info or pmid == "uids":
            continue

        authors = info.get("authors", [])
        first_author = authors[0].get("name", "") if authors else ""

        articles.append({
            "pmid": pmid,
            "title": info.get("title", ""),
            "first_author": first_author,
            "journal": info.get("source", ""),
            "pub_date": info.get("pubdate", ""),
            "doi": next((a.get("value", "") for a in info.get("articleids", [])
                        if a.get("idtype") == "doi"), ""),
        })

    summary = f"PubMed search '{used_query}': {total_count} total, showing {len(articles)}"
    if used_query != query:
        summary += f" (simplified from: '{query}')"

    return {
        "summary": summary,
        "query": used_query,
        "original_query": query,
        "total_count": total_count,
        "articles": articles,
    }


@registry.register(
    name="literature.chembl_query",
    description="Query ChEMBL for compound bioactivity, targets, and SAR data",
    category="literature",
    parameters={
        "query": "Compound name, SMILES, or ChEMBL ID",
        "query_type": "'molecule', 'target', 'activity', or 'similarity'",
        "max_results": "Maximum results (default 20)",
    },
    usage_guide="You want to look up known bioactivity data, find related compounds, or check if a target has known ligands. Use ChEMBL for chemical and pharmacological context.",
)
def chembl_query(query: str, query_type: str = "molecule", max_results: int = 20, **kwargs) -> dict:
    """Query ChEMBL database for compound/target/activity data."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    query = str(query or "").strip()
    query_type_raw = str(query_type or "molecule").strip().lower()
    query_type_aliases = {
        "compound": "molecule",
        "drug": "molecule",
        "molecules": "molecule",
        "compounds": "molecule",
        "protein": "target",
        "gene": "target",
        "bioactivity": "activity",
        "activities": "activity",
        "similar": "similarity",
    }
    query_type = query_type_aliases.get(query_type_raw, query_type_raw)
    base = "https://www.ebi.ac.uk/chembl/api/data"
    headers = {"Accept": "application/json"}

    # Accept common aliases
    if query_type == "compound":
        query_type = "molecule"

    try:
        if query_type == "molecule":
            url = f"{base}/molecule/search.json"
            params = {"q": query, "limit": max_results}
            data, error = request_json(
                "GET",
                url,
                params=params,
                headers=headers,
                timeout=30,
                retries=2,
            )
            if error:
                return {"error": f"ChEMBL query failed: {error}", "summary": f"ChEMBL query failed: {error}"}
            molecules = []
            for mol in data.get("molecules", []):
                props = mol.get("molecule_properties", {}) or {}
                molecules.append({
                    "chembl_id": mol.get("molecule_chembl_id", ""),
                    "pref_name": mol.get("pref_name", ""),
                    "molecule_type": mol.get("molecule_type", ""),
                    "max_phase": mol.get("max_phase", 0),
                    "mw": props.get("full_mwt"),
                    "logp": props.get("alogp"),
                    "smiles": (mol.get("molecule_structures", {}) or {}).get("canonical_smiles", ""),
                })

            return {
                "summary": f"ChEMBL molecule search '{query}': {len(molecules)} hits",
                "query": query,
                "molecules": molecules,
            }

        elif query_type == "target":
            url = f"{base}/target/search.json"
            params = {"q": query, "limit": max_results}
            data, error = request_json(
                "GET",
                url,
                params=params,
                headers=headers,
                timeout=30,
                retries=2,
            )
            if error:
                return {"error": f"ChEMBL query failed: {error}", "summary": f"ChEMBL query failed: {error}"}
            targets = []
            for tgt in data.get("targets", []):
                targets.append({
                    "chembl_id": tgt.get("target_chembl_id", ""),
                    "pref_name": tgt.get("pref_name", ""),
                    "organism": tgt.get("organism", ""),
                    "target_type": tgt.get("target_type", ""),
                })

            return {
                "summary": f"ChEMBL target search '{query}': {len(targets)} hits",
                "query": query,
                "targets": targets,
            }

        elif query_type == "activity":
            # Support both target and molecule ChEMBL IDs
            # If query starts with CHEMBL, determine if it's a target or molecule
            # Also support compound names: resolve to molecule ChEMBL ID first
            molecule_id = None
            target_id = None

            if query.startswith("CHEMBL"):
                # Could be target or molecule — try molecule activity first
                molecule_id = query
            else:
                # Try to resolve compound name to ChEMBL molecule ID
                search_url = f"{base}/molecule/search.json"
                search_params = {"q": query, "limit": 5}
                search_data, search_error = request_json(
                    "GET",
                    search_url,
                    params=search_params,
                    headers=headers,
                    timeout=30,
                    retries=2,
                )
                if not search_error:
                    mols = search_data.get("molecules", [])
                    if mols:
                        molecule_id = mols[0].get("molecule_chembl_id", "")

            # Query activities by molecule ChEMBL ID
            activities = []
            if molecule_id:
                url = f"{base}/activity.json"
                params = {
                    "molecule_chembl_id": molecule_id,
                    "limit": max_results,
                }
                data, error = request_json(
                    "GET",
                    url,
                    params=params,
                    headers=headers,
                    timeout=30,
                    retries=2,
                )
                if not error:
                    for act in data.get("activities", []):
                        activities.append({
                            "molecule_chembl_id": act.get("molecule_chembl_id", ""),
                            "molecule_name": act.get("molecule_pref_name", ""),
                            "target_chembl_id": act.get("target_chembl_id", ""),
                            "target_name": act.get("target_pref_name", ""),
                            "standard_type": act.get("standard_type", ""),
                            "standard_value": act.get("standard_value"),
                            "standard_units": act.get("standard_units", ""),
                            "pchembl_value": act.get("pchembl_value"),
                            "assay_type": act.get("assay_type", ""),
                            "assay_description": (act.get("assay_description", "") or "")[:200],
                        })

            # If no results from molecule lookup, try target lookup
            if not activities:
                target_id = query if query.startswith("CHEMBL") else None
                if target_id:
                    url = f"{base}/activity.json"
                    params = {
                        "target_chembl_id": target_id,
                        "limit": max_results,
                        "standard_type__in": "IC50,Ki,Kd,EC50",
                    }
                    data, error = request_json(
                        "GET",
                        url,
                        params=params,
                        headers=headers,
                        timeout=30,
                        retries=2,
                    )
                    if not error:
                        for act in data.get("activities", []):
                            activities.append({
                                "molecule_chembl_id": act.get("molecule_chembl_id", ""),
                                "molecule_name": act.get("molecule_pref_name", ""),
                                "target_chembl_id": act.get("target_chembl_id", ""),
                                "target_name": act.get("target_pref_name", ""),
                                "standard_type": act.get("standard_type", ""),
                                "standard_value": act.get("standard_value"),
                                "standard_units": act.get("standard_units", ""),
                                "pchembl_value": act.get("pchembl_value"),
                                "assay_type": act.get("assay_type", ""),
                                "assay_description": (act.get("assay_description", "") or "")[:200],
                            })

            resolved_id = molecule_id or target_id or query
            return {
                "summary": f"ChEMBL activities for {query} ({resolved_id}): {len(activities)} results",
                "query": query,
                "chembl_id": resolved_id,
                "activities": activities,
            }

        elif query_type == "similarity":
            url = f"{base}/similarity/{query}/70.json"
            params = {"limit": max_results}
            data, error = request_json(
                "GET",
                url,
                params=params,
                headers=headers,
                timeout=30,
                retries=2,
            )
            if error:
                return {"error": f"ChEMBL query failed: {error}", "summary": f"ChEMBL query failed: {error}"}
            hits = []
            for mol in data.get("molecules", []):
                hits.append({
                    "chembl_id": mol.get("molecule_chembl_id", ""),
                    "pref_name": mol.get("pref_name", ""),
                    "similarity": mol.get("similarity", 0),
                    "smiles": (mol.get("molecule_structures", {}) or {}).get("canonical_smiles", ""),
                })

            return {
                "summary": f"ChEMBL similarity search: {len(hits)} hits (>70% similar)",
                "query": query,
                "hits": hits,
            }

        else:
            return {"error": f"Unknown query_type: {query_type_raw}. Use 'molecule', 'target', 'activity', or 'similarity'", "summary": f"Unknown query_type: {query_type_raw}. Use 'molecule', 'target', 'activity', or 'similarity'"}
    except Exception as e:
        return {"error": f"ChEMBL query failed: {e}", "summary": f"ChEMBL query failed: {e}"}
@registry.register(
    name="literature.openalex_search",
    description="Search OpenAlex for academic publications with citation data and open access links",
    category="literature",
    parameters={
        "query": "Search query",
        "max_results": "Maximum results (default 20)",
    },
    usage_guide="You want academic publications with citation metrics and open access links. Broader than PubMed — covers all scientific literature. Use for comprehensive literature reviews.",
)
def openalex_search(query: str, max_results: int = 20, **kwargs) -> dict:
    """Search OpenAlex for publications with citation metrics."""
    try:
        import httpx
    except ImportError:
        return {"error": "httpx required (pip install httpx)", "summary": "httpx required (pip install httpx)"}
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": max_results,
        "sort": "relevance_score:desc",
        "mailto": "ct@celltype.bio",
    }

    data, error = request_json(
        "GET",
        url,
        params=params,
        timeout=30,
        retries=2,
    )
    if error:
        return {"error": f"OpenAlex search failed: {error}", "summary": f"OpenAlex search failed: {error}"}
    results_data = data.get("results", [])
    total_count = data.get("meta", {}).get("count", 0)

    articles = []
    for work in results_data:
        authorships = work.get("authorships", [])
        first_author = ""
        if authorships:
            author_info = authorships[0].get("author", {})
            first_author = author_info.get("display_name", "")

        primary_loc = work.get("primary_location") or {}
        source = primary_loc.get("source") or {}

        articles.append({
            "title": work.get("title", ""),
            "first_author": first_author,
            "publication_year": work.get("publication_year"),
            "cited_by_count": work.get("cited_by_count", 0),
            "doi": work.get("doi", ""),
            "open_access": (work.get("open_access") or {}).get("is_oa", False),
            "source": source.get("display_name", ""),
            "type": work.get("type", ""),
        })

    return {
        "summary": f"OpenAlex search '{query}': {total_count} total, showing {len(articles)}",
        "query": query,
        "total_count": total_count,
        "articles": articles,
    }


@registry.register(
    name="literature.patent_search",
    description="Search patent databases for drug discovery-relevant patents (Lens.org, EPO OPS, or PubMed fallback)",
    category="literature",
    parameters={
        "query": "Patent search query (e.g. 'CRBN molecular glue degrader')",
        "max_results": "Maximum number of results (default 20)",
    },
    usage_guide="You need to find relevant patents for a target, compound class, or technology. Use to assess patent landscape, freedom to operate, or find prior art. Tries Lens.org API first (if api.lens_key configured), then EPO Open Patent Services, then falls back to PubMed patent-related literature.",
)
def patent_search(query: str, max_results: int = 20, **kwargs) -> dict:
    """Search patent databases for drug discovery-relevant patents.

    Uses a tiered approach:
    1. Lens.org Patent API (if API key is configured via api.lens_key)
    2. EPO Open Patent Services (free, no key required for basic search)
    3. PubMed fallback (searches for patent-related publications)
    """
    # Try Lens.org first
    session = kwargs.get("_session", None)
    lens_key = None
    if session and hasattr(session, "config"):
        lens_key = session.config.get("api.lens_key", None)

    if lens_key:
        result = _patent_search_lens(query, max_results, lens_key)
        if result and "error" not in result:
            return result

    # Try EPO OPS (free, no key required)
    result = _patent_search_epo(query, max_results)
    if result and "error" not in result:
        return result

    # Fall back to PubMed patent search
    return _patent_search_pubmed_fallback(query, max_results)


def _patent_search_lens(query: str, max_results: int, api_key: str) -> dict:
    """Search Lens.org Patent API."""
    url = "https://api.lens.org/patent/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": {
            "match": query,
        },
        "size": max_results,
        "sort": [{"relevance": "desc"}],
        "include": [
            "lens_id", "title", "abstract", "applicant",
            "publication_date", "publication_key", "jurisdiction",
            "doc_number", "kind",
        ],
    }

    resp, error = request(
        "POST",
        url,
        json=payload,
        headers=headers,
        timeout=30,
        retries=2,
        raise_for_status=False,
    )
    if error:
        return {"error": f"Lens.org API request failed: {error}", "summary": f"Lens.org API request failed: {error}"}
    if resp.status_code != 200:
        return {"error": f"Lens.org API returned status {resp.status_code}", "summary": f"Lens.org API returned status {resp.status_code}"}
    try:
        data = resp.json()
    except Exception:
        return {"error": "Lens.org API returned invalid JSON", "summary": "Lens.org API returned invalid JSON"}
    results = data.get("data", [])
    total = data.get("total", 0)

    patents = []
    for item in results:
        title_obj = item.get("title", [])
        title = title_obj[0].get("text", "") if title_obj else ""

        abstract_obj = item.get("abstract", [])
        abstract = abstract_obj[0].get("text", "")[:300] if abstract_obj else ""

        applicants = item.get("applicant", [])
        applicant_names = [a.get("name", "") for a in applicants[:3]] if applicants else []

        patents.append({
            "lens_id": item.get("lens_id", ""),
            "title": title,
            "abstract": abstract,
            "applicants": applicant_names,
            "publication_date": item.get("publication_date", ""),
            "doc_number": item.get("doc_number", ""),
            "jurisdiction": item.get("jurisdiction", ""),
            "kind": item.get("kind", ""),
        })

    # Date range for summary
    dates = [p["publication_date"] for p in patents if p["publication_date"]]
    date_range = ""
    if dates:
        years = sorted(set(d[:4] for d in dates if len(d) >= 4))
        if years:
            date_range = f" ({years[0]}-{years[-1]})"

    return {
        "summary": f"Patent search '{query}': {total} total, showing {len(patents)}{date_range}",
        "source": "lens.org",
        "query": query,
        "total_count": total,
        "patents": patents,
    }


def _patent_search_epo(query: str, max_results: int) -> dict:
    """Search EPO Open Patent Services (Espacenet OPS) — free, no key required."""
    import xml.etree.ElementTree as ET

    # EPO OPS biblio search endpoint
    url = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"
    params = {
        "q": query,
        "Range": f"1-{min(max_results, 100)}",
    }
    headers = {
        "Accept": "application/xml",
    }

    resp, error = request(
        "GET",
        url,
        params=params,
        headers=headers,
        timeout=30,
        retries=0,
        raise_for_status=False,
    )
    if error:
        return {"error": f"EPO OPS request failed: {error}", "summary": f"EPO OPS request failed: {error}"}
    if resp.status_code == 404:
        return {"error": "No patents found via EPO OPS", "summary": "No patents found via EPO OPS"}
    if resp.status_code == 403:
        # Rate limited or auth required
        return {"error": "EPO OPS rate limited or requires authentication", "summary": "EPO OPS rate limited or requires authentication"}
    if resp.status_code != 200:
        return {"error": f"EPO OPS returned status {resp.status_code}", "summary": f"EPO OPS returned status {resp.status_code}"}
    # Validate Content-Type before XML parsing
    content_type = ""
    try:
        ct_raw = resp.headers.get("content-type", "")
        if isinstance(ct_raw, str):
            content_type = ct_raw.lower()
    except Exception:
        pass
    if content_type and "xml" not in content_type and "text/plain" not in content_type:
        return {"error": f"EPO OPS returned {content_type}, expected XML", "summary": "EPO OPS returned non-XML response"}

    # Parse XML response
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return {"error": f"Failed to parse EPO OPS XML: {e}", "summary": "Failed to parse EPO patent XML"}

    # EPO OPS XML namespaces
    ns = {
        "ops": "http://ops.epo.org",
        "epo": "http://www.epo.org/exchange",
        "exch": "http://www.epo.org/exchange",
    }

    patents = []
    total_count = 0

    # Try to get total count
    total_elem = root.find(".//ops:biblio-search", ns)
    if total_elem is not None:
        total_count = int(total_elem.get("total-result-count", 0))

    # Extract patent documents
    for doc in root.findall(".//exch:exchange-document", ns):
        doc_id = doc.get("doc-number", "")
        country = doc.get("country", "")
        kind = doc.get("kind", "")

        # Title
        title = ""
        for title_elem in doc.findall(".//exch:invention-title", ns):
            if title_elem.get("lang", "") == "en" or not title:
                title = title_elem.text or ""

        # Applicants
        applicants = []
        for app in doc.findall(".//exch:applicant/exch:applicant-name/exch:name", ns):
            if app.text:
                applicants.append(app.text)

        # Publication date
        pub_date = ""
        for pub_ref in doc.findall(".//exch:publication-reference//exch:date", ns):
            if pub_ref.text:
                pub_date = pub_ref.text
                break

        # Abstract
        abstract = ""
        for abs_elem in doc.findall(".//exch:abstract", ns):
            if abs_elem.get("lang", "") == "en" or not abstract:
                parts = []
                for p in abs_elem.findall(".//exch:p", ns):
                    if p.text:
                        parts.append(p.text)
                if parts:
                    abstract = " ".join(parts)[:300]

        patent_number = f"{country}{doc_id}{kind}" if country else doc_id

        patents.append({
            "patent_number": patent_number,
            "title": title,
            "abstract": abstract,
            "applicants": applicants[:3],
            "publication_date": pub_date,
            "country": country,
            "kind": kind,
        })

    if not patents:
        return {"error": "EPO OPS returned no parseable patents", "summary": "EPO OPS returned no parseable patents"}
    # Date range for summary
    dates = [p["publication_date"] for p in patents if p["publication_date"]]
    date_range = ""
    if dates:
        years = sorted(set(d[:4] for d in dates if len(d) >= 4))
        if years:
            date_range = f" ({years[0]}-{years[-1]})"

    return {
        "summary": f"Patent search '{query}': {total_count} patents found via EPO{date_range}",
        "source": "epo_ops",
        "query": query,
        "total_count": total_count,
        "patents": patents,
    }


def _patent_search_pubmed_fallback(query: str, max_results: int) -> dict:
    """Fall back to PubMed search for patent-related publications."""
    # Add patent-related terms to the query
    patent_query = f"({query}) AND (patent OR intellectual property OR claims OR USPTO OR EPO)"

    result = pubmed_search(query=patent_query, max_results=max_results)

    if "error" in result:
        return result

    # Re-label for clarity
    return {
        "summary": f"Patent search '{query}' (PubMed fallback): {result.get('total_count', 0)} "
                   f"patent-related publications found",
        "source": "pubmed_fallback",
        "query": query,
        "note": "No patent API available — showing patent-related PubMed publications. "
                "Configure api.lens_key for direct patent search via Lens.org.",
        "total_count": result.get("total_count", 0),
        "articles": result.get("articles", []),
    }
