"""
Network biology tools: protein-protein interaction analysis (STRING) and pathway crosstalk (Reactome).

These are REST API wrappers -- no local data required.
"""

from ct.tools import registry
from ct.tools.http_client import request, request_json


def _coerce_gene_list(value) -> list[str]:
    """Normalize gene input from str/list/tuple/set into a clean symbol list."""
    if value is None:
        return []

    items = []
    if isinstance(value, str):
        # Accept comma, semicolon, newline, or pipe separated strings.
        for chunk in value.replace("\n", ",").replace(";", ",").replace("|", ",").split(","):
            token = str(chunk).strip()
            if token:
                items.append(token)
    elif isinstance(value, (list, tuple, set)):
        for entry in value:
            if entry is None:
                continue
            token = str(entry).strip()
            if token:
                items.append(token)
    else:
        token = str(value).strip()
        if token:
            items.append(token)

    # De-duplicate while preserving order.
    seen = set()
    genes = []
    for gene in items:
        if gene in seen:
            continue
        seen.add(gene)
        genes.append(gene)
    return genes


@registry.register(
    name="network.ppi_analysis",
    description="Analyze protein-protein interaction network for a gene using STRING database",
    category="network",
    parameters={
        "gene": "Gene symbol or comma-separated list (e.g. 'CRBN' or 'CRBN,DDB1,CUL4A')",
        "min_score": "Minimum interaction confidence score 0-1 (default 0.4 = medium)",
        "network_depth": "1=direct partners only, 2=partners of partners (default 1)",
    },
    usage_guide="You want to understand what proteins interact with a target — maps the interaction neighborhood using STRING. Use for target validation, mechanism exploration, and finding co-complex members.",
)
def ppi_analysis(gene: str, min_score: float = 0.4, network_depth: int = 1, **kwargs) -> dict:
    """Analyze protein-protein interaction network via STRING API.

    Retrieves direct interaction partners and optionally second-shell neighbors.
    Computes network statistics and runs functional enrichment on the interactor set.
    """
    genes = _coerce_gene_list(gene)
    if not genes:
        return {"error": "No gene symbols provided", "summary": "No gene symbols provided"}
    string_score = int(min_score * 1000)  # STRING uses 0-1000 scale
    base = "https://string-db.org/api/json"

    # Step 1: Get direct interaction network
    interactions, error = request_json(
        "GET",
        f"{base}/network",
        params={
            "identifiers": "\r".join(genes),
            "species": 9606,
            "required_score": string_score,
            "caller_identity": "ct-celltype",
        },
        timeout=15,
        retries=2,
    )
    if error:
        return {"error": f"STRING network query failed: {error}", "summary": f"STRING network query failed: {error}"}
    if not interactions:
        return {
            "summary": f"No interactions found for {', '.join(genes)} at score >= {min_score}",
            "query_genes": genes,
            "interactions": [],
            "network_stats": {"node_count": len(genes), "edge_count": 0},
        }

    # Parse interactions
    edges = []
    all_nodes = set(genes)
    for ix in interactions:
        a = ix.get("preferredName_A", ix.get("stringId_A", ""))
        b = ix.get("preferredName_B", ix.get("stringId_B", ""))
        score = round(ix.get("score", 0), 3)
        edges.append({
            "gene_a": a,
            "gene_b": b,
            "score": score,
            "nscore": round(ix.get("nscore", 0), 3),
            "fscore": round(ix.get("fscore", 0), 3),
            "pscore": round(ix.get("pscore", 0), 3),
            "ascore": round(ix.get("ascore", 0), 3),
            "escore": round(ix.get("escore", 0), 3),
            "dscore": round(ix.get("dscore", 0), 3),
            "tscore": round(ix.get("tscore", 0), 3),
        })
        all_nodes.add(a)
        all_nodes.add(b)

    # Sort by score descending
    edges.sort(key=lambda x: x["score"], reverse=True)

    # Step 2: Depth-2 expansion (partners of partners)
    depth2_edges = []
    if network_depth >= 2:
        # Get first-shell partners (not query genes themselves)
        first_shell = all_nodes - set(genes)
        if first_shell:
            # Query top 10 first-shell partners to keep API calls reasonable
            expand_genes = sorted(first_shell, key=lambda g: max(
                (e["score"] for e in edges if g in (e["gene_a"], e["gene_b"])),
                default=0,
            ), reverse=True)[:10]

            depth2_data, depth2_error = request_json(
                "GET",
                f"{base}/network",
                params={
                    "identifiers": "\r".join(expand_genes),
                    "species": 9606,
                    "required_score": string_score,
                    "caller_identity": "ct-celltype",
                },
                timeout=15,
                retries=2,
            )
            if not depth2_error:
                existing_keys = {tuple(sorted([e["gene_a"], e["gene_b"]])) for e in edges}
                for ix in depth2_data:
                    a = ix.get("preferredName_A", ix.get("stringId_A", ""))
                    b = ix.get("preferredName_B", ix.get("stringId_B", ""))
                    score = round(ix.get("score", 0), 3)
                    # Only include edges not already seen
                    edge_key = tuple(sorted([a, b]))
                    if edge_key not in existing_keys:
                        depth2_edges.append({
                            "gene_a": a,
                            "gene_b": b,
                            "score": score,
                        })
                        all_nodes.add(a)
                        all_nodes.add(b)
                depth2_edges.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: Compute network statistics
    node_count = len(all_nodes)
    edge_count = len(edges) + len(depth2_edges)

    # Degree distribution
    degree = {}
    for e in edges + depth2_edges:
        degree[e["gene_a"]] = degree.get(e["gene_a"], 0) + 1
        degree[e["gene_b"]] = degree.get(e["gene_b"], 0) + 1

    avg_degree = sum(degree.values()) / max(len(degree), 1)

    # Approximate clustering coefficient (fraction of possible triangles)
    # For each node, count edges among its neighbors
    adjacency = {}
    for e in edges + depth2_edges:
        adjacency.setdefault(e["gene_a"], set()).add(e["gene_b"])
        adjacency.setdefault(e["gene_b"], set()).add(e["gene_a"])

    clustering_coefficients = []
    for node, neighbors in adjacency.items():
        n = len(neighbors)
        if n < 2:
            clustering_coefficients.append(0.0)
            continue
        neighbor_list = list(neighbors)
        triangles = 0
        for i in range(len(neighbor_list)):
            for j in range(i + 1, len(neighbor_list)):
                if neighbor_list[j] in adjacency.get(neighbor_list[i], set()):
                    triangles += 1
        possible = n * (n - 1) / 2
        clustering_coefficients.append(triangles / possible if possible > 0 else 0)

    avg_clustering = sum(clustering_coefficients) / max(len(clustering_coefficients), 1)

    # Hub genes (top by degree)
    hub_genes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]

    network_stats = {
        "node_count": node_count,
        "edge_count": edge_count,
        "avg_degree": round(avg_degree, 2),
        "clustering_coefficient": round(avg_clustering, 3),
        "hub_genes": [{"gene": g, "degree": d} for g, d in hub_genes],
    }

    # Step 4: Functional enrichment of the interactor set
    enrichment = []
    interactor_genes = list(all_nodes)
    if len(interactor_genes) >= 2:
        enrich_data, enrich_error = request_json(
            "GET",
            f"{base}/enrichment",
            params={
                "identifiers": "\r".join(interactor_genes),
                "species": 9606,
                "caller_identity": "ct-celltype",
            },
            timeout=15,
            retries=2,
        )
        if not enrich_error:
            for entry in enrich_data:
                enrichment.append({
                    "category": entry.get("category", ""),
                    "term": entry.get("term", ""),
                    "description": entry.get("description", ""),
                    "p_value": entry.get("p_value", 1.0),
                    "fdr": entry.get("fdr", 1.0),
                    "gene_count": entry.get("number_of_genes", 0),
                    "genes": entry.get("preferredNames", ""),
                })

            # Sort by FDR, keep top 20
            enrichment.sort(key=lambda x: x["fdr"])
            enrichment = enrichment[:20]

    # Build summary
    query_set = set(genes)
    seen_partners = set()
    top_partners = []
    for e in edges:
        partner = e["gene_b"] if e["gene_a"] in query_set else e["gene_a"]
        if partner not in seen_partners and partner not in query_set:
            seen_partners.add(partner)
            top_partners.append(partner)
        if len(top_partners) >= 5:
            break
    top_str = ", ".join(top_partners) if top_partners else "none"
    top_pathway = enrichment[0]["description"] if enrichment else "N/A"

    summary = (
        f"PPI network for {', '.join(genes)}: "
        f"{node_count} nodes, {edge_count} edges (score >= {min_score})\n"
        f"Top interactors: {top_str}\n"
        f"Avg clustering coefficient: {avg_clustering:.3f}\n"
        f"Top enriched pathway: {top_pathway}"
    )

    result = {
        "summary": summary,
        "query_genes": genes,
        "interactions": edges[:50],  # Cap to keep response manageable
        "network_stats": network_stats,
        "enrichment": enrichment,
    }
    if depth2_edges:
        result["depth2_interactions"] = depth2_edges[:30]

    return result


@registry.register(
    name="network.pathway_crosstalk",
    description="Analyze pathway membership and crosstalk for a gene set using Reactome",
    category="network",
    parameters={
        "genes": "Comma-separated gene symbols (e.g. 'CRBN,DDB1,CUL4A,RBX1')",
    },
    usage_guide="You want to understand which biological pathways a set of genes participate in and how those pathways overlap. Use for mechanism-of-action analysis and understanding pathway-level effects of perturbations.",
)
def pathway_crosstalk(genes: str, **kwargs) -> dict:
    """Analyze pathway membership and crosstalk via Reactome Content Service.

    Submits gene list for pathway over-representation analysis, then analyzes
    which genes appear in multiple pathways to identify crosstalk nodes.
    """
    gene_list = _coerce_gene_list(genes)
    if not gene_list:
        return {"error": "No gene symbols provided", "summary": "No gene symbols provided"}
    # Reactome analysis endpoint: POST gene list for over-representation
    reactome_url = "https://reactome.org/AnalysisService/identifiers/projection"
    body = "\n".join(gene_list)

    data, error = request_json(
        "POST",
        reactome_url,
        data=body,
        headers={"Content-Type": "text/plain"},
        params={"pageSize": 20, "page": 1},
        timeout=15,
        retries=2,
    )
    if error:
        return {"error": f"Reactome analysis failed: {error}", "summary": f"Reactome analysis failed: {error}"}
    # Parse pathway results
    pathways_raw = data.get("pathways", [])
    pathways = []
    gene_pathway_map = {}  # gene -> list of pathways

    for pw in pathways_raw:
        stid = pw.get("stId", "")
        name = pw.get("name", "")
        p_value = pw.get("entities", {}).get("pValue", 1.0)
        fdr = pw.get("entities", {}).get("fdr", 1.0)
        found = pw.get("entities", {}).get("found", 0)
        total = pw.get("entities", {}).get("total", 0)
        ratio = pw.get("entities", {}).get("ratio", 0)

        pathways.append({
            "pathway_id": stid,
            "name": name,
            "p_value": p_value,
            "fdr": fdr,
            "genes_found": found,
            "genes_total": total,
            "ratio": round(ratio, 4) if ratio else 0,
        })

    # Get identifiers mapping (which input genes map to which pathways)
    # Reactome returns this in the 'identifiers' section
    not_found = data.get("identifiersNotFound", 0)
    found_ids = data.get("foundEntities", 0)

    # Step 2: For each significant pathway, get the participant genes
    # Use Reactome content service to get contained participants
    significant_pathways = [p for p in pathways if p["fdr"] < 0.05][:10]

    for pw in significant_pathways:
        part_resp, part_error = request(
            "GET",
            f"https://reactome.org/ContentService/data/participants/{pw['pathway_id']}",
            headers={"Accept": "application/json"},
            timeout=10,
            raise_for_status=False,
        )
        if part_error or part_resp.status_code != 200:
            pw["matched_input_genes"] = []
            continue
        try:
            participants = part_resp.json()
        except Exception:
            pw["matched_input_genes"] = []
            continue

        pw_genes = set()
        for participant in participants:
            # Each participant has refEntities with gene names
            ref_entities = participant.get("refEntities", [])
            for ref in ref_entities:
                gene_name = ref.get("displayName", "")
                # Reactome format: "UniProt:XXXXX GENE_NAME"
                if " " in gene_name:
                    gene_name = gene_name.split(" ")[-1]
                if gene_name in gene_list:
                    pw_genes.add(gene_name)
                    gene_pathway_map.setdefault(gene_name, []).append(pw["name"])

        pw["matched_input_genes"] = sorted(pw_genes)

    # Crosstalk analysis: genes appearing in multiple pathways
    crosstalk_nodes = []
    for g, pws in gene_pathway_map.items():
        if len(pws) > 1:
            crosstalk_nodes.append({
                "gene": g,
                "pathway_count": len(pws),
                "pathways": pws,
            })
    crosstalk_nodes.sort(key=lambda x: x["pathway_count"], reverse=True)

    # Pathway overlap matrix: count shared genes between pathway pairs
    pathway_overlaps = []
    pathway_gene_sets = {}
    for pw in significant_pathways:
        matched = pw.get("matched_input_genes", [])
        if matched:
            pathway_gene_sets[pw["name"]] = set(matched)

    pw_names = list(pathway_gene_sets.keys())
    for i in range(len(pw_names)):
        for j in range(i + 1, len(pw_names)):
            shared = pathway_gene_sets[pw_names[i]] & pathway_gene_sets[pw_names[j]]
            if shared:
                pathway_overlaps.append({
                    "pathway_a": pw_names[i],
                    "pathway_b": pw_names[j],
                    "shared_genes": sorted(shared),
                    "shared_count": len(shared),
                })
    pathway_overlaps.sort(key=lambda x: x["shared_count"], reverse=True)

    # Build summary
    sig_count = len([p for p in pathways if p["fdr"] < 0.05])
    top_pathway = pathways[0]["name"] if pathways else "N/A"
    top_fdr = pathways[0]["fdr"] if pathways else "N/A"

    summary = (
        f"Reactome pathway analysis for {len(gene_list)} genes: "
        f"{len(pathways)} pathways enriched, {sig_count} significant (FDR < 0.05)\n"
        f"Top pathway: {top_pathway} (FDR={top_fdr})\n"
        f"Crosstalk nodes (multi-pathway genes): {len(crosstalk_nodes)}\n"
        f"Pathway pairs with shared genes: {len(pathway_overlaps)}"
    )

    return {
        "summary": summary,
        "query_genes": gene_list,
        "genes_not_found": not_found,
        "pathways": pathways,
        "crosstalk_nodes": crosstalk_nodes,
        "pathway_overlaps": pathway_overlaps,
    }
