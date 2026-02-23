"""
Protein analysis tools: embedding generation, function prediction, domain annotation.

Uses ESM-2 for embeddings (optional), UniProt API for function data, and InterPro API for domains.
"""

import re

from ct.tools import registry
from ct.tools.http_client import request


@registry.register(
    name="protein.embed",
    description="Generate protein sequence embeddings using ESM-2 (local) or ESMFold API",
    category="protein",
    parameters={
        "sequence": "Amino acid sequence (single-letter code, e.g. 'MKTL...')",
        "model": "Embedding model: 'esm2' (default) or 'esm2_small'",
    },
    usage_guide="You have a protein sequence and need a numerical representation for downstream analysis (similarity, clustering, property prediction). ESM-2 embeddings capture evolutionary and structural information. Use for comparing proteins, predicting function, or as features for ML models.",
)
def embed(sequence: str, model: str = "esm2", **kwargs) -> dict:
    """Generate ESM-2 protein embeddings.

    If torch + fair-esm are installed, generates embeddings locally using
    esm2_t33_650M_UR50D (or esm2_t6_8M_UR50D for 'esm2_small').
    Otherwise, returns an error with install instructions.
    """
    import numpy as np

    # Validate sequence
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    sequence = sequence.strip().upper()
    invalid_chars = set(sequence) - valid_aa - {"X", "U", "B", "Z", "O", "J"}
    if invalid_chars:
        return {
            "error": f"Invalid amino acid characters: {invalid_chars}",
            "summary": f"Sequence contains invalid characters: {invalid_chars}",
        }

    if len(sequence) == 0:
        return {"error": "Empty sequence provided", "summary": "No sequence to embed"}

    if len(sequence) > 2048:
        return {
            "error": f"Sequence too long ({len(sequence)} aa). Max 2048 for ESM-2 t33.",
            "summary": f"Sequence length {len(sequence)} exceeds limit of 2048 residues",
        }

    # Try local ESM-2
    try:
        import torch
        import esm

        if model == "esm2_small":
            esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            repr_layer = 6
            embed_dim = 320
        else:
            esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            repr_layer = 33
            embed_dim = 1280

        esm_model.eval()
        batch_converter = alphabet.get_batch_converter()

        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

        # Extract representations
        token_repr = results["representations"][repr_layer]
        # Remove BOS and EOS tokens: [0, 1:-1]
        per_residue = token_repr[0, 1:len(sequence) + 1].numpy()
        mean_pooled = per_residue.mean(axis=0)

        return {
            "summary": (
                f"ESM-2 embedding for sequence ({len(sequence)} aa): "
                f"{embed_dim}-dim representation generated"
            ),
            "sequence_length": len(sequence),
            "embedding_dim": embed_dim,
            "model": model,
            "embedding_shape": list(per_residue.shape),
            "mean_embedding_stats": {
                "mean": round(float(np.mean(mean_pooled)), 6),
                "std": round(float(np.std(mean_pooled)), 6),
                "min": round(float(np.min(mean_pooled)), 6),
                "max": round(float(np.max(mean_pooled)), 6),
                "norm": round(float(np.linalg.norm(mean_pooled)), 4),
            },
            "per_residue_stats": {
                "mean_norm": round(float(np.mean(np.linalg.norm(per_residue, axis=1))), 4),
                "shape": list(per_residue.shape),
            },
            "computed_locally": True,
        }

    except ImportError:
        return {
            "error": (
                "ESM-2 requires torch and fair-esm. Install with:\n"
                "  pip install torch fair-esm\n"
                "For GPU support: pip install torch --index-url https://download.pytorch.org/whl/cu118"
            ),
            "summary": (
                f"Cannot generate embedding for {len(sequence)} aa sequence — "
                "torch and fair-esm not installed"
            ),
            "sequence_length": len(sequence),
            "computed_locally": False,
        }


@registry.register(
    name="protein.function_predict",
    description="Predict protein function, localization, domains, PTMs, and disease associations from UniProt",
    category="protein",
    parameters={
        "gene": "Gene symbol (e.g. BRCA1) or UniProt ID (e.g. P38398)",
        "sequence": "Amino acid sequence (optional, used for basic analysis if API fails)",
    },
    usage_guide="You need comprehensive protein function information — GO terms, subcellular location, domains, PTMs, disease associations, and tissue specificity. Use for target characterization and understanding protein biology.",
)
def function_predict(gene: str, sequence: str = None, **kwargs) -> dict:
    """Query UniProt for comprehensive protein function data.

    Searches by gene symbol (human) or UniProt accession. Extracts function
    description, subcellular location, GO terms, domains, post-translational
    modifications, disease associations, and tissue specificity.
    """
    # Determine if input is UniProt ID or gene symbol
    is_uniprot_id = (
        len(gene) == 6
        and gene[0].isalpha()
        and all(c.isalnum() for c in gene)
    )

    if is_uniprot_id:
        resp, error = request(
            "GET",
            f"https://rest.uniprot.org/uniprotkb/{gene}.json",
            timeout=15,
            headers={"Accept": "application/json"},
            raise_for_status=False,
        )
        if error:
            return {"error": f"UniProt API error: {error}", "summary": f"Failed to query UniProt for {gene}"}
        if resp.status_code != 200:
            return {
                "error": f"UniProt entry not found for {gene} (HTTP {resp.status_code})",
                "summary": f"No UniProt entry for {gene}",
            }
        try:
            entry = resp.json()
        except Exception:
            return {"error": f"Invalid UniProt response for {gene}", "summary": f"Failed to parse UniProt response for {gene}"}
    else:
        resp, error = request(
            "GET",
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": f"{gene} AND organism_id:9606",
                "format": "json",
                "size": "1",
            },
            timeout=15,
            headers={"Accept": "application/json"},
            raise_for_status=False,
        )
        if error:
            return {"error": f"UniProt API error: {error}", "summary": f"Failed to query UniProt for {gene}"}
        if resp.status_code != 200:
            return {
                "error": f"UniProt search failed (HTTP {resp.status_code})",
                "summary": f"UniProt query failed for {gene}",
            }
        try:
            data = resp.json()
        except Exception:
            return {"error": f"Invalid UniProt response for {gene}", "summary": f"Failed to parse UniProt response for {gene}"}
        results = data.get("results", [])
        if not results:
            return {
                "error": f"No UniProt entry found for gene {gene} in human",
                "summary": f"Gene {gene} not found in UniProt (human)",
            }
        entry = results[0]

    # Extract basic info
    uniprot_id = entry.get("primaryAccession", "")
    protein_desc = entry.get("proteinDescription", {})
    rec_name = protein_desc.get("recommendedName", {})
    protein_name = rec_name.get("fullName", {}).get("value", gene)

    gene_names = entry.get("genes", [])
    gene_symbol = gene_names[0].get("geneName", {}).get("value", gene) if gene_names else gene

    seq_info = entry.get("sequence", {})
    seq_length = seq_info.get("length", 0)

    # Extract comments (function, location, tissue specificity, etc.)
    comments = entry.get("comments", [])

    function_text = ""
    subcellular_locations = []
    tissue_specificity = ""
    disease_associations = []
    catalytic_activity = []

    for comment in comments:
        ct = comment.get("commentType", "")

        if ct == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                function_text = texts[0].get("value", "")

        elif ct == "SUBCELLULAR LOCATION":
            for sl in comment.get("subcellularLocations", []):
                loc = sl.get("location", {}).get("value", "")
                if loc:
                    subcellular_locations.append(loc)

        elif ct == "TISSUE SPECIFICITY":
            texts = comment.get("texts", [])
            if texts:
                tissue_specificity = texts[0].get("value", "")

        elif ct == "DISEASE":
            disease = comment.get("disease", {})
            if disease:
                disease_associations.append({
                    "name": disease.get("diseaseId", ""),
                    "description": disease.get("description", ""),
                    "acronym": disease.get("acronym", ""),
                })

        elif ct == "CATALYTIC ACTIVITY":
            reaction = comment.get("reaction", {})
            if reaction:
                catalytic_activity.append(reaction.get("name", ""))

    # Extract features
    features = entry.get("features", [])
    domains = []
    ptms = []
    active_sites = []
    binding_sites = []

    for feat in features:
        ftype = feat.get("type", "")
        desc = feat.get("description", "")
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")

        if ftype == "Domain":
            domains.append({"name": desc, "start": start, "end": end})
        elif ftype in ("Modified residue", "Glycosylation", "Disulfide bond", "Cross-link", "Lipidation"):
            ptms.append({"type": ftype, "description": desc, "position": start})
        elif ftype == "Active site":
            active_sites.append({"description": desc, "position": start})
        elif ftype == "Binding site":
            binding_sites.append({"description": desc, "start": start, "end": end})

    # Extract GO terms from cross-references
    xrefs = entry.get("uniProtKBCrossReferences", [])
    go_terms = {"biological_process": [], "molecular_function": [], "cellular_component": []}
    for xref in xrefs:
        if xref.get("database") == "GO":
            props = xref.get("properties", [])
            go_id = xref.get("id", "")
            term_name = ""
            term_type = ""
            for p in props:
                if p.get("key") == "GoTerm":
                    val = p.get("value", "")
                    if val.startswith("P:"):
                        term_type = "biological_process"
                        term_name = val[2:]
                    elif val.startswith("F:"):
                        term_type = "molecular_function"
                        term_name = val[2:]
                    elif val.startswith("C:"):
                        term_type = "cellular_component"
                        term_name = val[2:]
            if term_type and term_name:
                go_terms[term_type].append({"id": go_id, "name": term_name})

    # Extract keywords
    keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]

    # Build summary
    location_str = ", ".join(subcellular_locations[:3]) if subcellular_locations else "Unknown"
    domain_str = f"{len(domains)} {'domain' if len(domains) == 1 else 'domains'}"
    if domains:
        domain_names = ", ".join(d["name"] for d in domains[:4])
        domain_str += f" ({domain_names})"

    disease_str = ""
    if disease_associations:
        disease_names = ", ".join(d["name"] for d in disease_associations[:3])
        disease_str = f" Associated with {disease_names}."

    func_short = function_text[:150] + "..." if len(function_text) > 150 else function_text

    summary = (
        f"{gene_symbol} ({uniprot_id}): {func_short} "
        f"{location_str}. {domain_str}.{disease_str}"
    )

    return {
        "summary": summary,
        "uniprot_id": uniprot_id,
        "gene": gene_symbol,
        "protein_name": protein_name,
        "sequence_length": seq_length,
        "function": function_text,
        "subcellular_locations": subcellular_locations,
        "tissue_specificity": tissue_specificity,
        "go_terms": go_terms,
        "domains": domains,
        "ptms": ptms[:30],
        "active_sites": active_sites,
        "binding_sites": binding_sites,
        "disease_associations": disease_associations,
        "catalytic_activity": catalytic_activity,
        "keywords": keywords,
    }


@registry.register(
    name="protein.domain_annotate",
    description="Annotate protein domains, families, and functional sites using InterPro",
    category="protein",
    parameters={
        "gene": "Gene symbol (e.g. TP53) or domain/family keyword (e.g. CAP superfamily)",
        "uniprot_id": "UniProt accession (e.g. P04637) — used directly if provided",
    },
    usage_guide="You need detailed domain architecture for a protein — domain boundaries, family classifications, active sites, binding sites. Can also search InterPro by domain/family keyword when no UniProt accession can be resolved.",
)
def domain_annotate(gene: str = None, uniprot_id: str = None, **kwargs) -> dict:
    """Annotate domains using InterPro API.

    Resolves gene to UniProt ID if needed, then queries InterPro for full
    domain architecture including Pfam, SMART, PROSITE, and other member databases.
    """
    if not gene and not uniprot_id:
        return {
            "error": "Provide either gene symbol or uniprot_id",
            "summary": "No gene or UniProt ID specified",
        }

    non_human_hints = (
        "helminth", "parasite", "schistosoma", "fasciola", "heligmosomoides",
        "nematode", "trematode", "cestode", "worm", "brugia", "filaria",
    )

    def _looks_non_human(text: str) -> bool:
        t = (text or "").lower()
        return any(h in t for h in non_human_hints)

    def _resolve_uniprot(gene_query: str) -> tuple[str, list[str]]:
        attempts: list[str] = []
        search_terms: list[str] = []
        if _looks_non_human(gene_query):
            search_terms.extend([gene_query, f"{gene_query} AND reviewed:true"])
        else:
            search_terms.extend(
                [
                    f"{gene_query} AND organism_id:9606",
                    gene_query,
                ]
            )

        for term in search_terms:
            if term in attempts:
                continue
            attempts.append(term)
            resp, error = request(
                "GET",
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": term,
                    "format": "json",
                    "size": "1",
                    "fields": "accession,gene_names",
                },
                timeout=15,
                headers={"Accept": "application/json"},
                raise_for_status=False,
            )
            if error or resp.status_code != 200:
                continue
            try:
                results = resp.json().get("results", [])
            except Exception:
                results = []
            if results:
                accession = results[0].get("primaryAccession", "")
                if accession:
                    return accession, attempts
        return "", attempts

    def _interpro_keyword_search(term: str) -> dict | None:
        cleaned = " ".join((term or "").split())
        if not cleaned:
            return None

        endpoints = (
            "https://www.ebi.ac.uk/interpro/api/entry/interpro/",
            "https://www.ebi.ac.uk/interpro/api/entry/all/",
        )
        for endpoint in endpoints:
            resp, error = request(
                "GET",
                endpoint,
                params={"search": cleaned, "page_size": "20"},
                timeout=15,
                headers={"Accept": "application/json"},
                raise_for_status=False,
            )
            if error or resp.status_code != 200:
                continue
            try:
                data = resp.json()
            except Exception:
                continue
            results = data.get("results", [])
            if not results:
                continue

            domains = []
            families = []
            for entry in results:
                md = entry.get("metadata", {}) or {}
                etype = md.get("type", "")
                annotation = {
                    "accession": md.get("accession", ""),
                    "name": md.get("name", ""),
                    "type": etype,
                    "source_database": md.get("source_database", ""),
                    "description": (
                        (md.get("description") or [{}])[0].get("text", "")
                        if isinstance(md.get("description"), list)
                        else ""
                    )[:200],
                    "locations": [],
                }
                if etype == "domain":
                    domains.append(annotation)
                elif etype == "family":
                    families.append(annotation)

            return {
                "summary": (
                    f"InterPro keyword search '{cleaned}': "
                    f"{len(domains)} domains, {len(families)} families (no single UniProt mapping)."
                ),
                "gene": gene,
                "uniprot_id": None,
                "n_domains": len(domains),
                "n_families": len(families),
                "n_sites": 0,
                "domains": domains[:30],
                "families": families[:30],
                "sites": [],
                "homologous_superfamilies": [],
                "total_annotations": len(results),
                "mode": "interpro_keyword_search",
            }
        return None

    # InterPro entry accession mode (e.g. IPR014044) for domain-family lookup.
    interpro_accession = None
    if isinstance(uniprot_id, str) and re.fullmatch(r"IPR\d{6,}", uniprot_id.strip().upper() or ""):
        interpro_accession = uniprot_id.strip().upper()
    elif isinstance(gene, str) and re.fullmatch(r"IPR\d{6,}", gene.strip().upper() or ""):
        interpro_accession = gene.strip().upper()

    if interpro_accession:
        resp, error = request(
            "GET",
            f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{interpro_accession}",
            timeout=15,
            headers={"Accept": "application/json"},
            raise_for_status=False,
        )
        if error or resp.status_code != 200:
            # Fallback through keyword search path
            keyword_result = _interpro_keyword_search(interpro_accession)
            if keyword_result is not None:
                return keyword_result
            return {
                "error": f"InterPro entry lookup failed for {interpro_accession}",
                "summary": f"No InterPro entry found for {interpro_accession}",
            }

        try:
            data = resp.json()
        except Exception:
            return {
                "error": f"Invalid InterPro response for {interpro_accession}",
                "summary": f"Failed to parse InterPro response for {interpro_accession}",
            }
        results = data.get("results", [])
        if not results:
            return {
                "error": f"No InterPro entry results for {interpro_accession}",
                "summary": f"No InterPro data for {interpro_accession}",
            }

        domains = []
        families = []
        for entry in results:
            md = entry.get("metadata", {}) or {}
            etype = md.get("type", "")
            annotation = {
                "accession": md.get("accession", ""),
                "name": md.get("name", ""),
                "type": etype,
                "source_database": md.get("source_database", ""),
                "description": (
                    (md.get("description") or [{}])[0].get("text", "")
                    if isinstance(md.get("description"), list)
                    else ""
                )[:200],
                "locations": [],
            }
            if etype == "domain":
                domains.append(annotation)
            elif etype == "family":
                families.append(annotation)
        return {
            "summary": (
                f"InterPro {interpro_accession}: {len(domains)} domains, {len(families)} families."
            ),
            "gene": gene,
            "uniprot_id": None,
            "n_domains": len(domains),
            "n_families": len(families),
            "n_sites": 0,
            "domains": domains,
            "families": families,
            "sites": [],
            "homologous_superfamilies": [],
            "total_annotations": len(results),
            "mode": "interpro_accession_lookup",
        }

    # Resolve gene to UniProt ID if needed
    if not uniprot_id and gene:
        uniprot_id, attempts = _resolve_uniprot(gene)

        if not uniprot_id:
            keyword_result = _interpro_keyword_search(gene)
            if keyword_result is not None:
                return keyword_result
            attempted = "; ".join(attempts[:4])
            return {
                "error": f"Could not resolve gene {gene} to UniProt ID",
                "summary": f"Gene {gene} not found in UniProt search",
                "resolution_attempts": attempts,
                "attempted_query_preview": attempted,
            }

    # Query InterPro for protein domain annotations
    resp, error = request(
        "GET",
        f"https://www.ebi.ac.uk/interpro/api/entry/all/protein/uniprot/{uniprot_id}",
        timeout=15,
        headers={"Accept": "application/json"},
        raise_for_status=False,
    )
    if error:
        # Final fallback: keyword search if a gene/domain term is available.
        if gene:
            keyword_result = _interpro_keyword_search(gene)
            if keyword_result is not None:
                return keyword_result
        return {"error": f"InterPro API error: {error}", "summary": f"Failed to query InterPro for {uniprot_id}"}
    if resp.status_code == 204:
        data = {"results": []}
    elif resp.status_code != 200:
        if gene:
            keyword_result = _interpro_keyword_search(gene)
            if keyword_result is not None:
                return keyword_result
        return {
            "error": f"InterPro query failed for {uniprot_id} (HTTP {resp.status_code})",
            "summary": f"No InterPro data for {uniprot_id}",
        }
    else:
        try:
            data = resp.json()
        except Exception:
            return {"error": f"Invalid InterPro response for {uniprot_id}", "summary": f"Failed to parse InterPro response for {uniprot_id}"}

    # Parse InterPro results
    entries = data.get("results", [])

    domains = []
    families = []
    sites = []
    homologous_superfamilies = []

    for entry in entries:
        metadata = entry.get("metadata", {})
        entry_type = metadata.get("type", "")
        entry_name = metadata.get("name", "")
        entry_accession = metadata.get("accession", "")
        source_db = metadata.get("source_database", "")
        description = metadata.get("description", [])
        desc_text = description[0].get("text", "") if description else ""

        # Get protein locations (domain positions)
        proteins = entry.get("proteins", [])
        locations = []
        for protein in proteins:
            for loc_group in protein.get("entry_protein_locations", []):
                for fragment in loc_group.get("fragments", []):
                    locations.append({
                        "start": fragment.get("start"),
                        "end": fragment.get("end"),
                    })

        annotation = {
            "accession": entry_accession,
            "name": entry_name,
            "type": entry_type,
            "source_database": source_db,
            "description": desc_text[:200],
            "locations": locations,
        }

        if entry_type == "domain":
            domains.append(annotation)
        elif entry_type == "family":
            families.append(annotation)
        elif entry_type in ("active_site", "binding_site", "conserved_site", "ptm"):
            sites.append(annotation)
        elif entry_type == "homologous_superfamily":
            homologous_superfamilies.append(annotation)

    # Build summary
    gene_label = gene or uniprot_id
    domain_strs = []
    for d in domains:
        loc_str = ""
        if d["locations"]:
            locs = d["locations"][0]
            loc_str = f" ({locs['start']}-{locs['end']})"
        domain_strs.append(f"{d['name']}{loc_str}")

    summary = (
        f"{gene_label}: {len(domains)} domain{'s' if len(domains) != 1 else ''}"
    )
    if domain_strs:
        summary += f" — {', '.join(domain_strs[:6])}"

    return {
        "summary": summary,
        "gene": gene,
        "uniprot_id": uniprot_id,
        "n_domains": len(domains),
        "n_families": len(families),
        "n_sites": len(sites),
        "domains": domains,
        "families": families,
        "sites": sites,
        "homologous_superfamilies": homologous_superfamilies,
        "total_annotations": len(entries),
    }
