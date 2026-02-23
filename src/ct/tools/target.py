"""
Target discovery tools: neosubstrate scoring, degron prediction, co-essentiality.
"""

import pandas as pd
import numpy as np
from ct.tools import registry
from ct.tools.http_client import request


@registry.register(
    name="target.neosubstrate_score",
    description="Score proteins as potential neosubstrate targets based on degradation selectivity and magnitude",
    category="target",
    parameters={"proteomics_path": "Path to proteomics LFC matrix", "top_n": "Number of top targets to return"},
    requires_data=["proteomics"],
    usage_guide="You want to discover new degradation targets from proteomics data — ranks proteins by selective, potent degradation across compounds. Use early in target discovery campaigns.",
)
def neosubstrate_score(proteomics_path: str = None, top_n: int = 50, **kwargs) -> dict:
    """Score proteins for neosubstrate potential."""
    # Load proteomics data
    if proteomics_path is None:
        try:
            from ct.data.loaders import load_proteomics
            prot = load_proteomics()
        except FileNotFoundError:
            return {
                "error": "Proteomics data not available.",
                "summary": "Proteomics data not available — skipping. Provide proteomics data for full analysis.",
            }
    else:
        prot = pd.read_csv(proteomics_path, index_col=0)

    # Score: selectivity × |mean_degradation| × log2(n_degraders + 1)
    results = []
    for protein in prot.index:
        values = prot.loc[protein].dropna()
        degraded = values[values < -0.5]
        if len(degraded) == 0:
            continue

        n_degraders = len(degraded)
        mean_deg = degraded.mean()
        # Selectivity: fraction of compounds that degrade it (lower = more selective)
        selectivity = 1.0 - (n_degraders / len(values))

        score = selectivity * abs(mean_deg) * np.log2(n_degraders + 1)

        results.append({
            "protein": protein,
            "score": score,
            "n_degraders": n_degraders,
            "mean_degradation": mean_deg,
            "selectivity": selectivity,
        })

    if not results:
        return {
            "summary": f"No neosubstrate candidates found in {len(prot)} proteins (none degraded below -0.5 LFC)",
            "top_targets": [],
            "n_proteins_scored": 0,
        }

    df = pd.DataFrame(results).sort_values("score", ascending=False).head(top_n)

    # Map UniProt IDs to gene symbols if protein IDs look like UniProt accessions
    top_proteins = df["protein"].tolist()
    if top_proteins and all(len(p) >= 6 and p[0].isalpha() and any(c.isdigit() for c in p) and " " not in p for p in top_proteins[:3]):
        try:
            import httpx
            # Batch lookup via UniProt ID mapping
            ids_str = ",".join(top_proteins)
            resp = httpx.get(
                "https://rest.uniprot.org/uniprotkb/accessions",
                params={"accessions": ids_str, "fields": "accession,gene_primary"},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if resp.status_code == 200:
                entries = resp.json().get("results", [])
                id_to_gene = {}
                for entry in entries:
                    acc = entry.get("primaryAccession", "")
                    genes = entry.get("genes", [])
                    if genes:
                        gene_name = genes[0].get("geneName", {}).get("value", "")
                        if gene_name:
                            id_to_gene[acc] = gene_name
                if id_to_gene:
                    df["gene_symbol"] = df["protein"].map(id_to_gene)
        except Exception:
            pass

    return {
        "summary": f"Top {min(top_n, len(results))} neosubstrate candidates scored from {len(prot)} proteins",
        "top_targets": df.to_dict("records"),
        "n_proteins_scored": len(results),
    }


@registry.register(
    name="target.degron_predict",
    description="Predict structural degron motifs in a protein (zinc fingers, disordered regions, surface accessibility) using UniProt features",
    category="target",
    parameters={"uniprot_id": "UniProt ID of target protein (e.g. P04637 for TP53)"},
    requires_data=[],
    usage_guide="You want to assess whether a protein has structural features (zinc fingers, disordered loops) that make it amenable to E3-mediated degradation. Use after identifying a target of interest.",
)
def degron_predict(uniprot_id: str, **kwargs) -> dict:
    """Predict degron features for a target protein using UniProt feature analysis."""
    # Fetch protein features from UniProt API
    resp, error = request(
        "GET",
        f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
        timeout=30,
        headers={"Accept": "application/json"},
        raise_for_status=False,
    )
    if error:
        return {"error": f"Failed to fetch UniProt data: {error}", "summary": f"Failed to fetch UniProt data: {error}"}
    if resp.status_code != 200:
        return {"error": f"UniProt entry not found for {uniprot_id} (HTTP {resp.status_code})", "summary": f"UniProt entry not found for {uniprot_id} (HTTP {resp.status_code})"}
    try:
        data = resp.json()
    except Exception:
        return {"error": f"Invalid UniProt JSON response for {uniprot_id}", "summary": f"Invalid UniProt JSON response for {uniprot_id}"}
    protein_name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", uniprot_id)
    gene_name = ""
    genes = data.get("genes", [])
    if genes:
        gene_name = genes[0].get("geneName", {}).get("value", "")
    sequence = data.get("sequence", {})
    seq_length = sequence.get("length", 0)

    # Extract structural features relevant to degradation
    features = data.get("features", [])
    zinc_fingers = [f for f in features if f.get("type") == "Zinc finger"]
    domains = [f for f in features if f.get("type") == "Domain"]
    disordered = [f for f in features if f.get("type") == "Region" and "Disordered" in f.get("description", "")]
    motifs = [f for f in features if f.get("type") == "Motif"]
    modifications = [f for f in features if f.get("type") in ("Modified residue", "Cross-link")]

    # Compute degron-relevant scores
    def _region_length(feat):
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value", 0)
        end = loc.get("end", {}).get("value", 0)
        return max(0, end - start + 1) if start and end else 0

    disordered_residues = sum(_region_length(f) for f in disordered)
    disorder_fraction = disordered_residues / seq_length if seq_length > 0 else 0

    # Known degron-associated domain types
    degron_domains = []
    for d in domains:
        desc = d.get("description", "").lower()
        if any(k in desc for k in ["zinc finger", "ring", "btb", "wd40", "kelch", "socs box", "f-box"]):
            degron_domains.append(d.get("description", "unknown"))

    # Lysine count from sequence (ubiquitination sites)
    raw_seq = sequence.get("value", "")
    lysine_count = raw_seq.count("K") if raw_seq else 0
    lysine_density = lysine_count / seq_length if seq_length > 0 else 0

    # Known ubiquitination sites from modifications
    ub_sites = [m for m in modifications if "ubiquit" in m.get("description", "").lower()]

    # Compute overall degradability score (0-1 heuristic)
    score = 0.0
    score_breakdown = {}

    # Zinc fingers are strong degron features (CRBN/IKZF-type)
    zf_score = min(len(zinc_fingers) * 0.15, 0.3)
    score += zf_score
    score_breakdown["zinc_fingers"] = zf_score

    # Disordered regions expose protein to E3 engagement
    disorder_score = min(disorder_fraction * 0.5, 0.25)
    score += disorder_score
    score_breakdown["disorder"] = disorder_score

    # Lysine density enables ubiquitination
    lys_score = min(lysine_density * 3.0, 0.2)
    score += lys_score
    score_breakdown["lysine_accessibility"] = lys_score

    # Known ubiquitination sites
    ub_score = min(len(ub_sites) * 0.05, 0.15)
    score += ub_score
    score_breakdown["known_ub_sites"] = ub_score

    # Small-medium proteins degrade more easily
    size_score = 0.1 if seq_length < 800 else 0.05 if seq_length < 1500 else 0.0
    score += size_score
    score_breakdown["protein_size"] = size_score

    score = min(score, 1.0)

    # Classify
    if score >= 0.5:
        classification = "high"
        rationale = "Strong structural features for E3-mediated degradation"
    elif score >= 0.25:
        classification = "moderate"
        rationale = "Some favorable features; may require linker/scaffold optimization"
    else:
        classification = "low"
        rationale = "Few structural degron features identified"

    return {
        "summary": (
            f"Degron prediction for {gene_name or uniprot_id} ({protein_name}): "
            f"{classification} degradability (score={score:.2f}). "
            f"{len(zinc_fingers)} zinc finger(s), {disordered_residues} disordered residues "
            f"({disorder_fraction:.0%}), {lysine_count} lysines, {len(ub_sites)} known Ub site(s)."
        ),
        "uniprot_id": uniprot_id,
        "gene": gene_name,
        "protein_name": protein_name,
        "seq_length": seq_length,
        "degradability_score": round(score, 3),
        "classification": classification,
        "rationale": rationale,
        "score_breakdown": {k: round(v, 3) for k, v in score_breakdown.items()},
        "features": {
            "zinc_fingers": len(zinc_fingers),
            "zinc_finger_details": [
                {"description": f.get("description", ""), "start": f.get("location", {}).get("start", {}).get("value"), "end": f.get("location", {}).get("end", {}).get("value")}
                for f in zinc_fingers
            ],
            "disordered_residues": disordered_residues,
            "disorder_fraction": round(disorder_fraction, 3),
            "domains": [d.get("description", "") for d in domains],
            "degron_associated_domains": degron_domains,
            "lysine_count": lysine_count,
            "lysine_density": round(lysine_density, 3),
            "known_ub_sites": len(ub_sites),
            "motifs": [m.get("description", "") for m in motifs],
        },
    }


@registry.register(
    name="target.coessentiality",
    description="Find co-essential and synthetic lethal partners for a target gene using DepMap CRISPR data",
    category="target",
    parameters={"gene": "Gene symbol", "top_n": "Number of partners to return"},
    requires_data=["depmap_crispr"],
    usage_guide="You need to validate a drug target by finding functionally related genes, or identify synthetic lethal partners for combination therapy. Also useful for understanding pathway context of a gene.",
)
def coessentiality(gene: str, top_n: int = 20, **kwargs) -> dict:
    """Compute co-essentiality network for a gene."""
    from ct.data.loaders import load_crispr

    crispr = load_crispr()

    if gene not in crispr.columns:
        return {"error": f"Gene {gene} not found in DepMap CRISPR data", "summary": f"Gene {gene} not found in DepMap CRISPR data"}
    target_vals = crispr[gene].dropna()

    correlations = []
    for other_gene in crispr.columns:
        if other_gene == gene:
            continue
        other_vals = crispr[other_gene].dropna()
        common = target_vals.index.intersection(other_vals.index)
        if len(common) < 50:
            continue

        from scipy import stats
        r, p = stats.pearsonr(target_vals[common], other_vals[common])
        correlations.append({"gene": other_gene, "r": r, "p": p})

    if not correlations:
        return {
            "summary": f"Co-essentiality network for {gene}: no genes with sufficient shared cell lines (>=50)",
            "gene": gene,
            "co_essential": [],
            "synthetic_lethal": [],
        }

    df = pd.DataFrame(correlations).sort_values("r", ascending=False)

    co_essential = df.head(top_n).to_dict("records")
    synthetic_lethal = df.tail(top_n).sort_values("r").to_dict("records")

    return {
        "summary": f"Co-essentiality network for {gene}: {len(correlations)} genes tested",
        "gene": gene,
        "co_essential": co_essential,
        "synthetic_lethal": synthetic_lethal,
    }


@registry.register(
    name="target.druggability",
    description="Assess the druggability of a protein target using UniProt annotations (protein family, domains, ligands, structural coverage)",
    category="target",
    parameters={"gene": "Gene symbol (e.g. BRAF, EGFR)"},
    requires_data=[],
    usage_guide="You want to evaluate whether a target protein is druggable — checks protein class, known ligands, structural data, and surface accessibility. Use early in target prioritization.",
)
def druggability(gene: str, **kwargs) -> dict:
    """Assess druggability of a protein target via UniProt annotations."""
    # Query UniProt for the gene
    resp, error = request(
        "GET",
        "https://rest.uniprot.org/uniprotkb/search",
        params={
            "query": f"gene_exact:{gene} AND organism_id:9606",
            "format": "json",
            "size": "1",
        },
        timeout=10,
        headers={"Accept": "application/json"},
        raise_for_status=False,
    )
    if error:
        return {"error": f"Failed to fetch UniProt data: {error}", "summary": f"UniProt API error for {gene}"}
    if resp.status_code != 200:
        return {"error": f"UniProt search failed for {gene} (HTTP {resp.status_code})", "summary": f"Failed to query UniProt for {gene}"}
    try:
        data = resp.json()
    except Exception:
        return {"error": f"Invalid UniProt response for {gene}", "summary": f"Failed to parse UniProt data for {gene}"}

    results = data.get("results", [])
    if not results:
        return {"error": f"No UniProt entry found for {gene} in human", "summary": f"Gene {gene} not found in UniProt (human)"}

    entry = results[0]

    # Extract protein info
    protein_name = (
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", gene)
    )
    uniprot_id = entry.get("primaryAccession", "")

    # Extract features
    features = entry.get("features", [])
    domains = [f.get("description", "") for f in features if f.get("type") == "Domain"]
    keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]

    # Subcellular location
    comments = entry.get("comments", [])
    subcellular_locs = []
    for c in comments:
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            for sl in c.get("subcellularLocations", []):
                loc_val = sl.get("location", {}).get("value", "")
                if loc_val:
                    subcellular_locs.append(loc_val)

    # Transmembrane regions
    transmembrane = [f for f in features if f.get("type") == "Transmembrane"]

    # Cross-references
    xrefs = entry.get("uniProtKBCrossReferences", [])

    # Check for ChEMBL cross-refs (known small molecule ligands)
    chembl_refs = [x for x in xrefs if x.get("database") == "ChEMBL"]
    known_drugs = [x.get("id", "") for x in chembl_refs]

    # Check for PDB cross-refs (structural coverage)
    pdb_refs = [x for x in xrefs if x.get("database") == "PDB"]
    pdb_ids = [x.get("id", "") for x in pdb_refs]

    # Determine protein class from keywords and domains
    protein_class = "other"
    class_score = 0.0
    keywords_lower = [k.lower() for k in keywords]
    domains_lower = [d.lower() for d in domains]
    all_annotations = " ".join(keywords_lower + domains_lower)

    if any(k in all_annotations for k in ["kinase", "protein kinase"]):
        protein_class = "kinase"
        class_score = 0.35
    elif any(k in all_annotations for k in ["g-protein coupled receptor", "gpcr"]):
        protein_class = "GPCR"
        class_score = 0.35
    elif any(k in all_annotations for k in ["ion channel", "voltage-gated"]):
        protein_class = "ion_channel"
        class_score = 0.30
    elif any(k in all_annotations for k in ["nuclear hormone receptor", "nuclear receptor"]):
        protein_class = "nuclear_receptor"
        class_score = 0.30
    elif any(k in all_annotations for k in ["protease", "peptidase"]):
        protein_class = "protease"
        class_score = 0.25
    elif any(k in all_annotations for k in ["phosphatase"]):
        protein_class = "phosphatase"
        class_score = 0.25
    elif any(k in all_annotations for k in ["transferase"]):
        protein_class = "transferase"
        class_score = 0.20
    elif any(k in all_annotations for k in ["transcription factor", "transcription"]):
        protein_class = "transcription_factor"
        class_score = 0.10
    elif any(k in all_annotations for k in ["scaffold", "adaptor"]):
        protein_class = "scaffold_adaptor"
        class_score = 0.05

    # Score: known ligands
    ligand_score = min(len(chembl_refs) * 0.10, 0.25)

    # Score: surface accessibility (extracellular / secreted / membrane)
    surface_keywords = ["secreted", "cell membrane", "extracellular"]
    is_surface = any(
        any(sk in loc.lower() for sk in surface_keywords)
        for loc in subcellular_locs
    )
    surface_score = 0.15 if is_surface else 0.0

    # Score: structural coverage (PDB entries)
    structure_score = min(len(pdb_refs) * 0.02, 0.15)

    # Score: has transmembrane (often druggable for membrane targets)
    tm_score = 0.10 if transmembrane else 0.0

    total_score = min(class_score + ligand_score + surface_score + structure_score + tm_score, 1.0)

    # Reasoning
    reasoning_parts = []
    if class_score > 0:
        reasoning_parts.append(f"Protein class '{protein_class}' is a {'highly ' if class_score >= 0.30 else ''}tractable target class")
    else:
        reasoning_parts.append(f"Protein class '{protein_class}' has limited druggability precedent")
    if chembl_refs:
        reasoning_parts.append(f"{len(chembl_refs)} ChEMBL entry/entries indicate known small-molecule interactions")
    else:
        reasoning_parts.append("No ChEMBL cross-references found (no known small-molecule ligands)")
    if pdb_refs:
        reasoning_parts.append(f"{len(pdb_refs)} PDB structure(s) available for structure-based drug design")
    else:
        reasoning_parts.append("No PDB structures available")
    if is_surface:
        reasoning_parts.append("Surface-accessible / extracellular localization supports biologic targeting")
    reasoning = ". ".join(reasoning_parts) + "."

    # Classify
    if total_score >= 0.6:
        classification = "highly druggable"
    elif total_score >= 0.35:
        classification = "druggable"
    elif total_score >= 0.15:
        classification = "challenging"
    else:
        classification = "undruggable (with current modalities)"

    return {
        "summary": (
            f"Druggability assessment for {gene} ({protein_name}): "
            f"{classification} (score={total_score:.2f}). "
            f"Class: {protein_class}. {len(pdb_ids)} PDB structures, "
            f"{len(known_drugs)} ChEMBL entries."
        ),
        "gene": gene,
        "uniprot_id": uniprot_id,
        "protein_name": protein_name,
        "druggability_score": round(total_score, 3),
        "classification": classification,
        "protein_class": protein_class,
        "known_drugs": known_drugs,
        "structural_coverage": {
            "pdb_count": len(pdb_ids),
            "pdb_ids": pdb_ids[:20],  # Cap at 20 for readability
        },
        "surface_accessible": is_surface,
        "subcellular_locations": subcellular_locs,
        "transmembrane_regions": len(transmembrane),
        "domains": domains,
        "reasoning": reasoning,
        "score_breakdown": {
            "protein_class": round(class_score, 3),
            "known_ligands": round(ligand_score, 3),
            "surface_accessibility": round(surface_score, 3),
            "structural_data": round(structure_score, 3),
            "transmembrane": round(tm_score, 3),
        },
    }


@registry.register(
    name="target.expression_profile",
    description="Get tissue expression profile for a gene using GTEx Portal API and Human Protein Atlas",
    category="target",
    parameters={
        "gene": "Gene symbol (e.g. TP53, EGFR, BRCA1)",
        "top_n": "Number of top tissues to return (default 10)",
    },
    requires_data=[],
    usage_guide="You want to understand where a target is expressed — tissue specificity, cancer vs normal, and cell type expression. Critical for safety assessment and indication selection.",
)
def expression_profile(gene: str, top_n: int = 10, **kwargs) -> dict:
    """Get tissue expression profile for a gene from GTEx and Human Protein Atlas.

    Resolves gene symbol to GENCODE ID via the GTEx reference API, then
    fetches median expression per tissue from GTEx v8. Also queries HPA
    for protein-level and single-cell expression. Computes a tissue
    specificity index (tau) from the GTEx TPM values.
    """
    # --- Step 1: Resolve gene symbol to GENCODE ID via GTEx reference API ---
    def _gene_symbol_candidates(input_gene: str) -> list[str]:
        alias_map = {
            "GBA1": "GBA",
            "PARK2": "PRKN",
        }
        token = (input_gene or "").strip()
        if not token:
            return []
        candidates = [token]
        mapped = alias_map.get(token.upper())
        if mapped:
            candidates.append(mapped)
        if token.endswith("1") and len(token) > 1:
            candidates.append(token[:-1])

        deduped = []
        seen = set()
        for c in candidates:
            k = c.upper()
            if k in seen:
                continue
            seen.add(k)
            deduped.append(c)
        return deduped

    gencode_id = None
    gene_symbol = gene
    gene_candidates = _gene_symbol_candidates(gene)

    for gene_candidate in gene_candidates:
        ref_resp, ref_error = request(
            "GET",
            "https://gtexportal.org/api/v2/reference/gene",
            params={"geneId": gene_candidate},
            timeout=10,
            raise_for_status=False,
        )
        if ref_error or ref_resp.status_code != 200:
            continue
        try:
            ref_data = ref_resp.json()
        except Exception:
            continue
        genes_list = ref_data.get("data", [])
        if genes_list:
            gene_info = genes_list[0]
            gencode_id = gene_info.get("gencodeId", "")
            gene_symbol = gene_info.get("geneSymbol", gene_candidate)
            break

    # --- Step 2: GTEx median gene expression per tissue ---
    gtex_expression = []

    if gencode_id:
        gtex_resp, gtex_error = request(
            "GET",
            "https://gtexportal.org/api/v2/expression/medianGeneExpression",
            params={
                "gencodeId": gencode_id,
                "datasetId": "gtex_v8",
            },
            timeout=10,
            raise_for_status=False,
        )
        if not gtex_error and gtex_resp.status_code == 200:
            try:
                gtex_data = gtex_resp.json()
                for entry in gtex_data.get("data", []):
                    gtex_expression.append({
                        "tissue": entry.get("tissueSiteDetailId", ""),
                        "median_tpm": entry.get("median", 0),
                    })
                gtex_expression.sort(key=lambda x: x.get("median_tpm", 0), reverse=True)
            except Exception:
                gtex_expression = []

    # --- Step 3: Compute tissue specificity index (tau) ---
    # Tau ranges from 0 (ubiquitous) to 1 (tissue-specific)
    tau = None
    if gtex_expression:
        tpm_values = [t["median_tpm"] for t in gtex_expression]
        max_tpm = max(tpm_values) if tpm_values else 0
        if max_tpm > 0 and len(tpm_values) > 1:
            n = len(tpm_values)
            tau = sum(1.0 - (x / max_tpm) for x in tpm_values) / (n - 1)
            tau = round(tau, 4)

    # --- Step 4: Human Protein Atlas ---
    hpa_data = {}
    tissue_rna = []
    tissue_protein = []
    cancer_expression = []
    cell_type_expression = []
    ensembl_id = None

    # Try to extract Ensembl ID from GENCODE ID (strip version suffix)
    if gencode_id:
        ensembl_id = gencode_id.split(".")[0]

    # Query HPA using Ensembl ID if available, then gene aliases.
    hpa_queries = []
    if ensembl_id:
        hpa_queries.append(ensembl_id)
    if gene_symbol:
        hpa_queries.append(gene_symbol)
    hpa_queries.extend(gene_candidates)

    # Stable de-dup for query candidates.
    deduped_hpa_queries = []
    seen_hpa = set()
    for q in hpa_queries:
        key = str(q).upper()
        if key in seen_hpa:
            continue
        seen_hpa.add(key)
        deduped_hpa_queries.append(q)

    for hpa_query in deduped_hpa_queries:
        hpa_resp, hpa_error = request(
            "GET",
            f"https://www.proteinatlas.org/{hpa_query}.json",
            timeout=10,
            headers={"Accept": "application/json"},
            raise_for_status=False,
        )
        if hpa_error or hpa_resp is None or hpa_resp.status_code != 200:
            continue
        try:
            hpa_data = hpa_resp.json()
        except Exception:
            hpa_data = {}
        if hpa_data:
            break

    if hpa_data:
        # RNA tissue expression
        for entry in hpa_data.get("RNATissue", {}).get("data", []):
            tissue_rna.append({
                "tissue": entry.get("Tissue", ""),
                "tpm": entry.get("TPM", 0),
                "ntpm": entry.get("nTPM", 0),
            })
        tissue_rna.sort(key=lambda x: x.get("tpm", 0), reverse=True)

        # Protein tissue expression
        for entry in hpa_data.get("ProteinTissue", {}).get("data", []):
            tissue_protein.append({
                "tissue": entry.get("Tissue", ""),
                "level": entry.get("Level", ""),
                "cell_type": entry.get("CellType", ""),
            })

        # Cancer expression
        for entry in hpa_data.get("RNACancer", {}).get("data", []):
            cancer_expression.append({
                "cancer": entry.get("Cancer", ""),
                "tpm": entry.get("TPM", 0),
                "ntpm": entry.get("nTPM", 0),
            })
        cancer_expression.sort(key=lambda x: x.get("tpm", 0), reverse=True)

        # Cell type expression
        for entry in hpa_data.get("RNASingleCell", {}).get("data", []):
            cell_type_expression.append({
                "cell_type": entry.get("CellType", ""),
                "ntpm": entry.get("nTPM", 0),
            })
        cell_type_expression.sort(key=lambda x: x.get("ntpm", 0), reverse=True)

    # --- Build response ---
    # Prefer GTEx for top tissues (quantitative TPM), fall back to HPA
    top_tissues = gtex_expression[:top_n] if gtex_expression else tissue_rna[:top_n]
    if not top_tissues and not hpa_data:
        return {
            "summary": f"No expression data found for {gene} from GTEx or Human Protein Atlas",
            "gene": gene,
            "error": "No data returned from GTEx or HPA APIs",
        }

    n_tissues = len(gtex_expression) if gtex_expression else len(tissue_rna)

    # Build summary line matching the spec format
    if gtex_expression:
        tissue_strs = [
            f"{t['tissue']} ({t['median_tpm']:.1f} TPM)" for t in gtex_expression[:top_n]
        ]
        summary = f"{gene_symbol} expression: highest in {', '.join(tissue_strs[:5])}"
    elif tissue_rna:
        tissue_strs = [
            f"{t['tissue']} ({t['tpm']:.1f} TPM)" for t in tissue_rna[:top_n]
        ]
        summary = f"{gene_symbol} expression: highest in {', '.join(tissue_strs[:5])}"
    else:
        summary = f"Expression profile for {gene_symbol}: {n_tissues} tissues profiled"

    if tau is not None:
        specificity_label = (
            "tissue-specific" if tau > 0.8
            else "tissue-enriched" if tau > 0.5
            else "broadly expressed"
        )
        summary += f". Specificity: {specificity_label} (tau={tau:.3f})"

    # RNA tissue specificity category from HPA
    rna_specificity = hpa_data.get("RNATissue", {}).get("summary", "")

    return {
        "summary": summary,
        "gene": gene_symbol,
        "gencode_id": gencode_id,
        "ensembl_id": ensembl_id,
        "tissue_specificity_tau": tau,
        "rna_specificity_hpa": rna_specificity,
        "gtex_expression": gtex_expression[:top_n],
        "tissue_rna_hpa": tissue_rna[:top_n],
        "tissue_protein": tissue_protein[:30],
        "cancer_expression": cancer_expression[:20],
        "cell_type_expression": cell_type_expression[:20],
        "n_tissues_profiled": n_tissues,
        "top_expressing_tissues": top_tissues[:top_n],
    }


@registry.register(
    name="target.disease_association",
    description="Query Open Targets Platform for disease associations of a gene target",
    category="target",
    parameters={"gene": "Gene symbol (e.g. BRAF, TP53)", "min_score": "Minimum association score (default 0.1)"},
    requires_data=[],
    usage_guide="You want to know which diseases a target is associated with — genetic evidence, drug evidence, literature support. Essential for indication selection and target validation.",
)
def disease_association(gene: str, min_score: float = 0.1, **kwargs) -> dict:
    """Query Open Targets for disease associations of a gene."""
    # Step 1: Resolve gene symbol to Ensembl ID
    ensembl_id = None
    ens_resp, ens_error = request(
        "GET",
        f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene}",
        params={"content-type": "application/json"},
        timeout=10,
        headers={"Content-Type": "application/json"},
        raise_for_status=False,
    )
    if ens_error:
        return {
            "error": f"Failed to resolve {gene} to Ensembl ID: {ens_error}",
            "summary": f"Could not resolve gene symbol {gene} via Ensembl REST API",
        }
    if ens_resp.status_code == 200:
        try:
            ens_data = ens_resp.json()
            ensembl_id = ens_data.get("id", "")
        except Exception:
            ensembl_id = None

    if not ensembl_id:
        return {
            "error": f"Gene {gene} not found in Ensembl (human)",
            "summary": f"Gene symbol {gene} could not be resolved to an Ensembl ID",
        }

    # Step 2: Query Open Targets GraphQL
    query = """
    query targetDiseases($ensemblId: String!, $size: Int!) {
      target(ensemblId: $ensemblId) {
        approvedSymbol
        approvedName
        associatedDiseases(page: {index: 0, size: $size}) {
          count
          rows {
            disease {
              id
              name
            }
            score
            datasourceScores {
              id
              score
            }
          }
        }
      }
    }
    """

    ot_resp, ot_error = request(
        "POST",
        "https://api.platform.opentargets.org/api/v4/graphql",
        json={
            "query": query,
            "variables": {
                "ensemblId": ensembl_id,
                "size": 50,
            },
        },
        timeout=10,
        headers={"Content-Type": "application/json"},
        raise_for_status=False,
    )
    if ot_error:
        return {
            "error": f"Open Targets API error: {ot_error}",
            "summary": f"Failed to query Open Targets for {gene}",
        }
    if ot_resp.status_code != 200:
        return {
            "error": f"Open Targets API returned HTTP {ot_resp.status_code}",
            "summary": f"Open Targets query failed for {gene} ({ensembl_id})",
        }
    try:
        ot_data = ot_resp.json()
    except Exception:
        return {
            "error": "Open Targets returned invalid JSON",
            "summary": f"Failed to parse Open Targets response for {gene}",
        }

    target_data = ot_data.get("data", {}).get("target")
    if not target_data:
        return {
            "error": f"No target data returned from Open Targets for {ensembl_id}",
            "summary": f"Open Targets has no entry for {gene} ({ensembl_id})",
        }

    approved_symbol = target_data.get("approvedSymbol", gene)
    approved_name = target_data.get("approvedName", "")
    assoc_data = target_data.get("associatedDiseases", {})
    total_count = assoc_data.get("count", 0)
    rows = assoc_data.get("rows", [])

    # Parse associations
    associations = []
    for row in rows:
        overall_score = row.get("score", 0)
        if overall_score < min_score:
            continue

        disease = row.get("disease", {})
        disease_id = disease.get("id", "")
        disease_name = disease.get("name", "")

        # Parse datasource scores into readable categories
        ds_scores = {}
        for ds in row.get("datasourceScores", []):
            comp_id = ds.get("id") or ds.get("componentId", "")
            ds_score = ds.get("score", 0)
            ds_scores[comp_id] = round(ds_score, 4)

        # Extract key evidence categories
        genetic_score = max(
            ds_scores.get("ot_genetics_portal", 0),
            ds_scores.get("gene_burden", 0),
            ds_scores.get("genomics_england", 0),
            ds_scores.get("eva", 0),
            ds_scores.get("uniprot_variants", 0),
        )
        drug_score = max(
            ds_scores.get("chembl", 0),
            ds_scores.get("europepmc", 0),
        )
        literature_score = ds_scores.get("europepmc", 0)

        associations.append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "overall_score": round(overall_score, 4),
            "genetic_association": round(genetic_score, 4),
            "known_drug": round(drug_score, 4),
            "literature": round(literature_score, 4),
            "all_datasource_scores": ds_scores,
        })

    associations.sort(key=lambda x: x["overall_score"], reverse=True)

    # Build summary
    n_filtered = len(associations)
    top_diseases = ", ".join(a["disease_name"] for a in associations[:5])
    if not top_diseases:
        top_diseases = "none"

    summary = (
        f"Disease associations for {approved_symbol} ({approved_name}): "
        f"{n_filtered} diseases above score {min_score} (out of {total_count} total). "
        f"Top: {top_diseases}."
    )

    return {
        "summary": summary,
        "gene": approved_symbol,
        "ensembl_id": ensembl_id,
        "approved_name": approved_name,
        "total_associations": total_count,
        "filtered_associations": n_filtered,
        "min_score": min_score,
        "associations": associations,
    }
