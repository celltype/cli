"""
Clean-room parity tools built from public/open APIs.

Adds practical connectors and utilities inspired by commonly requested platform
capabilities while staying implementation-original inside ct.
"""

from __future__ import annotations

from datetime import date
import re
import xml.etree.ElementTree as ET

from ct.tools import registry
from ct.tools.http_client import request, request_json


def _clip(value, n: int = 240) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _coerce_int(value, default: int, minimum: int = 1, maximum: int = 100) -> int:
    try:
        out = int(value)
    except Exception:
        out = default
    return min(max(out, minimum), maximum)


def _flatten_hits(payload, key: str = "hits") -> list[dict]:
    hits = payload.get(key, []) if isinstance(payload, dict) else []
    if isinstance(hits, dict):
        # my* APIs sometimes return dict hits keyed by ids
        hits = list(hits.values())
    return [h for h in hits if isinstance(h, dict)]


_MYGENE_SPECIES_MAP = {
    "human": "human",
    "mouse": "mouse",
    "rat": "rat",
    "zebrafish": "zebrafish",
    "drosophila": "fly",
    "yeast": "yeast",
    "schistosoma mansoni": "6183",
    "fasciola hepatica": "6192",
    "heligmosomoides polygyrus": "6337",
    "nippostrongylus brasiliensis": "27835",
    "trichuris muris": "70415",
    "brugia malayi": "6279",
}


def _normalize_mygene_species(species: str) -> str:
    s = (species or "human").strip().lower()
    if not s:
        return "human"
    if s.isdigit():
        return s
    if s in _MYGENE_SPECIES_MAP:
        return _MYGENE_SPECIES_MAP[s]

    # Extract likely binomial species from noisy planner text.
    m = re.search(r"([A-Za-z][a-z]+)\s+([a-z][a-z]+)", s)
    if m:
        candidate = f"{m.group(1)} {m.group(2)}".lower()
        if candidate in _MYGENE_SPECIES_MAP:
            return _MYGENE_SPECIES_MAP[candidate]

    # Fallback to original value; caller can still get API errors surfaced.
    return species


@registry.register(
    name="data_api.mygene_lookup",
    description="Lookup genes via MyGene.info",
    category="data_api",
    parameters={
        "query": "Gene symbol/name/identifier (e.g., TP53, ENSG00000141510)",
        "species": "Species filter (default human)",
        "size": "Maximum hits (default 10)",
    },
    usage_guide="Use for rapid gene identifier normalization and annotation via MyGene.info.",
)
def mygene_lookup(query: str, species: str = "human", size: int = 10, **kwargs) -> dict:
    """Query MyGene.info for gene-level metadata."""
    q = (query or "").strip()
    if not q:
        return {"summary": "query is required.", "error": "missing_query"}

    species_norm = _normalize_mygene_species(species)

    data, error = request_json(
        "GET",
        "https://mygene.info/v3/query",
        params={
            "q": q,
            "species": species_norm,
            "size": _coerce_int(size, 10),
            "fields": "symbol,name,entrezgene,ensembl.gene,taxid,type_of_gene",
        },
        timeout=20,
        retries=2,
    )
    if error:
        return {"summary": f"MyGene lookup failed: {error}", "error": "api_error"}

    hits = _flatten_hits(data)
    rows = []
    for hit in hits:
        ens = hit.get("ensembl")
        if isinstance(ens, list) and ens:
            ensembl_gene = ens[0].get("gene")
        elif isinstance(ens, dict):
            ensembl_gene = ens.get("gene")
        else:
            ensembl_gene = None
        rows.append(
            {
                "symbol": hit.get("symbol"),
                "name": hit.get("name"),
                "entrezgene": hit.get("entrezgene"),
                "ensembl_gene": ensembl_gene,
                "taxid": hit.get("taxid"),
                "type_of_gene": hit.get("type_of_gene"),
                "score": hit.get("_score"),
            }
        )

    return {
        "summary": f"MyGene: found {len(rows)} hits for '{q}'.",
        "query": q,
        "species": species_norm,
        "requested_species": species,
        "hits": rows,
        "count": len(rows),
        "source": "mygene.info",
    }


@registry.register(
    name="data_api.mydisease_lookup",
    description="Lookup diseases via MyDisease.info",
    category="data_api",
    parameters={
        "query": "Disease keyword/identifier",
        "size": "Maximum hits (default 10)",
    },
    usage_guide="Use for disease identifier mapping and cross-source disease metadata.",
)
def mydisease_lookup(query: str, size: int = 10, **kwargs) -> dict:
    """Query MyDisease.info for disease metadata."""
    q = (query or "").strip()
    if not q:
        return {"summary": "query is required.", "error": "missing_query"}

    data, error = request_json(
        "GET",
        "https://mydisease.info/v1/query",
        params={"q": q, "size": _coerce_int(size, 10)},
        timeout=20,
        retries=2,
    )
    if error:
        return {"summary": f"MyDisease lookup failed: {error}", "error": "api_error"}

    rows = []
    for hit in _flatten_hits(data):
        disease_name = hit.get("name")
        doid = None
        do_block = hit.get("disease_ontology")
        if isinstance(do_block, dict):
            doid = do_block.get("doid") or do_block.get("id")
            disease_name = disease_name or do_block.get("name")
        rows.append(
            {
                "name": disease_name,
                "doid": doid,
                "mondo": hit.get("mondo"),
                "score": hit.get("_score"),
                "id": hit.get("_id"),
            }
        )

    return {
        "summary": f"MyDisease: found {len(rows)} hits for '{q}'.",
        "query": q,
        "hits": rows,
        "count": len(rows),
        "source": "mydisease.info",
    }


@registry.register(
    name="data_api.myvariant_lookup",
    description="Lookup variants via MyVariant.info",
    category="data_api",
    parameters={
        "query": "Variant keyword/identifier (e.g., rs121913529, chr17:g.7673803G>A)",
        "size": "Maximum hits (default 10)",
    },
    usage_guide="Use for quick variant annotation triage from aggregated public sources.",
)
def myvariant_lookup(query: str, size: int = 10, **kwargs) -> dict:
    """Query MyVariant.info for variant annotations."""
    q = (query or "").strip()
    if not q:
        return {"summary": "query is required.", "error": "missing_query"}

    data, error = request_json(
        "GET",
        "https://myvariant.info/v1/query",
        params={
            "q": q,
            "size": _coerce_int(size, 10),
            "fields": "dbsnp.rsid,clinvar.hgvs,clinvar.clinsig,vcf.gene,vcf.position",
        },
        timeout=20,
        retries=2,
    )
    if error:
        return {"summary": f"MyVariant lookup failed: {error}", "error": "api_error"}

    rows = []
    for hit in _flatten_hits(data):
        dbsnp = hit.get("dbsnp")
        rsid = None
        if isinstance(dbsnp, dict):
            rsid = dbsnp.get("rsid")
        clinvar = hit.get("clinvar") if isinstance(hit.get("clinvar"), dict) else {}
        vcf = hit.get("vcf") if isinstance(hit.get("vcf"), dict) else {}
        rows.append(
            {
                "id": hit.get("_id"),
                "rsid": rsid,
                "hgvs": clinvar.get("hgvs"),
                "clinical_significance": clinvar.get("clinsig"),
                "gene": vcf.get("gene"),
                "position": vcf.get("position"),
                "score": hit.get("_score"),
            }
        )

    return {
        "summary": f"MyVariant: found {len(rows)} hits for '{q}'.",
        "query": q,
        "hits": rows,
        "count": len(rows),
        "source": "myvariant.info",
    }


@registry.register(
    name="data_api.mytaxon_lookup",
    description="Lookup taxonomy records via MyTaxon.info",
    category="data_api",
    parameters={
        "query": "Species/taxon keyword or taxonomy ID",
        "size": "Maximum hits (default 10)",
    },
    usage_guide="Use for organism/taxonomy normalization in multi-species analyses.",
)
def mytaxon_lookup(query: str, size: int = 10, **kwargs) -> dict:
    """Query MyTaxon.info for taxonomy metadata."""
    q = (query or "").strip()
    if not q:
        return {"summary": "query is required.", "error": "missing_query"}

    data, error = request_json(
        "GET",
        "https://mytaxon.info/v1/query",
        params={"q": q, "size": _coerce_int(size, 10)},
        timeout=20,
        retries=2,
    )
    if error:
        return {"summary": f"MyTaxon lookup failed: {error}", "error": "api_error"}

    rows = []
    for hit in _flatten_hits(data):
        rows.append(
            {
                "taxid": hit.get("_id") or hit.get("taxid"),
                "scientific_name": hit.get("scientific_name"),
                "common_name": hit.get("common_name"),
                "rank": hit.get("rank"),
                "parent_taxid": hit.get("parent_taxid"),
                "score": hit.get("_score"),
            }
        )

    return {
        "summary": f"MyTaxon: found {len(rows)} hits for '{q}'.",
        "query": q,
        "hits": rows,
        "count": len(rows),
        "source": "mytaxon.info",
    }


@registry.register(
    name="data_api.mychem_lookup",
    description="Lookup compounds/drugs via MyChem.info",
    category="data_api",
    parameters={
        "query": "Compound/drug keyword or identifier",
        "size": "Maximum hits (default 10)",
    },
    usage_guide="Use for rapid integrated compound metadata lookup across public sources.",
)
def mychem_lookup(query: str, size: int = 10, **kwargs) -> dict:
    """Query MyChem.info for compound metadata."""
    q = (query or "").strip()
    if not q:
        return {"summary": "query is required.", "error": "missing_query"}

    data, error = request_json(
        "GET",
        "https://mychem.info/v1/query",
        params={"q": q, "size": _coerce_int(size, 10)},
        timeout=20,
        retries=2,
    )
    if error:
        return {"summary": f"MyChem lookup failed: {error}", "error": "api_error"}

    rows = []
    for hit in _flatten_hits(data):
        chembl = hit.get("chembl") if isinstance(hit.get("chembl"), dict) else {}
        drugbank = hit.get("drugbank") if isinstance(hit.get("drugbank"), dict) else {}
        rows.append(
            {
                "id": hit.get("_id"),
                "name": hit.get("name") or hit.get("pref_name") or chembl.get("pref_name"),
                "chembl_id": chembl.get("molecule_chembl_id"),
                "drugbank_id": drugbank.get("id"),
                "inchi_key": hit.get("inchi_key"),
                "smiles": hit.get("smiles") or chembl.get("molecule_structures", {}).get("canonical_smiles") if isinstance(chembl.get("molecule_structures"), dict) else None,
                "score": hit.get("_score"),
            }
        )

    return {
        "summary": f"MyChem: found {len(rows)} hits for '{q}'.",
        "query": q,
        "hits": rows,
        "count": len(rows),
        "source": "mychem.info",
    }


@registry.register(
    name="data_api.pdbe_search",
    description="Search PDBe entries by keyword",
    category="data_api",
    parameters={
        "query": "PDBe keyword query (protein, ligand, organism, etc.)",
        "size": "Maximum hits (default 10)",
    },
    usage_guide="Use when you need PDBe-centric structure records and metadata.",
)
def pdbe_search(query: str, size: int = 10, **kwargs) -> dict:
    """Search PDBe Solr endpoint for structure entries."""
    q = (query or "").strip()
    if not q:
        return {"summary": "query is required.", "error": "missing_query"}

    params = {
        "q": q,
        "wt": "json",
        "rows": _coerce_int(size, 10),
        "fl": "pdb_id,title,experimental_method,resolution,organism_scientific_name",
    }
    data, error = request_json(
        "GET",
        "https://www.ebi.ac.uk/pdbe/search/pdb/select",
        params=params,
        timeout=20,
        retries=2,
    )
    if error:
        return {"summary": f"PDBe search failed: {error}", "error": "api_error"}

    docs = []
    if isinstance(data, dict):
        docs = ((data.get("response") or {}).get("docs") or [])
    rows = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        rows.append(
            {
                "pdb_id": doc.get("pdb_id"),
                "title": doc.get("title"),
                "experimental_method": doc.get("experimental_method"),
                "resolution": doc.get("resolution"),
                "organism": doc.get("organism_scientific_name"),
            }
        )

    return {
        "summary": f"PDBe: found {len(rows)} entries for '{q}'.",
        "query": q,
        "entries": rows,
        "count": len(rows),
        "source": "pdbe",
    }


@registry.register(
    name="data_api.reactome_pathway_search",
    description="Search Reactome pathways by keyword",
    category="data_api",
    parameters={
        "query": "Pathway or gene keyword",
        "size": "Maximum hits (default 10)",
    },
    usage_guide="Use to identify curated Reactome pathways related to a query.",
)
def reactome_pathway_search(query: str, size: int = 10, **kwargs) -> dict:
    """Search Reactome content service for pathways."""
    q = (query or "").strip()
    if not q:
        return {"summary": "query is required.", "error": "missing_query"}

    data, error = request_json(
        "GET",
        "https://reactome.org/ContentService/search/query",
        params={"query": q, "types": "Pathway", "cluster": "true"},
        timeout=20,
        retries=2,
    )
    if error:
        return {"summary": f"Reactome search failed: {error}", "error": "api_error"}

    results = []
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        results = data.get("results") or data.get("entries") or []

    rows = []
    for item in results[: _coerce_int(size, 10)]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "st_id": item.get("stId") or item.get("id"),
                "name": item.get("name") or item.get("displayName"),
                "species": item.get("species") or item.get("speciesName"),
                "type": item.get("type"),
                "url": (
                    f"https://reactome.org/content/detail/{item.get('stId')}"
                    if item.get("stId")
                    else None
                ),
            }
        )

    return {
        "summary": f"Reactome: found {len(rows)} pathway hits for '{q}'.",
        "query": q,
        "pathways": rows,
        "count": len(rows),
        "source": "reactome",
    }


@registry.register(
    name="literature.preprint_search",
    description="Search preprints across Europe PMC (bioRxiv/medRxiv) and arXiv",
    category="literature",
    parameters={
        "query": "Search query",
        "source": "Data source: europepmc, arxiv, or both (default both)",
        "max_results": "Maximum results (default 10)",
    },
    usage_guide="Use when you need latest preprint evidence that may not yet appear in PubMed.",
)
def preprint_search(query: str, source: str = "both", max_results: int = 10, **kwargs) -> dict:
    """Search preprints using EuropePMC and/or arXiv."""
    q = (query or "").strip()
    if not q:
        return {"summary": "query is required.", "error": "missing_query"}

    source_norm = str(source or "both").strip().lower()
    if source_norm not in {"both", "europepmc", "arxiv"}:
        return {"summary": "Invalid source. Use europepmc, arxiv, or both.", "error": "invalid_source"}

    limit = _coerce_int(max_results, 10, minimum=1, maximum=50)
    rows = []

    if source_norm in {"both", "europepmc"}:
        epmc_data, epmc_error = request_json(
            "GET",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": f"({q}) AND SRC:PPR",
                "format": "json",
                "pageSize": limit,
            },
            timeout=20,
            retries=2,
        )
        if epmc_error is None and isinstance(epmc_data, dict):
            result_list = (((epmc_data.get("resultList") or {}).get("result")) or [])
            for item in result_list:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or item.get("bookOrReportDetails", {}).get("title") if isinstance(item.get("bookOrReportDetails"), dict) else item.get("title")
                rows.append(
                    {
                        "source": "europepmc",
                        "id": item.get("id") or item.get("pmid") or item.get("doi"),
                        "title": _clip(title, 220),
                        "authors": _clip(item.get("authorString"), 180),
                        "journal": item.get("journalTitle") or item.get("pubType"),
                        "year": item.get("pubYear"),
                        "doi": item.get("doi"),
                        "url": item.get("fullTextUrl") or item.get("pmcid") or item.get("doi"),
                    }
                )

    if source_norm in {"both", "arxiv"}:
        resp, err = request(
            "GET",
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{q}",
                "start": 0,
                "max_results": limit,
            },
            timeout=20,
            retries=2,
            raise_for_status=False,
        )
        if err is None and resp is not None and int(resp.status_code) == 200:
            try:
                root = ET.fromstring(resp.text)
                ns = {"a": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("a:entry", ns):
                    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
                    published = (entry.findtext("a:published", default="", namespaces=ns) or "")
                    authors = [a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)]
                    rows.append(
                        {
                            "source": "arxiv",
                            "id": (entry.findtext("a:id", default="", namespaces=ns) or "").strip(),
                            "title": _clip(title, 220),
                            "authors": _clip(", ".join([a for a in authors if a]), 180),
                            "journal": "arXiv",
                            "year": published[:4] if published else None,
                            "doi": None,
                            "url": (entry.findtext("a:id", default="", namespaces=ns) or "").strip(),
                        }
                    )
            except Exception:
                pass

    # Deduplicate by title/url
    seen = set()
    deduped = []
    for row in rows:
        key = (row.get("title"), row.get("url"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    deduped = deduped[:limit]
    return {
        "summary": f"Found {len(deduped)} preprints for '{q}'.",
        "query": q,
        "source": source_norm,
        "articles": deduped,
        "count": len(deduped),
    }


def _count_aromatic_rings(mol) -> int:
    try:
        return sum(1 for ring in mol.GetRingInfo().AtomRings() if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
    except Exception:
        return 0


@registry.register(
    name="chemistry.sa_score",
    description="Estimate synthetic accessibility score (1 easy – 10 hard)",
    category="chemistry",
    parameters={"smiles": "Input SMILES"},
    usage_guide="Use during hit triage to reject compounds likely to be difficult to synthesize.",
)
def sa_score(smiles: str, **kwargs) -> dict:
    """Heuristic synthetic accessibility estimate based on molecular complexity."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
    except Exception:
        return {
            "summary": "RDKit is required for chemistry.sa_score.",
            "error": "missing_dependency",
        }

    smi = (smiles or "").strip()
    if not smi:
        return {"summary": "smiles is required.", "error": "missing_smiles"}

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"summary": f"Invalid SMILES: {smi}", "error": "invalid_smiles"}

    heavy = Descriptors.HeavyAtomCount(mol)
    rings = Lipinski.RingCount(mol)
    aromatic_rings = _count_aromatic_rings(mol)
    sp3 = Lipinski.FractionCSP3(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

    # Lightweight complexity model mapped into [1,10].
    complexity = 0.0
    complexity += min(heavy / 20.0, 2.5)
    complexity += min(rings * 0.35, 2.0)
    complexity += min(aromatic_rings * 0.25, 1.5)
    complexity += min(rot_bonds * 0.12, 1.2)
    complexity += min(stereo * 0.3, 1.8)
    complexity += max(0.0, (0.4 - float(sp3)) * 2.0)

    score = max(1.0, min(10.0, 1.5 + complexity))
    band = "easy" if score <= 3.5 else "moderate" if score <= 6.0 else "hard"

    return {
        "summary": f"Estimated synthetic accessibility score: {score:.2f}/10 ({band}).",
        "smiles": smi,
        "sa_score": round(score, 2),
        "difficulty": band,
        "features": {
            "heavy_atoms": int(heavy),
            "ring_count": int(rings),
            "aromatic_rings": int(aromatic_rings),
            "fraction_csp3": round(float(sp3), 3),
            "rotatable_bonds": int(rot_bonds),
            "stereocenters": int(stereo),
        },
        "note": "Heuristic estimate for prioritization; not a replacement for route planning.",
    }
