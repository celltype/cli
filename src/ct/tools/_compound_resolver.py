"""Compound name resolver — maps drug names to dataset-specific IDs.

The ct datasets use proprietary compound IDs:
- PRISM/L1000: YU-codes (e.g., YU254653)
- Proteomics: Cmpd format (e.g., Cmpd18_B10)

This module resolves common drug names (lenalidomide, pomalidomide, etc.)
to the most structurally similar compound in each dataset via Tanimoto similarity.
"""

import csv
import os
import re
from functools import lru_cache

# Data file paths
_DATA_DIR = "/mnt2/bronze/molecular_glue/crews_library"
_SMILES_CSV = os.path.join(_DATA_DIR, "all_compounds_smiles.csv")
_PROT_MAPPING_CSV = os.path.join(_DATA_DIR, "proteomics_to_yu_mapping.csv")

# Regex patterns
_YU_PATTERN = re.compile(r"^YU\d{6}$")
_CMPD_PATTERN = re.compile(r"^Cmpd\d+")

# Module-level caches (populated on first use)
_yu_smiles: dict | None = None
_prot_to_yu: dict | None = None
_yu_to_prot: dict | None = None


def _load_yu_smiles() -> dict:
    """Load YU compound → SMILES mapping (lazy, cached)."""
    global _yu_smiles
    if _yu_smiles is not None:
        return _yu_smiles
    _yu_smiles = {}
    if not os.path.exists(_SMILES_CSV):
        return _yu_smiles
    with open(_SMILES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            _yu_smiles[row["compound"]] = row["smiles"]
    return _yu_smiles


def _load_prot_mapping() -> tuple[dict, dict]:
    """Load proteomics Cmpd ↔ YU mapping (lazy, cached)."""
    global _prot_to_yu, _yu_to_prot
    if _prot_to_yu is not None:
        return _prot_to_yu, _yu_to_prot
    _prot_to_yu = {}
    _yu_to_prot = {}
    if not os.path.exists(_PROT_MAPPING_CSV):
        return _prot_to_yu, _yu_to_prot
    with open(_PROT_MAPPING_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            _prot_to_yu[row["cmpd_id"]] = row["yu_id"]
            _yu_to_prot[row["yu_id"]] = row["cmpd_id"]
    return _prot_to_yu, _yu_to_prot


@lru_cache(maxsize=64)
def _tanimoto_search(smiles: str, candidate_ids: frozenset) -> tuple[str, float] | None:
    """Find most similar YU compound to a SMILES string by Tanimoto similarity."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    yu_smiles = _load_yu_smiles()
    best_id, best_sim = None, 0.0

    for yu_id in candidate_ids:
        smi = yu_smiles.get(yu_id)
        if not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        if sim > best_sim:
            best_sim = sim
            best_id = yu_id

    if best_id is None:
        return None
    return best_id, round(best_sim, 4)


def resolve_to_smiles(name_or_smiles: str) -> str:
    """Resolve a compound name or SMILES string to a canonical SMILES string.

    Resolution order:
    1. Try parsing as SMILES with RDKit (if installed)
    2. Try PubChem lookup via API
    3. Try ChEMBL lookup via API
    4. Raise ValueError if all methods fail

    Parameters
    ----------
    name_or_smiles : str
        Drug name (e.g. "lenalidomide") or SMILES string.

    Returns
    -------
    str
        Canonical SMILES string.

    Raises
    ------
    ValueError
        If the input cannot be resolved to a SMILES string.
    """
    if not name_or_smiles or not isinstance(name_or_smiles, str):
        raise ValueError(f"Invalid input: {name_or_smiles}")

    name_or_smiles = name_or_smiles.strip()
    if not name_or_smiles:
        raise ValueError("Empty input")

    # 1. Try RDKit parse — is it already a valid SMILES?
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(name_or_smiles)
        if mol is not None:
            return name_or_smiles
    except ImportError:
        # No RDKit — heuristic: if it has typical SMILES characters, assume it's SMILES
        if any(c in name_or_smiles for c in "=#()[]"):
            return name_or_smiles

    # 2. Try PubChem lookup
    try:
        from ct.tools.chemistry import pubchem_lookup

        result = pubchem_lookup(name_or_smiles, query_type="name")
        smiles = (result.get("properties") or {}).get("canonical_smiles")
        if smiles:
            return smiles
    except Exception:
        pass

    # 3. Try ChEMBL lookup
    try:
        from ct.tools.literature import chembl_query

        result = chembl_query(name_or_smiles, query_type="molecule", max_results=1)
        molecules = result.get("molecules", [])
        if molecules and molecules[0].get("smiles"):
            return molecules[0]["smiles"]
    except Exception:
        pass

    raise ValueError(
        f"Could not resolve '{name_or_smiles}' to a SMILES string. "
        "Tried: RDKit SMILES parse, PubChem, ChEMBL."
    )


def resolve_compound(name_or_id: str, dataset: str = "prism") -> str:
    """Resolve a compound name or ID to a dataset-specific ID.

    Parameters
    ----------
    name_or_id : str
        Drug name (e.g. "lenalidomide"), YU code, or Cmpd ID.
    dataset : str
        Target dataset: "prism", "l1000", or "proteomics".

    Returns
    -------
    str
        Resolved compound ID for the target dataset.
        For L1000 with compound-named index, returns the compound name directly.
        For proteomics, returns the full Cmpd_well ID if possible.
        Falls back to the original input if resolution fails.
    """
    if not name_or_id or not isinstance(name_or_id, str):
        return name_or_id

    name_or_id = name_or_id.strip()

    # For L1000: check if the index uses compound names (new format) vs YU codes (legacy)
    if dataset == "l1000":
        try:
            from ct.data.loaders import load_l1000
            l1000 = load_l1000()
            sample_idx = str(l1000.index[0]) if len(l1000) > 0 else ""
            if not _YU_PATTERN.match(sample_idx):
                # New compound-named index: case-insensitive lookup
                name_lower = name_or_id.lower().strip()
                idx_lower = {c.lower(): c for c in l1000.index}
                if name_lower in idx_lower:
                    return idx_lower[name_lower]
                # Try partial match
                for key, original in idx_lower.items():
                    if name_lower in key or key in name_lower:
                        return original
                return name_or_id
        except (FileNotFoundError, ImportError):
            pass

    # Already a YU code — return as-is for PRISM/L1000, convert for proteomics
    if _YU_PATTERN.match(name_or_id):
        if dataset == "proteomics":
            return _yu_to_proteomics_col(name_or_id) or name_or_id
        return name_or_id

    # Already a Cmpd ID — return as-is for proteomics, convert for PRISM/L1000
    if _CMPD_PATTERN.match(name_or_id):
        if dataset == "proteomics":
            return name_or_id
        prot_to_yu, _ = _load_prot_mapping()
        base = name_or_id.split("_")[0]
        yu_id = prot_to_yu.get(base)
        return yu_id if yu_id else name_or_id

    # Drug name — try to resolve SMILES via API, then find closest match in dataset
    try:
        drug_smi = resolve_to_smiles(name_or_id)
    except ValueError:
        return name_or_id  # Cannot resolve to SMILES, return as-is

    # Dynamic SMILES-based resolution (cached)
    # Get candidate compounds from the target dataset
    candidates = _get_dataset_compounds(dataset)
    if not candidates:
        return name_or_id

    result = _tanimoto_search(drug_smi, frozenset(candidates))
    if result is None:
        return name_or_id

    yu_id, sim = result
    # Low-similarity proxies produce misleading data — return the original
    # name so the tool reports "not found" and synthesis uses LLM knowledge
    if sim < 0.65:
        return name_or_id
    if dataset == "proteomics":
        return _yu_to_proteomics_col(yu_id) or yu_id
    return yu_id


def resolve_proteomics_id(yu_id: str) -> str | None:
    """Convert a YU compound ID to the proteomics Cmpd_well column name.

    Returns None if no mapping exists.
    """
    return _yu_to_proteomics_col(yu_id)


def _yu_to_proteomics_col(yu_id: str) -> str | None:
    """Map YU ID → full proteomics column name (Cmpd##_well)."""
    _, yu_to_prot = _load_prot_mapping()
    base_cmpd = yu_to_prot.get(yu_id)
    if base_cmpd is None:
        return None

    # Find the full column name in proteomics data
    try:
        from ct.data.loaders import load_proteomics
        prot = load_proteomics()
        for col in prot.columns:
            if col.startswith(base_cmpd + "_"):
                return col
    except (FileNotFoundError, ImportError):
        pass

    return base_cmpd


def _get_dataset_compounds(dataset: str) -> set:
    """Get the set of YU compound IDs available in a dataset."""
    try:
        if dataset == "prism":
            from ct.data.loaders import load_prism
            prism = load_prism()
            return set(prism["pert_name"].unique())
        elif dataset == "l1000":
            from ct.data.loaders import load_l1000
            l1000 = load_l1000()
            return set(l1000.index.tolist())
        elif dataset == "proteomics":
            _, yu_to_prot = _load_prot_mapping()
            return set(yu_to_prot.keys())
    except (FileNotFoundError, ImportError):
        pass
    return set()
