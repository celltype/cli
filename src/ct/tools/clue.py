"""
L1000/CMap compound signature and connectivity tools.

Uses local L1000 Level 5 compound profiles (19,811 compounds × 978 landmark genes)
built from the Broad LINCS GSE92742 dataset. Falls back to CLUE API if configured.
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path

from ct.tools import registry
from ct.agent.config import Config


def _get_clue_key() -> str | None:
    """Get CLUE API key from config/environment (for API-based fallback)."""
    import os
    cfg = Config.load()
    return cfg.get("clue.api_key") or os.environ.get("CLUE_API_KEY")


# ── Local data paths ──

_LINCS_DIR = Path("/mnt2/bronze/lincs")
_PROFILES_PATH = _LINCS_DIR / "l1000_compound_profiles.parquet"
_METADATA_PATH = _LINCS_DIR / "l1000_pert_metadata.parquet"


@lru_cache(maxsize=1)
def _load_profiles() -> pd.DataFrame:
    """Load compound profiles (19,811 compounds × 978 landmark genes)."""
    # Check configured path first
    cfg = Config.load()
    path = cfg.get("data.l1000_profiles")
    if path:
        p = Path(path)
        if p.is_file():
            return pd.read_parquet(p)

    if _PROFILES_PATH.exists():
        return pd.read_parquet(_PROFILES_PATH)

    # Search in data.base
    base = cfg.get("data.base")
    if base:
        candidate = Path(base) / "lincs" / "l1000_compound_profiles.parquet"
        if candidate.exists():
            return pd.read_parquet(candidate)

    raise FileNotFoundError(
        "L1000 compound profiles not found. "
        "Expected at: /mnt2/bronze/lincs/l1000_compound_profiles.parquet"
    )


@lru_cache(maxsize=1)
def _load_pert_metadata() -> pd.DataFrame:
    """Load perturbagen metadata (SMILES, PubChem CID, etc.)."""
    if _METADATA_PATH.exists():
        return pd.read_parquet(_METADATA_PATH)
    return pd.DataFrame()


def _find_compound(name: str, profiles: pd.DataFrame) -> str | None:
    """Find a compound in profiles by case-insensitive name matching."""
    name_lower = name.lower().strip()
    # Build lowercase index for matching
    idx_lower = {c.lower(): c for c in profiles.index}
    if name_lower in idx_lower:
        return idx_lower[name_lower]
    # Try partial match
    for key, original in idx_lower.items():
        if name_lower in key or key in name_lower:
            return original
    return None


@registry.register(
    name="clue.compound_signature",
    description="Get the L1000 transcriptomic signature (up/down-regulated genes) for a compound",
    category="clue",
    parameters={
        "compound": "Compound name (e.g. 'vorinostat', 'lenalidomide', 'bortezomib')",
        "top_n": "Number of top up/down genes to return (default 50)",
    },
    usage_guide=(
        "You need the transcriptomic signature (up/down genes) of a compound from L1000/CMap. "
        "Use to understand a compound's mechanism of action or as input for connectivity queries. "
        "Covers ~19,800 compounds from the Broad LINCS dataset."
    ),
)
def compound_signature(compound: str, top_n: int = 50, **kwargs) -> dict:
    """Get the L1000 transcriptomic signature for a compound from local data."""
    if not compound or not isinstance(compound, str):
        return {
            "error": "compound parameter required",
            "summary": "Provide a compound name (e.g. 'lenalidomide').",
        }

    try:
        profiles = _load_profiles()
    except FileNotFoundError as e:
        return {"error": str(e), "summary": str(e)}

    # Find compound in profiles
    matched = _find_compound(compound, profiles)
    if matched is None:
        return {
            "error": f"Compound '{compound}' not found in L1000 data ({len(profiles)} compounds available).",
            "summary": f"Compound '{compound}' not found in L1000/CMap database.",
        }

    # Extract profile
    profile = profiles.loc[matched]

    # Get top up and down regulated genes
    sorted_genes = profile.sort_values(ascending=False)
    up_genes = [
        {"gene": g, "z_score": round(float(v), 4)}
        for g, v in sorted_genes.head(top_n).items()
    ]
    down_genes = [
        {"gene": g, "z_score": round(float(v), 4)}
        for g, v in sorted_genes.tail(top_n).iloc[::-1].items()
    ]

    # Get metadata if available
    meta = _load_pert_metadata()
    pert_id = ""
    smiles = ""
    pubchem_cid = ""
    if matched in meta.index:
        row = meta.loc[matched]
        pert_id = str(row.get("pert_id", "")) if pd.notna(row.get("pert_id")) else ""
        smiles = str(row.get("canonical_smiles", "")) if pd.notna(row.get("canonical_smiles")) else ""
        pubchem_cid = str(row.get("pubchem_cid", "")) if pd.notna(row.get("pubchem_cid")) else ""

    summary = (
        f"L1000 signature for {matched}: "
        f"{top_n} up-regulated genes (top: {up_genes[0]['gene']} z={up_genes[0]['z_score']}), "
        f"{top_n} down-regulated genes (top: {down_genes[0]['gene']} z={down_genes[0]['z_score']})"
    )

    return {
        "summary": summary,
        "compound": matched,
        "pert_id": pert_id,
        "smiles": smiles,
        "pubchem_cid": pubchem_cid,
        "n_signatures_aggregated": len(profiles),
        "up_genes": up_genes,
        "down_genes": down_genes,
    }


@registry.register(
    name="clue.connectivity_query",
    description="Find compounds with similar or opposing transcriptomic signatures to a gene set",
    category="clue",
    parameters={
        "gene_list": "Dict with 'up' and 'down' keys, each a list of gene symbols",
        "n_results": "Number of top results to return (default 20)",
    },
    usage_guide=(
        "You have a gene signature (up/down-regulated genes) and want to find compounds "
        "with similar or opposing transcriptomic effects. Core CMap analysis. "
        "Use to find drug repurposing candidates or understand mechanism of action."
    ),
)
def connectivity_query(gene_list: dict = None, n_results: int = 20, **kwargs) -> dict:
    """Query local L1000 profiles with a gene signature using weighted connectivity scoring."""
    if not gene_list or not isinstance(gene_list, dict):
        return {
            "error": "gene_list must be a dict with 'up' and 'down' keys",
            "summary": "Invalid input: provide gene_list={'up': [...], 'down': [...]}",
        }

    up_genes = gene_list.get("up", [])
    down_genes = gene_list.get("down", [])

    if not up_genes and not down_genes:
        return {
            "error": "gene_list must have at least one gene in 'up' or 'down'",
            "summary": "Provide at least one up- or down-regulated gene.",
        }

    try:
        profiles = _load_profiles()
    except FileNotFoundError as e:
        return {"error": str(e), "summary": str(e)}

    # Find which query genes are in the profile columns
    available_up = [g for g in up_genes if g in profiles.columns]
    available_down = [g for g in down_genes if g in profiles.columns]

    if not available_up and not available_down:
        return {
            "error": "None of the query genes found in L1000 landmark genes.",
            "summary": (
                f"0/{len(up_genes)} up genes and 0/{len(down_genes)} down genes "
                "matched L1000 landmark genes (978 genes)."
            ),
        }

    # Compute connectivity score for each compound:
    # score = mean(z-scores of up genes) - mean(z-scores of down genes)
    # Positive score = compound mimics the signature
    # Negative score = compound opposes the signature
    score = np.zeros(len(profiles))

    if available_up:
        up_matrix = profiles[available_up].values
        score += np.nanmean(up_matrix, axis=1)
    if available_down:
        down_matrix = profiles[available_down].values
        score -= np.nanmean(down_matrix, axis=1)

    # Rank compounds
    results_df = pd.DataFrame({
        "compound": profiles.index,
        "connectivity_score": score,
    }).sort_values("connectivity_score", ascending=False)

    # Top similar (positive scores) and top opposing (negative scores)
    top_similar = results_df.head(n_results).to_dict("records")
    top_opposing = results_df.tail(n_results).iloc[::-1].to_dict("records")

    for row in top_similar + top_opposing:
        row["connectivity_score"] = round(row["connectivity_score"], 4)

    top = top_similar[0] if top_similar else {}
    summary = (
        f"Connectivity query: {len(available_up)}/{len(up_genes)} up, "
        f"{len(available_down)}/{len(down_genes)} down genes matched. "
        f"Scored {len(profiles)} compounds. "
        f"Top mimicker: {top.get('compound', 'N/A')} (score={top.get('connectivity_score', 0)}). "
        f"Top opposer: {top_opposing[0]['compound']} (score={top_opposing[0]['connectivity_score']})"
    )

    return {
        "summary": summary,
        "n_up_matched": len(available_up),
        "n_down_matched": len(available_down),
        "n_compounds_scored": len(profiles),
        "top_similar": top_similar,
        "top_opposing": top_opposing,
    }
