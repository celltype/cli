"""
Data loaders for common drug discovery datasets.

Each loader checks configured paths, then the ct-data sister project,
and supports both CSV and Parquet formats.
Data paths are configured via ct config.
"""

import pandas as pd
from pathlib import Path
from functools import lru_cache

from ct.agent.config import Config


# Search order for data files (first match wins)
_DATA_SEARCH_PATHS = [
    Path.home() / "Projects" / "CellType" / "ct-data",         # Primary: ct-data repo
    Path.home() / "Projects" / "CellType" / "crews-glue-discovery",  # Legacy fallback
]


def _data_path(key: str) -> Path:
    """Get configured data path."""
    cfg = Config.load()
    path = cfg.get(f"data.{key}")
    if path:
        return Path(path)
    base = cfg.get("data.base")
    if base:
        return Path(base) / key
    return Path.home() / ".ct" / "data" / key


def _find_file(name: str, subdirs: list[str] = None) -> Path | None:
    """Search for a file across configured paths and common locations."""
    subdirs = subdirs or [""]
    stem = Path(name).stem

    # 1. Check configured data.base
    cfg = Config.load()
    base = cfg.get("data.base")
    search_dirs = []
    if base:
        search_dirs.append(Path(base))

    # 2. Check known data locations
    search_dirs.extend(_DATA_SEARCH_PATHS)

    # 3. Check ~/.ct/data
    search_dirs.append(Path.home() / ".ct" / "data")

    for base_dir in search_dirs:
        for sub in subdirs:
            d = base_dir / sub if sub else base_dir
            if not d.exists():
                continue
            # Exact match
            candidate = d / name
            if candidate.exists():
                return candidate
            # Try parquet variant
            parquet = d / f"{stem}.parquet"
            if parquet.exists():
                return parquet

    return None


def _resolve_path(p: Path, filenames: list[str]) -> Path | None:
    """If p is a directory, search for one of filenames inside it. If p is a file, return it."""
    if p.is_file():
        return p
    if p.is_dir():
        for name in filenames:
            candidate = p / name
            if candidate.exists():
                return candidate
    return None


def _read_tabular(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV or Parquet file based on extension."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **{k: v for k, v in kwargs.items() if k != "index_col"})
    return pd.read_csv(path, **kwargs)


@lru_cache(maxsize=1)
def load_crispr() -> pd.DataFrame:
    """Load DepMap CRISPR gene effect data."""
    # Try configured path first
    path = _find_file("CRISPRGeneEffect.csv", subdirs=["", "depmap"])
    if path is None:
        raise FileNotFoundError(
            "DepMap CRISPR data not found. "
            "Run: ct data pull depmap\n"
            "Or set: ct config set data.base /path/to/data"
        )
    df = _read_tabular(path, index_col=0)
    # Clean column names: "TP53 (7157)" → "TP53"
    df.columns = [c.split(' (')[0] for c in df.columns]
    return df


@lru_cache(maxsize=1)
def load_model_metadata() -> pd.DataFrame:
    """Load DepMap cell line metadata."""
    path = _find_file("Model.csv", subdirs=["", "depmap"])
    if path is None:
        raise FileNotFoundError(
            "Model metadata not found. "
            "Run: ct data pull depmap"
        )
    return _read_tabular(path)


@lru_cache(maxsize=1)
def load_proteomics() -> pd.DataFrame:
    """Load proteomics LFC matrix."""
    _prot_files = ["proteomics_log2fc_matrix.parquet", "proteomics_log2fc_matrix.csv", "merged_proteomics.csv"]
    # Check configured path
    cfg = Config.load()
    explicit = cfg.get("data.proteomics")
    if explicit:
        p = _resolve_path(Path(explicit), _prot_files)
        if p:
            return _read_tabular(p, index_col=0)

    # Search common locations
    for name in ["merged_proteomics.csv", "proteomics_log2fc_matrix.parquet",
                 "proteomics_log2fc_matrix.csv"]:
        path = _find_file(name)
        if path:
            return _read_tabular(path, index_col=0)

    raise FileNotFoundError(
        "Proteomics data not found. "
        "Set: ct config set data.proteomics /path/to/file"
    )


@lru_cache(maxsize=1)
def load_l1000() -> pd.DataFrame:
    """Load L1000 landmark gene expression data.

    Prefers the compound-named profiles parquet (19,811 compounds × 978 genes)
    built from the Broad LINCS GSE92742 Level 5 GCTX. Falls back to legacy
    formats (YU-indexed parquet/CSV).
    """
    cfg = Config.load()

    # 1. Prefer compound-named profiles (from LINCS GCTX, 19,811 compounds)
    lincs_path = Path("/mnt2/bronze/lincs/l1000_compound_profiles.parquet")
    if lincs_path.exists():
        return _read_tabular(lincs_path, index_col=0)

    # 2. Check data.base for compound profiles
    base = cfg.get("data.base")
    if base:
        candidate = Path(base) / "lincs" / "l1000_compound_profiles.parquet"
        if candidate.exists():
            return _read_tabular(candidate, index_col=0)

    # 3. Check explicit config (may point to legacy YU-indexed data)
    explicit = cfg.get("data.l1000")
    if explicit:
        _l1000_files = [
            "l1000_compound_profiles.parquet",
            "l1000_landmark_only.parquet",
            "L1000_landmark_LFC.csv",
            "l1000_expression_matrix.parquet",
            "l1000_landmark_only.csv",
        ]
        p = _resolve_path(Path(explicit), _l1000_files)
        if p:
            return _read_tabular(p, index_col=0)

    # 4. Fall back to legacy formats
    for name in ["l1000_compound_profiles.parquet", "L1000_landmark_LFC.csv",
                 "l1000_landmark_only.parquet", "l1000_expression_matrix.parquet",
                 "l1000_landmark_only.csv"]:
        path = _find_file(name, subdirs=["", "lincs", "l1000"])
        if path:
            return _read_tabular(path, index_col=0)

    raise FileNotFoundError(
        "L1000 data not found. "
        "Set: ct config set data.l1000 /path/to/file"
    )


@lru_cache(maxsize=1)
def load_prism() -> pd.DataFrame:
    """Load PRISM cell viability data."""
    _prism_files = ["prism_LFC_COLLAPSED.csv", "prism_LFC_COLLAPSED.parquet"]
    cfg = Config.load()
    explicit = cfg.get("data.prism")
    if explicit:
        p = _resolve_path(Path(explicit), _prism_files)
        if p:
            return _read_tabular(p)

    for name in ["prism_LFC_COLLAPSED.csv", "prism_LFC_COLLAPSED.parquet"]:
        path = _find_file(name)
        if path:
            return _read_tabular(path)

    raise FileNotFoundError(
        "PRISM data not found. "
        "Run: ct data pull prism\n"
        "Or set: ct config set data.prism /path/to/file"
    )


def load_mutations() -> pd.DataFrame:
    """Load DepMap somatic mutation data."""
    path = _find_file("OmicsSomaticMutationsMatrixDamaging.csv", subdirs=["", "depmap"])
    if path is None:
        raise FileNotFoundError(
            "Mutation data not found. "
            "Run: ct data pull depmap"
        )

    df = _read_tabular(path)
    meta_cols = ['Unnamed: 0', 'SequencingID', 'ModelConditionID',
                 'IsDefaultEntryForModel', 'IsDefaultEntryForMC']

    if 'IsDefaultEntryForModel' in df.columns:
        df = df[df['IsDefaultEntryForModel'] == 'Yes']

    if 'ModelID' in df.columns:
        df = df.set_index('ModelID')
    elif 'Unnamed: 0' in df.columns and df['Unnamed: 0'].astype(str).str.startswith('ACH-').any():
        df = df.set_index('Unnamed: 0')
        df.index.name = 'ModelID'
    df = df.drop(columns=[c for c in meta_cols if c in df.columns], errors='ignore')
    df.columns = [c.split(' (')[0] for c in df.columns]
    return df


def load_msigdb(collection: str = "h") -> dict:
    """Load MSigDB gene sets."""
    import json
    # Try both naming patterns: "h.all.v2024.1.Hs.json" and "c2.cp.kegg_legacy.v2024.1.Hs.json"
    for pattern in [f"{collection}.all.v2024.1.Hs.json", f"{collection}.v2024.1.Hs.json"]:
        path = _find_file(pattern, subdirs=["", "msigdb"])
        if path and path.exists():
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError("MSigDB data not found. Run: ct data pull msigdb")
