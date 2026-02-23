"""
Dataset downloader for ct.

Manages downloading and caching of common drug discovery datasets.
Supports automatic downloads for open-access datasets and guided
instructions for datasets requiring portal authentication.
"""

import gzip
import hashlib
import shutil
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, DownloadColumn, TransferSpeedColumn
from rich.table import Table

from ct.agent.config import Config

console = Console()

# Download timeout in seconds (large files like CRISPRGeneEffect ~700MB need more time)
DOWNLOAD_TIMEOUT = 600

DATASETS = {
    "depmap": {
        "description": "DepMap CRISPR gene dependencies, mutations, cell line metadata (24Q4)",
        "files": {
            "CRISPRGeneEffect.csv": "https://ndownloader.figshare.com/files/51064667",
            "Model.csv": "https://ndownloader.figshare.com/files/51065297",
            "OmicsSomaticMutationsMatrixDamaging.csv": "https://ndownloader.figshare.com/files/51065747",
        },
        "source": "https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248",
        "auto_download": True,
        "size_hint": "~580MB",
    },
    "prism": {
        "description": "PRISM cell viability screening data",
        "files": {
            "prism_LFC_COLLAPSED.csv": None,
        },
        "source": "https://depmap.org/repurposing/",
        "auto_download": False,
        "note": "PRISM data requires manual download from https://depmap.org/repurposing/ or symlink from existing data.",
        "size_hint": "~600MB",
    },
    "l1000": {
        "description": "L1000 landmark gene expression signatures (978 genes)",
        "files": {
            "l1000_landmark_only.parquet": None,
        },
        "source": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742",
        "auto_download": False,
        "note": "Run: python scripts/prepare_l1000.py to download from GEO and extract landmark subset.",
        "size_hint": "~200MB",
    },
    "msigdb": {
        "description": "MSigDB gene set collections (Hallmark, KEGG, Reactome, GO)",
        "files": {
            "h.all.v2024.1.Hs.json": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.json",
            "c2.cp.kegg_legacy.v2024.1.Hs.json": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.kegg_legacy.v2024.1.Hs.json",
            "c2.cp.reactome.v2024.1.Hs.json": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.reactome.v2024.1.Hs.json",
            "c5.go.bp.v2024.1.Hs.json": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c5.go.bp.v2024.1.Hs.json",
        },
        "source": "https://www.gsea-msigdb.org/gsea/msigdb/",
        "auto_download": True,
    },
    "string": {
        "description": "STRING protein-protein interaction network (human)",
        "files": {
            "9606.protein.links.v12.0.txt.gz": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        },
        "source": "https://string-db.org/",
        "auto_download": True,
    },
    "alphafold": {
        "description": "AlphaFold predicted protein structures (downloaded on demand per-protein)",
        "files": {},
        "source": "https://alphafold.ebi.ac.uk/",
        "auto_download": False,
        "note": "Structures are fetched on-demand by structure.alphafold_fetch tool.",
    },
}


def _download_file(url: str, dest: Path, desc: str = None) -> bool:
    """Download a file with progress bar. Returns True on success."""
    desc = desc or dest.name
    try:
        with httpx.stream("GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as resp:
            if resp.status_code != 200:
                console.print(f"  [red]HTTP {resp.status_code} for {url}[/red]")
                return False

            total = int(resp.headers.get("content-length", 0))

            with Progress(
                SpinnerColumn(),
                "[progress.description]{task.description}",
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"  {desc}", total=total or None)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        progress.advance(task, len(chunk))

        return True
    except httpx.HTTPError as e:
        console.print(f"  [red]Download failed: {e}[/red]")
        if dest.exists():
            dest.unlink()
        return False


def download_dataset(name: str, output: Path = None):
    """Download a dataset."""
    if name == "--all" or name == "all":
        download_all(output)
        return

    if name not in DATASETS:
        console.print(f"[red]Unknown dataset: {name}[/red]")
        console.print(f"Available: {', '.join(DATASETS.keys())}")
        return

    ds = DATASETS[name]
    cfg = Config.load()
    dest = output or Path(cfg.get("data.base")) / name
    dest.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[cyan]{name}:[/cyan] {ds['description']}")
    if ds.get("size_hint"):
        console.print(f"  Size: {ds['size_hint']}")
    console.print(f"  Destination: {dest}")

    if not ds.get("auto_download"):
        # Manual download required
        if "note" in ds:
            console.print(f"  [yellow]{ds['note']}[/yellow]")
        console.print(f"  Source: {ds['source']}")
        console.print(f"  Files needed:")
        for fname in ds["files"]:
            fpath = dest / fname
            status = "[green]found[/green]" if fpath.exists() else "[red]missing[/red]"
            console.print(f"    {fname} — {status}")
        console.print(f"\n  Download from {ds['source']} and place in {dest}/")
        console.print(f"  Then run: [cyan]ct config set data.{name} {dest}[/cyan]")
        return

    # Automatic download
    downloaded = 0
    skipped = 0
    failed = 0

    for fname, url in ds["files"].items():
        fpath = dest / fname
        if fpath.exists():
            console.print(f"  [dim]{fname} — already exists, skipping[/dim]")
            skipped += 1
            continue

        if url is None:
            console.print(f"  [yellow]{fname} — no download URL, skip[/yellow]")
            failed += 1
            continue

        if _download_file(url, fpath, fname):
            downloaded += 1
        else:
            failed += 1

    # Summary
    total = len(ds["files"])
    console.print(f"\n  [green]{downloaded} downloaded[/green], {skipped} skipped, ", end="")
    if failed:
        console.print(f"[red]{failed} failed[/red]")
    else:
        console.print(f"0 failed")

    # Auto-configure data path after successful download
    if downloaded > 0 or skipped > 0:
        cfg.set(f"data.{name}", str(dest))
        cfg.save()
        console.print(f"  [green]Auto-configured data.{name} = {dest}[/green]")


def download_all(output: Path = None):
    """Download all auto-downloadable datasets."""
    auto_datasets = [name for name, ds in DATASETS.items() if ds.get("auto_download")]
    console.print(f"[cyan]Downloading {len(auto_datasets)} datasets: {', '.join(auto_datasets)}[/cyan]")
    for name in auto_datasets:
        download_dataset(name, output=output)


def dataset_status() -> Table:
    """Check which datasets are available locally."""
    cfg = Config.load()
    base = Path(cfg.get("data.base"))

    table = Table(title="Dataset Status")
    table.add_column("Dataset", style="cyan")
    table.add_column("Status")
    table.add_column("Files", style="dim")
    table.add_column("Auto-DL")

    for name, ds in DATASETS.items():
        # Check custom config path first, then default location
        custom_path = cfg.get(f"data.{name}")
        path = Path(custom_path) if custom_path else base / name

        expected = set(ds["files"].keys())
        found = set()
        if path.exists():
            existing = {f.name for f in path.iterdir() if f.is_file()}
            found = expected & existing

        if not expected:
            status = "[dim]on-demand[/dim]"
            files_str = "-"
        elif found == expected:
            status = "[green]complete[/green]"
            files_str = f"{len(found)}/{len(expected)}"
        elif found:
            status = "[yellow]partial[/yellow]"
            files_str = f"{len(found)}/{len(expected)}"
        else:
            status = "[red]missing[/red]"
            files_str = f"0/{len(expected)}"

        auto = "[green]yes[/green]" if ds.get("auto_download") else "[dim]manual[/dim]"
        table.add_row(name, status, files_str, auto)

    return table
