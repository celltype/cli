#!/usr/bin/env python3
"""
Prepare large datasets for the ct Data API.

Converts h5ad (AnnData) files to queryable Parquet format for DuckDB.
Run this on EC2 where /mnt2/bronze/ is mounted.

Usage:
    python scripts/prepare_datasets.py --dataset scperturb --input /mnt2/bronze/scperturb
    python scripts/prepare_datasets.py --dataset sciplex3 --input /mnt2/bronze/sciplex3
    python scripts/prepare_datasets.py --dataset perturbatlas --input /mnt2/bronze/perturbatlas

Requirements:
    pip install scanpy anndata pandas pyarrow
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def prepare_scperturb(input_dir: Path, output_dir: Path):
    """Convert scPerturb h5ad files to Parquet."""
    try:
        import scanpy as sc
    except ImportError:
        print("scanpy required: pip install scanpy")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_files = list(input_dir.glob("*.h5ad"))

    if not h5ad_files:
        print(f"No h5ad files found in {input_dir}")
        return

    for h5ad_path in h5ad_files:
        print(f"Processing: {h5ad_path.name}")
        adata = sc.read_h5ad(h5ad_path)

        # Extract metadata
        obs_df = adata.obs.reset_index()
        obs_out = output_dir / f"{h5ad_path.stem}_metadata.parquet"
        obs_df.to_parquet(obs_out)
        print(f"  Metadata: {obs_out} ({len(obs_df)} cells)")

        # Extract DEG results if available in uns
        if "rank_genes_groups" in adata.uns:
            degs = sc.get.rank_genes_groups_df(adata, group=None)
            deg_out = output_dir / f"{h5ad_path.stem}_degs.parquet"
            degs.to_parquet(deg_out)
            print(f"  DEGs: {deg_out} ({len(degs)} genes)")

    print(f"Done! Output in {output_dir}")


def prepare_sciplex3(input_path: Path, output_dir: Path):
    """Convert sciPlex3 h5ad to Parquet."""
    try:
        import scanpy as sc
    except ImportError:
        print("scanpy required: pip install scanpy")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Reading sciPlex3 ({input_path.stat().st_size / 1e9:.1f} GB)...")
    adata = sc.read_h5ad(input_path)

    # Extract metadata (cell-level: compound, dose, cell_type, etc.)
    obs_df = adata.obs.reset_index()
    obs_out = output_dir / "sciplex3_metadata.parquet"
    obs_df.to_parquet(obs_out)
    print(f"Metadata: {obs_out} ({len(obs_df)} cells)")

    # Extract dose-response summary per compound × gene
    if "pert_name" in obs_df.columns or "product_name" in obs_df.columns:
        compound_col = "pert_name" if "pert_name" in obs_df.columns else "product_name"
        dose_col = "dose" if "dose" in obs_df.columns else "pert_dose"

        if dose_col in obs_df.columns:
            summary = obs_df.groupby([compound_col, dose_col]).size().reset_index(name="n_cells")
            summary_out = output_dir / "sciplex3_dose_response.parquet"
            summary.to_parquet(summary_out)
            print(f"Dose-response: {summary_out} ({len(summary)} rows)")

    print(f"Done! Output in {output_dir}")


def prepare_perturbatlas(input_dir: Path, output_dir: Path):
    """Index PerturbAtlas CSV.gz files (already in queryable format)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_gz_files = list(input_dir.rglob("degs.csv.gz"))

    if not csv_gz_files:
        print(f"No degs.csv.gz files found in {input_dir}")
        return

    # Build an index of all experiments
    experiments = []
    for f in csv_gz_files:
        # Path structure: Homo sapiens/{experiment}/degs.csv.gz
        parts = f.relative_to(input_dir).parts
        experiment = parts[-2] if len(parts) >= 2 else f.stem
        experiments.append({
            "experiment": experiment,
            "path": str(f.relative_to(input_dir)),
            "size_bytes": f.stat().st_size,
        })

    index_df = pd.DataFrame(experiments)
    index_out = output_dir / "perturbatlas_index.parquet"
    index_df.to_parquet(index_out)
    print(f"Indexed {len(experiments)} experiments → {index_out}")
    print(f"Files can be queried directly by DuckDB using glob patterns.")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for ct Data API")
    parser.add_argument("--dataset", required=True,
                        choices=["scperturb", "sciplex3", "perturbatlas"],
                        help="Dataset to prepare")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input directory or file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: <input>_parquet/)")
    args = parser.parse_args()

    output = args.output or args.input.parent / f"{args.input.name}_parquet"

    if args.dataset == "scperturb":
        prepare_scperturb(args.input, output)
    elif args.dataset == "sciplex3":
        prepare_sciplex3(args.input, output)
    elif args.dataset == "perturbatlas":
        prepare_perturbatlas(args.input, output)


if __name__ == "__main__":
    main()
