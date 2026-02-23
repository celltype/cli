#!/usr/bin/env python3
"""
Prepare L1000 landmark gene expression data from GEO.

Downloads the raw Level 5 GCTx file from GEO (GSE92742) and extracts
the 978 landmark gene subset into a compact Parquet file for use with ct.

Usage:
    python scripts/prepare_l1000.py [--output ~/.ct/data/l1000]

Requirements:
    pip install cmapPy pandas pyarrow
"""

import argparse
import gzip
import os
import sys
from pathlib import Path

import httpx
import pandas as pd


# GEO download URL for GSE92742 Level 5 (MODZ signatures)
GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
    "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
)

# 978 landmark genes (subset IDs)
LANDMARK_SPACE = "landmark"


def download_gctx(output_dir: Path) -> Path:
    """Download the raw GCTx file from GEO."""
    gz_path = output_dir / "GSE92742_Level5.gctx.gz"
    gctx_path = output_dir / "GSE92742_Level5.gctx"

    if gctx_path.exists():
        print(f"GCTx file already exists: {gctx_path}")
        return gctx_path

    if not gz_path.exists():
        print(f"Downloading from GEO (~2.3GB)...")
        print(f"URL: {GEO_URL}")
        with httpx.stream("GET", GEO_URL, timeout=3600, follow_redirects=True) as resp:
            if resp.status_code != 200:
                print(f"Download failed: HTTP {resp.status_code}")
                sys.exit(1)
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(gz_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded / 1e9:.1f} / {total / 1e9:.1f} GB ({pct:.0f}%)", end="")
            print()

    print("Decompressing...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(gctx_path, "wb") as f_out:
            while True:
                chunk = f_in.read(65536)
                if not chunk:
                    break
                f_out.write(chunk)

    # Clean up gz
    gz_path.unlink()
    return gctx_path


def extract_landmarks(gctx_path: Path, output_dir: Path) -> Path:
    """Extract landmark genes from GCTx into Parquet."""
    out_path = output_dir / "l1000_landmark_only.parquet"

    if out_path.exists():
        print(f"Landmark parquet already exists: {out_path}")
        return out_path

    try:
        from cmapPy.pandasGEXpress.parse import parse
    except ImportError:
        print("cmapPy required: pip install cmapPy")
        sys.exit(1)

    print("Parsing GCTx and extracting landmark genes...")
    gctoo = parse(str(gctx_path), rid=LANDMARK_SPACE)
    df = gctoo.data_df

    print(f"Extracted: {df.shape[0]} genes x {df.shape[1]} signatures")

    # Save as parquet
    df.to_parquet(out_path)
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare L1000 landmark data from GEO")
    parser.add_argument("--output", type=Path, default=Path.home() / ".ct" / "data" / "l1000",
                        help="Output directory (default: ~/.ct/data/l1000)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}")

    gctx_path = download_gctx(args.output)
    parquet_path = extract_landmarks(gctx_path, args.output)

    # Auto-configure ct
    try:
        from ct.agent.config import Config
        cfg = Config.load()
        cfg.set("data.l1000", str(args.output))
        cfg.save()
        print(f"\nAuto-configured data.l1000 = {args.output}")
    except ImportError:
        print(f"\nManual config: ct config set data.l1000 {args.output}")

    # Clean up raw GCTx (large)
    if gctx_path.exists() and parquet_path.exists():
        size_gb = gctx_path.stat().st_size / 1e9
        resp = input(f"\nDelete raw GCTx ({size_gb:.1f} GB) to save space? [y/N] ")
        if resp.lower() == "y":
            gctx_path.unlink()
            print("Deleted raw GCTx file.")

    print("\nDone! L1000 landmark data is ready.")


if __name__ == "__main__":
    main()
