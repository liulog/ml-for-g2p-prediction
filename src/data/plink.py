"""Utilities for reading PLINK bed dosages (additive 0/1/2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_fam(prefix: Path) -> pd.DataFrame:
    fam = pd.read_csv(
        f"{prefix}.fam",
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "father", "mother", "sex", "pheno"],
        dtype=str,
    )
    fam["genotype_id"] = fam["iid"].astype(str).str.strip()
    return fam


def read_bim(prefix: Path) -> pd.DataFrame:
    return pd.read_csv(
        f"{prefix}.bim",
        sep="\t",
        header=None,
        names=["chr", "snp", "cm", "bp", "a1", "a2"],
        dtype={"chr": str},
    )


def read_bed_dosages(
    prefix: Path,
    snp_idx: np.ndarray | None = None,
    sample_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Read additive dosages from PLINK bed (SNP-major). Missing -> nan.

    Returns float32 array shaped (n_snps_selected, n_samples_selected).
    """
    prefix = Path(prefix)
    fam = read_fam(prefix)
    bim = read_bim(prefix)
    n_samples = len(fam)
    n_snps = len(bim)
    if snp_idx is None:
        snp_idx = np.arange(n_snps)
    else:
        snp_idx = np.asarray(snp_idx, dtype=int)
    if sample_idx is None:
        sample_idx = np.arange(n_samples)
    else:
        sample_idx = np.asarray(sample_idx, dtype=int)

    # bytes per SNP (SNP-major): ceil(n_samples/4)
    bpp = (n_samples + 3) // 4
    bed_path = Path(f"{prefix}.bed")
    with open(bed_path, "rb") as f:
        magic = f.read(3)
        if magic != b"\x6c\x1b\x01":
            raise ValueError(f"Not a SNP-major PLINK bed: {magic!r}")
        out = np.empty((len(snp_idx), len(sample_idx)), dtype=np.float32)
        # PLINK bed genotype codes -> dosage of A1 (usually minor): 00=hom A1, 01=missing, 10=het, 11=hom A2
        # map to additive count of A1: 0->2? Wait PLINK: actually bit coding:
        # 00 -> 0 (homozygous A1), 01 -> missing, 10 -> 1 (het), 11 -> 2 (homozygous A2)
        # Standard additive dosage of A2 (alt-like): 0, nan, 1, 2 for codes 0,1,2,3 in little-endian bit pairs.
        code_to_dose = np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float32)
        for oi, si in enumerate(snp_idx):
            f.seek(3 + int(si) * bpp)
            raw = np.frombuffer(f.read(bpp), dtype=np.uint8)
            # unpack 2-bit genotypes for all samples
            # each byte holds 4 samples (low bits first)
            g = np.empty(n_samples, dtype=np.uint8)
            for b_i, byte in enumerate(raw):
                base = b_i * 4
                for k in range(4):
                    idx = base + k
                    if idx >= n_samples:
                        break
                    g[idx] = (byte >> (2 * k)) & 0b11
            out[oi] = code_to_dose[g[sample_idx]]
            if (oi + 1) % 5000 == 0:
                print(f"  bed read {oi + 1}/{len(snp_idx)} SNPs", flush=True)
    return out
