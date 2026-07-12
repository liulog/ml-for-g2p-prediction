#!/usr/bin/env python3
"""M4d: maize genotype PCA from a SNP subset (memory-safe)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.plink import read_bed_dosages, read_bim, read_fam  # noqa: E402


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)
    prefix = ROOT / cfg["paths"]["maize_plink_prefix"]
    out_dir = ROOT / cfg["paths"]["interim"] / "maize"
    out_dir.mkdir(parents=True, exist_ok=True)

    fam = read_fam(prefix)
    bim = read_bim(prefix)
    n_snps = len(bim)
    n_keep = min(20000, n_snps)
    snp_idx = np.sort(rng.choice(n_snps, size=n_keep, replace=False))
    print(f"Reading {n_keep} SNPs for {len(fam)} hybrids...", flush=True)
    X = read_bed_dosages(prefix, snp_idx=snp_idx)  # snp x sample
    # impute SNP means
    with np.errstate(all="ignore"):
        means = np.nanmean(X, axis=1)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    inds = np.where(~np.isfinite(X))
    X[inds] = means[inds[0]]
    # filter near-monomorphic
    sd = X.std(axis=1)
    keep = sd > 1e-6
    X = X[keep]
    snp_idx = snp_idx[keep]
    print(f"SNPs after variance filter: {X.shape[0]}", flush=True)

    # samples x snps, center
    Z = (X - X.mean(axis=1, keepdims=True)).T.astype(np.float32)
    n_comp = min(50, Z.shape[0] - 1, Z.shape[1])
    print(f"PCA n_components={n_comp} ...", flush=True)
    pca = PCA(n_components=n_comp, random_state=seed)
    pcs = pca.fit_transform(Z)
    pc_df = pd.DataFrame(pcs, columns=[f"G_PC{i+1}" for i in range(n_comp)])
    pc_df.insert(0, "genotype_id", fam["genotype_id"].to_numpy())
    pc_df.to_csv(out_dir / "maize_genotype_pca.csv", index=False)
    pd.DataFrame(
        {"snp_idx": snp_idx, "snp": bim.iloc[snp_idx]["snp"].to_numpy(), "chr": bim.iloc[snp_idx]["chr"].to_numpy()}
    ).to_csv(out_dir / "maize_pca_snp_subset.csv", index=False)

    summary = {
        "n_hybrids": int(len(fam)),
        "n_snps_sampled": int(n_keep),
        "n_snps_used": int(X.shape[0]),
        "n_pcs": int(n_comp),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "cumulative_top10": float(pca.explained_variance_ratio_[:10].sum()),
    }
    with open(out_dir / "maize_m4d_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("M4d OK")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
