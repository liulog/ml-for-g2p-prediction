#!/usr/bin/env python3
"""M1c: LD pruning, PCA, and genomic relationship matrix for wheat.

Uses QC dosage matrix from M1b. No PLINK required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def ld_prune(X: np.ndarray, window: int, step: int, r2_thresh: float) -> np.ndarray:
    """Approximate PLINK --indep-pairwise on SNP x sample matrix (already imputed).

    Returns boolean keep mask over SNPs.
    """
    n_snp = X.shape[0]
    # center / standardize for correlation
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-8] = 1.0
    Z = (X - mu) / sd

    keep = np.ones(n_snp, dtype=bool)
    start = 0
    while start < n_snp:
        end = min(start + window, n_snp)
        idx = np.arange(start, end)
        active = idx[keep[idx]]
        if len(active) > 1:
            # correlation among active SNPs in window
            C = np.corrcoef(Z[active])
            # greedy: walk left->right, drop later SNP if r2 high with earlier kept
            local_keep = np.ones(len(active), dtype=bool)
            for i in range(len(active)):
                if not local_keep[i]:
                    continue
                r2 = C[i, i + 1 :] ** 2
                drop = np.where(r2 >= r2_thresh)[0]
                local_keep[i + 1 :][drop] = False
            keep[active] = local_keep
        start += step
    return keep


def main() -> int:
    cfg = load_cfg()
    qc = cfg["qc"]
    interim = ROOT / cfg["paths"]["interim"] / "wheat"
    report_dir = ROOT / cfg["paths"]["reports"] / "data_audit"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading QC dosages...", flush=True)
    X = np.load(interim / "wheat_dosage_qc.npy")  # snp x sample
    snp = pd.read_parquet(interim / "wheat_snp_qc.parquet")
    samples = pd.read_csv(interim / "wheat_samples_kept.csv")["sample_id"].tolist()
    assert X.shape == (len(snp), len(samples))

    print(
        f"LD prune window={qc['ld_window']} step={qc['ld_step']} r2={qc['ld_r2']} ...",
        flush=True,
    )
    keep = ld_prune(X, qc["ld_window"], qc["ld_step"], qc["ld_r2"])
    print(f"  kept {int(keep.sum())}/{len(keep)} SNPs", flush=True)

    snp_pruned = snp.loc[keep].reset_index(drop=True)
    X_pruned = X[keep]
    snp_pruned.to_parquet(interim / "wheat_snp_ld_pruned.parquet", index=False)
    np.save(interim / "wheat_dosage_ld_pruned.npy", X_pruned.astype(np.float32))
    pd.DataFrame({"snp_id": snp["snp_id"], "keep_ld": keep}).to_csv(
        interim / "wheat_ld_prune_mask.csv", index=False
    )

    # Center for GRM / PCA
    Z = X_pruned - X_pruned.mean(axis=1, keepdims=True)
    m = Z.shape[0]
    print("Computing GRM K = Z'Z / m ...", flush=True)
    # K is sample x sample
    K = (Z.T @ Z) / m
    np.save(interim / "wheat_grm.npy", K.astype(np.float32))
    pd.DataFrame(K, index=samples, columns=samples).to_csv(interim / "wheat_grm.csv")

    print("PCA on LD-pruned SNPs (samples as rows)...", flush=True)
    pca = PCA(n_components=min(10, len(samples) - 1), random_state=cfg["project"]["seed"])
    # sklearn expects samples x features
    pcs = pca.fit_transform(Z.T)
    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pc_df = pd.DataFrame(pcs, columns=pc_cols)
    pc_df.insert(0, "sample_id", samples)
    pc_df.to_csv(interim / "wheat_pca.csv", index=False)
    evr = pca.explained_variance_ratio_

    summary = {
        "n_snp_qc": int(len(snp)),
        "n_snp_ld_pruned": int(keep.sum()),
        "ld": {"window": qc["ld_window"], "step": qc["ld_step"], "r2": qc["ld_r2"]},
        "n_samples": len(samples),
        "pca_explained_variance_ratio": [float(x) for x in evr],
        "pca_cumulative_top5": float(evr[:5].sum()),
        "grm_diag_mean": float(np.diag(K).mean()),
        "grm_offdiag_mean": float((K.sum() - np.diag(K).sum()) / (K.size - len(K))),
        "files": {
            "dosage_ld": "data/interim/wheat/wheat_dosage_ld_pruned.npy",
            "snp_ld": "data/interim/wheat/wheat_snp_ld_pruned.parquet",
            "grm": "data/interim/wheat/wheat_grm.npy",
            "pca": "data/interim/wheat/wheat_pca.csv",
        },
    }
    with open(interim / "wheat_m1c_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pc_df["PC1"], pc_df["PC2"], s=12, alpha=0.7, c="#4C78A8")
    axes[0].set_xlabel(f"PC1 ({100*evr[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({100*evr[1]:.1f}%)")
    axes[0].set_title("Wheat PCA (LD-pruned)")
    im = axes[1].imshow(K, cmap="viridis", aspect="auto")
    axes[1].set_title("Genomic relationship matrix")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(report_dir / "wheat_pca_grm.png", dpi=150)
    plt.close(fig)

    print("M1c OK")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
