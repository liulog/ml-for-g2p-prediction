from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def kinship_groups(K: np.ndarray, n_clusters: int) -> np.ndarray:
    dist = np.clip(1.0 - K, 0, None)
    np.fill_diagonal(dist, 0.0)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(dist)


def load_wheat_arrays(interim: Path) -> dict:
    interim = Path(interim)
    samples = pd.read_csv(interim / "wheat_samples_kept.csv")["sample_id"].tolist()
    pheno = pd.read_csv(interim / "wheat_pheno_aligned.csv")
    assert list(pheno["sample_id"]) == samples
    X_ld = np.load(interim / "wheat_dosage_ld_pruned.npy").T.astype(np.float32)
    X_qc = np.load(interim / "wheat_dosage_qc.npy").T.astype(np.float32)
    K = np.load(interim / "wheat_grm.npy").astype(np.float64)
    snp_ld = pd.read_parquet(interim / "wheat_snp_ld_pruned.parquet")
    snp_qc = pd.read_parquet(interim / "wheat_snp_qc.parquet")
    pca = pd.read_csv(interim / "wheat_pca.csv")
    assert list(pca["sample_id"]) == samples
    pcs = pca[[c for c in pca.columns if c.startswith("PC")]].to_numpy(dtype=float)
    return {
        "samples": samples,
        "pheno": pheno,
        "X_ld": X_ld,
        "X_qc": X_qc,
        "K": K,
        "snp_ld": snp_ld,
        "snp_qc": snp_qc,
        "pcs": pcs,
    }
