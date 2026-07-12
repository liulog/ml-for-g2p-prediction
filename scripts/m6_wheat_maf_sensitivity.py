#!/usr/bin/env python3
"""M6 robustness: wheat MAF 0.01 vs 0.05 sensitivity on pilot traits (kinship CV)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wheat import kinship_groups, load_wheat_arrays  # noqa: E402
from src.evaluation.metrics import regression_metrics  # noqa: E402
from src.models.baselines import GBLUP, MeanModel, fit_lightgbm  # noqa: E402


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    data = load_wheat_arrays(ROOT / cfg["paths"]["interim"] / "wheat")
    snp_qc = data["snp_qc"]
    X_qc = data["X_qc"]  # n x p
    K = data["K"]
    pheno = data["pheno"]
    # LD-pruned mask from interim
    snp_ld = data["snp_ld"]
    # maf from qc table aligned to X_qc columns
    maf = snp_qc["maf"].to_numpy(float) if "maf" in snp_qc.columns else None
    if maf is None:
        # compute quickly
        maf = np.minimum(X_qc.mean(0) / 2.0, 1.0 - X_qc.mean(0) / 2.0)

    # map ld snp ids to qc indices
    qc_id_to_idx = {s: i for i, s in enumerate(snp_qc["snp_id"])}
    ld_idx = np.array([qc_id_to_idx[s] for s in snp_ld["snp_id"] if s in qc_id_to_idx], dtype=int)

    groups = kinship_groups(K, cfg["cv"]["n_splits"])
    gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])
    rows = []
    for maf_thr in [0.01, 0.05]:
        keep = ld_idx[maf[ld_idx] >= maf_thr]
        X = X_qc[:, keep]
        print(f"MAF>={maf_thr}: {X.shape[1]} LD SNPs", flush=True)
        for trait in cfg["wheat"]["pilot_traits"]:
            y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(float)
            y = np.where(np.isfinite(y), y, np.nanmean(y))
            for fold, (tr, te) in enumerate(gkf.split(np.arange(len(y)), groups=groups)):
                # mean / gblup / lgbm
                for model_name, pred in [
                    ("mean", MeanModel().fit(y[tr]).predict(len(te))),
                    ("gblup", GBLUP(0.5).fit(K, y, tr).predict(te)),
                    ("lightgbm", fit_lightgbm(X[tr], y[tr], random_state=seed).predict(X[te])),
                ]:
                    m = regression_metrics(y[te], pred)
                    rows.append(
                        {
                            "maf_threshold": maf_thr,
                            "n_snps": int(X.shape[1]),
                            "trait": trait,
                            "fold": fold,
                            "model": model_name,
                            **m,
                        }
                    )
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "wheat_m6_maf_sensitivity_by_fold.csv", index=False)
    summary = (
        df.groupby(["maf_threshold", "trait", "model"], as_index=False)
        .agg(pearson_r_mean=("pearson_r", "mean"), rmse_mean=("rmse", "mean"), n_snps=("n_snps", "first"))
        .sort_values(["trait", "model", "maf_threshold"])
    )
    summary.to_csv(out / "wheat_m6_maf_sensitivity_summary.csv", index=False)
    # delta
    piv = summary.pivot_table(index=["trait", "model"], columns="maf_threshold", values="pearson_r_mean")
    if 0.01 in piv.columns and 0.05 in piv.columns:
        piv["delta_r_0p05_minus_0p01"] = piv[0.05] - piv[0.01]
    piv.to_csv(out / "wheat_m6_maf_sensitivity_delta.csv")
    print("M6 MAF OK")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
