#!/usr/bin/env python3
"""Plan remainder: wheat GWAS Top-K with vs without population-structure PCs."""
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
from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402
from src.features.gwas import gwas_neglog10_p, topk_indices  # noqa: E402
from src.models.baselines import fit_lightgbm  # noqa: E402


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def eval_pack(y_true, y_pred):
    m = regression_metrics(y_true, y_pred)
    m["top10_overlap"] = topk_overlap(y_true, y_pred, 0.10)
    return m


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    data = load_wheat_arrays(ROOT / "data" / "interim" / "wheat")
    X_qc, pcs, K, pheno = data["X_qc"], data["pcs"], data["K"], data["pheno"]
    groups = kinship_groups(K, cfg["cv"]["n_splits"])
    gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])
    rows = []
    for trait in cfg["wheat"]["pilot_traits"]:
        y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(float)
        y = np.where(np.isfinite(y), y, np.nanmean(y))
        print(f"[{trait}]", flush=True)
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(y)), groups=groups)):
            for use_pc, tag in [(False, "gwas_no_pc"), (True, "gwas_with_pc")]:
                cov = pcs[tr, :5] if use_pc else None
                scores = gwas_neglog10_p(X_qc[tr], y[tr], covariates=cov)
                idx = topk_indices(scores, 1000)
                model = fit_lightgbm(X_qc[tr][:, idx], y[tr], random_state=seed)
                p = model.predict(X_qc[te][:, idx])
                rows.append(
                    {
                        "trait": trait,
                        "fold": fold,
                        "features": tag,
                        "model": "lightgbm",
                        "scheme": "kinship_group",
                        **eval_pack(y[te], p),
                    }
                )
            print(f"  fold{fold}", flush=True)

    metrics = pd.DataFrame(rows)
    out = ROOT / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "wheat_m8_pc_ablation_by_fold.csv", index=False)
    summary = (
        metrics.groupby(["trait", "features"], as_index=False)
        .agg(pearson_r_mean=("pearson_r", "mean"), pearson_r_std=("pearson_r", "std"), rmse_mean=("rmse", "mean"))
        .sort_values(["trait", "features"])
    )
    # delta with_pc - no_pc
    piv = summary.pivot(index="trait", columns="features", values="pearson_r_mean")
    if "gwas_with_pc" in piv.columns and "gwas_no_pc" in piv.columns:
        piv["delta_with_minus_no"] = piv["gwas_with_pc"] - piv["gwas_no_pc"]
    piv.to_csv(out / "wheat_m8_pc_ablation_delta.csv")
    summary.to_csv(out / "wheat_m8_pc_ablation_summary.csv", index=False)
    print("M8 PC ablation OK")
    print(summary.to_string(index=False))
    print(piv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
