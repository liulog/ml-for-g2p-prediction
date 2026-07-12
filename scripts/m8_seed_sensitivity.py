#!/usr/bin/env python3
"""Plan §11.7: multi-seed sensitivity for wheat kinship CV (GBLUP / LightGBM)."""
from __future__ import annotations

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
from src.models.baselines import GBLUP, fit_lightgbm  # noqa: E402


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_cfg()
    base_seed = int(cfg["project"]["seed"])
    seeds = [base_seed, base_seed + 1, base_seed + 7, base_seed + 13, base_seed + 42]
    data = load_wheat_arrays(ROOT / "data" / "interim" / "wheat")
    X, K, pheno = data["X_ld"], data["K"], data["pheno"]
    # kinship groups depend on K clustering, not seed; seed affects LGBM + group label tie-breaks via n_splits path
    groups = kinship_groups(K, cfg["cv"]["n_splits"])
    gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])
    rows = []
    for trait in cfg["wheat"]["pilot_traits"]:
        y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(float)
        y = np.where(np.isfinite(y), y, np.nanmean(y))
        print(f"[{trait}]", flush=True)
        for seed in seeds:
            fold_rs = []
            for fold, (tr, te) in enumerate(gkf.split(np.arange(len(y)), groups=groups)):
                p_g = GBLUP(0.5).fit(K, y, tr).predict(te)
                p_l = fit_lightgbm(X[tr], y[tr], random_state=seed).predict(X[te])
                for name, p in [("gblup", p_g), ("lightgbm", p_l)]:
                    m = regression_metrics(y[te], p)
                    rows.append({"trait": trait, "seed": seed, "fold": fold, "model": name, "pearson_r": m["pearson_r"], "rmse": m["rmse"]})
                    if name == "gblup":
                        fold_rs.append(m["pearson_r"])
            print(f"  seed={seed} gblup_mean_r={np.nanmean(fold_rs):.3f}", flush=True)

    metrics = pd.DataFrame(rows)
    out = ROOT / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "wheat_m8_seed_sensitivity_by_fold.csv", index=False)
    seed_level = metrics.groupby(["trait", "model", "seed"], as_index=False).agg(pearson_r=("pearson_r", "mean"))
    seed_spread = seed_level.groupby(["trait", "model"], as_index=False).agg(
        r_mean=("pearson_r", "mean"),
        r_std_across_seeds=("pearson_r", "std"),
        r_min=("pearson_r", "min"),
        r_max=("pearson_r", "max"),
    )
    seed_spread.to_csv(out / "wheat_m8_seed_sensitivity_summary.csv", index=False)
    print("M8 seed sensitivity OK")
    print(seed_spread.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
