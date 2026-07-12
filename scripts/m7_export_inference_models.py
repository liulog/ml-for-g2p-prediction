#!/usr/bin/env python3
"""Export inference-ready models with explicit feature order metadata."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wheat import load_wheat_arrays  # noqa: E402
from src.models.baselines import RRBLUP, fit_lightgbm  # noqa: E402


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    model_dir = ROOT / "results" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Wheat: LD-pruned full-sample models for pilot traits
    wheat = load_wheat_arrays(ROOT / "data" / "interim" / "wheat")
    snp_ids = wheat["snp_ld"]["snp_id"].tolist()
    X = wheat["X_ld"]
    for trait in cfg["wheat"]["pilot_traits"]:
        y = pd.to_numeric(wheat["pheno"][trait], errors="coerce").to_numpy(float)
        y = np.where(np.isfinite(y), y, np.nanmean(y))
        rr = RRBLUP().fit(X, y)
        lgbm = fit_lightgbm(X, y, random_state=seed)
        joblib.dump(
            {"rrblup": rr, "lightgbm": lgbm, "snp_ids": snp_ids, "trait": trait},
            model_dir / f"wheat_{trait}_infer.joblib",
        )
        print(f"exported wheat_{trait}_infer.joblib", flush=True)

    # Maize Yield G+E with explicit feature order
    interim = ROOT / "data" / "interim" / "maize"
    gxe = pd.read_parquet(interim / "maize_routeB_yield_gxe.parquet")
    g_pc = pd.read_csv(interim / "maize_genotype_pca.csv")
    e_feat = pd.read_parquet(interim / "maize_env_features.parquet").reset_index()
    if "environment_id" not in e_feat.columns:
        e_feat = e_feat.rename(columns={e_feat.columns[0]: "environment_id"})
    e_pc = pd.read_csv(interim / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})
    d = gxe.merge(g_pc, on="genotype_id").merge(e_feat, on="environment_id").merge(e_pc, on="environment_id")
    d = d.dropna(subset=["y_raw_mean"])
    g_cols = [c for c in d.columns if c.startswith("G_PC")]
    e_cols = [
        c
        for c in d.columns
        if c.startswith("E_PC")
        or (
            c.startswith("full__")
            and any(k in c for k in ["mean", "sum", "hot_days", "cold_days", "frost", "high_vpd", "GDD"])
        )
    ]
    feats = g_cols + e_cols
    X = d[feats].to_numpy(float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = col_mean[inds[1]]
    y = d["y_raw_mean"].to_numpy(float)
    model = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=-1, force_col_wise=True,
    )
    model.fit(X, y)
    model.booster_.save_model(str(model_dir / "maize_yield_infer.txt"))
    meta = {"features": feats, "impute_mean": col_mean.tolist(), "target": "y_raw_mean"}
    (model_dir / "maize_yield_infer_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"exported maize_yield_infer.txt n_features={len(feats)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
