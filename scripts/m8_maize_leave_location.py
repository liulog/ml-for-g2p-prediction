#!/usr/bin/env python3
"""Plan §9.3: maize leave-Field_Location-out (new site) vs leave-year."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402


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
    interim = ROOT / "data" / "interim" / "maize"
    gxe = pd.read_parquet(interim / "maize_routeB_yield_gxe.parquet")
    g_pc = pd.read_csv(interim / "maize_genotype_pca.csv")
    e_pc = pd.read_csv(interim / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})
    d = gxe.merge(g_pc, on="genotype_id").merge(e_pc, on="environment_id").dropna(subset=["y_raw_mean"])
    y = d["y_raw_mean"].to_numpy(float)
    g_cols = [c for c in d.columns if c.startswith("G_PC")]
    e_cols = [c for c in d.columns if c.startswith("E_PC")]
    G = d[g_cols].to_numpy(float)
    E = d[e_cols].to_numpy(float)
    X = np.hstack([G, E])

    rows = []
    for scheme, groups in [
        ("leave_field_location", d["Field_Location"].astype(str).to_numpy()),
        ("leave_year", d["Year"].to_numpy()),
    ]:
        n_splits = min(5, pd.Series(groups).nunique())
        gkf = GroupKFold(n_splits=n_splits)
        print(f"{scheme} n_groups={pd.Series(groups).nunique()}", flush=True)
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(d)), groups=groups)):
            for name, feat in [
                ("G", G),
                ("E", E),
                ("G+E", X),
            ]:
                model = LGBMRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    verbosity=-1,
                    force_col_wise=True,
                )
                p = model.fit(feat[tr], y[tr]).predict(feat[te])
                rows.append({"scheme": scheme, "fold": fold, "features": name, "model": "lightgbm", **eval_pack(y[te], p)})
            print(f"  fold{fold}", flush=True)

    out = ROOT / "results" / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "maize_m8_leave_location_by_fold.csv", index=False)
    summary = (
        metrics.groupby(["scheme", "features"], as_index=False)
        .agg(pearson_r_mean=("pearson_r", "mean"), pearson_r_std=("pearson_r", "std"), rmse_mean=("rmse", "mean"))
        .sort_values(["scheme", "pearson_r_mean"], ascending=[True, False])
    )
    summary.to_csv(out / "maize_m8_leave_location_summary.csv", index=False)
    print("M8 leave-location OK")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
