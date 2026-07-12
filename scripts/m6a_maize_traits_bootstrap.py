#!/usr/bin/env python3
"""M6a: maize secondary traits + paired bootstrap CIs on fold metrics."""
from __future__ import annotations

import json
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
from src.models.baselines import MeanModel  # noqa: E402


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def eval_pack(y_true, y_pred):
    m = regression_metrics(y_true, y_pred)
    m["top10_overlap"] = topk_overlap(y_true, y_pred, 0.10)
    return m


def lgbm_fit_predict(Xtr, ytr, Xte, seed):
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
    return model.fit(Xtr, ytr).predict(Xte)


def bootstrap_ci(values: np.ndarray, seed: int = 2026, n_boot: int = 2000) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def load_merged(interim: Path) -> pd.DataFrame:
    g_pc = pd.read_csv(interim / "maize_genotype_pca.csv")
    e_feat = pd.read_parquet(interim / "maize_env_features.parquet").reset_index()
    if "environment_id" not in e_feat.columns:
        e_feat = e_feat.rename(columns={e_feat.columns[0]: "environment_id"})
    e_pc = pd.read_csv(interim / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})
    gxe = pd.read_parquet(interim / "maize_routeB_gxe_blue.parquet")
    return gxe.merge(g_pc, on="genotype_id", how="left").merge(e_feat, on="environment_id", how="left").merge(
        e_pc, on="environment_id", how="left"
    )


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    interim = ROOT / cfg["paths"]["interim"] / "maize"
    metrics_dir = ROOT / cfg["paths"]["results"] / "metrics"
    pred_dir = ROOT / cfg["paths"]["results"] / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_merged(interim)
    traits = [cfg["maize"]["primary_trait"]] + cfg["maize"]["secondary_traits"]
    g_cols = [c for c in df_all.columns if c.startswith("G_PC")]
    e_pc_cols = [c for c in df_all.columns if c.startswith("E_PC")]
    key_e = [
        c
        for c in df_all.columns
        if c.startswith("full__")
        and any(k in c for k in ["GDD__sum", "PRECTOT__sum", "T2M__mean", "VPD__mean", "hot_days", "RAD__sum"])
    ]
    e_cols = e_pc_cols + key_e

    metric_rows = []
    pred_rows = []
    for trait in traits:
        d = df_all[df_all.trait == trait].dropna(subset=["y_raw_mean"]).copy()
        y = d["y_raw_mean"].to_numpy(float)
        G = d[g_cols].to_numpy(float)
        E = d[e_cols].to_numpy(float)
        GE = np.einsum("ij,ik->ijk", G[:, :5], E[:, :5]).reshape(len(d), -1)
        mats = {"G": G, "E": E, "G+E": np.hstack([G, E]), "G+E+GxE": np.hstack([G, E, GE])}
        for scheme, groups in {
            "leave_genotype": d["genotype_id"].to_numpy(),
            "leave_environment": d["environment_id"].to_numpy(),
        }.items():
            splits = min(5, pd.Series(groups).nunique())
            gkf = GroupKFold(n_splits=splits)
            print(f"[{trait}] {scheme} n={len(d)}", flush=True)
            for fold, (tr, te) in enumerate(gkf.split(np.arange(len(d)), groups=groups)):
                ytr, yte = y[tr], y[te]
                p = MeanModel().fit(ytr).predict(len(te))
                metric_rows.append({"trait": trait, "scheme": scheme, "fold": fold, "model": "mean", "features": "none", **eval_pack(yte, p)})
                for feat, X in mats.items():
                    p = lgbm_fit_predict(X[tr], ytr, X[te], seed)
                    metric_rows.append(
                        {"trait": trait, "scheme": scheme, "fold": fold, "model": "lightgbm", "features": feat, **eval_pack(yte, p)}
                    )
                    pred_rows.append(
                        pd.DataFrame(
                            {
                                "trait": trait,
                                "scheme": scheme,
                                "fold": fold,
                                "features": feat,
                                "genotype_id": d.iloc[te]["genotype_id"].to_numpy(),
                                "environment_id": d.iloc[te]["environment_id"].to_numpy(),
                                "y_true": yte,
                                "y_pred": p,
                            }
                        )
                    )

    metrics = pd.DataFrame(metric_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    metrics.to_csv(metrics_dir / "maize_m6a_traits_metrics_by_fold.csv", index=False)
    preds.to_csv(pred_dir / "maize_m6a_traits_predictions.csv", index=False)

    # bootstrap CIs over folds for pearson_r
    ci_rows = []
    for keys, g in metrics.groupby(["trait", "scheme", "model", "features"]):
        ci = bootstrap_ci(g["pearson_r"].to_numpy(), seed=seed)
        ci_rows.append(
            {
                "trait": keys[0],
                "scheme": keys[1],
                "model": keys[2],
                "features": keys[3],
                "pearson_r_mean": ci["mean"],
                "pearson_r_ci95_low": ci["ci95_low"],
                "pearson_r_ci95_high": ci["ci95_high"],
                "rmse_mean": float(g["rmse"].mean()),
                "n_folds": int(len(g)),
            }
        )
    ci_df = pd.DataFrame(ci_rows).sort_values(["trait", "scheme", "pearson_r_mean"], ascending=[True, True, False])
    ci_df.to_csv(metrics_dir / "maize_m6a_traits_metrics_with_ci.csv", index=False)

    # also bootstrap CIs for existing wheat kinship metrics and maize m5b
    extra = []
    for path, crop in [
        (metrics_dir / "wheat_m3b_all_traits_metrics_by_fold.csv", "wheat"),
        (metrics_dir / "maize_m5b_raw_yield_metrics_by_fold.csv", "maize_yield"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        group_cols = [c for c in ["trait", "scheme", "model", "features"] if c in df.columns]
        for keys, g in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            ci = bootstrap_ci(g["pearson_r"].to_numpy(), seed=seed)
            row = {c: v for c, v in zip(group_cols, keys)}
            row.update({"crop": crop, **{f"pearson_r_{k}": v for k, v in ci.items()}, "n_folds": int(len(g))})
            extra.append(row)
    if extra:
        pd.DataFrame(extra).to_csv(metrics_dir / "m6a_bootstrap_ci_existing.csv", index=False)

    gate = {"traits": traits, "best_leave_environment": {}}
    for trait in traits:
        sub = ci_df[(ci_df.trait == trait) & (ci_df.scheme == "leave_environment") & (ci_df.model == "lightgbm")]
        if len(sub) == 0:
            continue
        best = sub.sort_values("pearson_r_mean", ascending=False).iloc[0]
        gate["best_leave_environment"][trait] = {
            "features": best["features"],
            "r": float(best["pearson_r_mean"]),
            "ci95": [float(best["pearson_r_ci95_low"]), float(best["pearson_r_ci95_high"])],
        }
    with open(metrics_dir / "maize_m6a_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    print("M6a OK")
    print(ci_df.head(30).to_string(index=False))
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
