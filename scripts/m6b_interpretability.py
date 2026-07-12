#!/usr/bin/env python3
"""M6b: interpretability — LightGBM importances for wheat SNPs and maize env features."""
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
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wheat import load_wheat_arrays  # noqa: E402
from src.features.gwas import gwas_neglog10_p, topk_indices  # noqa: E402


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def fit_lgbm(X, y, seed):
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=seed,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(X, y)
    return model


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    feat_dir = ROOT / cfg["paths"]["results"] / "features"
    fig_dir = ROOT / cfg["paths"]["reports"] / "figures"
    model_dir = ROOT / cfg["paths"]["results"] / "models"
    for d in (feat_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Wheat: GWAS top1000 + LGBM importance on full data (explanatory only; CV results already saved) ----
    wheat = load_wheat_arrays(ROOT / cfg["paths"]["interim"] / "wheat")
    wheat_rows = []
    for trait in cfg["wheat"]["pilot_traits"]:
        y = pd.to_numeric(wheat["pheno"][trait], errors="coerce").to_numpy(float)
        y = np.where(np.isfinite(y), y, np.nanmean(y))
        scores = gwas_neglog10_p(wheat["X_qc"], y, covariates=wheat["pcs"][:, :5])
        idx = topk_indices(scores, 1000)
        X = wheat["X_qc"][:, idx]
        model = fit_lgbm(X, y, seed)
        imp = model.feature_importances_
        snp = wheat["snp_qc"].iloc[idx].reset_index(drop=True)
        tab = snp[["snp_id", "chrom", "pos"]].copy()
        tab["trait"] = trait
        tab["neglog10p"] = scores[idx]
        tab["lgbm_gain_importance"] = imp
        tab = tab.sort_values("lgbm_gain_importance", ascending=False)
        wheat_rows.append(tab)
        # save model
        model.booster_.save_model(str(model_dir / f"wheat_{trait}_lgbm_gwas1000.txt"))
    wheat_imp = pd.concat(wheat_rows, ignore_index=True)
    wheat_imp.to_csv(feat_dir / "wheat_m6b_lgbm_snp_importance.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, trait in zip(axes.ravel(), cfg["wheat"]["pilot_traits"]):
        s = wheat_imp[wheat_imp.trait == trait].head(15)
        ax.barh(s["snp_id"][::-1], s["lgbm_gain_importance"][::-1], color="#4C78A8")
        ax.set_title(f"Wheat {trait} top SNP importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "wheat_m6b_snp_importance.png", dpi=150)
    plt.close(fig)

    # ---- Maize: env feature importance on Yield raw means ----
    interim = ROOT / cfg["paths"]["interim"] / "maize"
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
    e_cols = [c for c in d.columns if c.startswith("E_PC") or c.startswith("full__")]
    # limit env raw features to manageable set: E_PCs + full window summary stats
    e_cols = [c for c in e_cols if c.startswith("E_PC") or ("full__" in c and any(k in c for k in ["mean", "sum", "hot_days", "cold_days", "frost", "high_vpd", "GDD"]))]
    feature_names = g_cols + e_cols
    X = d[feature_names].to_numpy(float)
    y = d["y_raw_mean"].to_numpy(float)
    # impute nan cols
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = col_mean[inds[1]]
    model = fit_lgbm(X, y, seed)
    model.booster_.save_model(str(model_dir / "maize_yield_lgbm_GE.txt"))
    imp = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    imp.to_csv(feat_dir / "maize_m6b_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp.head(25)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#F58518")
    ax.set_title("Maize Yield G+E LightGBM importance")
    fig.tight_layout()
    fig.savefig(fig_dir / "maize_m6b_feature_importance.png", dpi=150)
    plt.close(fig)

    summary = {
        "wheat_traits": cfg["wheat"]["pilot_traits"],
        "wheat_top_snp_file": "results/features/wheat_m6b_lgbm_snp_importance.csv",
        "maize_importance_file": "results/features/maize_m6b_feature_importance.csv",
        "maize_top10_features": imp.head(10)["feature"].tolist(),
        "disclaimer": "Importance/GWAS are associative for hypothesis generation, not causal effects.",
    }
    with open(feat_dir / "m6b_interpretability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("M6b OK")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
