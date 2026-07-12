#!/usr/bin/env python3
"""M3a: wheat feature-set comparison on pilot traits (no leakage).

Compares inputs for Elastic Net / LightGBM:
  - ld_pruned
  - gwas_top{500,1000,5000} (GWAS only on train fold, QC SNPs + PC covariates)
  - pca10

Also reports Mean and GBLUP baselines on the same folds.
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
from sklearn.model_selection import GroupKFold, RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wheat import kinship_groups, load_wheat_arrays  # noqa: E402
from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402
from src.features.gwas import gwas_neglog10_p, topk_indices  # noqa: E402
from src.models.baselines import GBLUP, MeanModel, fit_elastic_net, fit_lightgbm  # noqa: E402


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def eval_pack(y_true, y_pred) -> dict[str, float]:
    m = regression_metrics(y_true, y_pred)
    m["top10_overlap"] = topk_overlap(y_true, y_pred, 0.10)
    return m


def standardize_train(X_tr, X_te):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X_tr - mu) / sd, (X_te - mu) / sd


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    np.random.seed(seed)

    interim = ROOT / cfg["paths"]["interim"] / "wheat"
    metrics_dir = ROOT / cfg["paths"]["results"] / "metrics"
    feat_dir = ROOT / cfg["paths"]["results"] / "features"
    pred_dir = ROOT / cfg["paths"]["results"] / "predictions"
    report_dir = ROOT / cfg["paths"]["reports"] / "figures"
    for d in (metrics_dir, feat_dir, pred_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = load_wheat_arrays(interim)
    samples = data["samples"]
    pheno = data["pheno"]
    X_ld, X_qc, K, pcs = data["X_ld"], data["X_qc"], data["K"], data["pcs"]
    snp_qc = data["snp_qc"]

    traits = cfg["wheat"]["pilot_traits"]
    n_splits = cfg["cv"]["n_splits"]
    # one random repeat for feature compare speed
    n_repeats = 1
    topk_list = [500, 1000, 5000]
    n_pcs_cov = min(5, pcs.shape[1])

    groups = kinship_groups(K, n_splits)
    schemes = []
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(rkf.split(np.arange(len(samples)))):
        schemes.append(("random", f"rep{i // n_splits}_fold{i % n_splits}", tr, te))
    gkf = GroupKFold(n_splits=n_splits)
    for i, (tr, te) in enumerate(gkf.split(np.arange(len(samples)), groups=groups)):
        schemes.append(("kinship_group", f"kinship_fold{i}", tr, te))

    metric_rows = []
    selected_rows = []
    pred_rows = []

    for trait in traits:
        y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            y = np.where(np.isfinite(y), y, np.nanmean(y))

        for scheme, fold_id, train_idx, test_idx in schemes:
            print(f"[{trait}] {scheme} {fold_id}", flush=True)
            y_tr, y_te = y[train_idx], y[test_idx]

            # Mean / GBLUP
            pred = MeanModel().fit(y_tr).predict(len(test_idx))
            metric_rows.append(
                {"trait": trait, "scheme": scheme, "fold": fold_id, "model": "mean", "features": "none", **eval_pack(y_te, pred)}
            )
            pred_rows.append(_preds(samples, test_idx, y_te, pred, trait, "mean", "none", scheme, fold_id))

            pred = GBLUP(h2=0.5).fit(K, y, train_idx).predict(test_idx)
            metric_rows.append(
                {"trait": trait, "scheme": scheme, "fold": fold_id, "model": "gblup", "features": "grm", **eval_pack(y_te, pred)}
            )
            pred_rows.append(_preds(samples, test_idx, y_te, pred, trait, "gblup", "grm", scheme, fold_id))

            # PCA features
            Xtr, Xte = standardize_train(pcs[train_idx, :10], pcs[test_idx, :10])
            for model_name, fit_fn, use_z in [
                ("elastic_net", fit_elastic_net, True),
                ("lightgbm", fit_lightgbm, False),
            ]:
                if use_z:
                    model = fit_fn(Xtr, y_tr, random_state=seed)
                    p = model.predict(Xte)
                else:
                    model = fit_fn(pcs[train_idx, :10], y_tr, random_state=seed)
                    p = model.predict(pcs[test_idx, :10])
                metric_rows.append(
                    {
                        "trait": trait,
                        "scheme": scheme,
                        "fold": fold_id,
                        "model": model_name,
                        "features": "pca10",
                        **eval_pack(y_te, p),
                    }
                )
                pred_rows.append(_preds(samples, test_idx, y_te, p, trait, model_name, "pca10", scheme, fold_id))

            # LD-pruned
            Xtr_ld, Xte_ld = X_ld[train_idx], X_ld[test_idx]
            Xtr_z, Xte_z = standardize_train(Xtr_ld, Xte_ld)
            for model_name, Xa, Xb in [
                ("elastic_net", Xtr_z, Xte_z),
                ("lightgbm", Xtr_ld, Xte_ld),
            ]:
                model = (fit_elastic_net if model_name == "elastic_net" else fit_lightgbm)(
                    Xa, y_tr, random_state=seed
                )
                p = model.predict(Xb)
                metric_rows.append(
                    {
                        "trait": trait,
                        "scheme": scheme,
                        "fold": fold_id,
                        "model": model_name,
                        "features": "ld_pruned",
                        **eval_pack(y_te, p),
                    }
                )
                pred_rows.append(_preds(samples, test_idx, y_te, p, trait, model_name, "ld_pruned", scheme, fold_id))

            # GWAS on train only (QC SNPs + PC covariates)
            scores = gwas_neglog10_p(X_qc[train_idx], y_tr, covariates=pcs[train_idx, :n_pcs_cov])
            for k in topk_list:
                idx = topk_indices(scores, k)
                feat_name = f"gwas_top{k}"
                for rank, j in enumerate(idx):
                    selected_rows.append(
                        {
                            "trait": trait,
                            "scheme": scheme,
                            "fold": fold_id,
                            "features": feat_name,
                            "rank": rank + 1,
                            "snp_idx": int(j),
                            "snp_id": snp_qc.iloc[j]["snp_id"],
                            "chrom": snp_qc.iloc[j]["chrom"],
                            "pos": int(snp_qc.iloc[j]["pos"]),
                            "neglog10p": float(scores[j]),
                        }
                    )
                Xtr = X_qc[train_idx][:, idx]
                Xte = X_qc[test_idx][:, idx]
                Xtr_z, Xte_z = standardize_train(Xtr, Xte)
                for model_name, Xa, Xb in [
                    ("elastic_net", Xtr_z, Xte_z),
                    ("lightgbm", Xtr, Xte),
                ]:
                    model = (fit_elastic_net if model_name == "elastic_net" else fit_lightgbm)(
                        Xa, y_tr, random_state=seed
                    )
                    p = model.predict(Xb)
                    metric_rows.append(
                        {
                            "trait": trait,
                            "scheme": scheme,
                            "fold": fold_id,
                            "model": model_name,
                            "features": feat_name,
                            **eval_pack(y_te, p),
                        }
                    )
                    pred_rows.append(
                        _preds(samples, test_idx, y_te, p, trait, model_name, feat_name, scheme, fold_id)
                    )

    metrics = pd.DataFrame(metric_rows)
    selected = pd.DataFrame(selected_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    metrics.to_csv(metrics_dir / "wheat_m3a_feature_metrics_by_fold.csv", index=False)
    selected.to_csv(feat_dir / "wheat_m3a_gwas_selected_snps.csv", index=False)
    preds.to_csv(pred_dir / "wheat_m3a_feature_predictions.csv", index=False)

    summary = (
        metrics.groupby(["trait", "scheme", "model", "features"], as_index=False)
        .agg(
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            rmse_mean=("rmse", "mean"),
            top10_overlap_mean=("top10_overlap", "mean"),
            n_folds=("rmse", "count"),
        )
        .sort_values(["trait", "scheme", "pearson_r_mean"], ascending=[True, True, False])
    )
    summary.to_csv(metrics_dir / "wheat_m3a_feature_metrics_summary.csv", index=False)

    # plot: LightGBM pearson by feature set
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
    for ax, trait in zip(axes.ravel(), traits):
        sub = summary[(summary.trait == trait) & (summary.model == "lightgbm")]
        for scheme, marker in [("random", "o"), ("kinship_group", "s")]:
            s2 = sub[sub.scheme == scheme]
            ax.errorbar(
                s2["features"],
                s2["pearson_r_mean"],
                yerr=s2["pearson_r_std"].fillna(0),
                fmt=marker + "-",
                label=scheme,
            )
        ax.set_title(trait)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=8)
    fig.suptitle("M3a LightGBM feature comparison")
    fig.tight_layout()
    fig.savefig(report_dir / "wheat_m3a_lgbm_feature_compare.png", dpi=150)
    plt.close(fig)

    gate = {
        "traits": traits,
        "topk_list": topk_list,
        "best_by_trait_scheme": {},
    }
    for trait in traits:
        for scheme in ["random", "kinship_group"]:
            sub = summary[(summary.trait == trait) & (summary.scheme == scheme)]
            # exclude mean nan pearson: rank by rmse then pearson
            sub2 = sub.copy()
            sub2["score"] = sub2["pearson_r_mean"].fillna(-1)
            best = sub2.sort_values(["score", "rmse_mean"], ascending=[False, True]).iloc[0]
            gate["best_by_trait_scheme"][f"{trait}|{scheme}"] = {
                "model": best["model"],
                "features": best["features"],
                "pearson_r": None if pd.isna(best["pearson_r_mean"]) else float(best["pearson_r_mean"]),
                "rmse": float(best["rmse_mean"]),
            }
    with open(metrics_dir / "wheat_m3a_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    print("M3a OK")
    print(summary.head(40).to_string(index=False))
    print(json.dumps(gate, indent=2))
    return 0


def _preds(samples, idx, y_true, y_pred, trait, model, features, scheme, fold):
    return pd.DataFrame(
        {
            "sample_id": [samples[i] for i in idx],
            "y_true": y_true,
            "y_pred": y_pred,
            "trait": trait,
            "model": model,
            "features": features,
            "scheme": scheme,
            "fold": fold,
        }
    )


if __name__ == "__main__":
    sys.exit(main())
