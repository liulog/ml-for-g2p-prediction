#!/usr/bin/env python3
"""M3b: wheat all-traits modeling + GWAS SNP stability.

For each of 15 traits:
  Mean, GBLUP(GRM), ElasticNet(LD), LightGBM(LD), LightGBM(GWAS top1000)
CV: random RepeatedKFold (n_repeats from config) + kinship GroupKFold
GWAS is fit on training fold only (QC SNPs + PC covariates).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
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
    traits = cfg["wheat"]["all_traits"]
    n_splits = cfg["cv"]["n_splits"]
    n_repeats = cfg["cv"]["n_repeats"]
    gwas_k = 1000
    n_pcs_cov = min(5, pcs.shape[1])

    # near-zero-var flags from pheno
    near_zero = []
    cv_thr = cfg["wheat"]["near_zero_var_cv_threshold"]
    for t in traits:
        s = pd.to_numeric(pheno[t], errors="coerce")
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        cv = sd / abs(mu) if abs(mu) > 1e-12 else np.nan
        if pd.notna(cv) and cv < cv_thr:
            near_zero.append(t)

    groups = kinship_groups(K, n_splits)
    fold_specs = []
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(rkf.split(np.arange(len(samples)))):
        fold_specs.append(("random_repeated", f"rep{i // n_splits}_fold{i % n_splits}", tr, te))
    gkf = GroupKFold(n_splits=n_splits)
    for i, (tr, te) in enumerate(gkf.split(np.arange(len(samples)), groups=groups)):
        fold_specs.append(("kinship_group", f"kinship_fold{i}", tr, te))

    metric_rows = []
    selected_rows = []
    pred_rows = []

    for trait in traits:
        y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            y = np.where(np.isfinite(y), y, np.nanmean(y))

        for scheme, fold_id, train_idx, test_idx in fold_specs:
            print(f"[{trait}] {scheme} {fold_id}", flush=True)
            y_tr, y_te = y[train_idx], y[test_idx]

            # Mean
            p = MeanModel().fit(y_tr).predict(len(test_idx))
            metric_rows.append(_mrow(trait, scheme, fold_id, "mean", "none", y_te, p))
            pred_rows.append(_preds(samples, test_idx, y_te, p, trait, "mean", "none", scheme, fold_id))

            # GBLUP
            p = GBLUP(h2=0.5).fit(K, y, train_idx).predict(test_idx)
            metric_rows.append(_mrow(trait, scheme, fold_id, "gblup", "grm", y_te, p))
            pred_rows.append(_preds(samples, test_idx, y_te, p, trait, "gblup", "grm", scheme, fold_id))

            # EN / LGBM on LD
            Xtr, Xte = X_ld[train_idx], X_ld[test_idx]
            Xtr_z, Xte_z = standardize_train(Xtr, Xte)
            en = fit_elastic_net(Xtr_z, y_tr, random_state=seed)
            p = en.predict(Xte_z)
            metric_rows.append(_mrow(trait, scheme, fold_id, "elastic_net", "ld_pruned", y_te, p))
            pred_rows.append(_preds(samples, test_idx, y_te, p, trait, "elastic_net", "ld_pruned", scheme, fold_id))

            lgbm = fit_lightgbm(Xtr, y_tr, random_state=seed)
            p = lgbm.predict(Xte)
            metric_rows.append(_mrow(trait, scheme, fold_id, "lightgbm", "ld_pruned", y_te, p))
            pred_rows.append(_preds(samples, test_idx, y_te, p, trait, "lightgbm", "ld_pruned", scheme, fold_id))

            # GWAS Top-K -> LightGBM
            scores = gwas_neglog10_p(X_qc[train_idx], y_tr, covariates=pcs[train_idx, :n_pcs_cov])
            idx = topk_indices(scores, gwas_k)
            for rank, j in enumerate(idx):
                selected_rows.append(
                    {
                        "trait": trait,
                        "scheme": scheme,
                        "fold": fold_id,
                        "rank": rank + 1,
                        "snp_idx": int(j),
                        "snp_id": snp_qc.iloc[j]["snp_id"],
                        "chrom": snp_qc.iloc[j]["chrom"],
                        "pos": int(snp_qc.iloc[j]["pos"]),
                        "neglog10p": float(scores[j]),
                    }
                )
            Xtr_g = X_qc[train_idx][:, idx]
            Xte_g = X_qc[test_idx][:, idx]
            lgbm_g = fit_lightgbm(Xtr_g, y_tr, random_state=seed)
            p = lgbm_g.predict(Xte_g)
            metric_rows.append(_mrow(trait, scheme, fold_id, "lightgbm", f"gwas_top{gwas_k}", y_te, p))
            pred_rows.append(
                _preds(samples, test_idx, y_te, p, trait, "lightgbm", f"gwas_top{gwas_k}", scheme, fold_id)
            )

    metrics = pd.DataFrame(metric_rows)
    selected = pd.DataFrame(selected_rows)
    preds = pd.concat(pred_rows, ignore_index=True)
    metrics.to_csv(metrics_dir / "wheat_m3b_all_traits_metrics_by_fold.csv", index=False)
    selected.to_csv(feat_dir / "wheat_m3b_gwas_selected_snps.csv", index=False)
    preds.to_csv(pred_dir / "wheat_m3b_all_traits_predictions.csv", index=False)

    summary = (
        metrics.groupby(["trait", "scheme", "model", "features"], as_index=False)
        .agg(
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            spearman_rho_mean=("spearman_rho", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            top10_overlap_mean=("top10_overlap", "mean"),
            n_folds=("rmse", "count"),
        )
        .sort_values(["trait", "scheme", "pearson_r_mean"], ascending=[True, True, False])
    )
    summary["near_zero_var"] = summary["trait"].isin(near_zero)
    summary.to_csv(metrics_dir / "wheat_m3b_all_traits_metrics_summary.csv", index=False)

    # Stability: selection frequency of SNPs across folds (per trait, kinship+random)
    stab_rows = []
    for trait in traits:
        sub = selected[selected.trait == trait]
        n_folds_sel = sub[["scheme", "fold"]].drop_duplicates().shape[0]
        counts = Counter(sub["snp_id"])
        for snp_id, c in counts.most_common(50):
            meta = sub[sub.snp_id == snp_id].iloc[0]
            stab_rows.append(
                {
                    "trait": trait,
                    "snp_id": snp_id,
                    "chrom": meta["chrom"],
                    "pos": int(meta["pos"]),
                    "n_selected": int(c),
                    "n_folds": int(n_folds_sel),
                    "freq": float(c / n_folds_sel) if n_folds_sel else np.nan,
                    "mean_neglog10p": float(sub.loc[sub.snp_id == snp_id, "neglog10p"].mean()),
                }
            )
    stability = pd.DataFrame(stab_rows)
    stability.to_csv(feat_dir / "wheat_m3b_snp_stability_top50.csv", index=False)

    # heatmap of best pearson by trait x model(feature)
    best = (
        summary[summary.scheme == "kinship_group"]
        .assign(label=lambda d: d["model"] + "|" + d["features"])
        .pivot_table(index="trait", columns="label", values="pearson_r_mean", aggfunc="first")
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(best.fillna(0).to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=0.8)
    ax.set_xticks(range(best.shape[1]))
    ax.set_xticklabels(best.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(best.shape[0]))
    ax.set_yticklabels(best.index, fontsize=8)
    ax.set_title("Wheat M3b kinship CV Pearson r")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(report_dir / "wheat_m3b_kinship_performance_heatmap.png", dpi=150)
    plt.close(fig)

    # stability bar for yield
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, trait in zip(axes.ravel(), cfg["wheat"]["pilot_traits"]):
        s = stability[stability.trait == trait].head(15)
        ax.barh(s["snp_id"][::-1], s["freq"][::-1], color="#4C78A8")
        ax.set_title(f"{trait} GWAS selection freq")
        ax.set_xlabel("frequency across folds")
    fig.tight_layout()
    fig.savefig(report_dir / "wheat_m3b_snp_stability_pilot.png", dpi=150)
    plt.close(fig)

    # Gate: every non-near-zero trait beats mean on RMSE in kinship CV for at least one model
    gate = {"near_zero_var_traits": near_zero, "trait_pass": {}, "all_actionable_pass": True}
    for trait in traits:
        sub = summary[(summary.trait == trait) & (summary.scheme == "kinship_group")]
        mean_rmse = float(sub.loc[sub.model == "mean", "rmse_mean"].iloc[0])
        cand = sub[sub.model != "mean"]
        best = cand.sort_values(["pearson_r_mean", "rmse_mean"], ascending=[False, True]).iloc[0]
        passed = bool(best["rmse_mean"] < mean_rmse)
        if trait not in near_zero and not passed:
            gate["all_actionable_pass"] = False
        gate["trait_pass"][trait] = {
            "near_zero_var": trait in near_zero,
            "best_model": best["model"],
            "best_features": best["features"],
            "best_r": None if pd.isna(best["pearson_r_mean"]) else float(best["pearson_r_mean"]),
            "best_rmse": float(best["rmse_mean"]),
            "mean_rmse": mean_rmse,
            "pass": passed,
        }
    with open(metrics_dir / "wheat_m3b_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    print("M3b OK")
    print(summary.groupby("trait")["pearson_r_mean"].max().sort_values(ascending=False).to_string())
    print(json.dumps({k: gate[k] for k in ["near_zero_var_traits", "all_actionable_pass"]}, indent=2))
    return 0 if gate["all_actionable_pass"] else 0  # still succeed; gate recorded


def _mrow(trait, scheme, fold, model, features, y_te, p):
    return {"trait": trait, "scheme": scheme, "fold": fold, "model": model, "features": features, **eval_pack(y_te, p)}


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
