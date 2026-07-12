#!/usr/bin/env python3
"""M2: wheat pilot baselines — Mean, GBLUP, Elastic Net, LightGBM.

Traits: yield, height, headingdate, tkw
CV: repeated random 5-fold + kinship-cluster GroupKFold (5)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GroupKFold, RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402
from src.models.baselines import (  # noqa: E402
    GBLUP,
    MeanModel,
    fit_elastic_net,
    fit_lightgbm,
)


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def kinship_groups(K: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    # Convert GRM similarity to distance
    dist = np.clip(1.0 - K, 0, None)
    np.fill_diagonal(dist, 0.0)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(dist)


def eval_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m = regression_metrics(y_true, y_pred)
    m["top10_overlap"] = topk_overlap(y_true, y_pred, 0.10)
    return m


def run_fold_models(
    *,
    trait: str,
    fold_id: str,
    scheme: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    K: np.ndarray,
    seed: int,
) -> tuple[list[dict], pd.DataFrame]:
    """X is samples x SNPs (LD-pruned). y full-length. K full GRM."""
    y_tr, y_te = y[train_idx], y[test_idx]
    X_tr, X_te = X[train_idx], X[test_idx]

    rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    # Mean
    mean_model = MeanModel().fit(y_tr)
    pred_mean = mean_model.predict(len(test_idx))
    rows.append({"trait": trait, "model": "mean", "scheme": scheme, "fold": fold_id, **eval_pack(y_te, pred_mean)})
    pred_frames.append(_pred_df(test_idx, y_te, pred_mean, trait, "mean", scheme, fold_id))

    # GBLUP with fixed h2 for pilot speed (inner search optional later)
    h2 = 0.5
    gblup = GBLUP(h2=h2).fit(K, y, train_idx)
    pred_g = gblup.predict(test_idx)
    m = eval_pack(y_te, pred_g)
    rows.append({"trait": trait, "model": "gblup", "scheme": scheme, "fold": fold_id, "h2": h2, **m})
    pred_frames.append(_pred_df(test_idx, y_te, pred_g, trait, "gblup", scheme, fold_id))

    # Elastic Net on LD-pruned dosages
    # standardize using train fold only
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    X_tr_z = (X_tr - mu) / sd
    X_te_z = (X_te - mu) / sd
    en = fit_elastic_net(X_tr_z, y_tr, random_state=seed)
    pred_en = en.predict(X_te_z)
    rows.append({"trait": trait, "model": "elastic_net", "scheme": scheme, "fold": fold_id, **eval_pack(y_te, pred_en)})
    pred_frames.append(_pred_df(test_idx, y_te, pred_en, trait, "elastic_net", scheme, fold_id))

    # LightGBM
    lgbm = fit_lightgbm(X_tr, y_tr, random_state=seed)
    pred_lg = lgbm.predict(X_te)
    rows.append({"trait": trait, "model": "lightgbm", "scheme": scheme, "fold": fold_id, **eval_pack(y_te, pred_lg)})
    pred_frames.append(_pred_df(test_idx, y_te, pred_lg, trait, "lightgbm", scheme, fold_id))

    return rows, pd.concat(pred_frames, ignore_index=True)


def _pred_df(idx, y_true, y_pred, trait, model, scheme, fold_id) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_idx": idx,
            "y_true": y_true,
            "y_pred": y_pred,
            "trait": trait,
            "model": model,
            "scheme": scheme,
            "fold": fold_id,
        }
    )


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    np.random.seed(seed)

    interim = ROOT / cfg["paths"]["interim"] / "wheat"
    results = ROOT / cfg["paths"]["results"]
    splits_dir = results / "splits"
    pred_dir = results / "predictions"
    metrics_dir = results / "metrics"
    for d in (splits_dir, pred_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(interim / "wheat_samples_kept.csv")["sample_id"].tolist()
    pheno = pd.read_csv(interim / "wheat_pheno_aligned.csv")
    assert list(pheno["sample_id"]) == samples

    # LD-pruned dosages: snp x sample -> sample x snp
    X = np.load(interim / "wheat_dosage_ld_pruned.npy").T.astype(np.float32)
    K = np.load(interim / "wheat_grm.npy").astype(np.float64)
    assert X.shape[0] == len(samples) == K.shape[0]

    traits = cfg["wheat"]["pilot_traits"]
    n_splits = cfg["cv"]["n_splits"]
    n_repeats = cfg["cv"]["n_repeats"]

    # kinship groups for GroupKFold
    groups = kinship_groups(K, n_clusters=n_splits, seed=seed)
    pd.DataFrame({"sample_id": samples, "kinship_group": groups}).to_csv(
        splits_dir / "wheat_kinship_groups.csv", index=False
    )

    all_metrics: list[dict] = []
    all_preds: list[pd.DataFrame] = []
    split_records: list[dict] = []

    # Repeated random CV (use 2 repeats for runtime in first pass if n_repeats large — honor config)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    print(f"Random RepeatedKFold: {n_splits}x{n_repeats}", flush=True)
    for trait in traits:
        y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            # fill trait missing with trait mean for pilot (n=998 complete expected)
            y = np.where(np.isfinite(y), y, np.nanmean(y))
        for rep_fold_i, (train_idx, test_idx) in enumerate(rkf.split(np.arange(len(samples)))):
            rep = rep_fold_i // n_splits
            fold = rep_fold_i % n_splits
            fold_id = f"rep{rep}_fold{fold}"
            split_records.append(
                {
                    "scheme": "random_repeated",
                    "trait": trait,
                    "fold": fold_id,
                    "train_idx": train_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                }
            )
            print(f"  [{trait}] random {fold_id}", flush=True)
            rows, preds = run_fold_models(
                trait=trait,
                fold_id=fold_id,
                scheme="random_repeated",
                train_idx=train_idx,
                test_idx=test_idx,
                y=y,
                X=X,
                K=K,
                seed=seed,
            )
            all_metrics.extend(rows)
            all_preds.append(preds)

    print("Kinship GroupKFold...", flush=True)
    gkf = GroupKFold(n_splits=n_splits)
    for trait in traits:
        y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y).any():
            y = np.where(np.isfinite(y), y, np.nanmean(y))
        for fold, (train_idx, test_idx) in enumerate(gkf.split(np.arange(len(samples)), groups=groups)):
            fold_id = f"kinship_fold{fold}"
            split_records.append(
                {
                    "scheme": "kinship_group",
                    "trait": trait,
                    "fold": fold_id,
                    "train_idx": train_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                }
            )
            print(f"  [{trait}] kinship {fold_id}", flush=True)
            rows, preds = run_fold_models(
                trait=trait,
                fold_id=fold_id,
                scheme="kinship_group",
                train_idx=train_idx,
                test_idx=test_idx,
                y=y,
                X=X,
                K=K,
                seed=seed,
            )
            all_metrics.extend(rows)
            all_preds.append(preds)

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True)
    # map sample_idx to sample_id
    preds_df["sample_id"] = preds_df["sample_idx"].map(lambda i: samples[int(i)])

    metrics_df.to_csv(metrics_dir / "wheat_pilot_metrics_by_fold.csv", index=False)
    preds_df.to_csv(pred_dir / "wheat_pilot_predictions.csv", index=False)
    with open(splits_dir / "wheat_pilot_splits.json", "w") as f:
        json.dump(split_records, f)

    summary = (
        metrics_df.groupby(["trait", "model", "scheme"], as_index=False)
        .agg(
            pearson_r_mean=("pearson_r", "mean"),
            pearson_r_std=("pearson_r", "std"),
            spearman_rho_mean=("spearman_rho", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            top10_overlap_mean=("top10_overlap", "mean"),
            n_folds=("pearson_r", "count"),
        )
        .sort_values(["trait", "scheme", "pearson_r_mean"], ascending=[True, True, False])
    )
    summary.to_csv(metrics_dir / "wheat_pilot_metrics_summary.csv", index=False)

    gate = {
        "traits": traits,
        "models": ["mean", "gblup", "elastic_net", "lightgbm"],
        "schemes": ["random_repeated", "kinship_group"],
        "n_metrics_rows": int(len(metrics_df)),
        "beats_mean": {},
    }
    for trait in traits:
        for scheme in ["random_repeated", "kinship_group"]:
            sub = summary[(summary.trait == trait) & (summary.scheme == scheme)]
            mean_rmse = float(sub.loc[sub.model == "mean", "rmse_mean"].iloc[0])
            cand = sub.loc[sub.model != "mean"].copy()
            # prefer higher pearson; fall back to lower rmse
            cand = cand.sort_values(["pearson_r_mean", "rmse_mean"], ascending=[False, True])
            best = cand.iloc[0]
            best_r = best["pearson_r_mean"]
            best_rmse = float(best["rmse_mean"])
            gate["beats_mean"][f"{trait}|{scheme}"] = {
                "mean_rmse": mean_rmse,
                "best_model": best["model"],
                "best_r": None if pd.isna(best_r) else float(best_r),
                "best_rmse": best_rmse,
                "pass": bool(best_rmse < mean_rmse),
            }
    with open(metrics_dir / "wheat_pilot_m2_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    print("M2 OK")
    print(summary.to_string(index=False))
    print(json.dumps(gate["beats_mean"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
