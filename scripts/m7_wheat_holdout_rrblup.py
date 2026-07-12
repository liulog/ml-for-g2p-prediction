#!/usr/bin/env python3
"""Holdout test + explicit RR-BLUP for wheat pilot traits (fast version)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wheat import kinship_groups, load_wheat_arrays  # noqa: E402
from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402
from src.models.baselines import GBLUP, MeanModel, RRBLUP, fit_lightgbm  # noqa: E402


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
    holdout_frac = float(cfg["cv"].get("test_size_holdout", 0.15))
    data = load_wheat_arrays(ROOT / cfg["paths"]["interim"] / "wheat")
    samples, pheno, X, K = data["samples"], data["pheno"], data["X_ld"], data["K"]
    # Use top 50 PCs for a fast EN-like linear baseline via RR on PCs is unnecessary;
    # models: mean, gblup, rrblup, lightgbm
    idx = np.arange(len(samples))
    train_idx, hold_idx = train_test_split(idx, test_size=holdout_frac, random_state=seed)
    train_idx, hold_idx = np.sort(train_idx), np.sort(hold_idx)

    splits_dir = ROOT / "results" / "splits"
    metrics_dir = ROOT / "results" / "metrics"
    model_dir = ROOT / "results" / "models"
    for d in (splits_dir, metrics_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"sample_id": [samples[i] for i in train_idx], "split": "train_dev"}).to_csv(
        splits_dir / "wheat_holdout_train_dev.csv", index=False
    )
    pd.DataFrame({"sample_id": [samples[i] for i in hold_idx], "split": "holdout"}).to_csv(
        splits_dir / "wheat_holdout_test.csv", index=False
    )

    groups_full = kinship_groups(K, cfg["cv"]["n_splits"])
    rows, hold_rows = [], []

    for trait in cfg["wheat"]["pilot_traits"]:
        print(f"=== {trait} ===", flush=True)
        y = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(float)
        y = np.where(np.isfinite(y), y, np.nanmean(y))
        gkf = GroupKFold(n_splits=cfg["cv"]["n_splits"])
        local = np.arange(len(train_idx))
        local_groups = groups_full[train_idx]
        cv_scores = {m: [] for m in ["mean", "gblup", "rrblup", "lightgbm"]}

        for fold, (tr_l, te_l) in enumerate(gkf.split(local, groups=local_groups)):
            tr, te = train_idx[tr_l], train_idx[te_l]
            print(f"  cv fold {fold}", flush=True)
            preds = {
                "mean": MeanModel().fit(y[tr]).predict(len(te)),
                "gblup": GBLUP(0.5).fit(K, y, tr).predict(te),
                "rrblup": RRBLUP().fit(X[tr], y[tr]).predict(X[te]),
                "lightgbm": fit_lightgbm(X[tr], y[tr], random_state=seed).predict(X[te]),
            }
            for name, p in preds.items():
                m = eval_pack(y[te], p)
                rows.append({"trait": trait, "fold": fold, "model": name, "split": "train_dev_cv", **m})
                cv_scores[name].append(m["pearson_r"] if np.isfinite(m["pearson_r"]) else -1e9)

        best = max(cv_scores, key=lambda k: float(np.nanmean(cv_scores[k])))
        final = {
            "mean": MeanModel().fit(y[train_idx]),
            "gblup": GBLUP(0.5).fit(K, y, train_idx),
            "rrblup": RRBLUP().fit(X[train_idx], y[train_idx]),
            "lightgbm": fit_lightgbm(X[train_idx], y[train_idx], random_state=seed),
        }
        joblib.dump(
            {"rrblup": final["rrblup"], "lightgbm": final["lightgbm"], "best_cv_model": best, "trait": trait},
            model_dir / f"wheat_{trait}_holdout_bundle.joblib",
        )
        for name, model in final.items():
            if name == "mean":
                p = model.predict(len(hold_idx))
            elif name == "gblup":
                p = model.predict(hold_idx)
            else:
                p = model.predict(X[hold_idx])
            m = eval_pack(y[hold_idx], p)
            hold_rows.append({"trait": trait, "model": name, "selected_by_cv": name == best, "best_cv_model": best, **m})
            print(f"  holdout {name}: r={m['pearson_r']:.3f} (best_cv={best})", flush=True)

    pd.DataFrame(rows).to_csv(metrics_dir / "wheat_holdout_cv_metrics.csv", index=False)
    hold_df = pd.DataFrame(hold_rows)
    hold_df.to_csv(metrics_dir / "wheat_holdout_test_metrics.csv", index=False)
    with open(metrics_dir / "wheat_holdout_gate.json", "w") as f:
        json.dump(
            {
                "holdout_frac": holdout_frac,
                "n_train_dev": int(len(train_idx)),
                "n_holdout": int(len(hold_idx)),
                "seed": seed,
                "models": ["mean", "gblup", "rrblup", "lightgbm"],
            },
            f,
            indent=2,
        )
    print("Holdout+RRBLUP OK")
    print(hold_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
