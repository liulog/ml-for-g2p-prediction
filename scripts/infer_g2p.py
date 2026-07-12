#!/usr/bin/env python3
"""Inference for new wheat / maize samples using exported infer bundles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd


def predict_wheat(trait: str, dosage_csv: Path, out_csv: Path) -> None:
    bundle = joblib.load(ROOT / "results" / "models" / f"wheat_{trait}_infer.joblib")
    snp_ids = bundle["snp_ids"]
    df = pd.read_csv(dosage_csv)
    if "sample_id" not in df.columns:
        raise ValueError("dosage_csv must contain sample_id")
    if set(snp_ids).issubset(df.columns):
        X = df[snp_ids].to_numpy(float)
    else:
        feat_cols = [c for c in df.columns if c != "sample_id"]
        if len(feat_cols) != len(snp_ids):
            raise ValueError("SNP columns missing and matrix width != training LD SNP count")
        X = df[feat_cols].to_numpy(float)
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "trait": trait,
            "pred_rrblup": bundle["rrblup"].predict(X),
            "pred_lightgbm": bundle["lightgbm"].predict(X),
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} n={len(out)}")


def predict_maize(g_pc_csv: Path, env_id: str, out_csv: Path) -> None:
    from lightgbm import Booster

    meta = json.loads((ROOT / "results" / "models" / "maize_yield_infer_meta.json").read_text())
    feats = meta["features"]
    impute = np.asarray(meta["impute_mean"], dtype=float)
    booster = Booster(model_file=str(ROOT / "results" / "models" / "maize_yield_infer.txt"))
    g = pd.read_csv(g_pc_csv)
    e_feat = pd.read_parquet(ROOT / "data" / "interim" / "maize" / "maize_env_features.parquet").reset_index()
    if "environment_id" not in e_feat.columns:
        e_feat = e_feat.rename(columns={e_feat.columns[0]: "environment_id"})
    e_pc = pd.read_csv(ROOT / "data" / "interim" / "maize" / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})
    env = e_feat.merge(e_pc, on="environment_id")
    env_row = env[env.environment_id.astype(str) == str(env_id)]
    if len(env_row) != 1:
        raise ValueError(f"environment_id={env_id} not found")
    er = env_row.iloc[0]
    rows = []
    for _, r in g.iterrows():
        vec = np.empty(len(feats), dtype=float)
        for i, f in enumerate(feats):
            if f.startswith("G_PC"):
                vec[i] = float(r[f]) if f in r else impute[i]
            else:
                vec[i] = float(er[f]) if f in er.index and pd.notna(er[f]) else impute[i]
        rows.append(
            {
                "genotype_id": r.get("genotype_id"),
                "environment_id": env_id,
                "pred_yield_raw": float(booster.predict(vec.reshape(1, -1))[0]),
            }
        )
    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} n={len(out)}")


def main():
    p = argparse.ArgumentParser(description="G2P inference")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wheat")
    w.add_argument("--trait", required=True, choices=["yield", "height", "headingdate", "tkw"])
    w.add_argument("--dosage-csv", required=True, type=Path)
    w.add_argument("--out", required=True, type=Path)
    m = sub.add_parser("maize")
    m.add_argument("--genotype-pca-csv", required=True, type=Path)
    m.add_argument("--env-id", required=True)
    m.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.cmd == "wheat":
        predict_wheat(args.trait, args.dosage_csv, args.out)
    else:
        predict_maize(args.genotype_pca_csv, args.env_id, args.out)


if __name__ == "__main__":
    main()
