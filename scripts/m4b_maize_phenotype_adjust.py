#!/usr/bin/env python3
"""M4b: maize field-design correction -> Hybrid×Env adjusted phenotypes.

Route B (primary for genetic comparison):
  For each trait, fit OLS:
    y ~ C(Env) + C(Replicate):C(Env) [if available] + optional Block
  Then Hybrid×Env BLUE ≈ mean of residuals + overall/trait structure,
  implemented as Hybrid×Env means of (y - Env effects) using a fixed-effects model:

    y = Env + Replicate(Env) + Hybrid:Env + e
  Practically we compute Hybrid×Env means after residualizing Env (+ Replicate nested).

Also exports Route A design covariates on plot-level table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def residualize(df: pd.DataFrame, ycol: str) -> pd.Series:
    """Residualize trait on Env + Replicate within Env via group means (fast)."""
    y = pd.to_numeric(df[ycol], errors="coerce")
    env = df["environment_id"].astype(str)
    rep = df["Replicate"].astype(str)
    # nested: y - mean(Env,Rep) + mean(Env) - mean(Env) => y - mean(Env,Rep)
    # keep Env main effect removed relative to overall by using Env,Rep demean only
    key = env + "|" + rep
    mu_er = y.groupby(key).transform("mean")
    # Also remove leftover Env mean differences already in er means; residual within Env-Rep
    resid = y - mu_er
    return resid


def main() -> int:
    cfg = load_cfg()
    out_dir = ROOT / cfg["paths"]["interim"] / "maize"
    plot = pd.read_parquet(out_dir / "maize_plot_level.parquet")
    traits = [cfg["maize"]["primary_trait"]] + cfg["maize"]["secondary_traits"]

    # Route A: keep plot-level design covariates
    design_cols = [
        "obs_id",
        "genotype_id",
        "environment_id",
        "Year",
        "Field_Location",
        "Experiment",
        "Replicate",
        "Block",
        "Plot",
        "Range",
        "Pass",
        "Date_Planted",
        "Date_Harvested",
    ] + traits
    route_a = plot[design_cols].copy()
    route_a.to_parquet(out_dir / "maize_routeA_plot.parquet", index=False)

    # Route B: Hybrid x Env adjusted means
    frames = []
    qc = []
    for trait in traits:
        print(f"Residualizing {trait} ...", flush=True)
        resid = residualize(plot, trait)
        tmp = plot[["genotype_id", "environment_id", "Year", "Field_Location"]].copy()
        tmp["trait"] = trait
        tmp["y_raw"] = pd.to_numeric(plot[trait], errors="coerce")
        tmp["y_resid"] = resid
        # Hybrid×Env aggregation
        g = (
            tmp.dropna(subset=["y_resid"])
            .groupby(["genotype_id", "environment_id", "Year", "Field_Location", "trait"], as_index=False)
            .agg(
                y_blue=("y_resid", "mean"),
                y_raw_mean=("y_raw", "mean"),
                n_plots=("y_resid", "size"),
                y_resid_std=("y_resid", "std"),
            )
        )
        # Also add Env-mean centered raw Hybrid×Env mean as alternative adjusted pheno
        env_mean = g.groupby("environment_id")["y_raw_mean"].transform("mean")
        g["y_env_centered"] = g["y_raw_mean"] - env_mean
        frames.append(g)
        qc.append(
            {
                "trait": trait,
                "n_gxe": int(len(g)),
                "median_plots": float(g["n_plots"].median()),
                "y_blue_std": float(g["y_blue"].std(ddof=1)),
                "y_env_centered_std": float(g["y_env_centered"].std(ddof=1)),
            }
        )

    route_b = pd.concat(frames, ignore_index=True)
    route_b.to_parquet(out_dir / "maize_routeB_gxe_blue.parquet", index=False)
    # wide primary trait table for modeling Yield
    primary = cfg["maize"]["primary_trait"]
    wide = route_b[route_b.trait == primary].copy()
    wide.to_parquet(out_dir / "maize_routeB_yield_gxe.parquet", index=False)
    pd.DataFrame(qc).to_csv(out_dir / "maize_m4b_trait_qc.csv", index=False)

    summary = {
        "n_plot_routeA": int(len(route_a)),
        "n_gxe_rows_long": int(len(route_b)),
        "n_gxe_yield": int(len(wide)),
        "traits": traits,
        "primary_trait": primary,
        "note": (
            "y_blue = Hybrid×Env mean after residualizing Env + Replicate(Env); "
            "y_env_centered = Hybrid×Env raw mean minus Env mean."
        ),
    }
    with open(out_dir / "maize_m4b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("M4b OK")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
