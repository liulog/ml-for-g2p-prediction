#!/usr/bin/env python3
"""M4a: maize G/P/E alignment and modeling-table construction."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.plink import read_fam  # noqa: E402


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_cfg()
    paths = cfg["paths"]
    out_dir = ROOT / paths["interim"] / "maize"
    report_dir = ROOT / paths["reports"] / "data_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    fam = read_fam(ROOT / paths["maize_plink_prefix"])
    pheno = pd.read_csv(ROOT / paths["maize_pheno"])
    weather = pd.read_csv(ROOT / paths["maize_env"])

    for c in ["Hybrid", "Hybrid_orig_name", "Env"]:
        pheno[c] = pheno[c].astype(str).str.strip()
    weather["env"] = weather["env"].astype(str).str.strip()
    fam_ids = set(fam["genotype_id"])

    # preferred genotype match
    match_hybrid = len(fam_ids & set(pheno["Hybrid"]))
    match_orig = len(fam_ids & set(pheno["Hybrid_orig_name"]))
    geno_col = "Hybrid" if match_hybrid >= match_orig else "Hybrid_orig_name"
    pheno["genotype_id"] = pheno[geno_col]
    pheno["environment_id"] = pheno["Env"]
    pheno["obs_id"] = (
        pheno["environment_id"].astype(str)
        + "|"
        + pheno["genotype_id"].astype(str)
        + "|"
        + pheno["Replicate"].astype(str)
        + "|"
        + pheno["Plot"].astype(str)
    )

    matched_g = pheno["genotype_id"].isin(fam_ids)
    matched_e = pheno["environment_id"].isin(set(weather["env"]))
    pheno["matched_genotype"] = matched_g
    pheno["matched_environment"] = matched_e
    pheno["matched_gpe"] = matched_g & matched_e

    model_df = pheno.loc[pheno["matched_gpe"]].copy()
    model_df.to_parquet(out_dir / "maize_plot_level.parquet", index=False)

    # GxE coverage
    gxe = (
        model_df.groupby(["genotype_id", "environment_id"], as_index=False)
        .size()
        .rename(columns={"size": "n_plots"})
    )
    gxe.to_csv(out_dir / "maize_gxe_counts.csv", index=False)

    cov = (
        model_df.groupby(["Year", "Field_Location"], as_index=False)
        .agg(n_plots=("obs_id", "count"), n_hybrids=("genotype_id", "nunique"), n_envs=("environment_id", "nunique"))
    )
    cov.to_csv(out_dir / "maize_year_location_coverage.csv", index=False)

    traits = [cfg["maize"]["primary_trait"]] + cfg["maize"]["secondary_traits"]
    trait_rows = []
    for t in traits:
        s = pd.to_numeric(model_df[t], errors="coerce")
        trait_rows.append(
            {
                "trait": t,
                "n_valid": int(s.notna().sum()),
                "missing_rate": float(s.isna().mean()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "max": float(s.max()),
            }
        )
    pd.DataFrame(trait_rows).to_csv(out_dir / "maize_trait_summary_matched.csv", index=False)

    summary = {
        "n_fam": int(len(fam)),
        "n_pheno_rows": int(len(pheno)),
        "genotype_match_col": geno_col,
        "n_matched_gpe_rows": int(len(model_df)),
        "n_genotypes_matched": int(model_df["genotype_id"].nunique()),
        "n_envs_matched": int(model_df["environment_id"].nunique()),
        "n_gxe": int(len(gxe)),
        "median_plots_per_gxe": float(gxe["n_plots"].median()),
        "years": sorted(model_df["Year"].dropna().unique().tolist()),
        "n_field_locations": int(model_df["Field_Location"].nunique()),
        "unmatched_genotype_rows": int((~matched_g).sum()),
        "unmatched_env_rows": int((~matched_e).sum()),
    }
    with open(out_dir / "maize_m4a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["rows", "G matched", "E matched", "G+E"],
        [len(pheno), int(matched_g.sum()), int(matched_e.sum()), int(len(model_df))],
        color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"],
    )
    axes[0].set_title("Maize alignment counts")
    axes[1].hist(gxe["n_plots"], bins=20, color="#4C78A8")
    axes[1].set_title("Plots per G×E")
    axes[1].set_xlabel("n_plots")
    fig.tight_layout()
    fig.savefig(report_dir / "maize_m4a_alignment.png", dpi=150)
    plt.close(fig)

    print("M4a OK")
    print(json.dumps(summary, indent=2))
    assert summary["n_matched_gpe_rows"] > 0
    assert summary["unmatched_genotype_rows"] == 0
    assert summary["unmatched_env_rows"] == 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
