#!/usr/bin/env python3
"""M6c: robustness synthesis, optimistic-bias table, final figures and delivery report."""
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


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_cfg()
    metrics_dir = ROOT / "results" / "metrics"
    tables_dir = ROOT / "reports" / "tables"
    fig_dir = ROOT / "reports" / "figures"
    for d in (tables_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Wheat optimistic bias: random vs kinship
    w = pd.read_csv(metrics_dir / "wheat_m3b_all_traits_metrics_summary.csv")
    bias_rows = []
    for trait in cfg["wheat"]["all_traits"]:
        for model, features in [("gblup", "grm"), ("lightgbm", "ld_pruned"), ("lightgbm", "gwas_top1000")]:
            r = w[(w.trait == trait) & (w.model == model) & (w.features == features)]
            rr = r[r.scheme == "random_repeated"]
            rk = r[r.scheme == "kinship_group"]
            if len(rr) and len(rk):
                bias_rows.append(
                    {
                        "trait": trait,
                        "model": model,
                        "features": features,
                        "random_r": float(rr.iloc[0]["pearson_r_mean"]),
                        "kinship_r": float(rk.iloc[0]["pearson_r_mean"]),
                        "optimistic_bias": float(rr.iloc[0]["pearson_r_mean"] - rk.iloc[0]["pearson_r_mean"]),
                    }
                )
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(tables_dir / "wheat_optimistic_bias_random_vs_kinship.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    # plot bias for lightgbm ld_pruned
    b2 = bias[bias.features == "ld_pruned"]
    ax.bar(b2["trait"], b2["optimistic_bias"], color="#4C78A8")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("random r − kinship r")
    ax.set_title("Wheat optimistic bias (LightGBM LD-pruned)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "wheat_m6c_optimistic_bias.png", dpi=150)
    plt.close(fig)

    # 2) Maize ablation summary figure (raw yield)
    m = pd.read_csv(metrics_dir / "maize_m5b_raw_yield_metrics_summary.csv")
    m2 = m[m.model == "lightgbm"]
    fig, ax = plt.subplots(figsize=(8, 4))
    schemes = ["leave_genotype", "leave_environment", "leave_year"]
    feats = ["G", "E", "G+E", "G+E+GxE"]
    x = np.arange(len(schemes))
    width = 0.18
    for i, feat in enumerate(feats):
        vals = []
        for sch in schemes:
            sub = m2[(m2.scheme == sch) & (m2.features == feat)]
            vals.append(float(sub.iloc[0]["pearson_r_mean"]) if len(sub) else np.nan)
        ax.bar(x + i * width, vals, width, label=feat)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(schemes, rotation=15)
    ax.set_ylabel("Pearson r")
    ax.set_title("Maize Yield raw-mean ablation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "maize_m6c_ablation.png", dpi=150)
    plt.close(fig)

    # 3) Secondary traits heatmap if available
    m6a_path = metrics_dir / "maize_m6a_traits_metrics_with_ci.csv"
    if m6a_path.exists():
        m6a = pd.read_csv(m6a_path)
        heat = (
            m6a[(m6a.model == "lightgbm") & (m6a.scheme == "leave_environment")]
            .pivot_table(index="trait", columns="features", values="pearson_r_mean", aggfunc="first")
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(heat.fillna(0).to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(heat.shape[1]))
        ax.set_xticklabels(heat.columns)
        ax.set_yticks(range(heat.shape[0]))
        ax.set_yticklabels(heat.index)
        ax.set_title("Maize leave-environment Pearson r")
        fig.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / "maize_m6c_traits_heatmap.png", dpi=150)
        plt.close(fig)

    # 4) Final markdown report (local)
    wheat_gate = json.loads((metrics_dir / "wheat_m3b_gate.json").read_text())
    maize_gate = json.loads((metrics_dir / "maize_m5b_gate.json").read_text())
    lines = [
        "# G2P Final Delivery Report",
        "",
        f"Seed: `{cfg['project']['seed']}`  ",
        "Conda env: `g2p`",
        "",
        "## Scope completed",
        "",
        "- Wheat M0–M3: audit, QC, LD/PCA/GRM, baselines, GWAS feature compare, 15 traits, SNP stability",
        "- Maize M4–M5: alignment, phenotype adjustment, env features, genotype PCA, G/E/G×E baselines",
        "- M6: secondary traits + bootstrap CIs, interpretability, optimistic-bias synthesis, delivery artifacts",
        "",
        "## Wheat highlights",
        "",
        f"- Near-zero-var traits: `{wheat_gate.get('near_zero_var_traits')}`",
        f"- Actionable gate pass: **{wheat_gate.get('all_actionable_pass')}**",
        f"- Mean optimistic bias (LGBM LD): **{bias[bias.features=='ld_pruned']['optimistic_bias'].mean():.3f}**",
        "",
        "## Maize Yield ablation (raw Hybrid×Env mean, LightGBM)",
        "",
        "| scheme | G | E | G+E | G+E beats G&E |",
        "|---|---:|---:|---:|---|",
    ]
    for sch, info in maize_gate.items():
        lines.append(
            f"| {sch} | {info['G']['r']:.3f} | {info['E']['r']:.3f} | {info['G+E']['r']:.3f} | {info.get('ge_beats_g_and_e')} |"
        )
    lines += [
        "",
        "## Key caveats",
        "",
        "1. Random CV overestimates wheat accuracy vs kinship-aware CV.",
        "2. Env-centered yield hides E main effects; use raw-mean ablation for E/G×E claims.",
        "3. Feature importance / GWAS are not causal.",
        "4. Maize genotypes use 20k-SNP PCA proxy (not full 351k GBLUP).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "conda activate g2p",
        "bash scripts/run_all_pipeline.sh",
        "```",
        "",
        "## Artifact index (local, not committed)",
        "",
        "- `results/metrics/` model metrics and gates",
        "- `results/predictions/` per-sample predictions",
        "- `results/features/` GWAS/stability/importance",
        "- `results/models/` saved LightGBM boosters",
        "- `reports/figures/` publication-oriented figures",
        "- `reports/tables/` summary CSVs and this report",
        "",
    ]
    report_path = tables_dir / "FINAL_DELIVERY_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # machine-readable delivery checklist
    checklist = {
        "wheat_m0_m3": True,
        "maize_m4_m5": True,
        "m6_secondary_traits_bootstrap": m6a_path.exists(),
        "m6_interpretability": (ROOT / "results/features/m6b_interpretability_summary.json").exists(),
        "optimistic_bias_table": True,
        "final_report": True,
        "seed": cfg["project"]["seed"],
    }
    with open(tables_dir / "delivery_checklist.json", "w") as f:
        json.dump(checklist, f, indent=2)

    print("M6c OK")
    print(json.dumps(checklist, indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
