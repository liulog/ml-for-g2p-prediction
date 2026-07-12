#!/usr/bin/env python3
"""Gap-fill: SHAP (or TreeExplainer fallback) + FDR-corrected paired model comparisons."""
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.wheat import load_wheat_arrays  # noqa: E402
from src.features.gwas import gwas_neglog10_p, topk_indices  # noqa: E402

try:
    import shap

    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    fdr = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        fdr[order[i]] = prev
    return np.clip(fdr, 0, 1)


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, seed=2026, n_boot=2000):
    """Paired bootstrap on per-fold metric vectors a vs b (a-b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return {"delta_mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_two_sided": np.nan}
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)])
    # two-sided p from bootstrap around 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = float(min(max(p, 1.0 / n_boot), 1.0))
    return {
        "delta_mean": float(d.mean()),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "p_two_sided": p,
    }


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    feat_dir = ROOT / "results" / "features"
    metrics_dir = ROOT / "results" / "metrics"
    fig_dir = ROOT / "reports" / "figures"
    for d in (feat_dir, metrics_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- SHAP on wheat height (pilot, GWAS top200 for speed) ----
    wheat = load_wheat_arrays(ROOT / "data" / "interim" / "wheat")
    trait = "height"
    y = pd.to_numeric(wheat["pheno"][trait], errors="coerce").to_numpy(float)
    y = np.where(np.isfinite(y), y, np.nanmean(y))
    scores = gwas_neglog10_p(wheat["X_qc"], y, covariates=wheat["pcs"][:, :5])
    idx = topk_indices(scores, 200)
    X = wheat["X_qc"][:, idx]
    snp = wheat["snp_qc"].iloc[idx].reset_index(drop=True)
    model = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, verbosity=-1, force_col_wise=True
    )
    model.fit(X, y)

    shap_method = "none"
    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        # subsample for speed
        rng = np.random.default_rng(seed)
        take = rng.choice(len(X), size=min(300, len(X)), replace=False)
        sv = explainer.shap_values(X[take])
        mean_abs = np.abs(sv).mean(0)
        shap_method = "TreeExplainer"
    else:
        # permutation importance fallback
        from sklearn.inspection import permutation_importance

        r = permutation_importance(model, X, y, n_repeats=5, random_state=seed, scoring="r2")
        mean_abs = r.importances_mean
        shap_method = "permutation_importance_fallback"

    shap_df = snp[["snp_id", "chrom", "pos"]].copy()
    shap_df["trait"] = trait
    shap_df["mean_abs_shap_or_perm"] = mean_abs
    shap_df["neglog10p"] = scores[idx]
    shap_df = shap_df.sort_values("mean_abs_shap_or_perm", ascending=False)
    shap_df.to_csv(feat_dir / "wheat_height_shap_or_perm.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    top = shap_df.head(20)
    ax.barh(top["snp_id"][::-1], top["mean_abs_shap_or_perm"][::-1], color="#54A24B")
    ax.set_title(f"Wheat {trait} ({shap_method})")
    fig.tight_layout()
    fig.savefig(fig_dir / "wheat_height_shap_or_perm.png", dpi=150)
    plt.close(fig)

    # Maize env SHAP on a compact G+E model
    interim = ROOT / "data" / "interim" / "maize"
    gxe = pd.read_parquet(interim / "maize_routeB_yield_gxe.parquet")
    g_pc = pd.read_csv(interim / "maize_genotype_pca.csv")
    e_pc = pd.read_csv(interim / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})
    d = gxe.merge(g_pc, on="genotype_id").merge(e_pc, on="environment_id").dropna(subset=["y_raw_mean"])
    # subsample rows for SHAP fit
    d = d.sample(n=min(8000, len(d)), random_state=seed)
    feats = [c for c in d.columns if c.startswith("G_PC") or c.startswith("E_PC")]
    Xm = d[feats].to_numpy(float)
    ym = d["y_raw_mean"].to_numpy(float)
    mmodel = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, verbosity=-1, force_col_wise=True
    )
    mmodel.fit(Xm, ym)
    if HAS_SHAP:
        explainer = shap.TreeExplainer(mmodel)
        take = np.random.default_rng(seed).choice(len(Xm), size=min(500, len(Xm)), replace=False)
        sv = explainer.shap_values(Xm[take])
        mean_abs = np.abs(sv).mean(0)
        method_m = "TreeExplainer"
    else:
        from sklearn.inspection import permutation_importance

        r = permutation_importance(mmodel, Xm, ym, n_repeats=3, random_state=seed, scoring="r2")
        mean_abs = r.importances_mean
        method_m = "permutation_importance_fallback"
    mimp = pd.DataFrame({"feature": feats, "mean_abs_shap_or_perm": mean_abs}).sort_values(
        "mean_abs_shap_or_perm", ascending=False
    )
    mimp.to_csv(feat_dir / "maize_yield_shap_or_perm.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    top = mimp.head(20)
    ax.barh(top["feature"][::-1], top["mean_abs_shap_or_perm"][::-1], color="#F58518")
    ax.set_title(f"Maize Yield ({method_m})")
    fig.tight_layout()
    fig.savefig(fig_dir / "maize_yield_shap_or_perm.png", dpi=150)
    plt.close(fig)

    # ---- FDR model comparisons on existing fold metrics ----
    comparisons = []
    # wheat: gblup vs lightgbm ld on kinship
    w = pd.read_csv(metrics_dir / "wheat_m3b_all_traits_metrics_by_fold.csv")
    for trait in cfg["wheat"]["all_traits"]:
        a = w[(w.trait == trait) & (w.scheme == "kinship_group") & (w.model == "lightgbm") & (w.features == "ld_pruned")][
            "pearson_r"
        ].to_numpy()
        b = w[(w.trait == trait) & (w.scheme == "kinship_group") & (w.model == "gblup") & (w.features == "grm")][
            "pearson_r"
        ].to_numpy()
        if len(a) and len(b) and len(a) == len(b):
            stats_d = paired_bootstrap_delta(a, b, seed=seed)
            comparisons.append({"crop": "wheat", "contrast": "lgbm_ld_vs_gblup", "trait_or_scheme": trait, **stats_d})

    # maize raw yield: G+E vs G / E
    m = pd.read_csv(metrics_dir / "maize_m5b_raw_yield_metrics_by_fold.csv")
    for scheme in m.scheme.unique():
        ge = m[(m.scheme == scheme) & (m.model == "lightgbm") & (m.features == "G+E")]["pearson_r"].to_numpy()
        for other in ["G", "E"]:
            o = m[(m.scheme == scheme) & (m.model == "lightgbm") & (m.features == other)]["pearson_r"].to_numpy()
            if len(ge) and len(o) and len(ge) == len(o):
                stats_d = paired_bootstrap_delta(ge, o, seed=seed)
                comparisons.append(
                    {"crop": "maize", "contrast": f"G+E_vs_{other}", "trait_or_scheme": scheme, **stats_d}
                )

    comp = pd.DataFrame(comparisons)
    if len(comp):
        comp["p_fdr"] = bh_fdr(comp["p_two_sided"].fillna(1.0).to_numpy())
        comp["significant_fdr_0.05"] = comp["p_fdr"] < 0.05
        comp.to_csv(metrics_dir / "m7_model_comparisons_fdr.csv", index=False)

    summary = {
        "has_shap_package": HAS_SHAP,
        "wheat_method": shap_method,
        "maize_method": method_m,
        "n_comparisons": int(len(comp)),
        "n_sig_fdr_0.05": int(comp["significant_fdr_0.05"].sum()) if len(comp) else 0,
    }
    with open(metrics_dir / "m7_shap_fdr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SHAP/FDR OK")
    print(json.dumps(summary, indent=2))
    if len(comp):
        print(comp.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
