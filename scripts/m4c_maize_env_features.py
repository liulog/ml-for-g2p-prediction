#!/usr/bin/env python3
"""M4c: maize weather window aggregation + environment PCA.

Uses fixed DAP windows only (no sample-specific silk/pollen dates) to avoid leakage:
  1-30, 31-60, 61-90, 91-120, and full available season (daysFromStart).
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


WINDOWS = {
    "d1_30": (1, 30),
    "d31_60": (31, 60),
    "d61_90": (61, 90),
    "d91_120": (91, 120),
    "full": (1, 10_000),
}

BASE_VARS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOT",
    "VPD",
    "RH2M",
    "WS2M",
    "EVPTRNS",
    "ALLSKY_SFC_SW_DWN",
    "GWETROOT",
]


def gdd_row(tmax, tmin, base=10.0, cap=30.0) -> float:
    if not np.isfinite(tmax) or not np.isfinite(tmin):
        return np.nan
    tavg = (min(cap, max(base, tmax)) + min(cap, max(base, tmin))) / 2.0
    return max(0.0, tavg - base)


def window_features(df: pd.DataFrame, tag: str) -> pd.Series:
    out = {}
    for v in BASE_VARS:
        if v not in df.columns:
            continue
        s = pd.to_numeric(df[v], errors="coerce")
        out[f"{tag}__{v}__mean"] = s.mean()
        out[f"{tag}__{v}__std"] = s.std(ddof=1)
        out[f"{tag}__{v}__min"] = s.min()
        out[f"{tag}__{v}__max"] = s.max()
        out[f"{tag}__{v}__p10"] = s.quantile(0.1)
        out[f"{tag}__{v}__p90"] = s.quantile(0.9)
    # cumulative / stress
    prec = pd.to_numeric(df.get("PRECTOT"), errors="coerce")
    rad = pd.to_numeric(df.get("ALLSKY_SFC_SW_DWN"), errors="coerce")
    et = pd.to_numeric(df.get("EVPTRNS"), errors="coerce")
    tmax = pd.to_numeric(df.get("T2M_MAX"), errors="coerce")
    tmin = pd.to_numeric(df.get("T2M_MIN"), errors="coerce")
    vpd = pd.to_numeric(df.get("VPD"), errors="coerce")
    frost = pd.to_numeric(df.get("FROST_DAYS"), errors="coerce")
    out[f"{tag}__PRECTOT__sum"] = prec.sum(min_count=1)
    out[f"{tag}__RAD__sum"] = rad.sum(min_count=1)
    out[f"{tag}__ET__sum"] = et.sum(min_count=1)
    gdd = np.array([gdd_row(a, b) for a, b in zip(tmax.fillna(np.nan), tmin.fillna(np.nan))], dtype=float)
    out[f"{tag}__GDD__sum"] = np.nansum(gdd)
    out[f"{tag}__hot_days_gt35"] = float((tmax > 35).sum())
    out[f"{tag}__cold_days_lt5"] = float((tmin < 5).sum())
    out[f"{tag}__frost_days"] = float(frost.fillna(0).sum())
    out[f"{tag}__high_vpd_days_gt2"] = float((vpd > 2.0).sum())
    out[f"{tag}__n_days"] = float(len(df))
    return pd.Series(out)


def main() -> int:
    cfg = load_cfg()
    out_dir = ROOT / cfg["paths"]["interim"] / "maize"
    report_dir = ROOT / cfg["paths"]["reports"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    weather = pd.read_csv(ROOT / cfg["paths"]["maize_env"])
    weather["env"] = weather["env"].astype(str).str.strip()
    weather["daysFromStart"] = pd.to_numeric(weather["daysFromStart"], errors="coerce")

    rows = []
    for env, g in weather.groupby("env"):
        feats = {"environment_id": env}
        # static location
        feats["LON"] = pd.to_numeric(g["LON"], errors="coerce").median()
        feats["LAT"] = pd.to_numeric(g["LAT"], errors="coerce").median()
        for tag, (lo, hi) in WINDOWS.items():
            sub = g[(g["daysFromStart"] >= lo) & (g["daysFromStart"] <= hi)]
            if len(sub) == 0:
                continue
            feats.update(window_features(sub, tag).to_dict())
        rows.append(feats)

    env_feat = pd.DataFrame(rows).set_index("environment_id").sort_index()
    # drop all-nan columns
    env_feat = env_feat.dropna(axis=1, how="all")
    env_feat.to_parquet(out_dir / "maize_env_features.parquet")

    # PCA on imputed standardized features
    X = env_feat.copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.to_numpy(dtype=float))
    n_comp = min(10, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=cfg["project"]["seed"])
    pcs = pca.fit_transform(Xs)
    pc_df = pd.DataFrame(pcs, index=env_feat.index, columns=[f"E_PC{i+1}" for i in range(n_comp)])
    pc_df.to_csv(out_dir / "maize_env_pca.csv")
    np.save(out_dir / "maize_env_feature_scaler_mean.npy", scaler.mean_)
    np.save(out_dir / "maize_env_feature_scaler_scale.npy", scaler.scale_)
    # persist column order
    pd.Series(X.columns, name="feature").to_csv(out_dir / "maize_env_feature_columns.csv", index=False)

    summary = {
        "n_envs": int(len(env_feat)),
        "n_features": int(env_feat.shape[1]),
        "windows": list(WINDOWS.keys()),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "pca_cumulative_top5": float(pca.explained_variance_ratio_[:5].sum()),
    }
    with open(out_dir / "maize_m4c_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    axes[0].set_title("Env PCA cumulative variance")
    axes[0].set_xlabel("component")
    axes[1].scatter(pc_df["E_PC1"], pc_df["E_PC2"], s=20, alpha=0.8, c="#4C78A8")
    axes[1].set_xlabel("E_PC1")
    axes[1].set_ylabel("E_PC2")
    axes[1].set_title("Environment PCA")
    fig.tight_layout()
    fig.savefig(report_dir / "maize_m4c_env_pca.png", dpi=150)
    plt.close(fig)

    print("M4c OK")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
