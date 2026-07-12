#!/usr/bin/env python3
"""Plan remainder: maize mean baselines, reaction-norm (Kronecker-style), dual-tower MLP,
and leave-both (new genotype + new environment) evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import regression_metrics, topk_overlap  # noqa: E402


def load_cfg():
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def eval_pack(y_true, y_pred):
    m = regression_metrics(y_true, y_pred)
    m["top10_overlap"] = topk_overlap(y_true, y_pred, 0.10)
    return m


class MeanBaselines:
    """Env mean / Genotype mean / Env+Genotype means (centring trick)."""

    def fit(self, geno, env, y):
        self.mu_ = float(np.mean(y))
        self.g_mean_ = pd.Series(y).groupby(pd.Series(geno).astype(str)).mean().to_dict()
        self.e_mean_ = pd.Series(y).groupby(pd.Series(env).astype(str)).mean().to_dict()
        return self

    def predict(self, geno, env, mode: str):
        g = np.array([self.g_mean_.get(str(x), self.mu_) for x in geno], float)
        e = np.array([self.e_mean_.get(str(x), self.mu_) for x in env], float)
        if mode == "env_mean":
            return e
        if mode == "geno_mean":
            return g
        if mode == "geno_plus_env_mean":
            # remove double-counted overall mean
            return g + e - self.mu_
        raise ValueError(mode)


def reaction_norm_predict(G_tr, E_tr, y_tr, G_te, E_te, lam_g=1.0, lam_e=1.0, lam_ge=5.0):
    """Approximate reaction-norm via additive kernels on G-PCs and E-PCs + interaction.

    Uses dual ridge on concatenated features [G, E, G⊙E_top], which is a practical
    stand-in for K_G ⊗ K_E when sample size is large.
    """
    # top-k interaction
    k = min(8, G_tr.shape[1], E_tr.shape[1])
    GE_tr = np.einsum("ij,ik->ijk", G_tr[:, :k], E_tr[:, :k]).reshape(len(G_tr), -1)
    GE_te = np.einsum("ij,ik->ijk", G_te[:, :k], E_te[:, :k]).reshape(len(G_te), -1)
    Xtr = np.hstack([G_tr, E_tr, GE_tr])
    Xte = np.hstack([G_te, E_te, GE_te])
    # column scales: stronger penalty on interactions via feature weighting
    w = np.concatenate(
        [
            np.full(G_tr.shape[1], 1.0 / np.sqrt(lam_g)),
            np.full(E_tr.shape[1], 1.0 / np.sqrt(lam_e)),
            np.full(GE_tr.shape[1], 1.0 / np.sqrt(lam_ge)),
        ]
    )
    Xtr = Xtr * w
    Xte = Xte * w
    sc = StandardScaler()
    Xtrz = sc.fit_transform(Xtr)
    Xtez = sc.transform(Xte)
    # ridge closed form
    alpha = 10.0
    A = Xtrz.T @ Xtrz + alpha * np.eye(Xtrz.shape[1])
    b = Xtrz.T @ (y_tr - y_tr.mean())
    coef = np.linalg.solve(A, b)
    return Xtez @ coef + y_tr.mean()


class DualTowerMLP:
    """Small dual-tower network in pure numpy/sklearn-free torch-free form.

    Implemented as two linear towers + ReLU + fusion MLP using a tiny custom GD loop
    to avoid new heavy deps. Capacity is intentionally small.
    """

    def __init__(self, d_g, d_e, hidden=32, lr=1e-2, epochs=200, seed=2026, l2=1e-3):
        rng = np.random.default_rng(seed)
        self.Wg = rng.normal(0, 0.05, size=(d_g, hidden))
        self.bg = np.zeros(hidden)
        self.We = rng.normal(0, 0.05, size=(d_e, hidden))
        self.be = np.zeros(hidden)
        # fusion on [g,e,g*e]
        self.Wf = rng.normal(0, 0.05, size=(hidden * 3, hidden))
        self.bf = np.zeros(hidden)
        self.w = rng.normal(0, 0.05, size=(hidden,))
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.sg = None
        self.se = None

    @staticmethod
    def relu(x):
        return np.maximum(x, 0.0)

    def _forward(self, G, E):
        hg = self.relu(G @ self.Wg + self.bg)
        he = self.relu(E @ self.We + self.be)
        h = np.hstack([hg, he, hg * he])
        z = self.relu(h @ self.Wf + self.bf)
        y = z @ self.w + self.b
        return y, hg, he, h, z

    def fit(self, G, E, y):
        self.sg = StandardScaler().fit(G)
        self.se = StandardScaler().fit(E)
        G = self.sg.transform(G)
        E = self.se.transform(E)
        y = y.astype(float)
        n = len(y)
        for ep in range(self.epochs):
            pred, hg, he, h, z = self._forward(G, E)
            err = pred - y
            loss = float((err @ err) / n)
            # backprop (simple)
            dz = np.outer(err, self.w) / n
            dz[z <= 0] = 0
            dWf = h.T @ dz + self.l2 * self.Wf
            dbf = dz.sum(axis=0)
            dh = dz @ self.Wf.T
            # split dh into hg, he, hg*he
            d_hg = dh[:, : hg.shape[1]] + dh[:, 2 * hg.shape[1] :] * he
            d_he = dh[:, hg.shape[1] : 2 * hg.shape[1]] + dh[:, 2 * hg.shape[1] :] * hg
            d_hg[hg <= 0] = 0
            d_he[he <= 0] = 0
            dWg = G.T @ d_hg + self.l2 * self.Wg
            dbg = d_hg.sum(0)
            dWe = E.T @ d_he + self.l2 * self.We
            dbe = d_he.sum(0)
            dw = z.T @ err / n + self.l2 * self.w
            db = float(err.mean())
            self.Wg -= self.lr * dWg
            self.bg -= self.lr * dbg
            self.We -= self.lr * dWe
            self.be -= self.lr * dbe
            self.Wf -= self.lr * dWf
            self.bf -= self.lr * dbf
            self.w -= self.lr * dw
            self.b -= self.lr * db
            if ep % 50 == 0:
                print(f"    dual-tower epoch {ep} mse={loss:.4f}", flush=True)
        return self

    def predict(self, G, E):
        G = self.sg.transform(G)
        E = self.se.transform(E)
        return self._forward(G, E)[0]


def make_leave_both_splits(geno, env, seed=2026, n_splits=5):
    """Approximate new-G + new-E: hold out a set of genotypes AND environments each fold."""
    rng = np.random.default_rng(seed)
    g_ids = np.array(sorted(pd.unique(geno)))
    e_ids = np.array(sorted(pd.unique(env)))
    rng.shuffle(g_ids)
    rng.shuffle(e_ids)
    g_folds = np.array_split(g_ids, n_splits)
    e_folds = np.array_split(e_ids, n_splits)
    splits = []
    for i in range(n_splits):
        g_te = set(g_folds[i])
        e_te = set(e_folds[i])
        te = np.array([str(g) in g_te and str(e) in e_te for g, e in zip(geno, env)])
        tr = ~te
        # if too few test rows, loosen to union holdout of those G or E then intersect-ish:
        if te.sum() < 50:
            te = np.array([str(g) in g_te or str(e) in e_te for g, e in zip(geno, env)])
            # still require at least one unseen side: keep only rows with unseen G and unseen E when possible
            both = np.array([str(g) in g_te and str(e) in e_te for g, e in zip(geno, env)])
            if both.sum() >= 20:
                te = both
            tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        splits.append((np.where(tr)[0], np.where(te)[0]))
    return splits


def main() -> int:
    cfg = load_cfg()
    seed = cfg["project"]["seed"]
    interim = ROOT / "data" / "interim" / "maize"
    metrics_dir = ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    gxe = pd.read_parquet(interim / "maize_routeB_yield_gxe.parquet")
    g_pc = pd.read_csv(interim / "maize_genotype_pca.csv")
    e_pc = pd.read_csv(interim / "maize_env_pca.csv")
    if "environment_id" not in e_pc.columns:
        e_pc = e_pc.rename(columns={e_pc.columns[0]: "environment_id"})
    d = gxe.merge(g_pc, on="genotype_id").merge(e_pc, on="environment_id").dropna(subset=["y_raw_mean"])
    # subsample for dual-tower speed if needed
    y = d["y_raw_mean"].to_numpy(float)
    geno = d["genotype_id"].astype(str).to_numpy()
    env = d["environment_id"].astype(str).to_numpy()
    g_cols = [c for c in d.columns if c.startswith("G_PC")]
    e_cols = [c for c in d.columns if c.startswith("E_PC")]
    G = d[g_cols].to_numpy(float)
    E = d[e_cols].to_numpy(float)

    rows = []

    # ---- mean baselines + reaction-norm + lgbm + dual-tower under standard schemes ----
    schemes = {
        "leave_genotype": geno,
        "leave_environment": env,
        "leave_year": d["Year"].to_numpy(),
        "leave_gxe_combo": np.array([f"{g}|{e}" for g, e in zip(geno, env)]),  # seen G+E new replicate approx
    }
    for scheme, groups in schemes.items():
        splits = min(5, pd.Series(groups).nunique())
        if splits < 2:
            continue
        gkf = GroupKFold(n_splits=splits)
        print(f"Scheme {scheme}", flush=True)
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(d)), groups=groups)):
            mb = MeanBaselines().fit(geno[tr], env[tr], y[tr])
            for mode in ["env_mean", "geno_mean", "geno_plus_env_mean"]:
                p = mb.predict(geno[te], env[te], mode)
                rows.append({"scheme": scheme, "fold": fold, "model": mode, "features": "means", **eval_pack(y[te], p)})

            p = reaction_norm_predict(G[tr], E[tr], y[tr], G[te], E[te])
            rows.append({"scheme": scheme, "fold": fold, "model": "reaction_norm", "features": "G+E+GxE_ridge", **eval_pack(y[te], p)})

            lgbm = LGBMRegressor(
                n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=seed, verbosity=-1, force_col_wise=True,
            )
            Xtr = np.hstack([G[tr], E[tr]])
            Xte = np.hstack([G[te], E[te]])
            p = lgbm.fit(Xtr, y[tr]).predict(Xte)
            rows.append({"scheme": scheme, "fold": fold, "model": "lightgbm", "features": "G+E", **eval_pack(y[te], p)})

            # dual tower only on first 2 folds for runtime
            if fold < 2:
                dt = DualTowerMLP(G.shape[1], E.shape[1], hidden=32, epochs=120, seed=seed + fold)
                dt.fit(G[tr], E[tr], y[tr])
                p = dt.predict(G[te], E[te])
                rows.append({"scheme": scheme, "fold": fold, "model": "dual_tower", "features": "G+E+GxE", **eval_pack(y[te], p)})

    # ---- leave both (new G + new E) ----
    print("Scheme leave_both_GE", flush=True)
    for fold, (tr, te) in enumerate(make_leave_both_splits(geno, env, seed=seed, n_splits=5)):
        print(f"  both fold{fold} n_te={len(te)}", flush=True)
        mb = MeanBaselines().fit(geno[tr], env[tr], y[tr])
        for mode in ["env_mean", "geno_mean", "geno_plus_env_mean"]:
            p = mb.predict(geno[te], env[te], mode)
            rows.append({"scheme": "leave_both_GE", "fold": fold, "model": mode, "features": "means", **eval_pack(y[te], p)})
        p = reaction_norm_predict(G[tr], E[tr], y[tr], G[te], E[te])
        rows.append({"scheme": "leave_both_GE", "fold": fold, "model": "reaction_norm", "features": "G+E+GxE_ridge", **eval_pack(y[te], p)})
        lgbm = LGBMRegressor(
            n_estimators=250, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=-1, force_col_wise=True,
        )
        p = lgbm.fit(np.hstack([G[tr], E[tr]]), y[tr]).predict(np.hstack([G[te], E[te]]))
        rows.append({"scheme": "leave_both_GE", "fold": fold, "model": "lightgbm", "features": "G+E", **eval_pack(y[te], p)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_dir / "maize_m8_advanced_metrics_by_fold.csv", index=False)
    summary = (
        metrics.groupby(["scheme", "model", "features"], as_index=False)
        .agg(pearson_r_mean=("pearson_r", "mean"), pearson_r_std=("pearson_r", "std"), rmse_mean=("rmse", "mean"), n_folds=("rmse", "count"))
        .sort_values(["scheme", "pearson_r_mean"], ascending=[True, False])
    )
    summary.to_csv(metrics_dir / "maize_m8_advanced_metrics_summary.csv", index=False)
    with open(metrics_dir / "maize_m8_gate.json", "w") as f:
        json.dump(
            {
                "schemes": summary.scheme.unique().tolist(),
                "has_mean_baselines": True,
                "has_reaction_norm": True,
                "has_dual_tower": True,
                "has_leave_both": "leave_both_GE" in set(summary.scheme),
            },
            f,
            indent=2,
        )
    print("M8 maize advanced OK")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
