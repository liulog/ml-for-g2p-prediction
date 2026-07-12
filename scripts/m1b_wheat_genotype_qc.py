#!/usr/bin/env python3
"""M1b: wheat genotype QC, dosage encoding, and aligned phenotype export.

Implements plan §4 without PLINK: load ALT dosages once, apply mind/geno/maf,
export float32 matrix + QC logs. Mean-imputation here is interim-only;
nested CV must re-impute within training folds.
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

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def normalize_id(x: object) -> str:
    return str(x).strip()


_GT_MAP = {
    "0/0": 0.0,
    "0|0": 0.0,
    "0/1": 1.0,
    "1/0": 1.0,
    "0|1": 1.0,
    "1|0": 1.0,
    "1/1": 2.0,
    "1|1": 2.0,
    "./.": np.nan,
    ".|.": np.nan,
    ".": np.nan,
}


def gt_to_dosage(gt: str) -> float:
    gt = gt.split(":", 1)[0]
    if gt in _GT_MAP:
        return _GT_MAP[gt]
    if not gt or gt[0] == ".":
        return np.nan
    alleles = gt.replace("|", "/").split("/")
    if len(alleles) != 2 or "." in alleles:
        return np.nan
    try:
        return float(int(alleles[0]) + int(alleles[1]))
    except ValueError:
        return np.nan


def load_vcf_dosages(vcf_path: Path, keep_idx: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Load dosages for keep_idx samples. Returns X (n_snp, n_sample) float32 and SNP meta."""
    # Count variants first (cheap)
    n_var = 0
    with open(vcf_path) as f:
        for line in f:
            if not line.startswith("#"):
                n_var += 1
    n_sample = len(keep_idx)
    print(f"Allocating dosage matrix: {n_var} x {n_sample}", flush=True)
    X = np.full((n_var, n_sample), np.nan, dtype=np.float32)
    chroms: list[str] = []
    positions: list[int] = []
    snp_ids: list[str] = []
    refs: list[str] = []
    alts: list[str] = []

    with open(vcf_path) as f:
        vi = 0
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            chroms.append(parts[0])
            positions.append(int(parts[1]) if parts[1].isdigit() else -1)
            snp_ids.append(parts[2])
            refs.append(parts[3])
            alts.append(parts[4])
            gts = parts[9:]
            for j, i in enumerate(keep_idx):
                X[vi, j] = gt_to_dosage(gts[i])
            vi += 1
            if vi % 20000 == 0:
                print(f"  loaded {vi}/{n_var} variants...", flush=True)

    meta = pd.DataFrame(
        {"chrom": chroms, "pos": positions, "snp_id": snp_ids, "ref": refs, "alt": alts}
    )
    return X, meta


def main() -> int:
    cfg = load_cfg()
    qc = cfg["qc"]
    paths = cfg["paths"]
    vcf_path = ROOT / paths["wheat_vcf"]
    pheno_path = ROOT / paths["wheat_pheno"]
    out_dir = ROOT / paths["interim"] / "wheat"
    report_dir = ROOT / paths["reports"] / "data_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#CHROM"):
                vcf_samples = [normalize_id(x) for x in line.rstrip("\n").split("\t")[9:]]
                break
        else:
            raise RuntimeError("VCF header missing")

    pheno = pd.read_csv(pheno_path, sep="\t")
    pheno["sample_id"] = pheno[cfg["wheat"]["sample_id_col"]].map(normalize_id)
    pheno_ids = set(pheno["sample_id"])
    keep_idx = np.array([i for i, s in enumerate(vcf_samples) if s in pheno_ids], dtype=np.int64)
    inter = [vcf_samples[i] for i in keep_idx]
    print(f"Intersection samples: {len(inter)} (vcf={len(vcf_samples)}, pheno={len(pheno_ids)})", flush=True)

    X, meta = load_vcf_dosages(vcf_path, keep_idx)

    # SNP filters
    called = np.isfinite(X)
    n_called = called.sum(axis=1)
    miss_rate = 1.0 - (n_called / X.shape[1])
    # mean dosage / 2 = ALT AF
    with np.errstate(invalid="ignore"):
        mean_d = np.nanmean(X, axis=1)
        af = mean_d / 2.0
        maf = np.minimum(af, 1.0 - af)
    snp_pass = (miss_rate <= qc["geno"]) & np.isfinite(maf) & (maf >= qc["maf"])
    print(
        f"SNPs: raw={X.shape[0]}; geno<={qc['geno']} & maf>={qc['maf']} -> {int(snp_pass.sum())}",
        flush=True,
    )

    # Sample missingness on passing SNPs
    X_cand = X[snp_pass]
    sample_miss = 1.0 - np.isfinite(X_cand).mean(axis=0)
    sample_pass = sample_miss <= qc["mind"]
    dropped_samples = [s for s, ok in zip(inter, sample_pass) if not ok]
    print(
        f"Samples: after mind<={qc['mind']} -> {int(sample_pass.sum())}/{len(inter)}; dropped={dropped_samples}",
        flush=True,
    )

    # Recompute SNP filters on retained samples (more precise)
    X2 = X[:, sample_pass]
    called2 = np.isfinite(X2)
    n_called2 = called2.sum(axis=1)
    miss_rate2 = 1.0 - (n_called2 / X2.shape[1])
    with np.errstate(invalid="ignore"):
        mean_d2 = np.nanmean(X2, axis=1)
        af2 = mean_d2 / 2.0
        maf2 = np.minimum(af2, 1.0 - af2)
    snp_pass2 = (miss_rate2 <= qc["geno"]) & np.isfinite(maf2) & (maf2 >= qc["maf"])
    print(
        f"SNPs recomputed on kept samples: {int(snp_pass2.sum())}",
        flush=True,
    )

    X_final = X2[snp_pass2].copy()
    mean_fill = np.nanmean(X_final, axis=1).astype(np.float32)
    mean_fill = np.where(np.isfinite(mean_fill), mean_fill, 0.0).astype(np.float32)
    inds = np.where(~np.isfinite(X_final))
    X_final[inds] = mean_fill[inds[0]]

    final_samples = [s for s, ok in zip(inter, sample_pass) if ok]
    snp_kept = meta.loc[snp_pass2].reset_index(drop=True)
    snp_kept["miss_rate"] = miss_rate2[snp_pass2]
    snp_kept["maf"] = maf2[snp_pass2]
    snp_kept["mean_dosage"] = mean_d2[snp_pass2]

    out_npy = out_dir / "wheat_dosage_qc.npy"
    np.save(out_npy, X_final.astype(np.float32))
    snp_kept.to_parquet(out_dir / "wheat_snp_qc.parquet", index=False)

    meta_all = meta.copy()
    meta_all["miss_rate_intersection"] = miss_rate
    meta_all["maf_intersection"] = maf
    meta_all["pass_qc"] = snp_pass2
    meta_all.to_csv(out_dir / "wheat_snp_qc_all.csv", index=False)

    pd.DataFrame(
        {
            "sample_id": inter,
            "miss_rate_on_first_pass_snps": sample_miss,
            "pass_mind": sample_pass,
        }
    ).to_csv(out_dir / "wheat_sample_qc.csv", index=False)
    pd.DataFrame({"sample_id": final_samples}).to_csv(out_dir / "wheat_samples_kept.csv", index=False)

    pheno_aligned = pheno.set_index("sample_id").loc[final_samples].reset_index()
    pheno_aligned.to_csv(out_dir / "wheat_pheno_aligned.csv", index=False)

    funnel = {
        "n_vcf_samples": len(vcf_samples),
        "n_pheno_samples": int(pheno["sample_id"].nunique()),
        "n_intersection": len(inter),
        "n_variants_raw": int(X.shape[0]),
        "n_variants_after_geno_maf_on_intersection": int(snp_pass.sum()),
        "n_samples_after_mind": int(sample_pass.sum()),
        "n_variants_after_recompute_on_kept_samples": int(snp_pass2.sum()),
        "dropped_samples_mind": dropped_samples,
        "thresholds": {"mind": qc["mind"], "geno": qc["geno"], "maf": qc["maf"]},
        "dosage_shape": [int(X_final.shape[0]), int(X_final.shape[1])],
        "dosage_dtype": "float32",
        "dosage_file": str(out_npy.relative_to(ROOT)),
        "imputation_note": (
            "Interim mean-imputation uses allele means over QC-kept samples. "
            "Training pipelines must re-impute using training-fold means only."
        ),
    }
    with open(out_dir / "wheat_qc_summary.json", "w") as f:
        json.dump(funnel, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(miss_rate2, bins=50, color="#4C78A8")
    axes[0].axvline(qc["geno"], color="red", ls="--")
    axes[0].set_title("SNP missing rate (kept samples)")
    axes[1].hist(maf2[np.isfinite(maf2)], bins=50, color="#F58518")
    axes[1].axvline(qc["maf"], color="red", ls="--")
    axes[1].set_title("MAF (kept samples)")
    axes[2].bar(
        ["raw SNP", "geno+maf", "samples"],
        [X.shape[0], int(snp_pass2.sum()), int(sample_pass.sum())],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    axes[2].set_title("QC funnel")
    fig.tight_layout()
    fig.savefig(report_dir / "wheat_qc_funnel.png", dpi=150)
    plt.close(fig)

    print("M1b OK")
    print(json.dumps(funnel, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
