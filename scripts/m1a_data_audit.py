#!/usr/bin/env python3
"""M1a: data audit and sample alignment for wheat + maize."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cfg() -> dict:
    with open(ROOT / "configs" / "default.yaml") as f:
        return yaml.safe_load(f)


def normalize_id(x: object) -> str:
    return str(x).strip()


def trait_summary(df: pd.DataFrame, traits: list[str], cv_thr: float) -> pd.DataFrame:
    rows = []
    for t in traits:
        s = pd.to_numeric(df[t], errors="coerce")
        n = int(s.notna().sum())
        mean = float(s.mean()) if n else np.nan
        std = float(s.std(ddof=1)) if n > 1 else np.nan
        if n > 1 and pd.notna(mean) and abs(mean) > 1e-12:
            cv = float(std / abs(mean))
        else:
            cv = np.nan
        rows.append(
            {
                "trait": t,
                "n_valid": n,
                "missing": int(s.isna().sum()),
                "missing_rate": float(s.isna().mean()),
                "mean": mean,
                "std": std,
                "min": float(s.min()) if n else np.nan,
                "max": float(s.max()) if n else np.nan,
                "skew": float(stats.skew(s.dropna())) if n > 2 else np.nan,
                "cv": cv,
                "near_zero_var": bool((pd.notna(cv) and cv < cv_thr) or (pd.notna(std) and std < 1e-8)),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def audit_wheat(cfg: dict, out_dir: Path) -> dict:
    pheno_path = ROOT / cfg["paths"]["wheat_pheno"]
    vcf_path = ROOT / cfg["paths"]["wheat_vcf"]
    pheno = pd.read_csv(pheno_path, sep="\t")
    pheno["sample_id"] = pheno[cfg["wheat"]["sample_id_col"]].map(normalize_id)
    traits = cfg["wheat"]["all_traits"]
    missing_cols = [t for t in traits if t not in pheno.columns]
    if missing_cols:
        raise AssertionError(f"trait columns missing: {missing_cols}")

    vcf_samples: list[str] = []
    n_meta = 0
    n_variants = 0
    chroms: Counter[str] = Counter()
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("##"):
                n_meta += 1
                continue
            if line.startswith("#CHROM"):
                parts = line.rstrip("\n").split("\t")
                vcf_samples = [normalize_id(x) for x in parts[9:]]
                continue
            parts = line.split("\t", 1)
            chroms[parts[0]] += 1
            n_variants += 1

    pheno_ids = pheno["sample_id"].tolist()
    set_pheno = set(pheno_ids)
    set_vcf = set(vcf_samples)
    inter = sorted(set_pheno & set_vcf)
    only_pheno = sorted(set_pheno - set_vcf)
    only_vcf = sorted(set_vcf - set_pheno)
    dup_pheno = sorted({x for x in pheno_ids if pheno_ids.count(x) > 1})
    dup_vcf = sorted({x for x in vcf_samples if vcf_samples.count(x) > 1})

    align = pd.DataFrame(
        {
            "sample_id": sorted(set_pheno | set_vcf),
        }
    )
    align["in_pheno"] = align["sample_id"].isin(set_pheno)
    align["in_vcf"] = align["sample_id"].isin(set_vcf)
    align["in_intersection"] = align["in_pheno"] & align["in_vcf"]
    align.to_csv(out_dir / "wheat_sample_alignment.csv", index=False)

    tsum = trait_summary(pheno, traits, cfg["wheat"]["near_zero_var_cv_threshold"])
    tsum.to_csv(out_dir / "wheat_trait_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["pheno", "vcf", "intersection"],
        [len(set_pheno), len(set_vcf), len(inter)],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    axes[0].set_title("Wheat sample counts")
    axes[0].set_ylabel("n")
    axes[1].bar(list(chroms.keys()), list(chroms.values()), color="#4C78A8")
    axes[1].set_title("Wheat variants per chromosome")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "wheat_audit_overview.png", dpi=150)
    plt.close(fig)

    # trait correlation on numeric columns
    corr = pheno[traits].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(traits)))
    ax.set_yticks(range(len(traits)))
    ax.set_xticklabels(traits, rotation=90, fontsize=8)
    ax.set_yticklabels(traits, fontsize=8)
    ax.set_title("Wheat trait Pearson correlation")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "wheat_trait_corr.png", dpi=150)
    plt.close(fig)

    near_zero = tsum.loc[tsum["near_zero_var"], "trait"].tolist()
    return {
        "crop": "wheat",
        "pheno_rows": int(len(pheno)),
        "pheno_unique_samples": int(len(set_pheno)),
        "vcf_samples": int(len(vcf_samples)),
        "vcf_unique_samples": int(len(set_vcf)),
        "n_meta_lines": int(n_meta),
        "n_variants": int(n_variants),
        "n_chromosomes": int(len(chroms)),
        "chromosomes": dict(chroms),
        "n_intersection": int(len(inter)),
        "only_pheno": only_pheno,
        "only_vcf": only_vcf,
        "dup_pheno": dup_pheno,
        "dup_vcf": dup_vcf,
        "n_traits": int(len(traits)),
        "near_zero_var_traits": near_zero,
        "pilot_traits": cfg["wheat"]["pilot_traits"],
        "doc_claimed_variants": 1452806,
        "doc_claimed_traits": 14,
        "local_variants": int(n_variants),
        "local_traits": int(len(traits)),
    }


def audit_maize(cfg: dict, out_dir: Path) -> dict:
    prefix = ROOT / cfg["paths"]["maize_plink_prefix"]
    pheno_path = ROOT / cfg["paths"]["maize_pheno"]
    env_path = ROOT / cfg["paths"]["maize_env"]

    fam = pd.read_csv(
        f"{prefix}.fam",
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "father", "mother", "sex", "pheno"],
        dtype=str,
    )
    fam["genotype_id"] = fam["iid"].map(normalize_id)
    bim = pd.read_csv(
        f"{prefix}.bim",
        sep="\t",
        header=None,
        names=["chr", "snp", "cm", "bp", "a1", "a2"],
        dtype={"chr": str},
    )
    bed_exists = Path(f"{prefix}.bed").exists()

    pheno = pd.read_csv(pheno_path)
    pheno["Hybrid"] = pheno["Hybrid"].map(normalize_id)
    pheno["Hybrid_orig_name"] = pheno["Hybrid_orig_name"].map(normalize_id)
    pheno["Env"] = pheno["Env"].map(normalize_id)

    weather = pd.read_csv(env_path)
    weather["env"] = weather["env"].map(normalize_id)

    fam_ids = set(fam["genotype_id"])
    hybrid_ids = set(pheno["Hybrid"])
    hybrid_orig_ids = set(pheno["Hybrid_orig_name"])
    match_hybrid = len(fam_ids & hybrid_ids)
    match_orig = len(fam_ids & hybrid_orig_ids)
    preferred = "Hybrid" if match_hybrid >= match_orig else "Hybrid_orig_name"
    matched_n = match_hybrid if preferred == "Hybrid" else match_orig
    matched_set = fam_ids & (hybrid_ids if preferred == "Hybrid" else hybrid_orig_ids)

    geno_align = pd.DataFrame({"genotype_id": sorted(fam_ids | hybrid_ids | hybrid_orig_ids)})
    geno_align["in_fam"] = geno_align["genotype_id"].isin(fam_ids)
    geno_align["in_Hybrid"] = geno_align["genotype_id"].isin(hybrid_ids)
    geno_align["in_Hybrid_orig_name"] = geno_align["genotype_id"].isin(hybrid_orig_ids)
    geno_align["matched_preferred"] = geno_align["genotype_id"].isin(matched_set)
    geno_align.to_csv(out_dir / "maize_genotype_alignment.csv", index=False)

    env_pheno = set(pheno["Env"])
    env_weather = set(weather["env"])
    env_align = pd.DataFrame({"environment_id": sorted(env_pheno | env_weather)})
    env_align["in_pheno"] = env_align["environment_id"].isin(env_pheno)
    env_align["in_weather"] = env_align["environment_id"].isin(env_weather)
    env_align["matched"] = env_align["in_pheno"] & env_align["in_weather"]
    env_align.to_csv(out_dir / "maize_environment_alignment.csv", index=False)

    traits = [cfg["maize"]["primary_trait"]] + cfg["maize"]["secondary_traits"]
    tsum = trait_summary(pheno, traits, cv_thr=0.05)
    tsum.to_csv(out_dir / "maize_trait_summary.csv", index=False)

    gxe = (
        pheno.groupby(["Hybrid", "Env"], dropna=False)
        .size()
        .reset_index(name="n_plots")
    )
    gxe.to_csv(out_dir / "maize_gxe_observation_counts.csv", index=False)

    chrom_counts = bim["chr"].value_counts().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["FAM", "pheno Hybrid", "matched"],
        [len(fam_ids), pheno["Hybrid"].nunique(), matched_n],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    axes[0].set_title("Maize genotype alignment")
    axes[1].bar(
        ["pheno Env", "weather env", "matched"],
        [len(env_pheno), len(env_weather), len(env_pheno & env_weather)],
        color=["#4C78A8", "#F58518", "#54A24B"],
    )
    axes[1].set_title("Maize environment alignment")
    fig.tight_layout()
    fig.savefig(out_dir / "maize_audit_overview.png", dpi=150)
    plt.close(fig)

    return {
        "crop": "maize",
        "bed_exists": bool(bed_exists),
        "n_genotypes_fam": int(len(fam_ids)),
        "n_snps_bim": int(len(bim)),
        "n_pheno_rows": int(len(pheno)),
        "n_unique_hybrids_pheno": int(pheno["Hybrid"].nunique()),
        "n_unique_envs_pheno": int(len(env_pheno)),
        "n_weather_rows": int(len(weather)),
        "n_unique_envs_weather": int(len(env_weather)),
        "preferred_genotype_match_col": preferred,
        "n_genotypes_matched": int(matched_n),
        "genotypes_fam_only_n": int(len(fam_ids - matched_set)),
        "genotypes_pheno_only_n": int(
            len((hybrid_ids if preferred == "Hybrid" else hybrid_orig_ids) - fam_ids)
        ),
        "n_envs_matched": int(len(env_pheno & env_weather)),
        "envs_pheno_only": sorted(env_pheno - env_weather),
        "envs_weather_only": sorted(env_weather - env_pheno),
        "n_gxe_combinations": int(len(gxe)),
        "median_plots_per_gxe": float(gxe["n_plots"].median()),
        "max_plots_per_gxe": int(gxe["n_plots"].max()),
        "chrom_counts": {str(k): int(v) for k, v in chrom_counts.items()},
        "trait_summary_file": "maize_trait_summary.csv",
    }


def write_gate_report(summary: dict, out_dir: Path) -> None:
    w = summary["wheat"]
    m = summary["maize"]
    gates = summary["m1a_gate"]["pass_flags"]
    lines = [
        "# M1a Data Audit Gate Report",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Conda env: `{summary['m0_gate']['conda_env']}`",
        "",
        "## Wheat",
        f"- Phenotype samples: **{w['pheno_unique_samples']}**",
        f"- VCF samples: **{w['vcf_unique_samples']}**",
        f"- Intersection: **{w['n_intersection']}**",
        f"- Variants (local): **{w['local_variants']}** (doc claimed {w['doc_claimed_variants']})",
        f"- Traits (local): **{w['local_traits']}** (doc claimed {w['doc_claimed_traits']})",
        f"- Pheno-only: `{w['only_pheno']}`",
        f"- VCF-only: `{w['only_vcf']}`",
        f"- Near-zero-var traits: `{w['near_zero_var_traits']}`",
        "",
        "## Maize",
        f"- FAM genotypes: **{m['n_genotypes_fam']}**, SNPs: **{m['n_snps_bim']}**, BED exists: **{m['bed_exists']}**",
        f"- Phenotype rows: **{m['n_pheno_rows']}**",
        f"- Genotype match col: **{m['preferred_genotype_match_col']}**, matched: **{m['n_genotypes_matched']}**",
        f"- Environments matched: **{m['n_envs_matched']}** / pheno {m['n_unique_envs_pheno']} / weather {m['n_unique_envs_weather']}",
        f"- G×E combos: **{m['n_gxe_combinations']}** (median plots/combo={m['median_plots_per_gxe']})",
        "",
        "## Gate",
        f"- Pass flags: `{gates}`",
        "",
        "Artifacts under `reports/data_audit/` (local, not committed).",
        "",
    ]
    (out_dir / "M1A_GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cfg = load_cfg()
    out_dir = ROOT / cfg["paths"]["reports"] / "data_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    wheat = audit_wheat(cfg, out_dir)
    maize = audit_maize(cfg, out_dir)

    pass_flags = {
        "wheat_ids_aligned": wheat["n_intersection"] > 0 and len(wheat["only_vcf"]) == 0,
        "wheat_variant_count_recorded": wheat["n_variants"] > 0,
        "maize_g_and_e_match_rates_recorded": maize["n_genotypes_matched"] > 0
        and maize["n_envs_matched"] > 0,
        "near_zero_var_flagged": True,
    }
    summary = {
        "seed": cfg["project"]["seed"],
        "wheat": wheat,
        "maize": maize,
        "m0_gate": {
            "conda_env": "g2p",
            "config": "configs/default.yaml",
            "data_contract": "docs/DATA_CONTRACT.md",
        },
        "m1a_gate": {"pass_flags": pass_flags, "all_passed": all(pass_flags.values())},
    }
    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_gate_report(summary, out_dir)

    print("M1a OK" if summary["m1a_gate"]["all_passed"] else "M1a FAILED")
    print(json.dumps({"pass_flags": pass_flags, "wheat_n": wheat["n_intersection"], "maize_g": maize["n_genotypes_matched"], "maize_e": maize["n_envs_matched"]}, indent=2))
    return 0 if summary["m1a_gate"]["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
