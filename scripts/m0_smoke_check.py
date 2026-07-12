#!/usr/bin/env python3
"""M0 smoke check: config, raw data paths, and core imports."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["project"]["seed"] == 2026
    required = [
        cfg["paths"]["wheat_vcf"],
        cfg["paths"]["wheat_pheno"],
        cfg["paths"]["maize_pheno"],
        cfg["paths"]["maize_env"],
        cfg["paths"]["maize_plink_prefix"] + ".bed",
        cfg["paths"]["maize_plink_prefix"] + ".bim",
        cfg["paths"]["maize_plink_prefix"] + ".fam",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        print("MISSING:", missing)
        return 1

    core = [
        "numpy",
        "pandas",
        "yaml",
        "sklearn",
        "scipy",
        "matplotlib",
        "seaborn",
        "lightgbm",
        "pyarrow",
        "statsmodels",
        "joblib",
    ]
    optional = ["optuna", "tqdm"]
    for name in core:
        __import__(name)
    optional_status = {}
    for name in optional:
        try:
            __import__(name)
            optional_status[name] = "ok"
        except ImportError:
            optional_status[name] = "missing (install later)"

    print("M0 OK")
    print(f"  seed={cfg['project']['seed']}")
    print(f"  wheat_traits={len(cfg['wheat']['all_traits'])}")
    print(f"  pilot_traits={cfg['wheat']['pilot_traits']}")
    print(f"  maize_primary={cfg['maize']['primary_trait']}")
    print(f"  optional={optional_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
