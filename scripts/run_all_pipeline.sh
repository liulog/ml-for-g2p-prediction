#!/usr/bin/env bash
# Full G2P pipeline: wheat + maize + M6 delivery.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"
PYTHON="${PYTHON:-/home/jingyu/miniconda3/envs/g2p/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python; fi

run() { echo "== $* =="; "$PYTHON" "$@"; }

run scripts/m0_smoke_check.py
run scripts/m1a_data_audit.py
run scripts/m1b_wheat_genotype_qc.py
run scripts/m1c_wheat_ld_pca_grm.py
run scripts/m2_wheat_pilot_baselines.py
run scripts/m3a_wheat_feature_compare.py
run scripts/m3b_wheat_all_traits.py
run scripts/m4a_maize_align.py
run scripts/m4b_maize_phenotype_adjust.py
run scripts/m4c_maize_env_features.py
run scripts/m4d_maize_genotype_pca.py
run scripts/m5_maize_gxe_baselines.py
run scripts/m5b_maize_raw_yield_ablation.py
run scripts/m6a_maize_traits_bootstrap.py
run scripts/m6b_interpretability.py
run scripts/m6c_final_delivery.py

echo "ALL PIPELINE COMPLETE"
echo "See reports/tables/FINAL_DELIVERY_REPORT.md"
