#!/usr/bin/env bash
# Reproduce wheat G2P pipeline (M0 checks + M1–M3). Requires conda env `g2p`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x /home/jingyu/miniconda3/envs/g2p/bin/python ]]; then
    PYTHON=/home/jingyu/miniconda3/envs/g2p/bin/python
  else
    PYTHON=python
  fi
fi

echo "== M0 smoke =="
"$PYTHON" scripts/m0_smoke_check.py
echo "== M1a audit =="
"$PYTHON" scripts/m1a_data_audit.py
echo "== M1b genotype QC =="
"$PYTHON" scripts/m1b_wheat_genotype_qc.py
echo "== M1c LD/PCA/GRM =="
"$PYTHON" scripts/m1c_wheat_ld_pca_grm.py
echo "== M2 pilot baselines =="
"$PYTHON" scripts/m2_wheat_pilot_baselines.py
echo "== M3a feature compare =="
"$PYTHON" scripts/m3a_wheat_feature_compare.py
echo "== M3b all traits =="
"$PYTHON" scripts/m3b_wheat_all_traits.py
echo "Wheat pipeline complete. See results/metrics/ and reports/ (local)."
