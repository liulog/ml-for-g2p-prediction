# ml-for-g2p-prediction

基因组预测（G2P）：小麦 G→P，玉米 G+E(+G×E)→P。

## 环境

```bash
conda activate g2p
# 或从零创建：
# conda env create -f environment.yml
```

依赖见 `requirements.txt` / `environment.yml`。随机种子固定为 `2026`（`configs/default.yaml`）。

## 数据

| 路径 | 说明 |
|---|---|
| `wheat1k/` | 小麦 VCF + 表型（本地，不入 git） |
| `trainingcleandata/` | 玉米 PLINK + 表型 + 天气（本地，不入 git） |

本地文档（不入 git）：`docs/DATA_CONTRACT.md`、`docs/G2P_EXECUTION_PLAN.md`。
实验产物 `results/`、`reports/` 与原始数据集亦不入 git。

## 目录

```text
configs/   YAML 配置
data/      interim / processed（清洗结果，本地）
reports/   审计报告与图表（本地）
results/   splits / predictions / metrics / models（本地）
scripts/   流水线入口
src/       可复用代码
```

## 当前阶段

- **小麦 M0–M3 已完成**：审计 → QC → LD/PCA/GRM → 试点基线 → 特征对比 → 15 性状 + SNP 稳定性
- **玉米 M4–M5 已完成**：G/P/E 对齐 → 表型校正 → 环境特征/PCA → 基因型 PCA → Yield G/E/G×E 基线（leave-genotype / leave-environment / leave-year）
- 本地汇总：`reports/tables/WHEAT_M3_SUMMARY.md`；指标见 `results/metrics/`（不入 git）
- 小麦复现：`bash scripts/run_wheat_pipeline.sh`
- 玉米复现：`python scripts/m4a_maize_align.py && python scripts/m4b_maize_phenotype_adjust.py && python scripts/m4c_maize_env_features.py && python scripts/m4d_maize_genotype_pca.py && python scripts/m5_maize_gxe_baselines.py`
