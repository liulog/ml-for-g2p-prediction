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

- **计划核心 M0–M8 已完成**（BayesA/B/C、天气序列、多任务、Optuna 等为可选扩展）
- 小麦：审计 → QC → 基线/特征/15 性状 → holdout/RR-BLUP → BayesianRidge/ARD → GWAS±PC → 多种子
- 玉米：对齐 → 环境/基因型表征 → G/E/G×E → 均值/反应规范/双塔 → leave-both / leave-location
- M6–M7：bootstrap CI、重要性、FDR、Route A、推理脚本
- 本地报告：`reports/tables/FINAL_DELIVERY_REPORT.md`
- 一键复现：`bash scripts/run_all_pipeline.sh`

