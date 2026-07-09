# DRP-Optima

> 基于数据驱动的**连锁零售药店需求预测与补货优化系统**
> 融合 TFT 时序预测 + PPO 强化学习 + Optuna 超参调优

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![SB3](https://img.shields.io/badge/SB3-2.8-0081A7?style=flat-square&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

---

## 项目亮点

| 特性 | 说明 |
|:---|:---|
| 双模型架构 | TFT 精准预测 + PPO 智能补货决策 |
| 不确定性量化 | 9 分位数输出 (P10~P90)，风险可知 |
| 真实业务约束 | 最小订货量、货架容量、保质期全支持 |
| 多维奖励函数 | 12 维奖励分量 + 裁剪，平衡 SOR/OF R/ITR |
| 经典补货策略 | (s,S) 补货策略推理，单 SKU 即用 |
| 多维评估 | 预测指标 (SMAPE/MAE/RMSE) + 业务指标 (ITR/SOR/HCR/OFR) |
| 模块化设计 | ARIMA/NHiTS/NBeats 对比模型即插即用 |
| 统计显著性检验 | Bootstrap 置信区间法验证模型优越性 (P<0.05) |
| 组件重要性分析 | 近似消融分析，量化注意力机制等组件贡献 |

---

## ⚠️ 发布版本说明（重要）

本目录为 **DRP-Optima v1.0 发布版**，仅包含运行系统所需的**核心源代码、最佳模型与配置**，**不包含**以下内容：

| 类别 | 已排除内容 | 说明 |
|:---|:---|:---|
| 训练产物 | `models/global_ppo*/`（PPO 检查点） | 训练过程与结果文件 |
| 训练日志 | `logs/`、`output/`、`records/output/` | 训练与评估输出 |
| 优化过程 | `optuna_results/`、`tuning_results/` | 超参搜索过程 |
| 归档文件 | `archive/` 及模型 `optimization/archive/` | 历史脚本与旧版源码 |
| 测试文件 | `tests/`、`data/test_data/` | 单元测试与测试数据 |
| 论文材料 | `docs/paper/`、`docs/paper_sections/` | 论文图表与草稿 |
| 开发脚本 | `scripts/` 及一次性数据处理脚本 | 绘图/分析/数据修补脚本 |
| 工作文件 | `.workbuddy/` | 编辑器/工具配置 |

**数据文件说明**：本发布版**不包含任何销售记录或训练数据**。仅保留了运行必需的少量主数据元数据：

- `data/goods_info.csv` — 商品主数据（必需）
- `data/shop_info.csv` — 门店主数据（必需）
- `data/main_dtypes_dict.json` — 数据列类型定义（必需）

要进行训练或完整推理，需用户**自行准备销售数据**（如 `data/sale_data.csv`）后再运行。

---

## 系统架构

DRP-Optima 采用「预测 → 决策」两段式架构：

```
        ┌─────────────────┐         ┌──────────────────────┐
 历史销售 │  TFT 时序预测   │ 分位数需求 │   PPO 强化学习补货   │ 补货决策
 数据  ─▶│(Temporal Fusion │────────▶│  决策 (Supply Chain   │────────▶
        │  Transformer)   │ 预测    │   Env + PPO Policy)  │ (s,Q)
        └─────────────────┘         └──────────────────────┘
                  │                            │
                  └─────── 集成层 integration/ ─┘
```

- **预测层**（`models/forecasting/`）：TFT 输出 9 个分位数（P10–P90）需求分布。
- **决策层**（`models/optimization/` + `models/rl/`）：以预测需求为状态，PPO 学习最优补货量。
- **集成层**（`integration/`）：将 TFT 预测注入强化学习环境，实现端到端优化。

---

## 目录结构（发布版真实结构）

```
DRP-Optima-v1.0/
├── README.md                  # 本文件
├── RELEASE_NOTE.md            # 发布说明
├── LICENSE                    # MIT 许可证
├── requirements.txt           # Python 依赖清单
├── .gitignore
│
├── cli.py                     # 统一命令行入口 (train/infer/evaluate/pipeline/graph/config)
├── inference.py               # 单步推理脚本
├── train_mvp.py               # 轻量训练（快速验证，默认 10 SKU）
├── train_v9.py                # 完整训练（默认 88 SKU / 82 门店）
├── minimal_run_test.py        # 环境自检（无需数据）
│
├── config/                    # 配置中心
│   ├── data_config.py
│   ├── main.yaml / train.yaml / inference.yaml / evaluate.yaml
│   ├── presets/               # 预设配置 (baseline / proposed / smoke_test ...)
│   └── loader.py / settings.py / validator.py / exceptions.py
│
├── data/                      # 数据管道（不含销售数据）
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── feature_engineering.py
│   ├── global_data_loader.py
│   ├── intersection_preprocessor.py
│   ├── goods_info.csv         # 商品主数据（必需）
│   ├── shop_info.csv          # 门店主数据（必需）
│   └── main_dtypes_dict.json
│
├── models/                    # 模型定义
│   ├── forecasting/           # 预测模型
│   │   ├── tft_model.py
│   │   ├── decision_aware_tft.py
│   │   ├── tft_model_best.ckpt   # ✅ 最佳 TFT 模型（已内置）
│   │   ├── arima_forecaster.py / lstm_forecaster.py / lightgbm_forecaster.py ...
│   │   └── hyperparameter_optimizer.py
│   ├── optimization/          # 供应链强化学习环境
│   │   ├── supply_chain_env.py / supply_chain_env_tft.py
│   │   ├── global_supply_chain_env.py / multi_store_weekly_env.py
│   │   ├── constraint_handler.py / reward_enhancer.py / reward_normalizer.py
│   │   └── vec_supply_chain_env.py
│   ├── rl/                    # 强化学习策略
│   │   ├── heuristic_policy.py / adaptive_heuristic_policy.py
│   │   └── tft_structure_wrapper.py
│   └── decision_cost.py
│
├── integration/               # TFT + DRL 集成层
│   ├── tft_drl_integration.py
│   ├── env_factory.py
│   ├── inference_api.py
│   └── multi_objective_optimizer.py
│
├── training/                  # 训练流程封装
│   ├── trainer.py / ppo_pipeline.py
│   ├── train_mvp.py / train_v9.py
│   └── callbacks.py / rl_callbacks.py / visualization_callbacks.py / data_utils.py
│
└── utils/                     # 工具模块
    ├── evaluation/            # 评估 (evaluator.py / tft_evaluator.py)
    ├── logger.py / plotting.py
    ├── service_level_diagnostics.py
    └── visualization_collector.py
```

> 说明：根目录的 `train_mvp.py` / `train_v9.py` 为便捷入口，内部调用 `training/` 中的同名训练逻辑。

---

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- pytorch-forecasting 1.7+
- stable-baselines3 2.8+
- Gymnasium 0.29+

---

## 安装

```bash
# 建议使用虚拟环境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 快速开始

### 1. 环境自检（无需数据）

发布版内置最佳模型与少量主数据，可直接验证代码可运行：

```bash
python minimal_run_test.py
```

### 2. 查看命令行帮助

```bash
python cli.py --help
python cli.py train --help
```

### 3. 推理（使用内置最佳 TFT 模型）

```bash
# 方式 A：通过 cli（推荐）
python cli.py infer --model-path models/forecasting/tft_model_best.ckpt

# 方式 B：直接调用推理脚本
python inference.py --product-code 1010101160 --store-code 16 --weeks 4
```

### 4. 评估模型

```bash
python cli.py evaluate --model-path models/forecasting/tft_model_best.ckpt
```

### 5. 训练模型

```bash
# 轻量 MVP 训练（默认 10 SKU，快速验证）
python train_mvp.py --total-timesteps 500000 --use-tft-predictions

# 完整 V9 训练（默认 88 SKU / 82 门店）
python train_v9.py --total-timesteps 5000000 --use-tft-predictions

# 或通过 cli 统一入口
python cli.py train --model integrated --total-steps 5000000
```

> **注意**：训练需要用户自备销售数据（如 `data/sale_data.csv`）。请参考 `data/data_loader.py` 中的数据路径约定准备数据。

### 6. 运行完整流程

```bash
python cli.py pipeline --skip-train   # 跳过训练，仅做推理/评估
```

---

## 内置最佳模型

发布版已包含训练好的最佳 TFT 模型，位于：

```
models/forecasting/tft_model_best.ckpt
```

- 该模型在验证集上取得最佳 `val_loss`，可直接用于推理，无需重新训练。
- 加载方式（代码内部已封装，一般无需手动加载）：

```python
from models.forecasting.tft_model import TFTModel
model = TFTModel.load("models/forecasting/tft_model_best.ckpt")
```

---

## 配置说明

所有运行参数集中在 `config/` 目录：

- `config/main.yaml` — 全局默认配置
- `config/train.yaml` / `config/inference.yaml` / `config/evaluate.yaml` — 各阶段配置
- `config/presets/` — 预设场景（如 `proposed.yaml` 为本文完整方案）

可通过 `python cli.py config show` 查看当前生效配置，或 `python cli.py config generate --output my.yaml` 生成示例配置。

---

## 许可证

本项目以 [MIT License](LICENSE) 发布。
