# DRP-Optima 发布版本说明

**版本**: v1.0
**日期**: 2026-07-09

## 目录结构（发布版）

```
DRP-Optima-v1.0/
├── README.md                  # 项目说明（已针对发布版更新）
├── RELEASE_NOTE.md            # 本文件
├── LICENSE                    # MIT 许可证
├── requirements.txt           # 依赖清单
├── .gitignore
│
├── cli.py / inference.py / train_mvp.py / train_v9.py / minimal_run_test.py
├── config/                    # 配置模块 + YAML
├── data/                      # 数据管道（仅含运行必需的少量主数据）
├── models/                    # 模型定义
│   └── forecasting/
│       └── tft_model_best.ckpt   # ✅ 最佳 TFT 模型
├── integration/               # TFT + DRL 集成
├── training/                  # 训练流程
└── utils/                     # 工具（含 utils/evaluation/）
```

## 本发布版包含

- ✅ 核心源代码：config / data / models / integration / training / utils
- ✅ 统一命令行入口 `cli.py` 及推理/训练便捷脚本
- ✅ 最佳 TFT 模型 `models/forecasting/tft_model_best.ckpt`
- ✅ 运行必需的主数据：`data/goods_info.csv`、`data/shop_info.csv`、`data/main_dtypes_dict.json`
- ✅ 配置文件、README、LICENSE

## 本发布版不包含

以减小体积并保护数据隐私，发布版**不包含**以下文件：

- ❌ 任何销售记录 / 训练数据（CSV、PKL 等原始数据）
- ❌ 训练产物（PPO 检查点 `models/global_ppo*/` 等）
- ❌ 训练日志与结果（`logs/`、`output/`、`records/output/`）
- ❌ 超参优化过程（`optuna_results/`、`tuning_results/`）
- ❌ 归档与历史脚本（`archive/` 及 `models/optimization/archive/`）
- ❌ 单元测试与测试数据（`tests/`、`data/test_data/`）
- ❌ 论文材料（`docs/paper/`、`docs/paper_sections/`）
- ❌ 开发辅助脚本（`scripts/` 及一次性数据处理脚本）
- ❌ 工作文件（`.workbuddy/`）

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 环境自检（无需数据）
python minimal_run_test.py

# 3. 推理（使用内置最佳模型）
python cli.py infer --model-path models/forecasting/tft_model_best.ckpt

# 4. 训练（需自备销售数据 data/sale_data.csv）
python train_mvp.py --total-timesteps 500000 --use-tft-predictions
```

## 模型文件

发布版内置最佳 TFT 模型：

- `models/forecasting/tft_model_best.ckpt` — 最佳 TFT 模型权重

> 注：早期版本曾将模型置于 `records/models/`，发布版 v1.0 已统一移至 `models/forecasting/` 并删除冗余副本。

## 技术栈

- Python 3.10+
- PyTorch 2.0+
- pytorch-forecasting 1.7+
- stable-baselines3 2.8+
- Gymnasium 0.29+

## 许可

MIT License
