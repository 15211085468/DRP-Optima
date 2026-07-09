# Models 目录说明

## 目录结构

### `forecasting/`
时间序列预测模型，包含：
- `tft_model.py` - TFT (Temporal Fusion Transformer) 模型实现
- `decision_aware_tft.py` - 决策感知 TFT 模型
- `tft_model_best.ckpt` - **最佳 TFT 模型**（2026-06-24 从 `records/models/` 复制）
- 其他预测模型（ARIMA, LSTM, LightGBM 等）

**使用场景**：生成需求预测，作为 PPO 训练的输入特征。

---

### `rl/`
强化学习模型，包含：
- `ppo_agent_sb3.py` - PPO 智能体实现（基于 Stable-Baselines3）
- `tft_structure_wrapper.py` - TFT 模型结构包装器

---

### `global_ppo_final_v7/`
当前最佳 PPO 模型（训练中），包含：
- `global_ppo_model.zip` - 最终模型（待保存）
- `checkpoints/` - 训练 checkpoint（每 50,000 步保存）
- `logs/` - 训练日志（Monitor CSV）
- `training_v7.log` - 训练输出日志

**训练配置**：
- `std` 约束：`[-5, -1.0]`（`std` ≤ 0.368）
- `max_reorder`：60（从 200 降至 60）
- 网络架构：`[512, 256, 128]`
- 重置策略头部（保留特征提取器）

---

### 其他目录（待归档）

| 目录 | 说明 | 状态 |
|------|------|------|
| `global_ppo/` | 初始 PPO 训练 | 待评估 |
| `global_ppo_final/` | V1 最终模型 | 待评估 |
| `global_ppo_final_v2/` ~ `v6/` | V2~V6 训练 | 待评估 |
| `global_ppo_smoke/` ~ `v4/` | 冒烟测试 | 可归档 |
| `global_ppo_test/` ~ `fixed_v2/` | 测试训练 | 可归档 |

---

## 模型文件命名规范

根据项目规范：
- **最佳模型**：文件名包含 `best` 字样（如 `tft_model_best.ckpt`）
- **Checkpoint**：`ppo_model_{steps}_steps.zip`（如 `ppo_model_950000_steps.zip`）
- **最终模型**：`global_ppo_model.zip`

---

## 清理记录

### 2026-06-24
- ✅ 复制 `tft_model_best.ckpt` 到 `models/forecasting/`
- ✅ 归档 `records/` 中的旧文件到 `records/archive/`
- ⏳ 待评估：V1~V6 模型是否可归档

---

## 使用说明

### 加载 TFT 最佳模型
```python
from models.forecasting.tft_model import TFTModel
model = TFTModel.load("models/forecasting/tft_model_best.ckpt")
```

### 加载 PPO 最佳模型
```python
from stable_baselines3 import PPO
model = PPO.load("models/global_ppo_final_v7/global_ppo_model.zip")
```

---

**最后更新**：2026-06-24
