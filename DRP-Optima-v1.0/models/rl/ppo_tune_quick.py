"""
简化版超参数调优 - 快速验证
"""
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
from config.data_config import EnvConfig
from models.optimization.supply_chain_env import SupplyChainEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

print("=" * 80, flush=True)
print("简化版超参数调优（快速验证）", flush=True)
print("=" * 80, flush=True)

# 测试不同学习率
learning_rates = [1e-4, 3e-4, 1e-3]
results = []

for i, lr in enumerate(learning_rates):
    print(f"\n试验 {i+1}/{len(learning_rates)}: learning_rate={lr}", flush=True)
    
    # 创建环境
    print(f"  1. 创建环境...", flush=True)
    env_config = EnvConfig(
        lead_time=3,
        order_cost=5.0,
        holding_cost_rate=0.1,
        store_capacity=1000,
    )
    
    env = SupplyChainEnv(
        num_skus=16,
        num_stores=1,
        env_config=env_config,
        tft_wrapper=None,
    )
    env = Monitor(env, filename=None)
    env = DummyVecEnv([lambda: env])
    
    # 创建模型
    print(f"  2. 创建PPO模型...", flush=True)
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=lr,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=0,
    )
    
    # 训练
    print(f"  3. 训练（5000步）...", flush=True)
    model.learn(total_timesteps=5000, progress_bar=False)
    print(f"     训练完成！", flush=True)
    
    # 简单评估（运行1个episode）
    print(f"  4. 评估...", flush=True)
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) >= 1:
        obs = reset_result[0]
    else:
        obs = reset_result
    
    total_reward = 0.0
    done = False
    steps = 0
    
    while not done and steps < 365:  # 最多365步（1个episode）
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        
        if len(step_result) >= 5:
            obs, reward, terminated, truncated, info = step_result[0], step_result[1], step_result[2], step_result[3], step_result[4]
            done = terminated or truncated
        elif len(step_result) >= 4:
            obs, reward, done, info = step_result[0], step_result[1], step_result[2], step_result[3]
            terminated = done
            truncated = False
        else:
            break
        
        total_reward += reward
        steps += 1
    
    avg_reward = total_reward / max(steps, 1)
    # 确保是标量
    if isinstance(avg_reward, np.ndarray):
        avg_reward = float(avg_reward.item())
    elif not isinstance(avg_reward, (int, float)):
        avg_reward = float(avg_reward)
    
    results.append({'lr': lr, 'avg_reward': avg_reward})
    print(f"     平均Reward: {avg_reward:.4f}", flush=True)
    
    # 清理
    env.close()

# 输出结果
print("\n" + "=" * 80, flush=True)
print("结果汇总：", flush=True)
print("=" * 80, flush=True)
for r in results:
    print(f"  learning_rate={r['lr']}: avg_reward={r['avg_reward']:.4f}", flush=True)

# 找出最佳
best = max(results, key=lambda x: x['avg_reward'])
print(f"\n最佳学习率: {best['lr']} (avg_reward={best['avg_reward']:.4f})", flush=True)
print("=" * 80, flush=True)
