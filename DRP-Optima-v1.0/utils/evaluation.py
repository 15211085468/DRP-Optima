"""
论文级评估函数
为最终验收准备：计算业务指标（CSL, ITO, OOS Cost）

使用方法：
1. 训练好 RL 模型后，加载模型
2. 运行 evaluate_policy(model, env, num_episodes=100)
3. 获取业务指标，用于论文 Table 1 (Baseline vs Proposed 对比表)

指标说明：
- CSL (Cycle Service Level / 订单满足率)：不缺货的周期占比
- ITO (Inventory Turnover / 库存周转率)：总销量 / 平均库存
- OOS Cost (缺货成本占比)：缺货罚款占总成本的比例
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def evaluate_policy(
    model,
    env,
    num_episodes: int = 100,
    deterministic: bool = True,
    output_path: str = None
) -> Dict[str, float]:
    """
    评估策略（论文级业务指标）
    
    Args:
        model: 训练好的 RL 模型（PPO）
        env: 供应链环境（SupplyChainEnv）
        num_episodes: 评估回合数
        deterministic: 是否使用确定性策略
        output_path: 评估结果保存路径（JSON）
    
    Returns:
        dict: 业务指标字典
    """
    print(f"[INFO] 开始评估策略：{num_episodes} 个回合")
    print(f"  - 确定性策略：{deterministic}")
    
    # 存储每个回合的指标
    episode_metrics = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        # 单个回合的数据
        episode_data = {
            'demand': [],
            'fulfilled': [],
            'inventory': [],
            'reward': [],
            'oOS_cost': [],  # 缺货成本
            'total_cost': [],  # 总成本
        }
        
        while not (done or truncated):
            # 使用模型选择动作
            if hasattr(model, 'predict'):
                # Stable-Baselines3 模型
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # 自定义模型（需要实现 predict 方法）
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = model(obs_tensor).cpu().numpy()[0]
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            # 收集数据（假设 info 中包含必要信息）
            # 注意：需要根据 SupplyChainEnv 的实际 info 结构来调整
            demand = info.get('demand', 0)
            fulilled = info.get('fulilled', 0)
            inventory = info.get('inventory', 0)
            oos_cost = info.get('oos_cost', 0)
            total_cost = info.get('total_cost', abs(reward))
            
            episode_data['demand'].append(demand)
            episode_data['fulfilled'].append(fulilled)
            episode_data['inventory'].append(inventory)
            episode_data['reward'].append(reward)
            episode_data['oOS_cost'].append(oos_cost)
            episode_data['total_cost'].append(total_cost)
        
        # 计算单个回合的指标
        demand_arr = np.array(episode_data['demand'])
        fulilled_arr = np.array(episode_data['fulfilled'])
        inventory_arr = np.array(episode_data['inventory'])
        oos_cost_arr = np.array(episode_data['oOS_cost'])
        total_cost_arr = np.array(episode_data['total_cost'])
        
        # CSL：不缺货的周期占比
        csl = np.mean(fulilled_arr >= demand_arr) if len(demand_arr) > 0 else 0.0
        
        # ITO：总销量 / 平均库存
        total_sales = np.sum(fulilled_arr)
        avg_inventory = np.mean(inventory_arr) if len(inventory_arr) > 0 else 1.0
        ito = total_sales / avg_inventory if avg_inventory > 0 else 0.0
        
        # OOS Cost：缺货成本占总成本的比例
        total_oos_cost = np.sum(oos_cost_arr)
        total_all_cost = np.sum(total_cost_arr)
        oos_cost_ratio = total_oos_cost / total_all_cost if total_all_cost > 0 else 0.0
        
        episode_metrics.append({
            'csl': csl,
            'ito': ito,
            'oos_cost_ratio': oos_cost_ratio,
            'total_reward': np.sum(episode_data['reward']),
        })
        
        if (episode + 1) % 10 == 0:
            print(f"  已完成 {episode + 1}/{num_episodes} 回合")
    
    # 汇总所有回合的指标
    csl_values = [m['csl'] for m in episode_metrics]
    ito_values = [m['ito'] for m in episode_metrics]
    oos_values = [m['oos_cost_ratio'] for m in episode_metrics]
    reward_values = [m['total_reward'] for m in episode_metrics]
    
    results = {
        'csl_mean': np.mean(csl_values),
        'csl_std': np.std(csl_values),
        'ito_mean': np.mean(ito_values),
        'ito_std': np.std(ito_values),
        'oos_cost_ratio_mean': np.mean(oos_values),
        'oos_cost_ratio_std': np.std(oos_values),
        'total_reward_mean': np.mean(reward_values),
        'total_reward_std': np.std(reward_values),
        'num_episodes': num_episodes,
    }
    
    print(f"[INFO] 评估完成")
    print(f"  - CSL: {results['csl_mean']:.4f} ± {results['csl_std']:.4f}")
    print(f"  - ITO: {results['ito_mean']:.4f} ± {results['ito_std']:.4f}")
    print(f"  - OOS Cost Ratio: {results['oos_cost_ratio_mean']:.4f} ± {results['oos_cost_ratio_std']:.4f}")
    print(f"  - Total Reward: {results['total_reward_mean']:.2f} ± {results['total_reward_std']:.2f}")
    
    # 保存结果
    if output_path:
        import json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 评估结果已保存：{output_path}")
    
    return results


def compare_baseline_proposed(
    baseline_results: Dict[str, float],
    proposed_results: Dict[str, float],
    output_path: str = None
) -> pd.DataFrame:
    """
    对比 Baseline 和 Proposed 方法，生成论文 Table 1
    
    Args:
        baseline_results: Baseline 方法的评估结果
        proposed_results: Proposed 方法的评估结果
        output_path: 对比表保存路径（CSV 或 LaTeX）
    
    Returns:
        DataFrame: 对比表
    """
    print(f"[INFO] 生成 Baseline vs Proposed 对比表...")
    
    # 创建对比表
    comparison = pd.DataFrame({
        'Metric': ['CSL (Cycle Service Level)', 'ITO (Inventory Turnover)', 'OOS Cost Ratio', 'Total Reward'],
        'Baseline': [
            f"{baseline_results['csl_mean']:.4f} ± {baseline_results['csl_std']:.4f}",
            f"{baseline_results['ito_mean']:.4f} ± {baseline_results['ito_std']:.4f}",
            f"{baseline_results['oos_cost_ratio_mean']:.4f} ± {baseline_results['oos_cost_ratio_std']:.4f}",
            f"{baseline_results['total_reward_mean']:.2f} ± {baseline_results['total_reward_std']:.2f}",
        ],
        'Proposed': [
            f"{proposed_results['csl_mean']:.4f} ± {proposed_results['csl_std']:.4f}",
            f"{proposed_results['ito_mean']:.4f} ± {proposed_results['ito_std']:.4f}",
            f"{proposed_results['oos_cost_ratio_mean']:.4f} ± {proposed_results['oos_cost_ratio_std']:.4f}",
            f"{proposed_results['total_reward_mean']:.2f} ± {proposed_results['total_reward_std']:.2f}",
        ],
    })
    
    print(f"[INFO] 对比表生成完成")
    print(comparison.to_string(index=False))
    
    # 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.csv'):
            comparison.to_csv(output_path, index=False, encoding='utf-8')
            print(f"[INFO] 对比表已保存（CSV）：{output_path}")
        
        elif output_path.endswith('.tex'):
            # 生成 LaTeX 代码
            latex_code = comparison.to_latex(index=False, caption='Baseline vs Proposed Comparison', label='tab:comparison')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"[INFO] 对比表已保存（LaTeX）：{output_path}")
    
    return comparison


def main():
    """
    主函数（示例用法）
    """
    print("=" * 60)
    print("论文级评估函数 - 示例用法")
    print("=" * 60)
    print()
    
    # 示例：创建随机策略进行评估
    print(f"[INFO] 示例：使用随机策略评估")
    
    # 注意：这里需要实际的 env 和 model
    # 这里只是示例框架
    
    print(f"[WARN] 这是示例框架，需要实际的 env 和 model 才能运行")
    print(f"[WARN] 请在 50 万步训练完成后，加载真实模型并调用 evaluate_policy()")
    
    print()
    print("=" * 60)
    print("示例结束")
    print("=" * 60)


if __name__ == "__main__":
    main()
