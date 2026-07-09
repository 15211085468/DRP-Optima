"""
满足率诊断工具 - 分析为什么PPO模型的订单满足率不足95%

诊断维度：
1. 奖励函数分析：缺货惩罚是否足够？
2. 库存分析：安全库存是否充足？
3. 预测分析：需求预测是否准确？
4. 动作分析：Agent是否学会了正确补货？
5. 训练分析：训练是否收敛？
"""

import numpy as np
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class ServiceLevelDiagnostics:
    """满足率诊断器"""
    
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent
        self.diagnostics = {}
    
    def run_full_diagnosis(self, num_episodes: int = 10) -> Dict[str, Any]:
        """运行完整诊断"""
        
        print("=" * 80)
        print("🔍 开始满足率诊断...")
        print("=" * 80)
        
        # 收集数据
        episode_data = self._collect_episode_data(num_episodes)
        
        # 诊断1: 奖励函数分析
        print("\n📊 诊断1: 奖励函数分析...")
        reward_diagnostics = self._diagnose_reward_function(episode_data)
        self.diagnostics['reward'] = reward_diagnostics
        
        # 诊断2: 库存分析
        print("\n📊 诊断2: 库存分析...")
        inventory_diagnostics = self._diagnose_inventory(episode_data)
        self.diagnostics['inventory'] = inventory_diagnostics
        
        # 诊断3: 预测分析
        print("\n📊 诊断3: 预测分析...")
        forecast_diagnostics = self._diagnose_forecast_accuracy(episode_data)
        self.diagnostics['forecast'] = forecast_diagnostics
        
        # 诊断4: 动作分析
        print("\n📊 诊断4: 动作分析...")
        action_diagnostics = self._diagnose_agent_actions(episode_data)
        self.diagnostics['action'] = action_diagnostics
        
        # 诊断5: 训练分析
        print("\n📊 诊断5: 训练分析...")
        training_diagnostics = self._diagnose_training_convergence()
        self.diagnostics['training'] = training_diagnostics
        
        # 生成诊断报告
        report = self._generate_diagnostic_report()
        
        print("\n" + "=" * 80)
        print("✅ 诊断完成！")
        print("=" * 80)
        
        return report
    
    def _collect_episode_data(self, num_episodes: int) -> List[Dict]:
        """收集多个episode的数据"""
        
        episode_data = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_steps = []
            
            while not done:
                if self.agent is not None:
                    action, _ = self.agent.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 记录数据
                step_data = {
                    'step': len(episode_steps),
                    'reward': float(reward),
                    'fill_rate': info.get('current_fill_rate', 0.0),
                    'shortage_cost': info.get('shortage_cost', 0.0),
                    'holding_cost': info.get('holding_cost', 0.0),
                    'service_level_bonus': info.get('service_level_bonus', 0.0),
                    'inventory': self.env.store_inventory.copy(),
                    'demand': self.env.current_demand.copy() if hasattr(self.env, 'current_demand') else None,
                    'action': action.copy() if action is not None else None,
                }
                episode_steps.append(step_data)
                
                obs = next_obs
                done = terminated or truncated
            
            episode_data.append({
                'episode_id': ep,
                'steps': episode_steps,
                'avg_fill_rate': np.mean([s['fill_rate'] for s in episode_steps]),
                'avg_reward': np.mean([s['reward'] for s in episode_steps]),
            })
        
        return episode_data
    
    def _diagnose_reward_function(self, episode_data: List[Dict]) -> Dict:
        """诊断1: 奖励函数是否合理"""
        
        all_fill_rates = []
        all_shortage_costs = []
        all_service_bonuses = []
        
        for ep in episode_data:
            for step in ep['steps']:
                all_fill_rates.append(step['fill_rate'])
                all_shortage_costs.append(step['shortage_cost'])
                all_service_bonuses.append(step['service_level_bonus'])
        
        # 分析
        avg_fill_rate = np.mean(all_fill_rates)
        fill_rate_above_95 = np.sum(np.array(all_fill_rates) >= 0.95) / len(all_fill_rates)
        
        diagnostics = {
            'avg_fill_rate': float(avg_fill_rate),
            'fill_rate_above_95_pct': float(fill_rate_above_95 * 100),
            'avg_shortage_cost': float(np.mean(all_shortage_costs)),
            'avg_service_bonus': float(np.mean(all_service_bonuses)),
            'reward_alignment': 'GOOD' if avg_fill_rate >= 0.95 else 'BAD',
        }
        
        # 建议
        if diagnostics['avg_fill_rate'] < 0.95:
            diagnostics['recommendations'] = [
                f"增加缺货惩罚（当前: {self.env.config.shortage_penalty_per_unit}）→ 建议: 15.0",
                f"增加满足率奖励（当前: 5.0）→ 建议: 10.0",
                "启用 reward_enhancer（增强版满足率奖励）",
            ]
        
        return diagnostics
    
    def _diagnose_inventory(self, episode_data: List[Dict]) -> Dict:
        """诊断2: 库存是否充足"""
        
        all_inventories = []
        all_demands = []
        
        for ep in episode_data:
            for step in ep['steps']:
                if step['inventory'] is not None:
                    all_inventories.append(step['inventory'])
                if step['demand'] is not None:
                    all_demands.append(step['demand'])
        
        if not all_inventories:
            return {'error': 'No inventory data'}
        
        all_inventories = np.array(all_inventories)
        all_demands = np.array(all_demands)
        
        # 计算库存覆盖率
        inventory_coverage = all_inventories / (all_demands + 1e-5)
        avg_coverage = np.mean(inventory_coverage)
        
        diagnostics = {
            'avg_inventory_coverage': float(avg_coverage),
            'inventory_below_demand_pct': float(np.mean(inventory_coverage < 1.0) * 100),
            'recommendation': 'INCREASE_SAFETY_STOCK' if avg_coverage < 1.5 else 'OK',
        }
        
        if diagnostics['avg_inventory_coverage'] < 1.5:
            diagnostics['recommendations'] = [
                f"增加安全库存因子（当前: 1.65）→ 建议: 2.33（99%服务水平）",
                "提高目标库存水平（target_inventory_level）",
                "检查Lead Time是否被正确考虑",
            ]
        
        return diagnostics
    
    def _diagnose_forecast_accuracy(self, episode_data: List[Dict]) -> Dict:
        """诊断3: 需求预测是否准确"""
        
        # 简化版：检查预测方法
        forecast_method = self.env.config.forecast_method
        
        diagnostics = {
            'forecast_method': forecast_method,
            'recommendation': 'OK',
        }
        
        if forecast_method == 'wma':
            diagnostics['recommendations'] = [
                "WMA预测可能不准确，建议改用TFT分位数预测",
                "如果使用TFT，确保用P90分位数（而非P50）",
            ]
        
        return diagnostics
    
    def _diagnose_agent_actions(self, episode_data: List[Dict]) -> Dict:
        """诊断4: Agent是否学会正确补货"""
        
        all_actions = []
        
        for ep in episode_data:
            for step in ep['steps']:
                if step['action'] is not None:
                    all_actions.append(step['action'])
        
        if not all_actions:
            return {'error': 'No action data'}
        
        all_actions = np.array(all_actions)
        
        diagnostics = {
            'avg_action': float(np.mean(all_actions)),
            'action_std': float(np.std(all_actions)),
            'action_min': float(np.min(all_actions)),
            'action_max': float(np.max(all_actions)),
        }
        
        # 判断Agent是否学会补货
        if diagnostics['action_std'] < 1e-5:
            diagnostics['recommendation'] = 'AGENT_NOT_LEARNING'  # Agent没有学习，动作不变
            diagnostics['recommendations'] = [
                "增加熵系数（ent_coef）鼓励探索",
                "检查奖励函数是否有梯度",
                "增加训练时长",
            ]
        
        return diagnostics
    
    def _diagnose_training_convergence(self) -> Dict:
        """诊断5: 训练是否收敛"""
        
        diagnostics = {
            'total_timesteps': getattr(self.env, 'total_training_steps', 0),
            'convergence_status': 'UNKNOWN',
        }
        
        # 简单判断：如果训练步数 < 500k，认为未收敛
        if diagnostics['total_timesteps'] < 500000:
            diagnostics['convergence_status'] = 'NOT_CONVERGED'
            diagnostics['recommendations'] = [
                f"当前训练步数: {diagnostics['total_timesteps']}",
                "建议: 至少训练 500,000 步",
                "使用 proposed_aggressive.yaml 配置",
            ]
        
        return diagnostics
    
    def _generate_diagnostic_report(self) -> str:
        """生成诊断报告"""
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("📊 满足率诊断报告")
        report.append("=" * 80)
        
        # 奖励函数诊断
        reward_diag = self.diagnostics.get('reward', {})
        report.append("\n1️⃣ 奖励函数诊断:")
        report.append(f"   平均满足率: {reward_diag.get('avg_fill_rate', 0):.2%}")
        report.append(f"   满足率>=95%的步数占比: {reward_diag.get('fill_rate_above_95_pct', 0):.1f}%")
        report.append(f"   对齐状态: {reward_diag.get('reward_alignment', 'UNKNOWN')}")
        
        if 'recommendations' in reward_diag:
            report.append("   💡 建议:")
            for rec in reward_diag['recommendations']:
                report.append(f"      - {rec}")
        
        # 库存诊断
        inv_diag = self.diagnostics.get('inventory', {})
        report.append("\n2️⃣ 库存诊断:")
        report.append(f"   平均库存覆盖率: {inv_diag.get('avg_inventory_coverage', 0):.2f}x")
        report.append(f"   库存低于需求的比例: {inv_diag.get('inventory_below_demand_pct', 0):.1f}%")
        report.append(f"   建议: {inv_diag.get('recommendation', 'OK')}")
        
        if 'recommendations' in inv_diag:
            report.append("   💡 建议:")
            for rec in inv_diag['recommendations']:
                report.append(f"      - {rec}")
        
        # 训练诊断
        train_diag = self.diagnostics.get('training', {})
        report.append("\n3️⃣ 训练诊断:")
        report.append(f"   总训练步数: {train_diag.get('total_timesteps', 0):,}")
        report.append(f"   收敛状态: {train_diag.get('convergence_status', 'UNKNOWN')}")
        
        if 'recommendations' in train_diag:
            report.append("   💡 建议:")
            for rec in train_diag['recommendations']:
                report.append(f"      - {rec}")
        
        # 总结
        report.append("\n" + "=" * 80)
        report.append("🎯 行动计划:")
        report.append("=" * 80)
        report.append("1. 使用 proposed_aggressive.yaml 配置重新训练")
        report.append("2. 训练至少 1,000,000 步")
        report.append("3. 启用 reward_enhancer（增强版满足率奖励）")
        report.append("4. 增加安全库存因子至 2.33（99%服务水平）")
        report.append("5. 训练完成后，再次运行此诊断工具验证")
        report.append("=" * 80)
        
        return "\n".join(report)


def run_diagnostics(env, agent=None, output_path: str = "diagnostics_report.json"):
    """
    运行满足率诊断
    
    使用示例：
    ```python
    # 注释：models.agents.ppo_agent 不存在，暂时注释
    # from models.optimization.supply_chain_env import SupplyChainEnv
    # from models.agents.ppo_agent import PPOAgent
    # from utils.diagnostics import run_diagnostics
    
    env = SupplyChainEnv(config=env_config)
    agent = PPOAgent.load("models/ppo_proposed.zip")
    
    report = run_diagnostics(env, agent)
    print(report)
    ```
    """
    
    diagnostics = ServiceLevelDiagnostics(env, agent)
    report = diagnostics.run_full_diagnosis(num_episodes=10)
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(diagnostics.diagnostics, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 诊断报告已保存至: {output_path}")
    
    return report
