"""
改进的Heuristic策略：滚动预测 + 动态调整
"""
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveHeuristicPolicy:
    """
    自适应Heuristic策略
    
    改进点：
    1. 使用滚动窗口计算统计特征（而非固定历史数据）
    2. 根据近期需求趋势动态调整(s, S)参数
    3. 考虑季节性因素（如果有）
    """
    
    def __init__(
        self,
        num_skus: int,
        lead_time_weeks: int = 1,
        history_window: int = 8,
        service_level: dict = None
    ):
        """
        初始化自适应Heuristic策略
        
        参数：
            num_skus: SKU数量
            lead_time_weeks: 提前期（周）
            history_window: 历史数据窗口
            service_level: 服务水平字典（A/B/C类对应的服务水平）
        """
        self.num_skus = num_skus
        self.lead_time_weeks = lead_time_weeks
        self.history_window = history_window
        
        # 服务水平（默认值）
        if service_level is None:
            service_level = {'A': 0.99, 'B': 0.95, 'C': 0.90}
        self.service_level = service_level
        self.z_score = {'A': 2.33, 'B': 1.65, 'C': 1.28}
        
        # 历史需求数据（滚动窗口）
        self.demand_history = []
        
        # ABC分类结果
        self.abc_class = None
        
        # (s, S)参数（动态更新）
        self.s = None
        self.S = None
        
        # 是否已初始化
        self.initialized = False
    
    def update(self, current_demand: np.ndarray):
        """
        更新需求历史
        
        参数：
            current_demand: 当前需求，shape: (num_skus,)
        """
        self.demand_history.append(current_demand.copy())
        
        # 保持窗口大小
        if len(self.demand_history) > self.history_window:
            self.demand_history = self.demand_history[-self.history_window:]
        
        # 动态更新(s, S)参数（每步都更新）
        if len(self.demand_history) >= 4:  # 至少4周数据
            self._update_params()
    
    def _update_params(self):
        """
        动态更新(s, S)参数
        """
        # 使用最近的历史数据
        demand_array = np.array(self.demand_history)  # shape: (weeks, num_skus)
        
        # 计算统计特征
        mean_demand = np.mean(demand_array, axis=0)
        std_demand = np.std(demand_array, axis=0)
        std_demand = np.maximum(std_demand, 0.1)
        
        # ABC分类（基于平均需求）
        if self.abc_class is None:
            self._abc_classification(mean_demand)
        
        # 计算(s, S)
        self.s = np.zeros(self.num_skus, dtype=np.float32)
        self.S = np.zeros(self.num_skus, dtype=np.float32)
        
        for i in range(self.num_skus):
            z = self.z_score[self.abc_class[i]]
            
            # s = mu_d * L + Z * sigma_d * sqrt(L)
            self.s[i] = mean_demand[i] * self.lead_time_weeks + z * std_demand[i] * np.sqrt(self.lead_time_weeks)
            
            # S = s + 2周平均需求（简化EOQ）
            self.S[i] = self.s[i] + 2 * mean_demand[i]
            
            # 下限
            self.s[i] = max(self.s[i], mean_demand[i])
            self.S[i] = max(self.S[i], self.s[i] + mean_demand[i])
        
        self.initialized = True
    
    def _abc_classification(self, mean_demand: np.ndarray):
        """
        ABC分类
        """
        sorted_indices = np.argsort(mean_demand)[::-1]
        cumulative_sales = np.cumsum(mean_demand[sorted_indices])
        total_sales = np.sum(mean_demand)
        cumulative_ratio = cumulative_sales / total_sales
        
        self.abc_class = np.array(['C'] * self.num_skus)
        a_mask = cumulative_ratio < 0.8
        self.abc_class[sorted_indices[a_mask]] = 'A'
        
        b_mask = (cumulative_ratio >= 0.8) & (cumulative_ratio < 0.95)
        self.abc_class[sorted_indices[b_mask]] = 'B'
    
    def predict(self, current_inventory: np.ndarray, on_order: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测动作
        
        参数：
            current_inventory: 当前库存，shape: (num_skus,)
            on_order: 在途订单，shape: (num_skus, lead_time)
        
        返回：
            action: 补货量，shape: (num_skus,)
        """
        if not self.initialized:
            # 未初始化时，使用简单策略：补货到2周需求
            target = 2 * np.mean(self.demand_history[-1]) if self.demand_history else 10
            action = np.maximum(0, target - current_inventory)
            return action
        
        # 计算有效库存
        if on_order is not None:
            effective_inventory = current_inventory + np.sum(on_order, axis=1)
        else:
            effective_inventory = current_inventory
        
        # 应用(s, S)策略
        action = np.zeros(self.num_skus, dtype=np.float32)
        for i in range(self.num_skus):
            if effective_inventory[i] < self.s[i]:
                action[i] = max(0, self.S[i] - effective_inventory[i])
        
        return action


if __name__ == "__main__":
    # 测试代码
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.optimization.supply_chain_env import SupplyChainEnv
    
    print("=" * 60)
    print("自适应Heuristic策略测试")
    print("=" * 60)
    
    # 创建环境
    env = SupplyChainEnv(num_stores=1, num_skus=10)
    obs, info = env.reset()
    
    # 创建策略
    policy = AdaptiveHeuristicPolicy(num_skus=10, lead_time_weeks=1)
    
    # 收集历史数据
    for step in range(8):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        policy.update(env.current_demand)
    
    print(f"\n策略已初始化: {policy.initialized}")
    print(f"平均s: {np.mean(policy.s):.2f}")
    print(f"平均S: {np.mean(policy.S):.2f}")
    
    # 运行Episode
    env.reset()
    total_reward = 0
    for step in range(52):
        # 获取当前状态
        current_inventory = env.store_inventory
        on_order = env.on_order
        
        # 预测动作
        action = policy.predict(current_inventory, on_order)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 更新策略
        policy.update(env.current_demand)
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode完成: {step+1}步")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {total_reward/(step+1):.2f}")
