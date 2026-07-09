#!/usr/bin/env python3
"""
多门店单SKU周度步进环境 V3（真实数据版）
==========================================

修改：
1. ✅ 从真实数据文件读取需求序列
2. ✅ 从真实数据读取初始库存和成本价
3. ✅ 移除所有模拟数据生成代码
4. ⚠️ 暂时不包括TFT预测（后续集成）

观测空间设计：
- 每店3维：[inventory, pending_arrivals, stockout]
- 全局1维：需求预测（从历史数据计算）
- 时间2维：周sin/cos编码
- 总成本价1维：所有SKU的平均成本价
=> 总维度 = num_stores * 3 + 1 + 2 + 1 = num_stores * 3 + 4

动作空间：
- 每个门店一个连续补货比例 [0, 1]
- 实际补货量 = action * max_order_qty
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os

class MultiStoreSingleSkuWeeklyEnvReal(gym.Env):
    """
    多门店单SKU周度步进环境（真实数据版）
    
    特点：
    1. 周度步进：step() = 一周（7天），解决信用分配问题
    2. 真实数据：从文件读取需求、库存、成本价
    3. 观测空间：当前库存 + 在途到货 + 需求预测 + 时间编码
    4. 动作空间：连续补货比例 [0, 1]（每个门店）
    5. 奖励：平衡缺货损失、持有成本、补货成本
    """
    
    def __init__(
        self,
        data_dir: str = "data/real_data_v2",
        sku_idx: int = 0,  # 使用第几个SKU
        lead_time_weeks: int = 1,
        max_steps: int = 57,  # 真实数据有57周
        tracking_penalty_rate: float = 0.0,  # 暂时禁用
        smoothness_penalty_rate: float = 0.0,  # 暂时禁用
        max_inventory_capacity: float = 1000.0,
        gamma: float = 0.95,
    ):
        """
        初始化环境
        
        Args:
            data_dir: 真实数据目录
            sku_idx: 使用第几个SKU（0到num_skus-1）
            lead_time_weeks: 交货周期（周）
            max_steps: 最大步数（周）
            tracking_penalty_rate: 预测跟踪惩罚权重（暂时禁用）
            smoothness_penalty_rate: 动作平滑惩罚权重（暂时禁用）
            max_inventory_capacity: 单个门店最大库存容量
            gamma: 势能函数折扣因子
        """
        super().__init__()
        
        # 1. 加载真实数据
        self.data_dir = data_dir
        self.sku_idx = sku_idx
        
        # 加载元数据
        with open(f"{data_dir}/metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 加载NumPy数组
        self.demand_series = np.load(f"{data_dir}/demand_series.npy")  # (weeks, num_stores, num_skus)
        self.initial_inventory = np.load(f"{data_dir}/initial_inventory.npy")  # (num_stores, num_skus)
        self.cost_price = np.load(f"{data_dir}/cost_price.npy")  # (num_skus,)
        self.selling_price = np.load(f"{data_dir}/selling_price.npy")  # (num_skus,)
        
        # 提取当前SKU的数据
        self.num_stores = self.metadata['num_stores']
        self.num_weeks = self.metadata['num_weeks']
        self.max_steps = min(max_steps, self.num_weeks - 1)
        
        # 当前SKU的需求序列 (weeks, num_stores)
        self.sku_demand = self.demand_series[:, :, sku_idx]
        
        # 当前SKU的成本价和销售价
        self.cost_price_sku = self.cost_price[sku_idx]
        self.selling_price_sku = self.selling_price[sku_idx]
        
        # 2. 环境参数
        self.lead_time = lead_time_weeks
        self.tracking_penalty_rate = tracking_penalty_rate
        self.smoothness_penalty_rate = smoothness_penalty_rate
        self.max_inventory_capacity = max_inventory_capacity
        self.gamma = gamma
        
        # 3. 定义观测空间和动作空间
        # 观测空间：每店3维 + 1维需求预测 + 2维周sin/cos + 1维成本价
        state_dim = self.num_stores * 3 + 1 + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 动作空间：每个门店一个连续补货比例 [0, 1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_stores,), dtype=np.float32
        )
        
        # 4. 初始化状态
        self.reset()
        
        print(f"✅ 真实数据环境创建成功")
        print(f"   - 门店数: {self.num_stores}")
        print(f"   - SKU索引: {sku_idx}")
        print(f"   - 总成本价: {self.cost_price_sku:.2f}")
        print(f"   - 总销售价: {self.selling_price_sku:.2f}")
        print(f"   - 最大步数: {self.max_steps} 周")
        print(f"   - 观测空间维度: {state_dim}")
    
    def reset(self, seed=None):
        """重置环境到初始状态"""
        # 1. 重置时间
        self.current_step = 0
        
        # 2. 重置库存（从真实数据读取）
        self.inventory = self.initial_inventory[:, self.sku_idx].copy().astype(np.float32)
        
        # 3. 重置在途到货
        self.pending_orders = np.zeros((self.num_stores, self.lead_time), dtype=np.float32)
        
        # 4. 重置上一周动作
        self.last_action = np.zeros(self.num_stores, dtype=np.float32)
        
        # 5. 初始化势能函数
        next_week_demand = self.sku_demand[(self.current_step + 1) % self.num_weeks, 0]
        self.last_potential = self._potential(self.inventory, next_week_demand)
        
        # 6. 返回初始观测
        return self._get_observation(), {}
    
    def _potential(self, inventory, demand):
        """势能函数：库存越接近需求，势能越高（负值越小）"""
        deviation = np.abs(inventory - demand)
        return -np.sum(deviation)
    
    def _get_observation(self):
        """构造当前观测"""
        # 1. 每店3维：[inventory, pending_arrivals, stockout]
        obs = []
        for i in range(self.num_stores):
            pending = np.sum(self.pending_orders[i])  # 在途到货总和
            stockout = max(0, -self.inventory[i])  # 缺货量（如果库存为负）
            self.inventory[i] = max(0, self.inventory[i])  # 库存不能为负
            
            obs.extend([self.inventory[i], pending, stockout])
        
        # 2. 全局1维：需求预测（从历史数据计算）
        if self.current_step < self.num_weeks - 1:
            next_week_demand = np.mean(self.sku_demand[self.current_step + 1, :])
        else:
            next_week_demand = np.mean(self.sku_demand[-1, :])
        obs.append(next_week_demand)
        
        # 3. 时间2维：周sin/cos编码
        week_of_year = self.current_step % 52
        sin_week = np.sin(2 * np.pi * week_of_year / 52)
        cos_week = np.cos(2 * np.pi * week_of_year / 52)
        obs.extend([sin_week, cos_week])
        
        # 4. 总成本价1维
        obs.append(self.cost_price_sku)
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        """
        执行一周的供应链动态
        
        Args:
            action: 补货比例，形状 (num_stores,)，范围 [0, 1]
        
        Returns:
            obs: 下一状态观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 1. 解析动作（将[0,1]映射为实际补货量）
        next_week_demand = self.sku_demand[(self.current_step + 1) % self.num_weeks, :]
        max_order_qty = next_week_demand * 2.0  # 最多补货2倍需求
        order_qty = action * max_order_qty
        
        # 2. 执行补货（更新在途到货）
        for i in range(self.num_stores):
            if order_qty[i] > 0:
                self.pending_orders[i, 0] += order_qty[i]
        
        # 3. 在途到货（交货周期）
        arrived_orders = self.pending_orders[:, -1].copy() if self.lead_time > 0 else np.zeros(self.num_stores)
        self.inventory += arrived_orders
        
        # 4. 滚动在途到货（向"未来"移动）
        if self.lead_time > 0:
            self.pending_orders = np.roll(self.pending_orders, shift=1, axis=1)
            self.pending_orders[:, 0] = 0.0
        
        # 5. 执行销售（需求实现）
        current_demand = self.sku_demand[self.current_step, :]
        actual_sales = np.minimum(self.inventory, current_demand)
        stockout = current_demand - actual_sales
        self.inventory -= actual_sales
        
        # 6. 计算奖励（使用真实成本价和销售价）
        # 6.1 持有成本（使用真实成本价）
        holding_cost = np.sum(self.inventory) * self.cost_price_sku * 0.01
        
        # 6.2 缺货惩罚
        shortage_penalty = np.sum(stockout) * self.selling_price_sku * 0.5
        
        # 6.3 销售收入
        revenue = np.sum(actual_sales) * self.selling_price_sku
        
        # 6.4 综合奖励
        reward = revenue - holding_cost - shortage_penalty
        
        # 6.5 归一化奖励（除以门店数）
        reward = reward / self.num_stores
        
        # 7. 更新状态
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # 我们不使用截断，只用terminated
        
        # 8. 构造返回信息
        info = {
            'total_demand': np.sum(current_demand),
            'total_sales': np.sum(actual_sales),
            'total_inventory': np.sum(self.inventory),
            'num_stores_ordering': np.sum(order_qty > 0.01),
            'mean_reward': reward,
        }
        
        # 9. 返回结果（符合gymnasium要求）
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """可视化当前状态"""
        print(f"Week {self.current_step}: Inventory={self.inventory[:5]}, Demand={self.sku_demand[self.current_step, :5]}")
    
    def close(self):
        pass


if __name__ == "__main__":
    # 测试环境
    env = MultiStoreSingleSkuWeeklyEnvReal(
        data_dir="data/real_data_v2",
        sku_idx=0,
        lead_time_weeks=1,
        max_steps=57,
    )
    
    # 测试reset
    obs = env.reset()
    print(f"\n✅ Reset成功！观测形状: {obs.shape}")
    
    # 测试step
    action = np.ones(env.num_stores) * 0.5  # 每个门店补货50%
    obs, reward, done, _, info = env.step(action)
    print(f"\n✅ Step成功！")
    print(f"   - 奖励: {reward:.2f}")
    print(f"   - 观测形状: {obs.shape}")
    print(f"   - 信息: {info}")
    
    print("\n✅ 环境测试完成！")
