"""
多门店单SKU周度步进环境（架构重塑版）
==========================================
移除固定订货成本，加入预测跟踪惩罚和动作平滑惩罚
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MultiStoreSingleSkuWeeklyEnv(gym.Env):
    """
    多门店单SKU的周度补货环境（架构重塑版）
    
    状态空间（每个门店3维 + 全局2维 + 时间编码2维）：
    - 各门店库存水平（相对值）
    - 各门店在途库存
    - 各门店上周销量
    - 全局：本周需求预测
    - 全局：当前周次（sin/cos编码）
    
    动作空间：
    - 连续动作：每个门店的补货比例 [0, 1]
    """

    def __init__(
        self,
        num_stores: int = 10,
        sku_id: str = "SKU_001",
        demand_series: Optional[np.ndarray] = None,
        cost_price: float = 22.03,  # 成本价（真实数据中的 current_move_avg_price）
        selling_price: Optional[float] = None,  # 销售价（若未提供则从成本价推算）
        lead_time_weeks: int = 1,
        max_steps: int = 52,  # 一年52周
        holding_cost_rate: float = 0.01,  # 持有成本率（占成本价的比例，例如 1%/周）
        shortage_penalty_rate: float = 2.0,  # 缺货惩罚系数（相对于销售价）
        tracking_penalty_rate: float = 0.5,  # 预测跟踪惩罚权重（降低，避免过于保守）
        smoothness_penalty_rate: float = 5.0,  # 动作平滑惩罚权重
        max_inventory_capacity: float = 200.0,
        gamma: float = 0.95,  # 势能函数折扣因子
        config: Optional[Any] = None,
    ):
        """
        初始化环境
        
        Args:
            num_stores: 门店数量
            sku_id: SKU编号（用于从TFT获取预测）
            demand_series: 周度需求序列（若未提供则使用模拟数据）
            cost_price: 成本价（真实数据中的 current_move_avg_price）
            selling_price: 销售价（若未提供则按毛利率 30% 推算）
            lead_time_weeks: 交货周期（周）
            max_steps: 最大步数（周）
            holding_cost_rate: 持有成本率（占成本价的比例，例如 0.01 = 1%/周）
            shortage_penalty_rate: 缺货惩罚系数（相对于销售价）
            tracking_penalty_rate: 预测跟踪惩罚权重（鼓励库存贴合预测）
            smoothness_penalty_rate: 动作平滑惩罚权重（抑制高频微调）
            max_inventory_capacity: 单个门店最大库存容量
            gamma: 势能函数折扣因子
            config: 环境配置对象
        """
        super().__init__()
        
        # 基础参数
        self.num_stores = num_stores
        self.sku_id = sku_id
        self.lead_time = lead_time_weeks
        self.max_steps = max_steps
        
        # ========= 新增：真实成本价和销售价 =========
        self.cost_price = cost_price  # 成本价
        # 若未提供销售价，则按毛利率 30% 推算
        if selling_price is None:
            self.selling_price = cost_price * 1.3  # 假设毛利率 30%
        else:
            self.selling_price = selling_price
        
        # 持有成本率（占成本价的比例）
        self.holding_cost_rate = holding_cost_rate
        # 实际持有成本 = 库存量 × 成本价 × 持有成本率
        # 单位：元/周
        
        # 缺货惩罚系数（相对于销售价）
        self.shortage_penalty_rate = shortage_penalty_rate
        # 实际缺货惩罚 = 缺货量 × 销售价 × 缺货惩罚系数
        # 单位：元/周
        
        self.tracking_penalty_rate = tracking_penalty_rate
        self.smoothness_penalty_rate = smoothness_penalty_rate
        self.max_inventory_capacity = max_inventory_capacity
        self.gamma = gamma
        
        # 需求数据：若未提供则模拟
        if demand_series is None:
            # 模拟3年的周度需求数据（带季节性）
            self.demand_series = self._generate_simulated_demand(
                weeks=52 * 3,
                base_demand=20.0,
                seasonality_strength=0.3
            )
        else:
            self.demand_series = demand_series
        
        # 状态空间维度
        # 每店3维（库存、在途、上周销量）+ 1维需求预测 + 2维周编码
        state_dim = num_stores * 3 + 1 + 2
        
        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_stores,),
            dtype=np.float32
        )
        
        # 最大补货量（基于需求预测的动态计算）
        self.max_order_coverage_weeks = 4  # 最多补4周的货
        
        # 内部状态
        self.inventory = None           # (num_stores,) 各门店库存
        self.in_transit_pipeline = None # List[(num_stores,)] 在途队列
        self.current_step = 0
        self.last_sales = None          # (num_stores,) 上周实际销量
        self.weekly_sales_history = []  # 历史销量（用于计算滑动平均）
        self.last_potential = 0.0      # 上一个状态的势能（用于奖励塑形）
        self.last_action = np.zeros(num_stores, dtype=np.float32)  # 上周动作（用于平滑惩罚）
        
        # 门店异质性因子（模拟不同门店的需求差异）
        self.store_factors = np.random.uniform(
            0.5, 1.5, size=num_stores
        ).astype(np.float32)
        
        logger.info(f"✅ MultiStoreSingleSkuWeeklyEnv 初始化完成")
        logger.info(f"   - 状态空间维度: {state_dim}")
        logger.info(f"   - 动作空间维度: {num_stores}")
        logger.info(f"   - 跟踪惩罚权重: {tracking_penalty_rate}")
        logger.info(f"   - 平滑惩罚权重: {smoothness_penalty_rate}")
        
    def _generate_simulated_demand(
        self,
        weeks: int = 156,
        base_demand: float = 20.0,
        seasonality_strength: float = 0.3,
        noise_std: float = 3.0
    ) -> np.ndarray:
        """生成模拟的周度需求数据（带季节性和噪声）"""
        t = np.arange(weeks)
        
        # 年度季节性（52周周期）
        seasonal = np.sin(2 * np.pi * t / 52)
        
        # 趋势项（轻微增长）
        trend = 0.01 * t
        
        # 组合
        demand = base_demand * (1 + seasonality_strength * seasonal) + trend
        
        # 添加随机噪声
        noise = np.random.normal(0, noise_std, size=weeks)
        demand = demand + noise
        
        # 确保需求为正
        demand = np.maximum(demand, 1.0)
        
        return demand.astype(np.float32)
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境到初始状态
        
        Returns:
            obs: 初始观测
            info: 额外信息（空字典）
        """
        super().reset(seed=seed)
        
        # 重置库存（每个门店20单位）
        self.inventory = np.ones(self.num_stores, dtype=np.float32) * 20.0
        
        # 重置在途队列
        self.in_transit_pipeline = [
            np.zeros(self.num_stores, dtype=np.float32)
            for _ in range(self.lead_time)
        ]
        
        # 重置当前步数
        self.current_step = 0
        
        # 上周销量：初始化为0
        self.last_sales = np.zeros(
            self.num_stores, dtype=np.float32
        )
        
        # 历史销量：清空
        self.weekly_sales_history = []
        
        # 重置上周动作
        self.last_action = np.zeros(self.num_stores, dtype=np.float32)
        
        # 计算初始状态的势能（用于奖励塑形）
        base_demand = self.demand_series[0]
        self.last_potential = self._potential(self.inventory, base_demand)
        
        logger.debug(f"环境重置: step=0, inventory={np.sum(self.inventory):.1f}, potential={self.last_potential:.3f}")
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一周的补货决策
        
        Args:
            action: 各门店的补货比例 [0, 1]，形状=(num_stores,)
        
        Returns:
            obs: 下一状态
            reward: 本周的累计奖励
            terminated: 是否终止（达到最大步数）
            truncated: 是否截断（本环境中不使用）
            info: 额外信息
        """
        # ========= 1. 解析动作 =========
        # 每周都可以补货（移除频率约束）
        # 将[0, 1]映射为实际补货量
        # 最大补货量 = 4周需求覆盖
        next_week_demand = self.demand_series[
            (self.current_step + 1) % len(self.demand_series)
        ]
        max_order_qty = next_week_demand * self.max_order_coverage_weeks
        max_order_qty = max(max_order_qty, 1.0)  # 防止除零
        
        # 实际补货量（每个门店独立）
        order_qty = action * max_order_qty  # (num_stores,)
        
        # ========= 2. 在途到货（FIFO） ==========
        arrived_this_week = self.in_transit_pipeline.pop(0)
        self.inventory += arrived_this_week
        
        # 强制截断库存（防止爆仓）
        self.inventory = np.clip(
            self.inventory,
            0,
            self.max_inventory_capacity
        )
        
        # ========= 3. 新订单加入管道 ==========
        self.in_transit_pipeline.append(order_qty.copy())
        
        # ========= 4. 生成各门店本周需求 ==========
        # 基础需求（从需求序列读取）
        base_demand = self.demand_series[
            self.current_step % len(self.demand_series)
        ]
        
        # 门店异质性（不同门店需求不同）
        weekly_demand_per_store = base_demand * self.store_factors
        
        # ========= 5. 执行销售 & 计算缺货 ==========
        actual_sales = np.minimum(self.inventory, weekly_demand_per_store)
        stockout = weekly_demand_per_store - actual_sales
        self.inventory -= actual_sales
        self.last_sales = actual_sales.copy()
        
        # 记录历史销量
        self.weekly_sales_history.append(np.sum(actual_sales))
        if len(self.weekly_sales_history) > 12:  # 保留最近12周
            self.weekly_sales_history.pop(0)
        
        # ========= 6. 周度奖励计算 ==========
        
        # 6.1 持有成本（使用真实成本价）
        # 持有成本 = 库存量 × 成本价 × 持有成本率
        holding_cost = np.sum(self.inventory) * self.cost_price * self.holding_cost_rate / self.num_stores
        
        # 6.2 缺货惩罚（使用真实销售价）
        # 缺货惩罚 = 缺货量 × 销售价 × 缺货惩罚系数
        shortage_penalty = np.sum(stockout) * self.selling_price * self.shortage_penalty_rate / self.num_stores
        
        # 6.3 销售收入（使用真实销售价）
        revenue = np.sum(actual_sales) * self.selling_price / self.num_stores
        
        # 6.4 预测跟踪惩罚（核心！鼓励库存贴合下周预测需求）
        next_week_demand_per_store = next_week_demand * self.store_factors
        tracking_error = np.abs(self.inventory - next_week_demand_per_store)
        tracking_penalty = np.sum(tracking_error) * self.tracking_penalty_rate / self.num_stores
        
        # 6.5 动作平滑惩罚（替代固定成本，抑制高频微调）
        action_change = np.abs(action - self.last_action)
        smoothness_penalty = np.sum(action_change) * self.smoothness_penalty_rate / self.num_stores
        
        # 6.6 综合奖励（未归一化）
        reward_raw = revenue - holding_cost - shortage_penalty - tracking_penalty - smoothness_penalty
        
        # 6.7 势能函数奖励塑形（基于文档 4.4.2 节）
        # 计算下一个状态的势能
        potential_next = self._potential(self.inventory, next_week_demand)
        # 计算塑形奖励：F(s_t, s_t+1) = gamma * Phi(s_t+1) - Phi(s_t)
        shaping_reward = self.gamma * potential_next - self.last_potential
        # 将塑形奖励加到原始奖励上
        reward_raw += shaping_reward
        # 更新上一个状态的势能
        self.last_potential = potential_next
        
        # 6.8 奖励归一化（关键！确保Reward在[-10, 10]区间内）
        # 除以 10.0，确保单步Reward绝对值在10以内
        reward = reward_raw / 10.0
        
        # ========= 7. 更新状态 ==========
        self.current_step += 1
        self.last_action = action.copy()  # 保存本周动作（用于下周平滑惩罚）
        
        # 检查是否终止
        terminated = self.current_step >= self.max_steps
        truncated = False  # 本环境中不使用
        
        # ========= 8. 构造info字典 ==========
        # 计算OFR（订单满足率）
        total_demand = np.sum(weekly_demand_per_store)
        total_sales = np.sum(actual_sales)
        ofr = total_sales / (total_demand + 1e-8)
        
        # 计算Action Activity (补货门店比例)
        action_activity = np.mean(order_qty > 0.01)
        
        # 监控指标：action_mean 和 action_activity_strict
        action_mean = np.mean(action)  # 动作均值
        action_activity_strict = np.mean(action > 0.3)  # 严格阈值
        
        info = {
            "total_sales": float(np.sum(actual_sales)),
            "total_stockout": float(np.sum(stockout)),
            "total_inventory": float(np.sum(self.inventory)),
            "total_demand": float(total_demand),
            "order_qty": order_qty.copy(),
            "actual_reward": float(reward),
            "ofr": float(ofr),
            "action_activity": float(action_activity),
            "action_mean": float(action_mean),
            "action_activity_strict": float(action_activity_strict),
            "num_stores_ordering": int(np.sum(order_qty > 0.01)),
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _potential(self, inventory: np.ndarray, demand: float) -> float:
        """
        势能函数：评估当前状态的"好坏"（越高越好）
        
        使用"区间惩罚"设计（而非单一高目标水位）：
        - 库存在区间 [0.5周需求, 1.5周需求] 内 -> 势能 = 0（最佳）
        - 库存低于0.5周需求 -> 势能 < 0（缺货风险）
        - 库存高于1.5周需求 -> 势能 < 0（过度库存）
        
        Args:
            inventory: 各门店库存，形状=(num_stores,)
            demand: 本周需求预测
        
        Returns:
            potential: 势能值（越高表示状态越好）
        """
        # 定义健康区间：[0.5周需求, 1.5周需求]
        low_bound = demand * 0.5 * self.store_factors
        high_bound = demand * 1.5 * self.store_factors
        
        # 只有超出区间才产生负势能（惩罚）
        under_stock = np.maximum(0, low_bound - inventory)
        over_stock = np.maximum(0, inventory - high_bound)
        
        # 势能 = 负偏差总和（越接近0越好）
        # 缺货惩罚权重 = 2.0，过度库存惩罚权重 = 1.0
        potential = -np.sum(under_stock * 2.0 + over_stock * 1.0)
        
        return potential
    
    def _get_obs(self) -> np.ndarray:
        """
        构造观测向量
        
        状态空间组成：
        - [0:num_stores]：各门店库存水平（相对值）
        - [num_stores:2*num_stores]：各门店在途库存（相对值）
        - [2*num_stores:3*num_stores]：各门店上周销量（相对值）
        - [3*num_stores]：本周需求预测（相对值）
        - [3*num_stores+1]：周次sin编码
        - [3*num_stores+2]：周次cos编码
        
        Returns:
            obs: 观测向量，形状=(state_dim,)
        """
        # 归一化参数
        max_inv = self.max_inventory_capacity
        max_demand = np.max(self.demand_series) * 2.0
        
        # 1. 各门店库存水平
        inv_normalized = self.inventory / max_inv
        
        # 2. 各门店在途库存
        in_transit_current = self.in_transit_pipeline[0] if self.in_transit_pipeline else np.zeros(self.num_stores)
        in_transit_normalized = in_transit_current / max_inv
        
        # 3. 各门店上周销量
        sales_normalized = self.last_sales / (max_demand + 1e-8)
        
        # 4. 本周需求预测
        base_demand = self.demand_series[
            self.current_step % len(self.demand_series)
        ]
        demand_normalized = base_demand / max_demand
        
        # 5. 周次编码（sin/cos）
        week_angle = 2 * np.pi * (self.current_step % 52) / 52
        week_encoding = np.array([np.sin(week_angle), np.cos(week_angle)])
        
        # 拼接
        obs = np.concatenate([
            inv_normalized,
            in_transit_normalized,
            sales_normalized,
            np.array([demand_normalized]),
            week_encoding
        ]).astype(np.float32)
        
        return obs
