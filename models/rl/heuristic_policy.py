"""
启发式策略 (Heuristic Policy) 模块

实现经典的 (s, S) 策略（Base-Stock策略），用于作为Baseline对比。

核心逻辑：
1. ABC分类：按历史销量将SKU分为A类(爆款)、B类(常规)、C类(长尾)
2. 统计计算：使用运筹学公式计算每个SKU的s（再订货点）和S（目标库存）
3. (s, S)策略：当库存低于s时，补货到S

参考：
- Zipkin, P. H. (2000). Foundations of Inventory Management.
- Silver, E. A., Pyke, D. F., & Peterson, R. (1998). Inventory Management and Production Planning and Scheduling.
"""

import numpy as np
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class HeuristicPolicy:
    """
    启发式策略类（(s, S)策略）
    
    参数：
        num_skus: SKU数量
        lead_time_weeks: 提前期（周）
        history_window: 历史数据窗口（用于计算统计特征）
        service_level: 服务水平（A类=0.99, B类=0.95, C类=0.90）
    """
    
    def __init__(
        self,
        num_skus: int,
        lead_time_weeks: int = 1,
        history_window: int = 8,
        service_level: Dict[str, float] = None
    ):
        self.num_skus = num_skus
        self.lead_time_weeks = lead_time_weeks
        self.history_window = history_window
        
        # 服务水平（对应Z分数）
        # A类：99% → Z=2.33
        # B类：95% → Z=1.65
        # C类：90% → Z=1.28
        if service_level is None:
            service_level = {'A': 0.99, 'B': 0.95, 'C': 0.90}
        self.service_level = service_level
        self.z_score = {
            'A': 2.33,
            'B': 1.65,
            'C': 1.28
        }
        
        # 历史需求数据（用于计算统计特征）
        self.demand_history = []  # List[np.ndarray]，每个元素是一周的需求
        
        # ABC分类结果
        self.abc_class = None  # np.ndarray，每个SKU的分类（'A', 'B', 'C'）
        
        # (s, S)参数
        self.s = None  # 再订货点（Reorder Point）
        self.S = None  # 目标库存水平（Order-up-to Level）
        
        # 是否已初始化
        self.initialized = False
    
    def initialize(self, env=None):
        """
        占位方法：兼容标准 RL 训练/评估循环的初始化调用。
        Heuristic 策略不需要神经网络权重初始化，因此直接 pass。
        
        参数：
            env: 环境对象（可选）
        """
        pass
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        兼容 SB3 评估接口的 predict 方法
        
        参数：
            observation: 当前观察（未使用，因为Heuristic策略只使用库存水平）
            state: 状态（未使用）
            episode_start: episode开始标志（未使用）
            deterministic: 是否确定性预测（Heuristic策略总是确定性的）
        
        返回：
            action: 补货量，shape: (num_skus,)
            state: 状态（未使用，返回None）
        """
        # 获取当前库存
        if hasattr(self, 'env') and self.env is not None:
            if hasattr(self.env, 'store_inventory'):
                current_inventory = self.env.store_inventory
            else:
                current_inventory = np.zeros(self.num_skus, dtype=np.float32)
            
            # 获取在途订单
            on_order = None
            if hasattr(self.env, 'pipeline'):
                on_order = self.env.pipeline
            elif hasattr(self.env, 'on_order'):
                on_order = self.env.on_order
            
            # 使用Heuristic策略预测动作
            action = self.compute_action(current_inventory, on_order)
        else:
            # 如果没有环境，返回零动作
            action = np.zeros(self.num_skus, dtype=np.float32)
        
        return action, None
    
    def reset(self, demand_history: Optional[np.ndarray] = None):
        """
        重置策略，计算ABC分类和(s, S)参数
        
        参数：
            demand_history: 历史需求数据，shape: (weeks, num_skus)
                           如果为None，则使用self.demand_history
        """
        if demand_history is not None:
            self.demand_history = list(demand_history)
        
        if len(self.demand_history) < self.history_window:
            logger.warning(f"[Heuristic] 历史数据不足（{len(self.demand_history)}周 < {self.history_window}周），使用默认值")
            self._init_default_params()
            return
        
        # 1. 计算历史需求的统计特征
        demand_array = np.array(self.demand_history)  # shape: (weeks, num_skus)
        mean_demand = np.mean(demand_array, axis=0)  # shape: (num_skus,)
        std_demand = np.std(demand_array, axis=0)  # shape: (num_skus,)
        
        # 防止标准差为0
        std_demand = np.maximum(std_demand, 0.1)
        
        # 2. ABC分类（基于平均需求）
        self._abc_classification(mean_demand)
        
        # 3. 计算每个SKU的s和S
        self._calculate_s_S(mean_demand, std_demand)
        
        self.initialized = True
        logger.info(f"[Heuristic] 策略初始化完成")
        logger.info(f"  - A类SKU: {np.sum(self.abc_class == 'A')}")
        logger.info(f"  - B类SKU: {np.sum(self.abc_class == 'B')}")
        logger.info(f"  - C类SKU: {np.sum(self.abc_class == 'C')}")
        logger.info(f"  - 平均s: {np.mean(self.s):.2f}")
        logger.info(f"  - 平均S: {np.mean(self.S):.2f}")
    
    def _abc_classification(self, mean_demand: np.ndarray):
        """
        ABC分类（基于历史销量）
        
        A类（爆款）：前20%的SKU，贡献80%的销量
        B类（常规）：中间30%的SKU，贡献15%的销量
        C类（长尾）：后50%的SKU，贡献5%的销量
        
        参数：
            mean_demand: 平均需求，shape: (num_skus,)
        """
        # 按平均需求排序
        sorted_indices = np.argsort(mean_demand)[::-1]  # 从大到小
        cumulative_sales = np.cumsum(mean_demand[sorted_indices])
        total_sales = np.sum(mean_demand)
        cumulative_ratio = cumulative_sales / total_sales
        
        # 分类
        self.abc_class = np.array(['C'] * self.num_skus)
        
        # A类：累计占比 < 80%
        a_mask = cumulative_ratio < 0.8
        self.abc_class[sorted_indices[a_mask]] = 'A'
        
        # B类：累计占比 80%~95%
        b_mask = (cumulative_ratio >= 0.8) & (cumulative_ratio < 0.95)
        self.abc_class[sorted_indices[b_mask]] = 'B'
        
        # C类：累计占比 >= 95%（默认）
        
        logger.info(f"[Heuristic] ABC分类完成")
        logger.info(f"  - A类（爆款）: {np.sum(self.abc_class == 'A')} SKU")
        logger.info(f"  - B类（常规）: {np.sum(self.abc_class == 'B')} SKU")
        logger.info(f"  - C类（长尾）: {np.sum(self.abc_class == 'C')} SKU")
    
    def _calculate_s_S(self, mean_demand: np.ndarray, std_demand: np.ndarray):
        """
        计算每个SKU的s和S
        
        公式：
        s = mu_d * L + Z * sigma_d * sqrt(L)
        S = s + EOQ (经济订货批量) 或 S = s + 2周平均需求
        
        其中：
        - mu_d: 周均需求
        - sigma_d: 周需求标准差
        - L: 提前期（周）
        - Z: 服务水平系数（A类=2.33, B类=1.65, C类=1.28）
        
        参数：
            mean_demand: 平均需求，shape: (num_skus,)
            std_demand: 需求标准差，shape: (num_skus,)
        """
        # 初始化
        self.s = np.zeros(self.num_skus, dtype=np.float32)
        self.S = np.zeros(self.num_skus, dtype=np.float32)
        
        # 对每个SKU计算
        for i in range(self.num_skus):
            # 获取该SKU的Z分数
            z = self.z_score[self.abc_class[i]]
            
            # 计算s（再订货点）
            # s = mu_d * L + Z * sigma_d * sqrt(L)
            self.s[i] = mean_demand[i] * self.lead_time_weeks + z * std_demand[i] * np.sqrt(self.lead_time_weeks)
            
            # 计算S（目标库存）
            # 简化：S = s + 2周平均需求
            self.S[i] = self.s[i] + 2 * mean_demand[i]
            
            # 下限：至少满足1周需求
            self.s[i] = max(self.s[i], mean_demand[i])
            self.S[i] = max(self.S[i], self.s[i] + mean_demand[i])
    
    def compute_action(self, current_inventory: np.ndarray, on_order: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测动作（补货决策）
        
        核心逻辑：(s, S)策略
        - 如果当前库存 + 在途订单 < s，则补货到S
        - 否则，不补货
        
        参数：
            current_inventory: 当前库存，shape: (num_skus,)
            on_order: 在途订单，shape: (lead_time, num_skus) 或 None
        
        返回：
            action: 补货量，shape: (num_skus,)
        """
        if not self.initialized:
            logger.warning("[Heuristic] 策略未初始化，使用默认动作（不补货）")
            return np.zeros(self.num_skus, dtype=np.float32)
        
        # 计算有效库存（当前库存 + 在途订单）
        if on_order is not None:
            # on_order shape: (num_skus, lead_time)
            effective_inventory = current_inventory + np.sum(on_order, axis=1)
        else:
            effective_inventory = current_inventory
        
        # 初始化动作
        action = np.zeros(self.num_skus, dtype=np.float32)
        
        # 对每个SKU应用(s, S)策略
        for i in range(self.num_skus):
            if effective_inventory[i] < self.s[i]:
                # 有效库存低于再订货点，补货到目标库存
                action[i] = max(0, self.S[i] - effective_inventory[i])
            # 否则，不补货
        
        return action
    
    def update(self, demand: np.ndarray):
        """
        更新历史需求数据
        
        参数：
            demand: 本周需求，shape: (num_skus,)
        """
        self.demand_history.append(demand.copy())
        
        # 只保留最近history_window周的数据
        if len(self.demand_history) > self.history_window:
            self.demand_history = self.demand_history[-self.history_window:]
    
    def _init_default_params(self):
        """
        使用默认参数初始化（当历史数据不足时）
        """
        # 默认：每个SKU的s=10, S=20
        self.s = np.ones(self.num_skus, dtype=np.float32) * 10.0
        self.S = np.ones(self.num_skus, dtype=np.float32) * 20.0
        
        # 默认：所有SKU都是B类
        self.abc_class = np.array(['B'] * self.num_skus)
        
        self.initialized = True
        logger.info(f"[Heuristic] 使用默认参数初始化（s=10, S=20）")


class HeuristicBaselineWrapper:
    """
    Heuristic Baseline包装器
    
    将HeuristicPolicy包装为与SB3/自定义PPO兼容的接口，
    以便在相同环境下运行和评估。
    """
    
    def __init__(
        self,
        env,
        lead_time_weeks: int = 1,
        history_window: int = 8
    ):
        """
        初始化Heuristic Baseline
        
        参数：
            env: 环境对象（SupplyChainEnv）
            lead_time_weeks: 提前期（周）
            history_window: 历史数据窗口
        """
        self.env = env
        self.num_skus = env.num_skus
        
        # 创建HeuristicPolicy
        self.policy = HeuristicPolicy(
            num_skus=self.num_skus,
            lead_time_weeks=lead_time_weeks,
            history_window=history_window
        )
        
        # 收集历史数据，初始化策略
        self._init_policy()
    
    def _init_policy(self):
        """
        初始化策略（收集历史数据）
        """
        logger.info("[HeuristicWrapper] 初始化Heuristic策略...")
        
        # 运行几个Episode，收集历史需求数据
        obs, info = self.env.reset()
        
        for episode in range(3):  # 收集3个Episode的历史数据
            for step in range(self.policy.history_window):
                # 随机动作
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 更新需求历史
                if hasattr(self.env, 'current_demand'):
                    self.policy.update(self.env.current_demand)
                
                if terminated or truncated:
                    obs, info = self.env.reset()
                    break
        
        # 重置策略
        self.policy.reset()
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        预测动作
        
        参数：
            obs: 当前观察（未使用，因为Heuristic策略只使用库存水平）
            deterministic: 是否确定性预测（Heuristic策略总是确定性的）
        
        返回：
            action: 补货量，shape: (num_skus,)
        """
        # 获取当前库存
        if hasattr(self.env, 'store_inventory'):
            current_inventory = self.env.store_inventory
        else:
            logger.warning("[HeuristicWrapper] 无法获取当前库存，使用默认值")
            current_inventory = np.zeros(self.num_skus, dtype=np.float32)
        
        # 获取在途订单（如果存在）
        on_order = None
        if hasattr(self.env, 'pipeline'):
            # pipeline shape: (lead_time, num_skus)
            on_order = self.env.pipeline
        elif hasattr(self.env, 'on_order'):
            on_order = self.env.on_order
        
        # 使用Heuristic策略预测动作（考虑在途订单）
        action = self.policy.compute_action(current_inventory, on_order)
        
        # 更新需求历史（用于下一个step）
        if hasattr(self.env, 'current_demand'):
            self.policy.update(self.env.current_demand)
        
        return action
    
    def reset(self):
        """重置策略和环境"""
        obs, info = self.env.reset()
        self.policy.reset()
        return obs, info


if __name__ == "__main__":
    # 测试代码
    import sys
    import os
    
    # 添加项目根目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.optimization.supply_chain_env import SupplyChainEnv
    
    print("=" * 60)
    print("Heuristic Policy 测试")
    print("=" * 60)
    
    # 创建环境
    env = SupplyChainEnv(num_stores=1, num_skus=10)
    print(f"环境创建成功: {env.num_skus} SKU")
    
    # 创建Heuristic Baseline
    heuristic = HeuristicBaselineWrapper(env, lead_time_weeks=1, history_window=8)
    print("Heuristic Baseline创建成功")
    
    # 测试预测
    obs, info = env.reset()
    action = heuristic.predict(obs, deterministic=True)
    print(f"\n预测动作: {action}")
    print(f"  - 动作形状: {action.shape}")
    print(f"  - 有补货动作的SKU数: {np.sum(action > 0)}")
    
    # 运行一个Episode
    print("\n运行一个Episode...")
    obs, info = env.reset()
    total_reward = 0
    for step in range(52):  # 52周
        action = heuristic.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Episode完成: {step+1}步, 总奖励: {total_reward:.2f}")
    print(f"  - 平均奖励: {total_reward/(step+1):.2f}")
