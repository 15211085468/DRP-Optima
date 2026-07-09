"""
全局多门店供应链环境（Global Multi-Store Supply Chain Environment）— V8 终极无坑版
用于支持82门店大一统PPO模型训练

核心改进（V8 终极无坑版）：
1. 前置预警惩罚：在动作执行前计算超出容量/资金的幅度，提前给予梯度
2. 纯经济惩罚驱动，完全移除单SKU硬截断
3. 精简奖励函数：6项核心信号，避免多目标冲突
4. 修复梯度黑洞：使用 np.round 替代 astype(int)
5. 完善的监控与熔断：自定义 Callback 每 60 万步校验关键业务指标
"""

import sys
import os
from pathlib import Path
import math

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
import gymnasium as gym
from gymnasium import spaces

from config import EnvConfig
from utils.logger import get_logger
from data.global_data_loader import GlobalDataLoader
from stable_baselines3.common.callbacks import BaseCallback

logger = get_logger(__name__)


class GlobalSupplyChainEnv(gym.Env):
    """
    全局多门店供应链环境 (V8 终极无坑版)
    
    功能：
    1. 支持82门店统一训练
    2. reset()时随机采样门店
    3. State包含门店画像 + 全局容量/资金特征
    4. 使用GlobalDataLoader极速查询需求
    5. 奖励函数：前置预警 + Sigmoid满足率 + 增量资金成本 + 动态0补货奖励
    6. 全局业务硬约束（容量 + 资金）
    """
    
    def __init__(
        self,
        data_loader: GlobalDataLoader,
        env_config: EnvConfig,
        num_skus: int = 94,
        num_stores: int = 82,
        max_weeks: int = 104,
        curriculum_stage: int = 0,  # 0=前期, 1=中期, 2=后期
    ):
        """
        初始化全局多门店环境 (V8 终极无坑版)
        
        Args:
            data_loader: 全局数据加载器
            env_config: 环境配置
            num_skus: SKU数量
            num_stores: 门店数量
            max_weeks: 最大周数
            curriculum_stage: 课程学习阶段（0=前期, 1=中期, 2=后期）
        """
        logger.info("=" * 80)
        logger.info("GlobalSupplyChainEnv 初始化 (V8 终极无坑版)")
        logger.info("=" * 80)
        
        self.data_loader = data_loader
        self.env_config = env_config
        self.num_stores = num_stores
        self.max_weeks = max_weeks
        self.curriculum_stage = curriculum_stage
        
        # 获取有效的门店和SKU列表
        self.valid_stores = data_loader.valid_stores
        self.valid_skus = data_loader.valid_skus
        self.num_skus = len(self.valid_skus)  # 修复：使用实际valid_skus数量，而非参数默认值
        
        # 当前门店（reset()时随机采样）
        self.current_store = None
        self.current_week = 0
        
        # 🌟 V8新增：加载SKU单价（用于资金占用计算）
        self.sku_unit_prices = self._load_sku_prices()
        
        # 🌟 V8新增：全局容量和资金特征维度
        # 观测空间：num_skus个SKU库存 + num_skus个SKU在途 + 2维门店画像 + 2维全局特征
        self.global_features_dim = 2  # capacity_usage_ratio + budget_usage_ratio
        state_dim = self.num_skus * 2 + 2 + self.global_features_dim  # 修复：使用self.num_skus
        
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # 动作空间：valid_skus个SKU的补货量（连续值0-1，表示补货比例）
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_skus,),  # 修复：使用self.num_skus而非参数num_skus
            dtype=np.float32
        )
        
        logger.info(f"\n✅ GlobalSupplyChainEnv 初始化完成 (V8 终极无坑版)")
        logger.info(f"  - 门店数: {len(self.valid_stores)}")
        logger.info(f"  - SKU数: {len(self.valid_skus)}")
        logger.info(f"  - 状态空间维度: {state_dim} (含{self.global_features_dim}维全局特征)")
        logger.info(f"  - 动作空间维度: {self.num_skus}")  # 修复：使用self.num_skus而非参数num_skus
        logger.info(f"  - 课程学习阶段: {curriculum_stage}")
        logger.info(f"  - 最大补货量: {env_config.base_reorder_unit * env_config.max_order_quantity_ratio} 件")
        logger.info(f"  - 最小起订量: {env_config.min_reorder_qty} 件")
        
    def _load_sku_prices(self) -> np.ndarray:
        """
        加载SKU单价（用于资金占用计算）
        
        Returns:
            SKU单价数组 (num_skus,)，如果无法获取则返回全1数组
        """
        try:
            # 尝试从数据加载器获取SKU价格
            prices = self.data_loader.get_sku_prices(self.valid_skus)
            # 🌟 V8新增：脏数据防御，防止价格为0或负数
            safe_prices = np.maximum(prices, 1e-4)
            logger.info(f"  - SKU单价已加载（均值: {np.mean(safe_prices):.2f}）")
            return safe_prices
        except Exception as e:
            logger.warning(f"  - 无法加载SKU单价: {e}，使用默认值1.0")
            return np.ones(self.num_skus)  # 默认单价为1元
        
    def reset(self, seed=None, options=None):
        """
        重置环境，随机采样门店
        
        核心：实现"随机抽店"，让模型学会适应不同门店
        """
        # 1. 根据课程学习阶段选择门店采样策略
        if self.curriculum_stage == 0:
            # 前期：只采样需求稳定的"中等门店"
            store_weights = self._get_medium_store_weights()
        elif self.curriculum_stage == 1:
            # 中期：均匀采样所有门店
            store_weights = None  # 均匀采样
        else:
            # 后期：增加"极端门店"的采样权重
            store_weights = self._get_extreme_store_weights()
        
        # 2. 随机采样门店
        self.current_store = np.random.choice(self.valid_stores, p=store_weights)
        self.current_week = 0
        
        # 3. 获取初始库存和初始在途
        try:
            initial_stock, transit_stock = self.data_loader.get_initial_state(
                self.current_store, self.current_week
            )
            self.current_stock = initial_stock.astype(np.float32)
            self.transit_stock = transit_stock.astype(np.float32)
            logger.info(f"  - 初始库存已加载（门店: {self.current_store}）")
        except Exception as e:
            logger.warning(f"  - 无法加载初始库存: {e}，使用随机初始化的库存")
            #  fallback：随机初始化
            self.current_stock = np.random.uniform(10, 50, self.num_skus).astype(np.float32)
            self.transit_stock = np.zeros(self.num_skus, dtype=np.float32)
        
        # 4. 获取门店画像
        store_profile = self.data_loader.get_store_profile(self.current_store)
        store_scale = store_profile['scale']
        store_type = store_profile['type']
        
        # 5. 计算初始全局特征
        self.capacity_usage_ratio, self.budget_usage_ratio = self._calculate_global_features(self.current_stock)
        
        # 6. 构建初始State向量
        self.state = np.concatenate([
            self.current_stock,
            self.transit_stock,
            [store_scale, store_type],
            [self.capacity_usage_ratio, self.budget_usage_ratio]
        ]).astype(np.float32)
        
        # 7. 重置统计信息
        self.total_sales = 0
        self.total_stockout = 0
        self.total_holding_cost = 0.0
        self.current_step_added_value = 0.0
        
        return self.state, {}
    
    def _calculate_global_features(self, stock_levels):
        """
        计算全局特征（容量使用率、资金使用率）
        
        用于观测空间，让模型感知全局约束
        """
        # 计算总库存单位数和总库存价值
        total_units = np.sum(np.maximum(stock_levels, 0))
        total_value = np.sum(np.maximum(stock_levels, 0) * self.sku_unit_prices)
        
        # 计算使用率 (0~1) 并挂载到 self，供 Reward 使用
        capacity_usage_ratio = min(1.0, total_units / max(self.env_config.max_store_capacity_units, 1))
        budget_usage_ratio = min(1.0, total_value / max(self.env_config.max_store_budget, 1))
        
        return capacity_usage_ratio, budget_usage_ratio
    
    def step(self, action):
        """
        V8 执行一步（前置预警 + 消除梯度黑洞）
        """
        self.current_week += 1
        
        # 1. 解析动作（比例 → 实际补货量）
        max_reorder = self.env_config.base_reorder_unit * self.env_config.max_order_quantity_ratio
        proposed_qty_float = np.clip(action * max_reorder, 0, max_reorder)
        
        # 2. 🌟 V8核心：使用 np.round 消除低值区梯度黑洞
        proposed_qty = np.round(proposed_qty_float).astype(int)
        
        # 3. 🌟 V8核心：前置预警（检查若执行该补货量会否超限）
        projected_units = np.sum(np.maximum(self.current_stock, 0) + proposed_qty)
        projected_value = np.sum((np.maximum(self.current_stock, 0) + proposed_qty) * np.maximum(self.sku_unit_prices, 1e-4))
        
        unit_overflow = max(0.0, projected_units - self.env_config.max_store_capacity_units)
        budget_overflow = max(0.0, projected_value - self.env_config.max_store_budget)
        
        # 4. 执行库存更新（无硬截断，纯经济惩罚）
        self.current_stock += proposed_qty
        
        # 5. 获取当周的真实需求（极速字典查询）
        current_demand = self._get_weekly_demand(
            self.current_store, 
            self.current_week
        )
        
        # 6. 执行销售（计算满足的销售和缺货）
        sales, stockout = self._execute_sales(current_demand)
        
        # 7. 更新库存状态
        self._update_inventory(proposed_qty)
        
        # 8. 🌟 V8.2 重构：计算Reward（传入前置预警值）
        # 返回标量奖励（已内部加权）
        reward = self._calculate_normalized_reward(sales, stockout, proposed_qty, unit_overflow, budget_overflow)
        
        # 获取奖励项字典（用于分析和日志）
        reward_components = self.get_reward_components(sales, stockout, proposed_qty, unit_overflow, budget_overflow)
        
        # 🔧 V8.2 修改：移除固定缩放 /10000.0，改用自适应归一化（见阶段 3）
        # reward = reward / 10000.0  # 已移除
        
        # 奖励裁剪（保留作为安全阀）
        cfg = self.env_config
        reward = np.clip(reward, cfg.reward_clip_min, cfg.reward_clip_max)
        
        # 9. 更新周索引
        done = self.current_week >= self.max_weeks
        
        # 10. 🌟 V8新增：更新全局特征
        self.capacity_usage_ratio, self.budget_usage_ratio = self._calculate_global_features(self.current_stock)
        
        # 11. 更新State向量（V8：包含全局特征）
        store_profile = self.data_loader.get_store_profile(self.current_store)
        self.state = np.concatenate([
            self.current_stock,
            self.transit_stock,
            [store_profile['scale'], store_profile['type']],
            [self.capacity_usage_ratio, self.budget_usage_ratio]
        ]).astype(np.float32)
        
        # 12. 构建info字典
        info = {
            'store_id': self.current_store,
            'week_index': self.current_week,
            'sales': sales.sum(),
            'stockout': stockout.sum(),
            'reward': reward,
            'fill_rate': sales.sum() / (sales.sum() + stockout.sum() + 1e-8),
            'capacity_usage_ratio': self.capacity_usage_ratio,
            'budget_usage_ratio': self.budget_usage_ratio,
            'reorder_qty_median': float(np.median(proposed_qty)),  # 🌟 V8新增：供Callback聚合使用
            'actual_qty_list': proposed_qty.tolist(),  # 🌟 V8新增：供Callback聚合使用
            'unit_overflow': unit_overflow,  # 🌟 V8新增：前置预警值
            'budget_overflow': budget_overflow,  # 🌟 V8新增：前置预警值
        }
        
        return self.state, reward, done, False, info
    
    def _get_weekly_demand(self, store_id: str, week_index: int) -> np.ndarray:
        """
        极速查询：获取指定门店+周的需求
        
        使用哈希字典，时间复杂度 O(num_skus)
        """
        demand = np.zeros(self.num_skus)
        
        for i, sku_id in enumerate(self.valid_skus):
            demand[i] = self.data_loader.get_demand(
                store_id, sku_id, week_index
            )
        
        return demand
    
    def _execute_sales(self, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行销售，计算满足的销售和缺货
        
        Returns:
            sales: 实际销售量
            stockout: 缺货量
        """
        # 销售 = min(库存, 需求)
        sales = np.minimum(np.maximum(self.current_stock, 0), demand)
        stockout = demand - sales
        
        # 更新库存
        self.current_stock -= sales
        
        # 更新统计
        self.total_sales += sales.sum()
        self.total_stockout += stockout.sum()
        
        return sales, stockout
    
    def _update_inventory(self, reorder_quantities: np.ndarray):
        """
        更新库存状态（简化版）
        
        实际实现需要考虑：
        1. 在途库存到达
        2. 补货提前期
        3. 库存周转
        """
        # 简化：假设补货立即到达（实际应考虑lead time）
        # self.current_stock += reorder_quantities  # 已在 step 方法中执行
        
        # 计算持有成本
        holding_cost_rate = self.env_config.holding_cost_per_unit_weekly or 0.01
        self.total_holding_cost += np.sum(np.maximum(self.current_stock, 0)) * holding_cost_rate
    
    def _calculate_normalized_reward(
        self, 
        sales: np.ndarray, 
        stockout: np.ndarray, 
        proposed_qty: np.ndarray,
        unit_overflow: float,
        budget_overflow: float
    ) -> float:
        """
        V8.2 奖励函数：使用奖励权重进行加权求和
        
        返回：
            float: 加权后的总奖励
        """
        # 获取原始奖励项
        components = self.get_reward_components(sales, stockout, proposed_qty, unit_overflow, budget_overflow)
        
        # 应用奖励权重（如果已计算）
        cfg = self.env_config
        if hasattr(cfg, 'reward_weights') and cfg.reward_weights:
            reward = sum(components[k] * cfg.reward_weights.get(k, 1.0) for k in components)
        else:
            # 默认等权求和
            reward = sum(components.values())
        
        # 奖励裁剪
        reward = np.clip(reward, cfg.reward_clip_min, cfg.reward_clip_max)
        
        return reward
    def get_reward_components(
        self, 
        sales: np.ndarray, 
        stockout: np.ndarray, 
        proposed_qty: np.ndarray,
        unit_overflow: float,
        budget_overflow: float
    ) -> dict:
        """
        V8.2 新增：返回各奖励项原始值（未缩放，用于分析）
        
        此方法用于：
        1. 记录各奖励项到日志/TensorBoard
        2. 计算 P95 分位数（用于奖励量级对齐）
        3. 调试奖励函数行为
        """
        cfg = self.env_config
        components = {}
        
        # 1. 满足率奖励
        total_demand = np.sum(sales + stockout) + 1e-8
        fill_rate = np.sum(sales) / total_demand
        
        sigmoid_raw = 1.0 / (1.0 + math.exp(-cfg.sigmoid_steepness * (fill_rate - cfg.sigmoid_threshold)))
        log_raw = math.log(1 + cfg.log_k * fill_rate)
        log_norm = log_raw / math.log(1 + cfg.log_k)
        
        mixed_fill = cfg.w_sig * sigmoid_raw + (1 - cfg.w_sig) * log_norm
        components['fill_rate'] = cfg.fill_reward_scale * mixed_fill
        
        # 2. 持有成本
        holding_cost = 0.0
        unit_hold = cfg.holding_cost_per_unit_weekly
        max_inv = cfg.max_sku_inventory or 200.0
        
        for i in range(self.num_skus):
            inv_ratio = np.maximum(self.current_stock[i], 0) / max_inv
            cost = unit_hold * inv_ratio
            
            if inv_ratio > 0.7:
                excess = inv_ratio - 0.7
                cost += unit_hold * (excess ** 2) * (cfg.holding_excess_penalty_multiplier or 20.0)
            
            holding_cost += cost
        
        components['holding_cost'] = -holding_cost * cfg.holding_cost_penalty_scale
        
        # 3. 缺货惩罚
        components['stockout'] = -np.sum(stockout) * cfg.stockout_penalty_per_unit
        
        # 4. 前置超载惩罚
        components['unit_overflow'] = 0.0
        components['budget_overflow'] = 0.0
        if unit_overflow > 0:
            components['unit_overflow'] = -unit_overflow * (cfg.unit_overflow_penalty_rate or 0.5)
        if budget_overflow > 0:
            components['budget_overflow'] = -budget_overflow * (cfg.budget_overflow_penalty_rate or 0.01)
        
        # 5. 不补货奖励
        components['no_reorder'] = 0.0
        total_inv = np.sum(np.maximum(self.current_stock, 0))
        if total_inv > (cfg.no_reorder_inventory_threshold or 0.7) * cfg.max_store_capacity_units:
            zero_count = np.sum(proposed_qty == 0)
            components['no_reorder'] = zero_count * (cfg.no_reorder_bonus_per_sku or 0.001)
        
        # 6. 小额补货惩罚
        components['small_order'] = 0.0
        min_qty = cfg.min_reorder_qty
        small_mask = (proposed_qty > 0) & (proposed_qty < min_qty)
        if np.any(small_mask):
            components['small_order'] = -np.sum(min_qty - proposed_qty[small_mask]) * (cfg.small_reorder_penalty_rate or 1.5)
        
        return components
    

    def _get_medium_store_weights(self) -> List[float]:
        """
        获取"中等门店"的采样权重（前期训练）
        
        选择需求稳定的中等规模门店
        """
        weights = []
        
        for store_id in self.valid_stores:
            profile = self.data_loader.get_store_profile(store_id)
            scale = profile['scale']
            
            # 中等门店：scale在0.3-0.7之间
            if 0.3 <= scale <= 0.7:
                weights.append(1.0)
            else:
                weights.append(0.1)
        
        # 归一化
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return weights
    
    def _get_extreme_store_weights(self) -> List[float]:
        """
        获取"极端门店"的采样权重（后期训练）
        
        增加需求波动极大或极小的门店权重
        """
        weights = []
        
        for store_id in self.valid_stores:
            profile = self.data_loader.get_store_profile(store_id)
            scale = profile['scale']
            
            # 极端门店：scale < 0.2 或 scale > 0.8
            if scale < 0.2 or scale > 0.8:
                weights.append(2.0)  # 高权重
            else:
                weights.append(1.0)
        
        # 归一化
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return weights
    
    def set_curriculum_stage(self, stage: int):
        """设置课程学习阶段"""
        self.curriculum_stage = stage
        logger.info(f"[Curriculum] 切换到阶段 {stage}")


# ==================== 自定义 Callback ====================

class FuseCallbackV8(BaseCallback):
    """
    V8 熔断回调
    
    每 60 万步校验关键业务指标：
    - 补货量中位数 > 40
    - 门店超载 step 占比 > 10%
    
    连续两次不达标则熔断停止训练
    """
    
    def __init__(self, check_freq=600000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.fail_count = 0
        self.last_median = 0.0
        self.last_overflow_rate = 0.0
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # 收集所有环境中的补货量，计算真实全局中位数
            all_qty = []
            overflows = []
            
            # 注意：stable-baselines3 的 info 结构
            for info in self.locals.get('infos', []):
                if 'actual_qty_list' in info:
                    all_qty.extend(info['actual_qty_list'])
                if 'unit_overflow' in info:
                    overflows.append(info['unit_overflow'])
            
            if all_qty:
                global_median = np.median(all_qty)
            else:
                global_median = 0.0
            
            avg_overflow = np.mean(overflows) if overflows else 0.0
            
            # 计算溢出率（溢出 step 占比）
            overflow_steps = sum(1 for x in overflows if x > 0)
            total_steps = len(overflows) if overflows else 1
            overflow_rate = overflow_steps / total_steps
            
            self.last_median = global_median
            self.last_overflow_rate = overflow_rate
            
            print(f"[V8 Monitor] Step {self.n_calls}: global_median={global_median:.1f}, avg_overflow={avg_overflow:.1f}, overflow_rate={overflow_rate:.2%}")
            
            # 熔断条件：连续两次不达标
            # 条件：中位数 > 40 且 溢出率 > 10%
            if global_median > 40 and overflow_rate > 0.1:
                self.fail_count += 1
                print(f"⚠️  熔断警告：第 {self.fail_count} 次不达标 (median={global_median:.1f}, overflow_rate={overflow_rate:.2%})")
                
                if self.fail_count >= 2:
                    print("❌ 熔断：连续两次中位数>40且溢出率>10%，停止训练！")
                    return False
            else:
                self.fail_count = 0  # 达标则重置计数
        
        return True


# ==================== 简易评估脚本 ====================

def evaluate_v8_model(model_path, env_config, n_episodes=1000):
    """
    评估 V8 模型性能
    
    参数：
        model_path: 模型文件路径
        env_config: 环境配置
        n_episodes: 评估回合数
    
    返回：
        评估指标字典
    """
    from stable_baselines3 import PPO
    from data.global_data_loader import GlobalDataLoader
    
    # 加载数据加载器
    data_loader = GlobalDataLoader()
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建环境
    env = GlobalSupplyChainEnv(data_loader, env_config)
    
    # 评估
    all_qty = []
    all_fill_rates = []
    all_overflows = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            all_qty.extend(info['actual_qty_list'])
            all_fill_rates.append(info['fill_rate'])
            all_overflows.append(info['unit_overflow'])
    
    # 计算指标
    qty = np.array(all_qty)
    metrics = {
        'reorder_qty_median': float(np.median(qty)),
        'reorder_rate': float(np.mean(qty > 0) * 100),
        'fill_rate_mean': float(np.mean(all_fill_rates)),
        'overflow_rate': float(np.mean(np.array(all_overflows) > 0) * 100),
        'qty_mean': float(np.mean(qty)),
        'qty_std': float(np.std(qty)),
    }
    
    # 打印结果
    print('=' * 70)
    print('V8 模型评估结果')
    print('=' * 70)
    print(f"补货量中位数: {metrics['reorder_qty_median']:.2f} 件")
    print(f"补货率 (>0): {metrics['reorder_rate']:.2f}%")
    print(f"平均满足率: {metrics['fill_rate_mean']:.2%}")
    print(f"门店超载 step 占比: {metrics['overflow_rate']:.2f}%")
    print('=' * 70)
    
    return metrics
