"""
供应链 Gym 环境 - models/optimization/supply_chain_env.py
实现强化学习所需的连锁零售药店供应链模拟环境

核心设计：
1. 多维度奖励函数：销售奖励 + OFR 满足率奖励 + ITR 周转率奖励 + SOR 缺货惩罚
2. 有效库存评估：低库存惩罚基于"当前库存 + 在途订单"，避免延迟奖励问题
3. 即时补货激励：补货动作立即获得奖励，克服 lead_time 导致的信用分配难题
4. 奖励裁剪：[-5, 5] 防止尺度爆炸，保证训练稳定性
5. 低库存预警：初始库存仅 1 周需求量，强迫 Agent 从第一步学习补货
"""

import json
import os
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from config import EnvConfig
from utils.logger import get_logger
from models.optimization.reward_normalizer import RunningRewardNormalizer
from models.optimization.constraint_handler import ConstraintHandler

# 项目本地导入
from data.global_data_loader import GlobalDataLoader

# 可选导入：TFT预测支持
try:
    from models.forecasting.tft_model import TFTForecaster
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    TFTForecaster = None

# 可选导入：WMA预测支持
try:
    from models.forecasting.weighted_moving_average import WeightedMovingAverageForecaster
    WMA_AVAILABLE = True
except ImportError:
    WMA_AVAILABLE = False
    WeightedMovingAverageForecaster = None

logger = get_logger(__name__)


class SupplyChainEnv(gym.Env):
    """连锁零售药店供应链环境
    
    ================================================================
    方法目录（按功能分组）：
    ----------------------------------------------------------------
    1. 初始化与状态管理
       __init__          (行48)
       _init_state        (行396)
       _prepare_historical_data  (行508)
       _inject_real_world_state  (行523)
       demand_forecasts   (行635)
    
    2. 环境接口（Gym）
       reset             (行648)
       step              (行735)
       render            (行1916)
       close             (行1925)
    
    3. 动作处理
       _compute_max_reorder    (行910)
       _apply_action_mask     (行945)
       _execute_replenishment (行966)
       _update_deliveries    (行991)
    
    4. 状态更新
       _update_demand       (行1082)
       _update_target_inventory  (行1146)
       _execute_sales      (行1202)
       _update_performance  (行1595)
    
    5. 奖励计算
       _potential              (行1224)
       _calculate_reward      (行1262)
       _calculate_stockout_penalty_piecewise  (行1397)
       _calculate_order_cost   (行1426)
       _calculate_low_stock_penalty  (行1440)
       _calculate_holding_cost  (行1481)
       _calculate_expiry_loss  (行1489)
       _calculate_critical_drug_reward  (行1511)
       _calculate_ofr_reward  (行1537)
       _calculate_itr_reward  (行1552)
       _calculate_sor_penalty  (行1570)
       _calculate_over_stock_penalty  (行1584)
    
    6. 工具方法
       _get_stockout_penalty_coef  (行1474)
       _get_expiry_ratio         (行1502)
       _decay_expiry_ratios     (行1506)
       _is_critical_drug        (行1533)
       get_inventory_state      (行1631)
       get_warehouse_state     (行1641)
       _get_state              (行1658)
       _get_info               (行1881)
    ----------------------------------------------------------------
    ================================================================
    """
    
    def __init__(self,
                 num_skus: int = None,  # 改为None，动态计算
                 num_stores: int = None,  # 改为None，动态计算
                 forecast_method: str = 'tft_quantile',  # 预测方法：'tft_quantile' 或 'wma'
                 demand_forecasts: Optional[np.ndarray] = None,  # 已废弃，请使用demand_forecasts_quantiles
                 env_config: Optional[EnvConfig] = None,
                 initial_inventory: Optional[np.ndarray] = None,
                 sku_list: Optional[list] = None,
                 store_list: Optional[list] = None,
                 use_action_mask: bool = True,  # 默认启用Action Masking
                 # TFT分位数预测参数
                 demand_forecasts_quantiles: Optional[np.ndarray] = None,  # TFT分位数预测 (T x num_skus x num_quantiles)
                 quantile_levels: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],  # 分位数级别
                 # WMA预测参数
                 historical_sales_data: Optional[np.ndarray] = None,  # 历史销量数据 (T x num_skus)，用于WMA
                 wma_weights: Optional[List[float]] = None,  # WMA权重，默认[0.1, 0.2, 0.3, 0.4]
                 warehouse_inventory: Optional[np.ndarray] = None,  # 仓库库存数据
                 historical_data: Optional[Any] = None,  # 🔥 新增：真实历史数据（DataFrame）
                 abc_category_path: str = "data/sku_abc_categories.json",  # 🌟 新增：ABC分类文件路径
                 # 🔥 新增：真实库存数据（时间序列）
                 real_shop_stock_data: Optional[np.ndarray] = None,  # 真实门店库存数据 (num_dates, num_stores, num_skus)
                 real_dc_stock_data: Optional[np.ndarray] = None,  # 真实仓库库存数据 (num_dates, num_skus)
                 real_sales_data: Optional[np.ndarray] = None,  # 真实销售数据 (num_dates, num_stores, num_skus)
                 store_id: Optional[str] = None):  # 🌟 架构师新增：门店ID（用于GlobalDataLoader）
        """
        初始化供应链环境
        
        Args:
            num_skus: SKU数量（如果为None，则从sku_list或demand_forecasts推断）
            num_stores: 门店数量（如果为None，则从store_list推断）
            forecast_method: 预测方法，'tft_quantile'（TFT分位数预测）或 'wma'（加权移动平均）
            demand_forecasts: 需求预测数据 (T x num_skus) - 已废弃，请使用demand_forecasts_quantiles
            env_config: 环境配置
            initial_inventory: 初始库存数据 (num_skus,), 如果提供则使用真实数据
            sku_list: SKU列表（用于匹配 initial_inventory）
            store_list: 门店列表（用于匹配 initial_inventory）
            use_action_mask: 是否使用Action Masking（默认True）
            demand_forecasts_quantiles: TFT分位数预测 (T x num_skus x num_quantiles)，forecast_method='tft_quantile'时必需
            quantile_levels: 分位数级别列表，默认[0.1, 0.25, 0.5, 0.75, 0.9]
            historical_sales_data: 历史销量数据 (T x num_skus)，forecast_method='wma'时必需
            wma_weights: WMA权重，默认[0.1, 0.2, 0.3, 0.4]
            warehouse_inventory: 仓库库存数据 (num_skus,) 或 None
            historical_data: 真实历史数据（DataFrame）
            abc_category_path: ABC分类文件路径（用于业务感知Reward Shaping）
            real_shop_stock_data: 真实门店库存数据 (num_dates, num_stores, num_skus)
            real_dc_stock_data: 真实仓库库存数据 (num_dates, num_skus)
            real_sales_data: 真实销售数据 (num_dates, num_stores, num_skus)
            store_id: 门店ID（用于GlobalDataLoader获取真实初始库存）
        """
        super().__init__()
        
        self.config = env_config or EnvConfig()
        
        # 动态计算 num_skus 和 num_stores
        if num_skus is None:
            if sku_list is not None:
                num_skus = len(sku_list)
            elif demand_forecasts is not None:
                num_skus = demand_forecasts.shape[1]
            elif demand_forecasts_quantiles is not None:
                num_skus = demand_forecasts_quantiles.shape[1]
            else:
                raise ValueError("无法推断 num_skus，请提供 sku_list、demand_forecasts、demand_forecasts_quantiles 或显式设置 num_skus")
        
        if num_stores is None:
            if store_list is not None:
                num_stores = len(store_list)
            else:
                num_stores = 1  # 默认单门店
                logger.warning(f"num_stores 未指定，默认设为 1")
        
        self.num_skus = num_skus
        self.num_stores = num_stores
        
        # 🌟 架构师新增：门店ID和GlobalDataLoader（用于真实初始库存）
        self.store_id = str(store_id) if store_id is not None else None
        self.data_loader = None  # 懒加载，在_reset()中初始化
        logger.info(f"  ✓ store_id: {self.store_id}")
        
        # 🔥 新增：真实库存数据（时间序列）
        self.real_shop_stock_data = real_shop_stock_data  # 真实门店库存数据 (num_dates, num_stores, num_skus)
        self.real_dc_stock_data = real_dc_stock_data  # 真实仓库库存数据 (num_dates, num_skus)
        self.real_sales_data = real_sales_data  # 真实销售数据 (num_dates, num_stores, num_skus)
        
        # 标记是否使用真实数据
        # 🌟 修正：优先使用 env_config.use_real_data，否则根据数据是否提供自动判断
        if env_config is not None and hasattr(env_config, 'use_real_data'):
            self.use_real_data = env_config.use_real_data
            logger.info(f"  - use_real_data from config: {self.use_real_data}")
        else:
            self.use_real_data = (real_shop_stock_data is not None) and (real_dc_stock_data is not None) and (real_sales_data is not None)
        
        if self.use_real_data:
            logger.info(f"✓ 使用真实数据（GlobalDataLoader 模式）")
        else:
            logger.info(f"  - 使用模拟数据（参数未提供或 config.use_real_data=False）")
        
        # 预测方法
        self.forecast_method = forecast_method
        if forecast_method not in ['tft_quantile', 'wma', 'real_data']:
            raise ValueError(f"forecast_method必须是 'tft_quantile'、'wma' 或 'real_data'，当前为 {forecast_method}")
        
        # 初始库存数据（如果提供则使用真实数据）
        self.initial_inventory = initial_inventory
        self.sku_list = sku_list
        self.store_list = store_list
        
        # ========== 初始化真实数据（如果使用） ==========
        if self.use_real_data:
            logger.info(f"✓ 使用真实数据")
            
            # 🌟 确保 data_loader 已初始化
            if self.data_loader is None:
                logger.info(f"  - data_loader 未初始化，正在初始化...")
                self.data_loader = GlobalDataLoader()
                logger.info(f"  ✓ GlobalDataLoader 初始化完成")
            
            # 🌟 修正：如果使用 GlobalDataLoader，设置默认 Episode 长度
            if self.data_loader is not None:
                self.max_time = 52  # 默认52周
                logger.info(f"  - Episode长度已设置为: {self.max_time} 周（GlobalDataLoader 模式）")
                
                # 🔥 架构师防线：确保 num_skus 与 GlobalDataLoader 对齐
                expected_num_skus = len(self.data_loader.valid_skus)
                if self.num_skus != expected_num_skus:
                    logger.warning(f"⚠️ num_skus 不匹配！预期 {expected_num_skus} (from GlobalDataLoader), 实际 {self.num_skus}. 强制对齐...")
                    self.num_skus = expected_num_skus
                    num_skus = self.num_skus  # 🌟 同步局部变量，防止后续代码使用旧值
            else:
                self.max_time = self.real_shop_stock_data.shape[0]
                logger.info(f"  - Episode长度已设置为: {self.max_time} 周")
            
            # 🔥 初始化historical_sales_data，用于预测方法
            # 🌟 架构师特供：支持多门店，从 GlobalDataLoader 获取特定门店的历史需求
            if hasattr(self, 'data_loader') and self.data_loader is not None and hasattr(self, 'store_id') and self.store_id is not None:
                # 🌟 多门店情况：从 GlobalDataLoader 获取特定门店的历史需求
                logger.info(f"  - 多门店情况：从 GlobalDataLoader 获取门店 {self.store_id} 的历史需求...")
                try:
                    self.historical_sales_data = self.data_loader.get_historical_demand_array(
                        store_id=self.store_id,
                        target_skus=self.data_loader.valid_skus,  # 🌟 修正：使用 data_loader.valid_skus
                        time_steps=52  # 默认52周
                    )
                    
                    # 🚨 架构师防线：绝不允许全零数据流入！
                    if np.max(self.historical_sales_data) < 1e-5:
                        raise ValueError(f"🚨 致命错误：门店 {self.store_id} 的历史需求全为 0！数据流水线断裂！")
                    
                    logger.info(f"  ✓ 历史需求加载成功！Shape: {self.historical_sales_data.shape}, 均值: {np.mean(self.historical_sales_data):.2f}")
                    
                    # 🌟 修正：只使用前 num_skus 个SKU
                    if self.historical_sales_data.shape[1] > self.num_skus:
                        self.historical_sales_data = self.historical_sales_data[:, :self.num_skus]
                        logger.info(f"  - 切片 historical_sales_data 至: {self.historical_sales_data.shape}")
                except Exception as e:
                    logger.error(f"  ❌ 从 GlobalDataLoader 获取历史需求失败: {e}")
                    raise
            else:
                # 单门店情况：将真实销售数据转换为 (num_dates, num_skus) 格式
                self.historical_sales_data = self.real_sales_data[:, 0, :]  # (num_dates, num_skus)
                logger.info(f"  - historical_sales_data已初始化（单门店）: shape={self.historical_sales_data.shape}")
        
        # ========== 根据预测方法初始化 ==========
        # 🔥 核心修正：无论是否使用真实数据，都初始化预测方法
        # 原因：需求应该通过预测方法生成，而非直接使用销售数据
        if forecast_method == 'tft_quantile':
            # -------- TFT分位数预测 --------
            if demand_forecasts_quantiles is None:
                raise ValueError(
                    "forecast_method='tft_quantile' 时，demand_forecasts_quantiles 是必需的！"
                    "请提供TFT分位数预测数据，形状为 (T, num_skus, num_quantiles)"
                )
            
            self.demand_forecasts_quantiles = demand_forecasts_quantiles
            self.quantile_levels = quantile_levels
            
            # 计算分位数索引
            self.q10_idx = quantile_levels.index(0.1) if 0.1 in quantile_levels else None
            self.q50_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else None
            self.q90_idx = quantile_levels.index(0.9) if 0.9 in quantile_levels else None
            
            if any(idx is None for idx in [self.q10_idx, self.q50_idx, self.q90_idx]):
                raise ValueError(f"quantile_levels必须包含0.1, 0.5, 0.9，当前为{quantile_levels}")
            
            logger.info(f"✓ TFT分位数预测已加载: shape={demand_forecasts_quantiles.shape}, quantile_levels={quantile_levels}")
            
            # 扩展 episode 长度
            original_len = demand_forecasts_quantiles.shape[0]
            target_len = max(84, original_len)
            if original_len < target_len:
                extended = np.zeros((target_len, num_skus, len(quantile_levels)), dtype=np.float32)
                for i in range(target_len):
                    src_idx = i % original_len
                    weekly = demand_forecasts_quantiles[src_idx].copy()
                    if i >= original_len:
                        noise = np.random.uniform(0.95, 1.05, size=weekly.shape).astype(np.float32)
                        weekly = weekly * noise
                    extended[i] = weekly
                self.demand_forecasts_quantiles = extended
                logger.info(f"TFT分位数预测已扩展: {original_len} → {target_len} 周")
            self.max_time = target_len
            
        elif forecast_method == 'wma':
            # -------- WMA预测 --------
            # 🔥 修正：如果没有提供 historical_sales_data，但 self.historical_sales_data 已存在（从真实数据生成），则使用它
            if historical_sales_data is None:
                if hasattr(self, 'historical_sales_data') and self.historical_sales_data is not None:
                    historical_sales_data = self.historical_sales_data
                    logger.info(f"  - 使用 self.historical_sales_data 作为 WMA 输入")
                else:
                    raise ValueError(
                        "forecast_method='wma' 时，historical_sales_data 是必需的！"
                        "请提供历史销量数据，形状为 (T, num_skus)"
                    )
            
            if not WMA_AVAILABLE:
                raise ImportError("WMA预测器不可用，请检查 models/forecasting/weighted_moving_average.py")
            
            # ========== 数据清洗管道（Data Sanitization Protocol） ==========
            if historical_sales_data is not None:
                logger.info(f" [DataClean] 开始清洗 historical_sales_data: shape={historical_sales_data.shape}")
                
                # 步骤1：截断负数（Clip Negatives）
                min_before = np.min(historical_sales_data)
                historical_sales_data = np.clip(historical_sales_data, a_min=0, a_max=None)
                min_after = np.min(historical_sales_data)
                if min_before < 0:
                    logger.info(f"  ✓ [DataClean] 截断负数: {min_before:.4f} → {min_after:.4f}")
                
                # 步骤2：Winsorize 极端 Spike（缩尾处理）
                # 排除0计算99分位数
                positive_data = historical_sales_data[historical_sales_data > 0]
                if len(positive_data) > 0:
                    p99 = np.percentile(positive_data, 99)
                    max_before = np.max(historical_sales_data)
                    historical_sales_data = np.clip(historical_sales_data, a_min=0, a_max=p99)
                    max_after = np.max(historical_sales_data)
                    if max_before > p99:
                        logger.info(f"  ✓ [DataClean] Winsorize Spike: {max_before:.4f} → {p99:.4f} (P99)")
                
                # 打印清洗报告
                logger.info(f"  [DataClean] 清洗后统计: min={np.min(historical_sales_data):.4f}, max={np.max(historical_sales_data):.4f}, mean={np.mean(historical_sales_data):.4f}")
            
            self.historical_sales_data = historical_sales_data
            self.wma_weights = wma_weights or [0.1, 0.2, 0.3, 0.4]
            
            # 初始化WMA预测器（每个SKU一个）
            self.wma_forecasters = []
            for sku_idx in range(num_skus):
                forecaster = WeightedMovingAverageForecaster(name=f"WMA_SKU{sku_idx}", weights=self.wma_weights)
                self.wma_forecasters.append(forecaster)
            
            logger.info(f"✓ WMA预测器已初始化: {num_skus} 个预测器，权重={self.wma_weights}")
            
            # 扩展 episode 长度
            original_len = historical_sales_data.shape[0]
            target_len = max(84, original_len)
            if original_len < target_len:
                extended = np.zeros((target_len, num_skus), dtype=np.float32)
                for i in range(target_len):
                    src_idx = i % original_len
                    weekly = historical_sales_data[src_idx].copy()
                    if i >= original_len:
                        noise = np.random.uniform(0.95, 1.05, size=weekly.shape).astype(np.float32)
                        weekly = weekly * noise
                    extended[i] = weekly
                self.historical_sales_data = extended
                logger.info(f"历史销量数据已扩展: {original_len} → {target_len} 周")
            self.max_time = target_len
        
        # 仓库库存数据（如果提供则使用真实数据）
        self.warehouse_inventory_data = warehouse_inventory
        
        # 🔥 新增：挂载真实历史数据
        self.historical_data = historical_data
        self.sku_mapping = None
        self.available_weeks = None
        
        if self.historical_data is not None:
            self._prepare_historical_data()
        
        # ========== 状态空间维度计算（修正版：只看下一周预测） ==========
        # 状态组成（特征解耦）：
        # 1. 需求侧特征（下一周TFT分位数预测）: 3 × num_skus
        #    - expected_demand (q50)
        #    - uncertainty (q90 - q10)
        #    - skewness ((q90 - q50) - (q50 - q10))
        # 2. 当前库存 (num_skus,)
        # 3. 在途库存 (num_skus,)
        # 4. 距上次下单周数 (num_skus,)
        # 5. 过去3周销量 (3 × num_skus,)
        # 6. 过去3周缺货量 (3 × num_skus,)
        # 7. 时间特征 (3,)
        # 总维度 = 11 × num_skus + 3
        self.state_components = {
            'demand_features': 3 * num_skus,  # 下一周TFT分位数特征（3个特征）
            'current_inventory': num_skus,  # 当前库存水位
            'in_transit': num_skus,  # 在途库存
            'days_since_order': num_skus,  # 距上次下单周数
            'recent_sales': 3 * num_skus,  # 过去3周销量
            'recent_stockouts': 3 * num_skus,  # 过去3周缺货量
            'time_features': 3,  # 时间特征
        }
        self.state_dim = sum(self.state_components.values())
        
        # 更新维度校验
        expected_state_dim = 12 * num_skus + 3  # 3+1+1+1+3+3=12个分量，每个num_skus维度，+3时间特征
        if self.state_dim != expected_state_dim:
            raise ValueError(
                f"状态维度计算错误: 实际={self.state_dim}, 预期={expected_state_dim}"
            )
        
        logger.info(f"✓ 状态空间维度: {self.state_dim} (需求特征: 3×{num_skus}={3*num_skus}, 总维度降低{(1 - 1037/1517)*100:.1f}%)")
        
        self.total_training_steps = 0
        self.curriculum_threshold = 10_000  # 前 10,000 步为"新手期"
        logger.info(f"  ✓ 课程学习已启用: 新手期={self.curriculum_threshold} 步")
        
        # 动作空间: 有限连续空间（优化一：用 tanH 做平滑映射）
        # 修改：将 high 设置为"最大周需求量的 3 倍"，防止 PPO 输出极端值
        # 首先计算最大可能需求量（用于设置动作空间上限）
        if self.demand_forecasts is not None and len(self.demand_forecasts) > 0:
            max_weekly_demand_per_sku = np.max(self.demand_forecasts, axis=0)  # 每个SKU的最大周需求
            max_possible_order = max_weekly_demand_per_sku * 3.0  # 最多补 3 周需求量
        else:
            max_possible_order = np.ones(self.num_skus, dtype=np.float32) * 300.0  # 默认上限 300
        
        # 动作空间：PPO 输出 [-10, 10]，tanh 映射到 [0, 1]，再乘以 max_possible_order
        # 因此实际补货量范围：[0, max_possible_order]
        # 为了兼容 SB3 的 Box 空间，我们设置 low=0, high=1（表示归一化的补货比例）
        # 然后在 step() 中乘以 max_possible_order
        self.action_space = spaces.Box(
            low=0.0,           # 最少补 0 个（改为 0，不使用负数）
            high=1.0,           # 最多补 1 倍（归一化比例）
            shape=(num_skus,),
            dtype=np.float32
        )
        
        # 保存最大补货量（用于 step() 中的反归一化）
        self.max_possible_order = max_possible_order.astype(np.float32)
        
        # 观察空间（Z-score归一化后，范围约为[-10, 10]）
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )
        
        # 🔧 修复：预分配 State Buffer（防止引用污染）
        self._state_buffer = np.zeros(self.state_dim, dtype=np.float32)
        logger.info(f"  ✓ State Buffer 已预分配: {self.state_dim} 维")
        
        # 默认参数
        self.lead_time_mean = self.config.lead_time_weeks  # 使用周粒度的 lead_time
        self.reorder_frequency = getattr(self.config, 'reorder_frequency_weeks', 1)  # 补货频率（周）
        self.last_reorder_step = -self.reorder_frequency  # 上次补货时间步（初始化为允许第一周补货）
        self.performance_window = 10
        self.use_action_mask = False
        self.min_reorder_threshold = 1.0
        self.stockout_penalty_critical = 10.0
        self.stockout_penalty_base = 5.0
        
        # 奖励权重
        self.weight_sales = 0.5              # 缺货惩罚权重
        self.weight_sales_positive = 1.0     # 销售正奖励权重
        self.weight_inventory = 0.2          # 持有成本权重
        self.weight_expiry = 0.1            # 效期损耗权重
        self.weight_critical = 0.5          # 关键药品权重
        self.weight_order_cost = 0.03        # 订购成本权重
        self.weight_low_stock = 0.5         # 低库存预警权重
        self.weight_replenish_action = 0.3   # 即时补货动作奖励权重
        self.weight_ofr_reward = 2.0        # 订单满足率奖励权重
        self.weight_itr_reward = 1.0        # 库存周转率奖励权重
        self.weight_sor_penalty = 3.0       # 缺货率惩罚权重
        self.weight_warehouse_stockout = 3.0  # 仓库缺货惩罚权重
        
        # 奖励裁剪范围
        self.reward_clip_min = -5.0
        self.reward_clip_max = 5.0
        
        # 优化二：成本归一化所需的最大可能成本（预计算）
        # 缺货惩罚：每个SKU最大缺货 = 最大周需求，乘以缺货惩罚单价
        self.max_possible_shortage_cost = float(self.num_skus * 100.0)  # 假设缺货惩罚 100/件
        # 持有成本：每个SKU最大库存 = store_capacity，乘以持有成本率
        self.max_possible_holding_cost = float(self.config.store_capacity * self.config.holding_cost_rate * self.num_skus)
        # 塑形权重
        self.shaping_weight = 0.01  # 势能塑形权重，不宜过大
        
        # ========= PBRS（势能函数奖励塑形）参数 =========
        self.gamma = 0.95  # 势能函数折扣因子
        self.last_potential = 0.0  # 上一个状态的势能值（用于PBRS计算）
        # 🔧 修复：从配置读取，支持消融实验（使用getattr保证向下兼容）
        self.pbrs_enabled = getattr(self.config, 'enable_pbrs', True)
        
        # ========= ConstraintHandler（业务约束处理）参数 =========
        # 🔧 修复：从配置读取，支持消融实验
        self.constraints_enabled = getattr(self.config, 'enable_constraints', True)
        
        if self.constraints_enabled:
            self.constraint_handler = ConstraintHandler(store_capacity=self.config.store_capacity)
            logger.info(f"  ✓ ConstraintHandler 已初始化（门店容量: {self.config.store_capacity}）")
        else:
            self.constraint_handler = None
            logger.info(f"  ⚠️ ConstraintHandler 已禁用（消融实验模式）")
        
        # 目标库存水平（Base-Stock Level）：由 TFT 预测驱动，在_step()中动态更新
        self.target_inventory_level = np.zeros(self.num_skus, dtype=np.float32)          # Reward Normalizer - 动态归一化12个奖励组件
        # 使用3个核心维度聚合（Profitability, Service Level, Efficiency）
        # 🔧 修复：从配置读取，支持消融实验
        self.use_dynamic_normalization = getattr(self.config, 'use_dynamic_normalization', True)
        
        if self.use_dynamic_normalization:
            self.reward_normalizer = RunningRewardNormalizer(num_components=3, gamma=0.99)
            logger.info(f"  ✓ 动态Reward归一化已启用")
        else:
            self.reward_normalizer = None
            logger.info(f"  ⚠️ 动态Reward归一化已禁用（消融实验模式）")
        
        # 动态权重（训练初期→后期可调整）
        # 初期：强调服务水平和避免缺货
        # 后期：强调利润和效率
        self.dynamic_weights = np.array([0.3, 0.6, 0.1], dtype=np.float32)  # [Profit, Service, Efficiency]
        self.weight_adjustment_step = 0
        self.weight_adjustment_interval = 50000  # 每50000步调整一次权重
        
        # 🌟 加载 ABC 分类先验（用于业务感知 Reward Shaping）
        self.sku_penalty_multiplier = np.ones(self.num_skus, dtype=np.float32)  # 默认乘数为 1
        
        # 创建 SKU 代码列表（用于映射）
        if self.sku_list is not None:
            sku_codes = [str(code) for code in self.sku_list]
        else:
            sku_codes = [str(i) for i in range(self.num_skus)]
        
        if os.path.exists(abc_category_path):
            try:
                with open(abc_category_path, 'r', encoding='utf-8') as f:
                    categories = json.load(f)
                
                # 根据 ABC 分类设置惩罚乘数
                for idx, sku_code in enumerate(sku_codes):
                    cat = categories.get(str(sku_code), 'C')  # 默认 C 类
                    if cat == 'A':
                        self.sku_penalty_multiplier[idx] = 3.0  # A类：缺货重罚
                    elif cat == 'B':
                        self.sku_penalty_multiplier[idx] = 1.5  # B类：中等惩罚
                    else:
                        self.sku_penalty_multiplier[idx] = 1.0  # C类：基础惩罚
                
                logger.info(f"✅ 业务感知 Reward 已实装：A类x3.0, B类x1.5, C类x1.0")
                logger.info(f"   - ABC 分类文件: {abc_category_path}")
                logger.info(f"   - 加载了 {len(categories)} 个 SKU 的分类")
                
            except Exception as e:
                logger.warning(f"⚠️ 加载 ABC 分类文件失败: {e}，使用默认乘数")
        else:
            logger.info(f"ℹ️ ABC 分类文件不存在: {abc_category_path}，使用默认乘数（全1）")
        
        # ========= 🔥 满足率追踪变量（用于阶梯奖励） =========
        # 用于计算累积满足率，触发悬崖式阶梯奖金
        self.cumulative_filled = 0.0      # 当前episode累计满足需求量
        self.cumulative_demand = 0.0      # 当前episode累计总需求量
        self.episode_fill_rates = []      # 每步满足率记录（用于平滑）
        self.current_episode_service_level = 0.0  # 当前episode满足率
        
        # 🚨 架构师修复：初始化 current_week_idx 属性
        self.current_week_idx = 0  # 当前周索引（用于需求生成）
        
        logger.info(f"  ✓ 满足率追踪器已初始化（用于95%突破策略）")
        logger.info(f"  ✓ current_week_idx 已初始化: {self.current_week_idx}")
        
        # 初始化状态变量
        self._init_state()
    
    def _init_state(self):
        """初始化状态变量（如果使用真实库存数据则加载，否则使用1周需求量）"""
        # 门店实时库存 — 优先使用GlobalDataLoader（架构师全局净化协议）
        if self.store_id is not None:
            # 🌟 使用GlobalDataLoader获取真实初始库存
            try:
                # 懒加载：第一次使用时初始化data_loader
                if self.data_loader is None:
                    from data.global_data_loader import GlobalDataLoader
                    self.data_loader = GlobalDataLoader()
                    logger.info(f"  ✓ GlobalDataLoader已初始化")
                
                # 获取真实初始库存 (长度=88的数组)
                # 🌟 修复：get_initial_state()返回的是元组 (initial_stock, transit_stock)
                initial_stock, transit_stock = self.data_loader.get_initial_state(self.store_id)
                real_initial_stock = initial_stock
                
                # 截取前num_skus个（如果num_skus < 88）
                if len(real_initial_stock) >= self.num_skus:
                    self.store_inventory = real_initial_stock[:self.num_skus].astype(np.float32)
                else:
                    # 如果真实SKU数量不足，用均值填充
                    self.store_inventory = np.zeros(self.num_skus, dtype=np.float32)
                    self.store_inventory[:len(real_initial_stock)] = real_initial_stock
                    # 剩余SKU用平均需求×2估算
                    avg_demand = np.mean(real_initial_stock) if len(real_initial_stock) > 0 else 10.0
                    self.store_inventory[len(real_initial_stock):] = avg_demand * 2.0
                
                logger.info(f"  ✓ 门店库存初始化（GlobalDataLoader）: {self.store_inventory[:5]}...")
                logger.info(f"  - 库存均值: {np.mean(self.store_inventory):.2f}")
                
            except Exception as e:
                logger.error(f"  ❌ GlobalDataLoader初始化失败: {e}")
                logger.warning(f"  ⚠️ 降级为随机初始化")
                np.random.seed(42)
                self.store_inventory = np.random.uniform(5, 15, size=(self.num_skus,)).astype(np.float32)
        
        # 门店实时库存 — 如果使用真实库存数据
        elif self.use_real_data:
            # 🔥 使用真实库存数据（第一周）
            logger.info(f"使用真实库存数据初始化（时间序列）: 第一周数据")
            self.store_inventory = self.real_shop_stock_data[0, 0, :].astype(np.float32)  # (num_skus,)
            logger.info(f"  门店库存初始化: {self.store_inventory[:5]}...")
        elif self.initial_inventory is not None:
            # 使用提供的真实库存数据
            logger.info(f"使用真实库存数据初始化: shape={self.initial_inventory.shape}")
            self.store_inventory = self.initial_inventory.astype(np.float32)
        elif self.demand_forecasts is not None and len(self.demand_forecasts) > 0:
            # 原有逻辑：初始库存 = 1周需求量
            logger.info("使用简化逻辑初始化库存: 1周需求量")
            first_week = self.demand_forecasts[:min(7, len(self.demand_forecasts))]
            mean_demand = np.mean(first_week, axis=0)
            self.store_inventory = np.clip(
                mean_demand * 1.0, 0, self.config.store_capacity * 0.1
            ).astype(np.float32)
        else:
            # 修复：使用 historical_sales_data 初始化库存（避免全0）
            if self.historical_sales_data is not None and self.historical_sales_data.shape[0] > 0:
                # 使用第一周的平均销量作为初始库存（1周需求量）
                first_week = self.historical_sales_data[0, :]
                self.store_inventory = np.clip(
                    first_week * 1.0,  # 1周需求量
                    0, self.config.store_capacity * 0.1
                ).astype(np.float32)
                logger.info(f"  门店库存初始化（使用historical_sales_data）: {self.store_inventory[:5]}...")
            else:
                # 降级方案：使用随机初始库存
                np.random.seed(42)
                self.store_inventory = np.random.uniform(5, 15, size=(self.num_skus,)).astype(np.float32)
                logger.warning(f"  门店库存初始化（随机）: {self.store_inventory[:5]}...")

        # 中心仓库存 — 优先使用GlobalDataLoader（架构师要求）
        if self.store_id is not None and self.data_loader is not None:
            # 🌟 使用GlobalDataLoader获取仓库库存（或基于平均需求计算）
            try:
                # 获取该门店的平均需求
                avg_demand_list = []
                for sku_id in self.data_loader.valid_skus[:self.num_skus]:
                    demand = self.data_loader.get_demand(self.store_id, sku_id, 0)
                    avg_demand_list.append(demand)
                
                if avg_demand_list:
                    avg_demand = np.array(avg_demand_list, dtype=np.float32)
                    # 仓库初始库存 = 该门店所有SKU的4周总需求
                    self.warehouse_inventory = (avg_demand * 4).astype(np.float32)
                else:
                    # 如果无法获取需求，使用默认值
                    self.warehouse_inventory = np.ones(self.num_skus, dtype=np.float32) * 1000.0
                
                logger.info(f"  ✓ 仓库库存初始化（GlobalDataLoader）: {self.warehouse_inventory[:5]}...")
                logger.info(f"  - 仓库库存均值: {np.mean(self.warehouse_inventory):.2f}")
                
            except Exception as e:
                logger.error(f"  ❌ 仓库库存初始化失败: {e}")
                logger.warning(f"  ⚠️ 降级为默认值")
                self.warehouse_inventory = np.ones(self.num_skus, dtype=np.float32) * 1000.0
        
        # 中心仓库存 — 【修改】如果使用真实仓库库存数据
        elif self.use_real_data:
            # 🔥 使用真实仓库库存数据（第一周）
            logger.info(f"使用真实仓库库存数据初始化（时间序列）: 第一周数据")
            self.warehouse_inventory = self.real_dc_stock_data[0, :].astype(np.float32)  # (num_skus,)
            logger.info(f"  仓库库存初始化: {self.warehouse_inventory[:5]}...")
        elif self.warehouse_inventory_data is not None:
            # 使用提供的真实仓库库存数据
            logger.info(f"使用真实仓库库存数据初始化: shape={self.warehouse_inventory_data.shape}")
            self.warehouse_inventory = self.warehouse_inventory_data.astype(np.float32)
        elif self.demand_forecasts is not None and len(self.demand_forecasts) > 0:
            # 原有逻辑：按每个SKU需求量 × lead_time × 2 分配
            logger.info("使用模拟逻辑初始化仓库库存: 需求量 × lead_time × 2")
            first_week = self.demand_forecasts[:min(7, len(self.demand_forecasts))]
            mean_demand = np.mean(first_week, axis=0)
            self.warehouse_inventory = np.clip(
                mean_demand * self.lead_time_mean * 2,
                0, self.config.warehouse_capacity
            ).astype(np.float32)
        else:
            self.warehouse_inventory = np.zeros(self.num_skus, dtype=np.float32) + 1000

        # 在途订单状态
        self.on_order = np.zeros((self.num_skus, self.lead_time_mean + 1), dtype=np.float32)  # 历史销量和历史缺货量（用于PPO状态空间）
        # 过去 N 周的实际销量，形状为 (history_window, num_skus)
        # 使用 performance_window 作为历史窗口（默认10周）
        history_window = max(3, self.performance_window)  # 至少保留3周
        self.sales_history_weekly = np.zeros((history_window, self.num_skus), dtype=np.float32)
        
        # 过去 N 周的缺货量，形状为 (history_window, num_skus)
        self.stockout_history_weekly = np.zeros((history_window, self.num_skus), dtype=np.float32)
        
        # 距上次下单周数，形状为 (num_skus,)
        self.weeks_since_last_order = np.zeros(self.num_skus, dtype=np.int32)
        
        # 上次下单的日期，形状为 (num_skus,)
        self.last_order_week = np.ones(self.num_skus, dtype=np.int32) * (-self.lead_time_mean)  # 初始值为 -lead_time，表示很久以前
        
        # 历史绩效 - 使用滚动窗口
        self.stockout_events = np.zeros((self.num_skus, self.performance_window), dtype=np.float32)
        self.turnover_buffer = np.zeros((self.num_skus, self.performance_window), dtype=np.float32)
        self.stockout_history = np.zeros(self.num_skus, dtype=np.float32)
        self.turnover_history = np.zeros(self.num_skus, dtype=np.float32)
        self.current_demand = np.zeros(self.num_skus, dtype=np.float32)

        # 外部需求标志
        self.use_external_demand = False

        # 累计统计
        self.total_stockout = 0
        self.total_holding_cost = 0
        self.total_expiry_loss = 0
        self.total_reward = 0
        self.episode_steps = 0
        self.total_sales_amount = 0.0

        # 演示可视化
        self.inventory_history = []

        # 效期比例
        self._expiry_ratios = np.random.uniform(0.6, 1.0, self.num_skus).astype(np.float32)
        self._expiry_decay_rate = 0.01
        
        # 归一化所需的最大周需求量
        if self.demand_forecasts is not None and len(self.demand_forecasts) > 0:
            self._max_weekly_demand = max(float(np.max(self.demand_forecasts)), 1.0)
            self._mean_demand_per_sku = np.mean(self.demand_forecasts, axis=0).astype(np.float32)
        else:
            self._max_weekly_demand = 50.0
            self._mean_demand_per_sku = np.ones(self.num_skus, dtype=np.float32)
        
        # ========== 新增：初始化全局基准成本（将在reset中计算） ==========
        self._global_baseline_cost = None
        self.historical_mean_demand = None
        self.historical_std_demand = None
        # ========== 结束新增 ==========
        
        # 记录上一步动作和销售额
        self.last_action = np.zeros(self.num_skus, dtype=np.float32)
        self._last_step_actual_sales = np.zeros(self.num_skus, dtype=np.float32)
        
        # 补货前仓库库存快照（用于 OFR 计算）
        self._warehouse_before_replenish = np.zeros(self.num_skus, dtype=np.float32)
        self._step_delivered = np.zeros(self.num_skus, dtype=np.float32)
        self._step_requested = np.zeros(self.num_skus, dtype=np.float32)
        
        # 上一步订购成本（用于评估指标）
        self._last_order_cost = 0.0
    
    def _prepare_historical_data(self):
        """预处理历史数据，确保 SKU 顺序与环境严格对齐"""
        # 假设环境有一个固定的 SKU 列表，比如 self.sku_list = ['SKU_0', 'SKU_1', ...]
        # 我们需要确保查询出来的数据顺序永远是这个顺序
        
        # 1. 获取所有唯一的周索引，并排序
        self.available_weeks = sorted(self.historical_data['week_index'].unique())
        
        # 2. 建立 SKU 映射 (防止 Pandas 查询出来的顺序错乱)
        # 假设您的 DataFrame 有 'sku_id' 列
        unique_skus = sorted(self.historical_data['sku_id'].unique())
        self.sku_mapping = {sku: i for i, sku in enumerate(unique_skus)}
        
        logger.info(f"✓ 历史数据已加载: {len(self.available_weeks)} 周, {len(unique_skus)} SKU")
    
    def _inject_real_world_state(self, current_week):
        """从真实数据中注入当前周及过去3周的状态"""
        
        # 1. 查询当前周的数据 (用于初始化 当前库存 和 在途库存)
        current_snapshot = self.historical_data[
            self.historical_data['week_index'] == current_week
        ].copy()
        
        # 强制按 SKU 排序，确保索引对齐！
        current_snapshot['sku_sort_key'] = current_snapshot['sku_id'].map(self.sku_mapping)
        current_snapshot = current_snapshot.sort_values('sku_sort_key')
        
        # --- 注入 当前库存 (对应 _get_state 中的 self.store_inventory) ---
        if 'store_inventory' in current_snapshot.columns:
            real_inventory = current_snapshot['store_inventory'].values
            real_inventory = np.nan_to_num(real_inventory, nan=0.0)
            self.store_inventory = np.clip(real_inventory, 0.0, None).astype(np.float32)
        else:
            # 如果没有 store_inventory 列，使用 initial_inventory 或默认值
            if self.initial_inventory is not None:
                self.store_inventory = self.initial_inventory.astype(np.float32)
            else:
                self.store_inventory = np.ones(self.num_skus, dtype=np.float32) * 50.0
        
        # --- 注入 在途库存 (对应 _get_state 中的 self.on_order) ---
        if 'in_transit' in current_snapshot.columns:
            real_transit = current_snapshot['in_transit'].values
            real_transit = np.nan_to_num(real_transit, nan=0.0)
            # on_order 的形状是 (num_skus, lead_time_mean + 1)
            self.on_order = np.zeros((self.num_skus, self.lead_time_mean + 1), dtype=np.float32)
            self.on_order[:, 0] = np.clip(real_transit, 0.0, None).astype(np.float32)
        else:
            self.on_order = np.zeros((self.num_skus, self.lead_time_mean + 1), dtype=np.float32)

        # 2. 查询过去 3 周的数据 (用于初始化 销量历史 和 缺货历史)
        # 找到 current_week 在列表中的位置，往前推3周
        current_pos = self.available_weeks.index(current_week)
        past_weeks = self.available_weeks[current_pos-3 : current_pos]
        
        past_snapshot = self.historical_data[
            self.historical_data['week_index'].isin(past_weeks)
        ].copy()
        past_snapshot['sku_sort_key'] = past_snapshot['sku_id'].map(self.sku_mapping)
        past_snapshot = past_snapshot.sort_values(['sku_sort_key', 'week_index'])
        
        # --- 注入 过去3周销量 (对应 self.sales_history_weekly) ---
        # 假设 self.sales_history_weekly 的 shape 是 (history_window, num_skus)
        history_window = max(3, self.performance_window)
        self.sales_history_weekly = np.zeros((history_window, self.num_skus), dtype=np.float32)
        if 'sales' in past_snapshot.columns:
            sales_pivot = past_snapshot.pivot(index='week_index', columns='sku_sort_key', values='sales')
            sales_pivot = sales_pivot.reindex(past_weeks).fillna(0.0)
            self.sales_history_weekly[-3:, :] = sales_pivot.values.astype(np.float32)
            
        # --- 注入 过去3周缺货量 (对应 self.stockout_history_weekly) ---
        self.stockout_history_weekly = np.zeros((history_window, self.num_skus), dtype=np.float32)
        if 'stockouts' in past_snapshot.columns:
            stockouts_pivot = past_snapshot.pivot(index='week_index', columns='sku_sort_key', values='stockouts')
            stockouts_pivot = stockouts_pivot.reindex(past_weeks).fillna(0.0)
            self.stockout_history_weekly[-3:, :] = stockouts_pivot.values.astype(np.float32)
        
        # --- 仓库库存设为足够大，不限制前期补货 ---
        # 🚨 架构师禁用：不要在 reset() 中强制设置仓库库存，而是使用外部注入的值
        # self.warehouse_inventory = np.full(self.num_skus, 10000.0, dtype=np.float32)
        # 🌟 如果外部没有设置仓库库存，才使用默认值
        if not hasattr(self, '_warehouse_inventory_injected') or not self._warehouse_inventory_injected:
            self.warehouse_inventory = np.full(self.num_skus, 10000.0, dtype=np.float32)
            logger.warning(f"[Reset] 仓库库存使用默认值 10000.0（外部未注入）")
        else:
            logger.info(f"[Reset] 仓库库存保留外部注入值: 均值 = {np.mean(self.warehouse_inventory):.2f}")
        
        # --- 重置其他非数据相关的计数器 ---
        self.weeks_since_last_order = np.zeros(self.num_skus, dtype=np.int32)
        self.last_order_week = np.ones(self.num_skus, dtype=np.int32) * (-self.lead_time_mean)
        self.stockout_events = np.zeros((self.num_skus, self.performance_window), dtype=np.float32)
        self.turnover_buffer = np.zeros((self.num_skus, self.performance_window), dtype=np.float32)
        self.stockout_history = np.zeros(self.num_skus, dtype=np.float32)
        self.turnover_history = np.zeros(self.num_skus, dtype=np.float32)
        self.current_demand = np.zeros(self.num_skus, dtype=np.float32)
        
        # 外部需求标志
        self.use_external_demand = False
        
        # 累计统计
        self.total_stockout = 0
        self.total_holding_cost = 0
        self.total_expiry_loss = 0
        self.total_reward = 0
        self.episode_steps = 0
        self.total_sales_amount = 0.0
        
        # 演示可视化
        self.inventory_history = []
        
        # 效期比例
        self._expiry_ratios = np.random.uniform(0.6, 1.0, self.num_skus).astype(np.float32)
        self._expiry_decay_rate = 0.01
        
        # 归一化所需的最大周需求量
        if self.demand_forecasts is not None and len(self.demand_forecasts) > 0:
            self._max_weekly_demand = max(float(np.max(self.demand_forecasts)), 1.0)
            self._mean_demand_per_sku = np.mean(self.demand_forecasts, axis=0).astype(np.float32)
        else:
            self._max_weekly_demand = 50.0
            self._mean_demand_per_sku = np.ones(self.num_skus, dtype=np.float32)
        
        # 记录上一步动作和销售额
        self.last_action = np.zeros(self.num_skus, dtype=np.float32)
        self._last_step_actual_sales = np.zeros(self.num_skus, dtype=np.float32)
        
        # 补货前仓库库存快照（用于 OFR 计算）
        self._warehouse_before_replenish = np.zeros(self.num_skus, dtype=np.float32)
        self._step_delivered = np.zeros(self.num_skus, dtype=np.float32)
        self._step_requested = np.zeros(self.num_skus, dtype=np.float32)
        self._last_order_cost = 0.0
        
        logger.info(f"✓ 已从真实数据注入初始状态: 当前周={current_week}, 库存={self.store_inventory[:3]}...")
    
    @property
    def demand_forecasts(self):
        """
        向后兼容属性：返回TFT分位数预测的中位数（q50）
        
        这样所有旧代码（使用self.demand_forecasts）都能自动获取中位数预测
        
        Returns:
            np.ndarray or None: 形状 (T, num_skus)，TFT中位数预测
        """
        if hasattr(self, 'demand_forecasts_quantiles') and self.demand_forecasts_quantiles is not None:
            return self.demand_forecasts_quantiles[:, :, self.q50_idx]
        return None
    
    def reset(self, 
             seed: Optional[int] = None,
             options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境（基于真实数据分布）
        
        Args:
            seed: 随机种子
            options: 重置选项
        
        Returns:
            Tuple: (state, info)
        """
        super().reset(seed=seed)
        
        # 1. 确定本 Episode 的起始周 (Start Week)
        # 必须确保剩下的周数足够跑完一个完整的 Episode (假设 episode_length = 52)
        episode_length = getattr(self, 'episode_length', 52) 
        
        if self.historical_data is not None and self.available_weeks is not None:
            # 🔥 使用真实历史数据：采样起始周
            max_start_idx = len(self.available_weeks) - episode_length - 3  # 预留3周给历史回溯
            
            if max_start_idx <= 0:
                raise ValueError(f"历史数据太短！需要至少 {episode_length + 3} 周的数据。")
            
            # 🚨 架构师控制变量协议：支持通过 options 传入固定的 start_idx
            if options is not None and 'start_idx' in options:
                # 使用固定的起始周索引
                start_idx = options['start_idx']
                if start_idx < 0 or start_idx >= max_start_idx:
                    raise ValueError(f"start_idx={start_idx} 超出范围 [0, {max_start_idx})")
                logger.info(f"[Reset] 使用固定起始周索引: {start_idx}")
            else:
                # 随机采样一个起始周索引
                start_idx = self.np_random.integers(0, max_start_idx)
            
            self.current_week_idx = start_idx + 3  # +3 是因为我们需要查询前3周的历史
            current_week = self.available_weeks[self.current_week_idx]
            
            # ==========================================
            # 🔥 核心操作：从真实数据中"抽取"初始状态
            # ==========================================
            self._inject_real_world_state(current_week)
            
            self.current_time = 0
            self.total_stockout = 0
            self.total_holding_cost = 0
            self.total_expiry_loss = 0
            self.total_reward = 0
            self.episode_steps = 0
            self.total_sales_amount = 0.0
            
            # ========= 🔥 重置满足率追踪变量 =========
            self.cumulative_filled = 0.0
            self.cumulative_demand = 0.0
            self.episode_fill_rates = []
            self.current_episode_service_level = 0.0
            logger.debug("[Reset] 满足率追踪变量已重置")
            
            # 生成初始需求（基于真实数据的当前周）
            self._update_demand()
            
            # 计算全局统计基准成本（用于奖励归一化）
            if self._global_baseline_cost is None:
                self._calculate_global_baseline_cost()
            
            # ========== 初始化PBRS（势能函数奖励塑形） ==========
            if self.pbrs_enabled:
                # 计算初始状态的势能
                initial_potential = self._potential(self.store_inventory, self.current_demand)
                self.last_potential = initial_potential
                logger.info(f"[PBRS] 初始化势能: {initial_potential:.4f}")
            
            return self._get_state(), {"start_week": current_week}
        
        else:
            # 降级方案：如果没有传入真实数据，使用原有的逻辑
            self._init_state()
            self.current_time = 0
            self.total_stockout = 0
            self.total_holding_cost = 0
            self.total_expiry_loss = 0
            self.total_reward = 0
            self.episode_steps = 0
            self.total_sales_amount = 0.0
            
            # 生成初始需求
            self._update_demand()
            
            # 计算全局统计基准成本（用于奖励归一化）
            if self._global_baseline_cost is None:
                self._calculate_global_baseline_cost()
            
            # ========== 初始化PBRS（势能函数奖励塑形） ==========
            if self.pbrs_enabled:
                # 计算初始状态的势能
                initial_potential = self._potential(self.store_inventory, self.current_demand)
                self.last_potential = initial_potential
                logger.info(f"[PBRS] 初始化势能: {initial_potential:.4f}")
            
            return self._get_state(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作（业务对齐：符合连锁零售药店实际补货流程）
        
        业务流程：
        1. 智能体观察当前状态 s_t
        2. 执行动作 a_t（下发补货订单，将在 t + lead_time 时到货）
        3. 环境状态转移：
           - 在途订单到货
           - 生成当周需求
           - 执行销售
        4. 计算奖励 r_t
        5. 返回下一状态 s_{t+1}
        
        Args:
            action: 归一化补货决策 (num_skus,)，范围 [0, 1]
        
        Returns:
            Tuple: (next_state, reward, terminated, truncated, info)
        """
        # 🌟 手术 2：强行截断 Action，防止 PPO 输出非法值（双保险）
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 🚨 终极修复：初始化 actual_action（否则会被设为 [0, 0, 0,...]！）
        actual_action = action.copy()
        
        # [DEBUG] 打印动作执行前的状态（架构师真值审计）
        if self.episode_steps < 3:  # 只打印前3步
            print(f"[DEBUG] Step {self.episode_steps}: action={action[:3]}, store_inv={self.store_inventory[:3]}, warehouse_inv={self.warehouse_inventory[:3]}")
        
        # ========= 🔥 第三阶段：动作约束（物理外挂） =========
        # 强制安全库存底线 (Action Masking)
        # 🚨【暗坑2修正】计算考虑 Lead Time 的真实安全库存
        if getattr(self.config, 'enable_action_masking', False):
            # 计算覆盖 Lead Time + 1 周的需求
            coverage_weeks = self.config.lead_time_weeks + 1  # 必须覆盖"提前期内" + "下一个周期"的需求
            
            # 获取未来 coverage_weeks 周的预测需求
            if hasattr(self, 'demand_forecast') and self.demand_forecast is not None:
                # demand_forecast shape: (forecast_horizon, num_skus)
                future_demand_coverage = np.sum(self.demand_forecast[:, :coverage_weeks], axis=0)  # shape: (num_skus,)
            else:
                # 降级：使用当前需求 × coverage_weeks
                if hasattr(self, 'current_demand') and self.current_demand is not None:
                    future_demand_coverage = self.current_demand * coverage_weeks
                else:
                    future_demand_coverage = np.ones(self.num_skus) * 10.0 * coverage_weeks  # 默认值
            
            # 计算安全库存（考虑预测需求 × 安全系数）
            safety_stock = future_demand_coverage * self.config.action_masking_multiplier
            
            # 检查哪些SKU低于安全库存
            # 注意：store_inventory shape可能是 (num_stores, num_skus) 或 (num_skus,)
            if len(self.store_inventory.shape) > 1:
                current_inventory = self.store_inventory[0]  # 取第一个门店
            else:
                current_inventory = self.store_inventory
            
            low_stock_mask = current_inventory < safety_stock
            
            # 对低于安全库存的SKU，强制补货（动作设为0.5，即50%最大补货量）
            if np.any(low_stock_mask):
                action[low_stock_mask] = np.maximum(action[low_stock_mask], 0.5)
                
                # 调试日志（每20步打印一次）
                num_forced = np.sum(low_stock_mask)
                if self.episode_steps % 20 == 0:
                    logger.info(f"[Action Masking] Step {self.episode_steps}: "
                               f"强制 {num_forced} 个SKU补货（库存低于安全线，考虑Lead Time {coverage_weeks}周）")
        

        # 0. 反归一化：将 [0, 1] 的归一化动作映射为实际补货量
        # 修改：不再使用 tanh（因为 action_space 已经改为 [0, 1]）
        # 直接使用线性映射：actual_action = action * max_possible_order
        max_reorder = self._compute_max_reorder()
        actual_action = action * max_reorder
        
        # 🌟 关键防御：再次 Clip，确保补货量不超过仓库库存上限
        # 🔧 临时禁用：为了公平对比 Heuristic Baseline，允许无限仓库库存
        # actual_action = np.minimum(actual_action, self.warehouse_inventory)
        pass  # 禁用仓库库存限制
        
        # 🚨 架构师调试：暂时注释掉约束逻辑，看策略是否工作
        # 目的：隔离 Bug，确定是否是约束逻辑把 actual_action 清零了
        # actual_action = action.copy()  # ❌ 删除此行！它会覆盖正确的转换逻辑！
        
        # # 1. 应用动作掩码（过滤掉小于最小批量的无效动作）
        # actual_action = self._apply_action_mask(actual_action)
        
        # # 1.5 业务约束检查与调整（如果启用）
        # constraint_penalty = 0.0
        # if self.constraints_enabled and hasattr(self, 'constraint_handler'):
        #     # 调用ConstraintHandler检查和调整动作
        #     constraint_result = self.constraint_handler.check_and_adjust(
        #         actual_action,
        #         self.store_inventory,
        #         self.warehouse_inventory_data if self.warehouse_inventory_data is not None else np.ones(self.num_skus) * 999999,
        #         min_batch_size=10.0,  # 最小补货批量
        #         max_order=1000.0  # 最大订单量
        #     )
            
        #     # 调整动作
        #     actual_action = constraint_result.adjusted_action
            
        #     # 记录约束惩罚
        #     constraint_penalty = constraint_result.penalty
            
        #     # 记录违规信息（调试用）
        #     if len(constraint_result.violations) > 0:
        #         logger.debug(f"[ConstraintHandler] 发现 {len(constraint_result.violations)} 个约束违规: {constraint_result.violations}")
        
        # [DEBUG] 打印注释掉约束后的 actual_action
        if self.episode_steps < 3:
            print(f"[DEBUG] After comment out constraints: actual_action[:5]={actual_action[:5]}")
        
        # 🚨 架构师调试：注释掉第二个约束逻辑块（它把 actual_action 清零了！）
        # constraint_penalty = 0.0
        # if self.constraints_enabled and hasattr(self, 'constraint_handler'):
        #     constraint_result = self.constraint_handler.check_and_adjust(
        #         actual_action,
        #         self.store_inventory,
        #         self.warehouse_inventory_data if self.warehouse_inventory_data is not None else np.ones(self.num_skus) * 999999,
        #         min_batch_size=10.0,
        #         max_order=1000.0
        #     )
        #     actual_action = constraint_result.adjusted_action
        #     constraint_penalty = constraint_result.penalty
        
        # [DEBUG] 打印注释掉第二个约束块后的 actual_action
        if self.episode_steps < 3:
            print(f"[DEBUG] After commenting 2nd constraint: actual_action[:5]={actual_action[:5]}")
            
        
        # ========== 核心逻辑：周度步进（每步代表一周） ==========
        # 周粒度环境下，每步都进行补货决策！
        
        # [DEBUG] 打印 actual_action（架构师真值审计）
        if self.episode_steps < 3:
            print(f"[DEBUG] Before _execute: actual_action[:5]={actual_action[:5]}")
        
        # 保存补货动作
        self.last_action = actual_action.copy()
        
        # 记录补货前的仓库库存快照（用于 OFR 计算）        self._warehouse_before_replenish = self.warehouse_inventory.copy()
        self._step_delivered = np.zeros(self.num_skus, dtype=np.float32)
        self._step_requested = actual_action.copy()
        
        # 2. 【执行动作】下发补货订单（将在 lead_time 后到达）
        self._execute_replenishment(actual_action)
        
        # 3. 【状态转移】在途订单到货
        self._update_deliveries()
        
        # 4. 【状态转移】生成当周需求
        self._update_demand()
        
        # 4.5 优化二：更新目标库存水平（Base-Stock Level），用于势能塑形
        self._update_target_inventory()
        
        # 5. 【状态转移】执行销售
        self._execute_sales()  # 更新历史销量和历史缺货量（用于PPO状态空间）
        # 将本周的销量记录到 sales_history_weekly 中（滚动窗口）
        if hasattr(self, 'sales_history_weekly'):
            # 滚动更新：将新数据插入到第一行
            self.sales_history_weekly = np.roll(self.sales_history_weekly, 1, axis=0)
            self.sales_history_weekly[0, :] = self._last_step_actual_sales.copy()
        
        # 将本周的缺货量记录到 stockout_history_weekly 中
        if hasattr(self, 'stockout_history_weekly'):
            # 计算本周的缺货量
            stockout_today = np.maximum(self.current_demand - self._last_step_actual_sales, 0)
            # 滚动更新
            self.stockout_history_weekly = np.roll(self.stockout_history_weekly, 1, axis=0)
            self.stockout_history_weekly[0, :] = stockout_today  # 更新距上次下单周数
        if hasattr(self, 'weeks_since_last_order'):
            # 所有 SKU 的 weeks_since_last_order 加 1
            self.weeks_since_last_order += 1
            
            # 如果某个 SKU 本周有补货动作，则重置为 0
            for i in range(self.num_skus):
                if self.last_action[i] > 0:
                    self.weeks_since_last_order[i] = 0
                    self.last_order_week[i] = self.current_time
        
        # 6. 【计算奖励】基于销售结果和库存状态计算RL奖励
        # 创建info字典，用于传递奖励分解信息
        info = {}
        reward = self._calculate_reward(info=info)
        self.total_reward += reward
        self.episode_steps += 1
        
        # 6.2 应用约束惩罚（如果启用）
#        if self.constraints_enabled and constraint_penalty > 0:
#            reward -= constraint_penalty
#            info['constraint_penalty'] = float(constraint_penalty)
#            logger.debug(f"[ConstraintHandler] 应用约束惩罚: {constraint_penalty:.4f}")
        
        # ========== 6.5. PBRS（势能函数奖励塑形） ==========
        if self.pbrs_enabled:
            # 计算当前状态的势能（在状态转移之前）
            # 注意：self.last_potential 已经在 reset() 或上一步中设置为当前状态的势能
            potential_current = self.last_potential
            
            # 计算下一个状态的势能（状态转移之后）
            # 使用转移后的库存和下一阶段的需求
            next_demand = self.current_demand  # 当前需求（状态转移后未变化）
            potential_next = self._potential(self.store_inventory, next_demand)
            
            # 计算塑形奖励：F(s_t, s_t+1) = gamma * Phi(s_t+1) - Phi(s_t)
            shaping_reward = self.gamma * potential_next - potential_current
            
            # 将塑形奖励加到原始奖励上
            reward_with_pbrs = reward + self.shaping_weight * shaping_reward
            
            # 记录PBRS信息（调试用）
            info['pbrs_potential_current'] = float(potential_current)
            info['pbrs_potential_next'] = float(potential_next)
            info['pbrs_shaping_reward'] = float(self.shaping_weight * shaping_reward)
            info['pbrs_reward_before'] = float(reward)
            info['pbrs_reward_after'] = float(reward_with_pbrs)
            
            # 更新奖励
            reward = reward_with_pbrs
            
            # 更新上一个状态的势能
            self.last_potential = potential_next
        
        # 7. 【更新绩效统计】
        self._update_performance()
        
        # 8. 效期衰减
        self._decay_expiry_ratios()
        
        # 记录库存历史（用于演示可视化）
        self.inventory_history.append(self.store_inventory.copy())
        
        # 9. 时间推进与终止判断
        self.current_time += 1
        terminated = self.current_time >= self.max_time
        truncated = False
        
        # 获取基础info字典
        base_info = self._get_info()
        
        # 将奖励分解信息合并到基础info中
        info.update(base_info)
        
        # 存储原始 Reward（未归一化）到 info 中
        # 这样 Callback 就能从 infos[i]["original_reward"] 中获取真实的业务 Reward
        info["original_reward"] = float(reward)
        
        return self._get_state(), reward, terminated, truncated, info
    
    def _compute_max_reorder(self) -> np.ndarray:
        """
        计算每个SKU的最大补货量（动态缩放）
        
        策略：基于需求预测的 P95 × lead_time × 安全系数(1.5)
        使用P95而不是均值，避免极端spike导致max_reorder过大
        
        Returns:
            np.ndarray: 每个SKU的最大补货量 (num_skus,)
        """
        # 使用当前需求预测的P95 × lead_time × 1.5 作为上限
        if self.demand_forecasts is not None and self.current_time < len(self.demand_forecasts):
            # 取未来 lead_time 周的预测
            future_end = min(self.current_time + self.lead_time_mean, len(self.demand_forecasts))
            if future_end > self.current_time:
                future_demand = self.demand_forecasts[self.current_time:future_end]
                # 修改：使用P95分位数而不是均值
                p95_weekly_demand = np.percentile(future_demand, 95, axis=0)  # (num_skus,)
            else:
                p95_weekly_demand = self.current_demand.copy()
        else:
            p95_weekly_demand = self.current_demand.copy()
        
        # 最大补货量 = P95周需求 × lead_time × 1.5（安全系数）
        max_reorder = p95_weekly_demand * self.lead_time_mean * 1.5
        
        # 下限至少为1，避免零值
        max_reorder = np.maximum(max_reorder, 1.0)
        
        # 上限不超过仓库当前库存（【修改】使用真实库存限制）
        # 原来：max_reorder = np.minimum(max_reorder, float(self.config.warehouse_capacity))
        # 修改：使用当前仓库库存作为上限，让智能体感知库存约束
        max_reorder = np.minimum(max_reorder, self.warehouse_inventory)
        
        return max_reorder.astype(np.float32)
    
    def _apply_action_mask(self, action: np.ndarray) -> np.ndarray:
        """
        应用动作掩码
        
        Args:
            action: 原始动作
        
        Returns:
            np.ndarray: 掩码后的动作
        """
        if not self.use_action_mask:
            return action
        
        # 动作掩码: 0 或 [L_min, A_max]
        masked_action = np.zeros_like(action)
        valid_mask = action >= self.min_reorder_threshold
        
        masked_action[valid_mask] = action[valid_mask]
        
        return masked_action
    
    def _execute_replenishment(self, action: np.ndarray):
        """
        执行补货决策
        
        Args:
            action: 补货量（可能是归一化值或实际量）
        """
        # [DEBUG] 打印 action 的值（架构师真值审计）
        if self.episode_steps < 3:
            print(f"[DEBUG] _execute_replenishment: action[:5]={action[:5]}, warehouse_inv[:5]={self.warehouse_inventory[:5]}")
        
        # 确保action是1维数组
        action = np.array(action).flatten()
        
        for i in range(self.num_skus):
            reorder_qty = float(action[i])  # 转换为标量
            
            # 检查仓库可用库存
            available = min(reorder_qty, float(self.warehouse_inventory[i]))  # 转换为标量
            
            # 记录实际发货量
            self._step_delivered[i] = available
            
            if available > 0:
                # 更新在途订单（订单将在 lead_time 周后到达）
                lead_time = int(self.lead_time_mean)
                if lead_time >= self.on_order.shape[1]:
                    lead_time = self.on_order.shape[1] - 1
                
                self.warehouse_inventory[i] -= available
                self.on_order[i, lead_time] += available
    
    def _update_deliveries(self):
        """更新交货（在途订单到达门店）"""
        for i in range(self.num_skus):
            # 第一个时间窗口的订单到达
            delivered = float(self.on_order[i, 0])  # 转换为标量
            
            # 检查门店容量（store_inventory是1维数组）
            space_available = float(self.config.store_capacity - self.store_inventory[i])  # 转换为标量
            actual_delivery = min(delivered, space_available)
            
            self.store_inventory[i] += actual_delivery  # store_inventory是1维数组
            
            # 剩余货物留在仓库（或产生惩罚）
            if actual_delivery < delivered:
                self.warehouse_inventory[i] += (delivered - actual_delivery)
            
            # 更新在途订单（移位）
            self.on_order[i, :-1] = self.on_order[i, 1:]
            self.on_order[i, -1] = 0
    
    def _calculate_global_baseline_cost(self):
        """
        计算全局统计基准成本（用于奖励归一化）
        
        业务逻辑：基准成本 = "如果采用极简策略，导致的安全库存缺货成本" + "基础持有成本"
        Z=1.65 代表 95% 服务水平下的潜在缺货风险
        
        这个方法在环境初始化时调用一次，后续所有奖励计算都使用这个基准
        """
        # 获取历史需求数据（用于计算均值和标准差）
        # 优先使用 sales_history_weekly（实际销售数据）
        if hasattr(self, 'sales_history_weekly') and np.sum(np.abs(self.sales_history_weekly)) > 0:
            historical_demand = self.sales_history_weekly
            logger.info("[Baseline Cost] 使用 sales_history_weekly 计算基准成本")
        elif hasattr(self, 'demand_forecasts') and self.demand_forecasts is not None:
            # 如果没有实际销售数据，使用需求预测数据
            historical_demand = self.demand_forecasts
            logger.info("[Baseline Cost] 使用 demand_forecasts 计算基准成本")
        else:
            # 如果没有任何数据，使用模拟数据（基于当前需求或默认值）
            if hasattr(self, 'current_demand') and self.current_demand is not None:
                # 使用当前需求生成模拟历史数据
                # 使用historical_data_weeks配置，如果不存在则使用默认值4
                history_weeks = getattr(self.config, 'historical_data_weeks', 4)
                historical_demand = np.tile(self.current_demand, (history_weeks, 1))
            else:
                # 默认：每个SKU每周需求=10
                historical_demand = np.ones((4, self.num_skus)) * 10.0
            logger.info("[Baseline Cost] 使用模拟数据计算基准成本")
        
        # 计算历史需求的均值和标准差
        # 确保historical_demand是2D数组 (weeks, num_skus)
        if len(historical_demand.shape) == 1:
            historical_demand = historical_demand.reshape(1, -1)
        
        self.historical_mean_demand = np.mean(historical_demand, axis=0)  # shape: (num_skus,)
        self.historical_std_demand = np.std(historical_demand, axis=0)  # shape: (num_skus,)
        
        # 防止标准差为0（需求完全稳定）或过小
        self.historical_std_demand = np.maximum(self.historical_std_demand, 0.1)
        
        # Z分数：95%服务水平（对应1.65标准差）
        # 对于医药零售，使用95%服务水平是合理的
        z_score = 1.65
        
        # 估算每周的基准缺货量（需求波动带来的风险）
        # 使用np.mean()而非np.sum()，确保与total_cost在同一量级
        baseline_shortage_units = np.mean(self.historical_std_demand * z_score)
        
        # 估算每周的基准持有量（安全库存）
        # 假设维持半周销量的安全库存（这是较为保守的估计）
        baseline_holding_units = np.mean(self.historical_mean_demand * 0.5)
        
        # 计算基准成本（单SKU平均成本）
        self._global_baseline_cost = (
            baseline_shortage_units * self.config.shortage_penalty_per_unit +
            baseline_holding_units * self.config.holding_cost_per_unit_weekly
        )
        
        # 加上固定订货成本（平均每个SKU的订货成本）
        self._global_baseline_cost += self.config.fixed_cost_per_order
        
        # 防止除零，并设定一个下限
        self._global_baseline_cost = max(self._global_baseline_cost, 1.0)
        
        logger.info(f"[Baseline Cost] 全局统计基准成本: {self._global_baseline_cost:.2f}")
        logger.info(f"  - 基准缺货量: {baseline_shortage_units:.2f}")
        logger.info(f"  - 基准持有量: {baseline_holding_units:.2f}")
        logger.info(f"  - 历史需求均值: {np.mean(self.historical_mean_demand):.2f}")
        logger.info(f"  - 历史需求标准差: {np.mean(self.historical_std_demand):.2f}")
    
    def _update_demand(self):
        """
        更新当前需求（根据预测方法生成）
        
        核心修正：需求 ≠ 销售量！
        - 需求：独立的客户需求（不受库存限制）
        - 销售量：min(需求, 可用库存)（受库存限制）
        
        对于TFT分位数预测：使用q50（中位数）作为确定性需求
        对于WMA预测：使用WMA预测值作为需求
        
        🔥 关键修正：不再将销售数据直接作为需求！
        真实销售数据只用于：1) 初始化预测器 2) 计算奖励时作为实际销售量
        """
        t = self.current_time
        # 🚨 调试：验证方法是否被调用
        print(f"🔬 [Demand X-Ray] _update_demand() called! t={t}, forecast_method={self.forecast_method}")
        
        # 🚨 架构师"维度 X 光"：在需求生成后立即打印 Shape（调试用）
        log_demand_shape = True  # 设置为 False 可关闭日志
        
        # 统一使用预测方法生成需求（无论是否使用真实数据）
        if self.forecast_method == 'tft_quantile':
            # -------- TFT分位数预测：使用确定性中位数（q50） --------
            if self.demand_forecasts_quantiles is not None and t < self.demand_forecasts_quantiles.shape[0]:
                # 直接使用q50（中位数）作为确定性需求，不使用采样
                self.current_demand = self.demand_forecasts_quantiles[t, :, self.q50_idx].astype(np.float32)
                
                logger.debug(f"[_update_demand] TFT确定性需求（q50）: t={t}, demand_range=[{self.current_demand.min():.1f}, {self.current_demand.max():.1f}]")
            else:
                raise ValueError(
                    f"_update_demand失败：demand_forecasts_quantiles为None或t={t}超出范围！"
                )
        
        elif self.forecast_method == 'real_data':
            # -------- 真实数据模式：直接使用real_sales_data作为需求 --------
            # 注意：这里的"需求"应该是独立的客户需求，不是销售量
            # 但如果没有删失需求估计，只能暂时用销售量代替
            if self.real_sales_data is not None and t < self.real_sales_data.shape[0]:
                # 使用真实销售数据作为需求（第一层门店）
                self.current_demand = self.real_sales_data[t, 0, :].astype(np.float32)
                logger.debug(f"[_update_demand] 真实数据需求: t={t}, demand_range=[{self.current_demand.min():.1f}, {self.current_demand.max():.1f}]")
            else:
                raise ValueError(
                    f"_update_demand失败：real_sales_data为None或t={t}超出范围！"
                )
        
        else:  # forecast_method == 'wma'
            # -------- WMA预测：使用WMA预测值 + 随机噪声 --------
            if self.historical_sales_data is not None and t < self.historical_sales_data.shape[0]:
                # 使用过去4周的历史销量计算WMA预测
                if t >= 4:
                    past_sales = self.historical_sales_data[t-4:t, :]
                else:
                    past_sales = self.historical_sales_data[:t+1, :]
                    if len(past_sales) < 4:
                        padding = np.zeros((4 - len(past_sales), self.num_skus), dtype=np.float32)
                        past_sales = np.concatenate([padding, past_sales], axis=0)
                
                # 计算WMA预测（对每个SKU）
                wma_predictions = np.zeros(self.num_skus, dtype=np.float32)
                for sku_idx in range(self.num_skus):
                    forecaster = self.wma_forecasters[sku_idx]
                    forecaster.fit(past_sales[:, sku_idx])
                    wma_pred = forecaster.predict(horizon=1)[0]
                    wma_predictions[sku_idx] = wma_pred
                
                # 添加随机噪声（模拟预测误差）
                noise = np.random.normal(0, 0.1 * wma_predictions, size=self.num_skus).astype(np.float32)
                self.current_demand = np.maximum(wma_predictions + noise, 0)  # 确保非负
                
                # 🚨 调试：打印需求 Shape（在方法结束前）
                print(f"🔬 [WMA Debug] t={t}, current_demand shape = {self.current_demand.shape}, sum = {np.sum(self.current_demand):.2f}")
                
                logger.debug(f"[_update_demand] WMA预测: t={t}, demand_range=[{self.current_demand.min():.1f}, {self.current_demand.max():.1f}]")
            else:
                raise ValueError(
                    f"_update_demand失败：historical_sales_data为None或t={t}超出范围！"
                )
        
        # 🚨 架构师"维度 X 光"：在需求生成后打印 Shape（调试用）
        if hasattr(self, 'log_demand_shape') and self.log_demand_shape:
            demand_shape = self.current_demand.shape
            demand_sum = np.sum(self.current_demand)
            demand_mean = np.mean(self.current_demand)
            # 使用 print() 确保输出（不受日志级别影响）
            print(f"🔬 [Demand X-Ray] Week {self.current_week_idx}: current_demand shape = {demand_shape}, sum = {demand_sum:.2f}, mean = {demand_mean:.2f}")
            print(f"🔬 [Demand X-Ray]   - self.forecast_method = {self.forecast_method}")
            if hasattr(self, 'store_id'):
                print(f"🔬 [Demand X-Ray]   - self.store_id = {self.store_id}")
    
    def _update_target_inventory(self):
        """
        更新目标库存水平（Base-Stock Level），用于势能塑形
        
        目标库存 = 未来 lead_time 周需求均值 + 安全库存
        安全库存 = 需求标准差 × sqrt(lead_time) × z_score（95% 服务水平对应 z=1.65）
        
        根据预测方法选择不同的需求估算方式
        """
        t = self.current_time
        
        if self.forecast_method == 'tft_quantile':
            # -------- TFT分位数预测 --------
            if self.demand_forecasts_quantiles is not None:
                future_end = min(t + self.lead_time_mean, self.demand_forecasts_quantiles.shape[0])
                if future_end > t:
                    # 提取未来lead_time周的q50 (future_lead_time_weeks, num_skus)
                    future_q50 = self.demand_forecasts_quantiles[t:future_end, :, self.q50_idx]
                    mean_future_demand = np.mean(future_q50, axis=0)
                    
                    # 计算需求标准差（使用q10和q90估算）
                    future_q10 = self.demand_forecasts_quantiles[t:future_end, :, self.q10_idx]
                    future_q90 = self.demand_forecasts_quantiles[t:future_end, :, self.q90_idx]
                    # 近似标准差 = (q90 - q10) / 2.56 (对于正态分布)
                    std_future_demand = np.mean((future_q90 - future_q10) / 2.56, axis=0)
                else:
                    # 如果超出预测范围，使用当前周的q50
                    mean_future_demand = self.demand_forecasts_quantiles[t-1, :, self.q50_idx].copy()
                    std_future_demand = np.ones(self.num_skus, dtype=np.float32) * 5.0
            else:
                raise ValueError("_update_target_inventory失败：demand_forecasts_quantiles为None！")
        
        else:  # forecast_method == 'wma'
            # -------- WMA预测：使用历史销量数据 --------
            if self.historical_sales_data is not None:
                # 使用过去lead_time周的历史销量估算需求均值和标准差
                if t >= self.lead_time_mean:
                    past_demand = self.historical_sales_data[t-self.lead_time_mean:t, :]
                else:
                    # 如果历史数据不足，用可用数据
                    past_demand = self.historical_sales_data[:t+1, :]
                
                mean_future_demand = np.mean(past_demand, axis=0)
                std_future_demand = np.std(past_demand, axis=0) + 1e-6
            else:
                raise ValueError("_update_target_inventory失败：historical_sales_data为None！")
        
        # 安全库存（95% 服务水平，z=1.65）
        safety_stock = 1.65 * std_future_demand * np.sqrt(self.lead_time_mean)
        
        # 目标库存水平
        self.target_inventory_level = mean_future_demand * self.lead_time_mean + safety_stock
        
        # 下限：至少满足 1 天需求
        self.target_inventory_level = np.maximum(self.target_inventory_level, mean_future_demand)
    
    def _execute_sales(self):
        """执行销售（跟踪每步实际销售额）"""
        self._last_step_actual_sales = np.zeros(self.num_skus, dtype=np.float32)
        for i in range(self.num_skus):
            demand = self.current_demand[i]
            
            if self.store_inventory[i] >= demand:
                # 满足需求
                sale = demand
                self.store_inventory[i] -= sale
                self.stockout_events[i, self.episode_steps % self.performance_window] = 0
            else:
                sale = self.store_inventory[i]
                shortage = demand - self.store_inventory[i]
                self.stockout_events[i, self.episode_steps % self.performance_window] = 1
                self.total_stockout += shortage
                self.store_inventory[i] = 0
            
            self._last_step_actual_sales[i] = sale
            self.total_sales_amount += sale
            self._last_order_cost = self._calculate_order_cost()
        
        # ========= 🔥 更新满足率追踪（用于阶梯奖励） =========
        # 计算当前步的满足情况
        step_total_demand = np.sum(self.current_demand)
        step_total_filled = np.sum(self._last_step_actual_sales)
        
        # 更新累积统计
        self.cumulative_filled += step_total_filled
        self.cumulative_demand += step_total_demand
        
        # 计算当前episode的满足率
        if self.cumulative_demand > 1e-5:
            self.current_episode_service_level = self.cumulative_filled / self.cumulative_demand
        else:
            self.current_episode_service_level = 1.0
        
        # 记录每步满足率（用于平滑和调试）
        if step_total_demand > 1e-5:
            step_fill_rate = step_total_filled / step_total_demand
        else:
            step_fill_rate = 1.0
        self.episode_fill_rates.append(step_fill_rate)
        
        # 调试日志（每10步打印一次）
        if self.episode_steps % 10 == 0:
            logger.debug(f"[Service Level] Episode Step {self.episode_steps}: "
                        f"当前步满足率={step_fill_rate:.2%}, "
                        f"累积满足率={self.current_episode_service_level:.2%}, "
                        f"累积满足={self.cumulative_filled:.0f}/{self.cumulative_demand:.0f}")
    
    def _potential(self, inventory: np.ndarray, demand: np.ndarray) -> float:
        """
        势能函数：评估当前状态的"好坏"（越高越好）
        
        使用"区间惩罚"设计（而非单一高目标水位）：
        - 库存在区间 [0.5周需求, 1.5周需求] 内 -> 势能 = 0（最佳）
        - 库存低于0.5周需求 -> 势能 < 0（缺货风险）
        - 库存高于1.5周需求 -> 势能 < 0（过度库存）
        
        Args:
            inventory: 各SKU库存，形状=(num_skus,)
            demand: 本周需求预测，形状=(num_skus,)
        
        Returns:
            potential: 势能值（越高表示状态越好）
        """
        if not self.pbrs_enabled:
            return 0.0
        
        # 定义健康区间：[0.5周需求, 1.5周需求]
        low_bound = demand * 0.5
        high_bound = demand * 1.5
        
        # 只有超出区间才产生负势能（惩罚）
        under_stock = np.maximum(0, low_bound - inventory)
        over_stock = np.maximum(0, inventory - high_bound)
        
        # 应用ABC差异化惩罚权重
        # A类：缺货惩罚更重（乘以惩罚乘数）
        # C类：过度库存惩罚更重（乘以惩罚乘数的倒数）
        under_stock_weighted = under_stock * self.sku_penalty_multiplier * 2.0
        over_stock_weighted = over_stock * (1.0 / self.sku_penalty_multiplier) * 1.0
        
        # 势能 = 负偏差总和（越接近0越好）
        potential = -np.sum(under_stock_weighted + over_stock_weighted)
        
        return float(potential)
    
    def _calculate_reward(self, info: dict = None) -> float:
        """
        重构后的周频财务成本奖励函数（集成动态归一化）
        
        核心思想：Reward = -(持有成本 + 缺货惩罚 + 订货固定成本)
        
        改进点：
        1. 废除 service_bonus：服务水平通过缺货惩罚体现，不缺货是底线
        2. 废除硬裁剪：使用软截断（tanh），保留极端情况下的梯度信号
        3. 动态归一化：使用 RunningRewardNormalizer 进行 Z-score 归一化
        4. 多分量归一化：分别归一化各成本分量，避免量级差异
        
        参数：
            info: 可选，包含额外信息的字典（如当前库存、需求、动作等）
        
        返回：
            归一化后的奖励值（通过tanh软截断在[-10, 10]区间）
        """
        # 更新训练步数
        self.total_training_steps += 1
        
        # ========== 1. 获取当前周的真实业务数据 ==========
        # 当前库存（shape: (num_stores, num_skus) 或 (num_skus,)）
        current_inventory = self.inventory if hasattr(self, 'inventory') else self.store_inventory
        
        # 当前需求
        current_demand = self.current_demand if hasattr(self, 'current_demand') else None
        
        # 当前动作（本周的补货决策量）
        current_action = self.last_action if hasattr(self, 'last_action') else None
        
        # 修复：定义 current_service_level（避免作用域Bug）
        current_service_level = self.current_episode_service_level if hasattr(self, 'current_episode_service_level') else 0.0
        
        # 如果没有需求或动作数据，返回0（不应发生）
        if current_demand is None or current_action is None:
            logger.warning("[Reward] current_demand or current_action is None, returning 0")
            return 0.0
        
        # ========== 2. 计算持有成本 (Holding Cost) - 应用ABC差异化惩罚 ==========
        if hasattr(self, '_last_step_actual_sales'):
            ending_inventory = np.maximum(current_inventory - self._last_step_actual_sales, 0)
        else:
            ending_inventory = np.maximum(current_inventory - current_demand, 0)
        
        # 每个SKU的持有成本
        holding_cost_per_sku = ending_inventory * self.config.holding_cost_per_unit_weekly
        # 应用ABC差异化惩罚：C类积压罚得更重（乘以惩罚乘数的倒数）
        # 注意：self.sku_penalty_multiplier 是缺货惩罚乘数（A类=3.0, B类=1.5, C类=1.0）
        # C类持有成本惩罚 = 1.0 / 1.0 = 1.0（基础）
        # B类持有成本惩罚 = 1.0 / 1.5 ≈ 0.67（较轻）
        # A类持有成本惩罚 = 1.0 / 3.0 ≈ 0.33（最轻）
        holding_cost_weighted = holding_cost_per_sku * (1.0 / self.sku_penalty_multiplier)
        total_holding_cost = np.sum(holding_cost_weighted)
        
        # ========== 3. 计算缺货惩罚 (Shortage Penalty) - 应用ABC差异化惩罚 ==========
        if hasattr(self, '_last_step_actual_sales'):
            unmet_demand = np.maximum(current_demand - self._last_step_actual_sales, 0)
        else:
            unmet_demand = np.maximum(current_demand - current_inventory, 0)
        
        # 🔥 核心修改1：指数级缺货惩罚（让缺货变得极其痛苦）
        # 线性惩罚 + 指数放大（缺货越多，惩罚呈几何级数爆炸）
        shortage_penalty_per_sku = unmet_demand * self.config.shortage_penalty_per_unit
        
        # 添加指数项（缺货^1.5，让大额缺货的惩罚爆炸式增长）
        shortage_exponent = getattr(self.config, 'shortage_exponent', 1.5)
        exponential_penalty = 3.0 * (unmet_demand ** shortage_exponent)
        
        # 总缺货成本 = 线性惩罚 + 指数惩罚
        shortage_cost_per_sku = shortage_penalty_per_sku + exponential_penalty
        
        # 应用ABC差异化惩罚：A类缺货罚得更重（乘以惩罚乘数）
        # A类缺货惩罚 = 3.0倍
        # B类缺货惩罚 = 1.5倍
        # C类缺货惩罚 = 1.0倍（基础）
        shortage_cost_weighted = shortage_cost_per_sku * self.sku_penalty_multiplier
        total_shortage_cost = np.sum(shortage_cost_weighted)
        
        # 调试日志：打印指数惩罚的贡献（每100步打印一次）
        if self.episode_steps % 100 == 0 and np.sum(unmet_demand) > 0:
            logger.info(f"[Exponential Penalty] Step {self.episode_steps}: "
                       f"线性惩罚={np.sum(shortage_penalty_per_sku):.2f}, "
                       f"指数惩罚={np.sum(exponential_penalty):.2f}, "
                       f"总缺货成本={total_shortage_cost:.2f}")
        
        # ========== 4. 计算过度补货惩罚 (Overstock Penalty) - 【暗坑3防御】 ==========
        # 🚨 防止PPO为了刷满足率奖金而"无脑囤货"
        # 逻辑：如果补货后的库存，远超未来4周的预测需求，说明在制造死库存
        if hasattr(self, 'demand_forecast') and self.demand_forecast is not None:
            # 获取未来4周的预测需求
            future_4_weeks_demand = np.sum(self.demand_forecast[:, :4], axis=1)  # shape: (num_skus,)
            # 计算过度补货量（库存超过未来4周需求的1.2倍）
            overstock = np.maximum(0, current_inventory - future_4_weeks_demand * 1.2)
            # 计算过度补货惩罚（重罚死库存）
            overstock_penalty = np.sum(overstock * self.config.holding_cost_per_unit_weekly * self.config.overstock_penalty_multiplier)
        else:
            overstock_penalty = 0.0
        
        # ========== 5. 计算原始奖励 (Raw Reward) ==========
        # 🚨【暗坑1防御】先计算原始奖励，然后整体归一化
        # 移除 fixed_cost（业务修正：一周发一次车是硬规则）
        raw_holding_cost = total_holding_cost
        raw_shortage_cost = total_shortage_cost
        raw_overstock_cost = overstock_penalty
        
        # 基础奖励 = -(持有成本 + 缺货惩罚 + 过度补货惩罚)
        base_reward = -(raw_holding_cost + raw_shortage_cost + raw_overstock_cost)
        
        # ========== 6. 满足率阶梯奖励 (Service Level Bonus) ==========
        sl_bonus = 0.0
        if getattr(self.config, 'enable_service_level_bonus', False):
            # 获取当前Episode的累积满足率
            current_service_level = self.current_episode_service_level
            
            # 从EnvConfig读取阈值和奖励
            threshold_1 = getattr(self.config, 'sl_bonus_threshold_1', 0.90)
            reward_1 = getattr(self.config, 'sl_bonus_reward_1', 30.0)
            threshold_2 = getattr(self.config, 'sl_bonus_threshold_2', 0.95)
            reward_2 = getattr(self.config, 'sl_bonus_reward_2', 150.0)
            penalty_threshold = getattr(self.config, 'sl_penalty_threshold', 0.85)
            penalty_reward = getattr(self.config, 'sl_penalty_reward', -50.0)
            
            # 阶梯判断（悬崖式奖励）
            if current_service_level >= threshold_2:
                sl_bonus = reward_2  # 🔥 95%满足率：巨额奖金（150.0）
                if self.episode_steps % 10 == 0:
                    logger.info(f"[Service Level BONUS] 🎉 达成{threshold_2:.0%}满足率！奖金={sl_bonus:.0f}")
            elif current_service_level >= threshold_1:
                sl_bonus = reward_1  # 90%满足率：中等奖励（30.0）
            elif current_service_level < penalty_threshold:
                sl_bonus = penalty_reward  # 低于85%：重罚（-50.0）
        
        # 最终原始奖励
        raw_reward = base_reward + sl_bonus
        
        # ========== 7. 动态归一化（对最终total reward归一化） ==========
        if self.use_dynamic_normalization and hasattr(self, 'reward_normalizer'):
            # 🚨【暗坑1防御】对最终total reward归一化（而非对分量归一化）
            # 将 raw_reward 减去历史均值，除以历史标准差，确保输入 PPO 的值在 [-3, 3] 左右
            self.reward_normalizer.update(np.array([raw_reward], dtype=np.float32))
            normalized_reward = self.reward_normalizer.normalize(np.array([raw_reward], dtype=np.float32))[0]
            
            # 软截断（保留梯度信号）
            final_reward = np.tanh(normalized_reward / self.config.soft_clip_range) * self.config.soft_clip_range
            
            # 记录调试信息
            if info is None:
                info = {}
            
            info['holding_cost'] = float(raw_holding_cost)
            info['shortage_cost'] = float(raw_shortage_cost)
            info['exponential_penalty'] = float(np.sum(exponential_penalty)) if 'exponential_penalty' in locals() else 0.0
            info['overstock_penalty'] = float(raw_overstock_cost)
            info['sl_bonus'] = float(sl_bonus)
            info['current_sl'] = float(current_service_level)
            info['raw_reward'] = float(raw_reward)
            info['normalized_reward'] = float(normalized_reward)
            info['final_reward'] = float(final_reward)
            
            # 调试日志
            if self.total_training_steps % 1000 == 0:
                logger.info(f"[Reward] Step {self.total_training_steps}: "
                           f"holding={raw_holding_cost:.2f}, shortage={raw_shortage_cost:.2f}, "
                           f"overstock={raw_overstock_cost:.2f}, sl_bonus={sl_bonus:.0f}, "
                           f"raw={raw_reward:.2f}, normalized={normalized_reward:.4f}, final={final_reward:.4f}")
            
        else:
            # 降级方案：使用固定缩放因子
            # 🚨 注意：即使降级，也要对raw_reward进行缩放（而非对分量）
            scale_factor = 100.0  # 增大缩放因子，应对sl_bonus的大数值（150.0）
            normalized_reward = raw_reward / scale_factor
            final_reward = np.tanh(normalized_reward) * self.config.soft_clip_range
            
            if info is None:
                info = {}
            
            info['holding_cost'] = float(raw_holding_cost)
            info['shortage_cost'] = float(raw_shortage_cost)
            info['exponential_penalty'] = float(np.sum(exponential_penalty)) if 'exponential_penalty' in locals() else 0.0
            info['overstock_penalty'] = float(raw_overstock_cost)
            info['sl_bonus'] = float(sl_bonus)
            info['current_sl'] = float(current_service_level)
            info['raw_reward'] = float(raw_reward)
            info['normalized_reward'] = float(normalized_reward)
            info['final_reward'] = float(final_reward)
        
        # ========== 8. 定期记录奖励分解（调试用） ==========
        if self.total_training_steps % 1000 == 0:
            logger.info(f"[Reward] Step {self.total_training_steps}: "
                       f"holding={raw_holding_cost:.2f}, shortage={raw_shortage_cost:.2f}, "
                       f"overstock={raw_overstock_cost:.2f}, sl_bonus={sl_bonus:.0f}, "
                       f"raw={raw_reward:.2f}, final={final_reward:.4f}")
        
        return float(final_reward)
    
    def _calculate_stockout_penalty_piecewise(self) -> float:
        """
        分段缺货惩罚
        
        零售药店行业目标 SOR 约 5-10%：
        - SOR < 5%: 不惩罚（已达到目标服务水平）
        - 5% ≤ SOR < 15%: 适中惩罚（6×sor）
        - SOR ≥ 15%: 陡增惩罚（15×超出门槛）
        
        Returns:
            float: 缺货惩罚值
        """
        penalty = 0.0
        
        for i in range(self.num_skus):
            sor_i = float(self.stockout_history[i])
            
            if sor_i < 0.05:
                # 目标服务水平内，不惩罚
                continue
            elif sor_i < 0.15:
                # 适中惩罚
                penalty += 6.0 * sor_i
            else:
                # 陡增惩罚
                penalty += 6.0 * 0.15 + 15.0 * (sor_i - 0.15)
        
        return penalty / self.num_skus
    
    def _calculate_order_cost(self) -> float:
        """
        计算订购固定成本（新增：A4）
        
        每次下单的固定成本，防止频繁少量订购。
        只要某SKU的补货量超过最小阈值，就计为一次订购。
        
        Returns:
            float: 订购成本
        """
        active_orders = np.sum(self.last_action > self.min_reorder_threshold)
        order_cost = active_orders * self.config.order_cost / self.num_skus
        return order_cost
    
    def _calculate_low_stock_penalty(self) -> float:
        """
        低库存预警惩罚（基于有效库存 = 当前库存 + 在途订单）
        
        当有效库存不足以覆盖 lead_time 期间需求时给予惩罚，
        促使 Agent 在库存耗尽前提前补货。
        
        Returns:
            float: 低库存惩罚值
        """
        penalty = 0.0
        threshold_weeks = float(self.lead_time_mean)
        
        if self.demand_forecasts is not None and self.current_time < len(self.demand_forecasts):
            future_end = min(self.current_time + self.lead_time_mean, len(self.demand_forecasts))
            if future_end > self.current_time:
                future_demand = self.demand_forecasts[self.current_time:future_end]
                mean_future_demand = np.mean(future_demand, axis=0)
            else:
                mean_future_demand = self.current_demand.copy()
        else:
            mean_future_demand = self.current_demand.copy()
        
        threshold_inventory = mean_future_demand * threshold_weeks
        
        # 有效库存 = 门店库存 + 在途订单
        in_transit = np.sum(self.on_order, axis=1)
        effective_inventory = self.store_inventory + in_transit
        
        low_stock_ratio = np.maximum(threshold_inventory - effective_inventory, 0.0) / (threshold_inventory + 1e-6)
        penalty = float(np.mean(low_stock_ratio))
        
        return penalty
    
    def _get_stockout_penalty_coef(self, sku_id: int) -> float:
        """获取缺货惩罚系数"""
        # 急救药品惩罚更高
        if sku_id % 50 < 10:  # 假设前10%是关键药品
            return self.stockout_penalty_critical
        return self.stockout_penalty_base
    
    def _calculate_holding_cost(self) -> float:
        """计算持有成本（归一化到合理范围）"""
        # 门店库存持有成本
        total_inventory = np.sum(self.store_inventory)
        holding_cost = total_inventory * self.config.holding_cost_rate / self.num_skus
        self.total_holding_cost += holding_cost
        return holding_cost
    
    def _calculate_expiry_loss(self) -> float:
        """计算效期损耗惩罚"""
        expiry_loss = 0.0
        
        for i in range(self.num_skus):
            # 假设有部分库存接近效期
            expiry_ratio = self._get_expiry_ratio(i)
            if expiry_ratio < 0.25:  # 剩余有效期不足12周
                expiry_loss += expiry_ratio * self.store_inventory[i]
        
        self.total_expiry_loss += expiry_loss
        return expiry_loss / self.num_skus
    
    def _get_expiry_ratio(self, sku_id: int) -> float:
        """获取效期比例（基于初始化值，随时间递减）"""
        return float(self._expiry_ratios[sku_id])

    def _decay_expiry_ratios(self):
        """每步衰减效期比例，模拟药品接近效期"""
        self._expiry_ratios -= self._expiry_decay_rate
        self._expiry_ratios = np.clip(self._expiry_ratios, 0.0, 1.0)
    
    def _calculate_critical_drug_reward(self) -> float:
        """
        关键药品保障：只在库存无法覆盖接下来预期需求时给予惩罚。
        避免奖励单纯的高库存，引导"刚好够"的策略。
        
        Returns:
            float: 惩罚值（负值）
        """
        penalty = 0.0
        for i in range(self.num_skus):
            if self._is_critical_drug(i):
                # 预期每周需求（至少为1，避免除零）
                weekly_demand = self.current_demand[i] if self.current_demand[i] > 0 else 1.0
                # 库存能支撑的周数
                weeks_of_stock = self.store_inventory[i] / weekly_demand
                # 若库存不足3周需求，线性增加惩罚
                if weeks_of_stock < 3:
                    shortage = max(0.0, 3 - weeks_of_stock)
                    penalty += shortage * 2.0   # 调节惩罚强度
        # 归一化并返回负值（惩罚）
        return -penalty / max(1, self.num_skus * 0.2)
    
    def _is_critical_drug(self, sku_id: int) -> bool:
        """判断是否关键药品"""
        return sku_id % 50 < 10  # 简化判断
    
    def _calculate_ofr_reward(self) -> float:
        """
        订单满足率（OFR）奖励
        
        计算当前步的订单满足情况，鼓励 Agent 提高满足率。
        OFR = 实际销售 / 需求
        """
        if hasattr(self, '_last_step_actual_sales') and hasattr(self, 'current_demand'):
            actual_sales = self._last_step_actual_sales
            demand = self.current_demand
            demand_safe = demand + 1e-6
            ofr_per_sku = np.minimum(actual_sales / demand_safe, 1.0)
            return float(np.mean(ofr_per_sku))
        return 0.0
    
    def _calculate_itr_reward(self) -> float:
        """
        库存周转率（ITR）奖励
        
        鼓励 Agent 保持适度的库存周转率，
        使用 sigmoid 函数以目标 ITR=10 为中心。
        """
        if hasattr(self, '_last_step_actual_sales') and hasattr(self, 'store_inventory'):
            actual_sales = self._last_step_actual_sales
            inventory = self.store_inventory
            inventory_safe = inventory + 1e-6
            itr_per_sku = actual_sales / inventory_safe
            target_itr = 10.0
            itr_normalized = itr_per_sku / target_itr
            itr_reward = 2.0 / (1.0 + np.exp(-itr_normalized)) - 1.0  # 范围 [-1, 1]
            return float(np.mean(itr_reward))
        return 0.0
    
    def _calculate_sor_penalty(self) -> float:
        """
        缺货率（SOR）惩罚
        
        返回负值，惩罚缺货行为，迫使 Agent 降低缺货率。
        """
        if hasattr(self, 'current_demand') and hasattr(self, '_last_step_actual_sales'):
            demand = self.current_demand
            actual_sales = self._last_step_actual_sales
            stockout = np.maximum(demand - actual_sales, 0.0)
            stockout_ratio = stockout / (demand + 1e-6)
            return -float(np.mean(stockout_ratio))  # 返回负值（惩罚）
        return 0.0
    
    def _calculate_overstock_penalty(self) -> float:
        """计算超容惩罚"""
        penalty = 0.0
        
        for i in range(self.num_skus):
            if self.store_inventory[i] > self.config.store_capacity:
                overflow = self.store_inventory[i] - self.config.store_capacity
                penalty += overflow * 0.5  # 超容惩罚系数
        
        return penalty / self.num_skus
    
    def _update_performance(self):
        """更新历史绩效 - 使用滚动窗口"""
        # 计算当前步的缺货标志
        current_stockout = (self.store_inventory < self.current_demand).astype(np.float32)
        
        # 计算当前步各SKU周转率 = min(销售量, 库存) / (库存 + 1)
        sales = np.minimum(self.store_inventory, self.current_demand)
        turnover = sales / (self.store_inventory + 1.0)
        
        # 滚动窗口更新
        window = self.performance_window
        
        # 将当前步的数据插入缓冲区（简化：使用循环缓冲区）
        idx = self.episode_steps % window
        
        # 更新缺货事件缓冲区
        self.stockout_events[:, idx] = current_stockout
        
        # 更新周转率缓冲区
        self.turnover_buffer[:, idx] = turnover
        
        # 更新状态中使用滚动均值（对所有SKU取平均后扩展到每个SKU）
        # 这里简化：每个SKU使用自己的历史均值
        self.stockout_history = np.mean(self.stockout_events, axis=1)
        self.turnover_history = np.mean(self.turnover_buffer, axis=1)
        
        # 仓库补货：每隔 lead_time 周按每个SKU的需求比例补充中心仓
        if self.episode_steps > 0 and self.episode_steps % self.lead_time_mean == 0:
            # 按每个SKU的需求量按比例分配，补充量 = 周均需求 × lead_time × 安全系数(2.0)
            replenish_per_sku = self._mean_demand_per_sku * self.lead_time_mean * 2.0
            # 不超过仓库剩余容量
            remaining_capacity = self.config.warehouse_capacity - self.warehouse_inventory
            actual_replenish = np.minimum(replenish_per_sku, np.maximum(remaining_capacity, 0))
            self.warehouse_inventory += actual_replenish
            self.warehouse_inventory = np.clip(self.warehouse_inventory, 0, self.config.warehouse_capacity)
    
    def get_inventory_state(self) -> np.ndarray:
        """
        获取当前库存状态（归一化）
        
        Returns:
            np.ndarray: 归一化后的库存状态，形状 (num_skus,)
        """
        inventory_state = self.store_inventory.copy() / self.config.store_capacity
        return np.clip(inventory_state, 0.0, 1.0).astype(np.float32)
    
    def get_warehouse_state(self) -> np.ndarray:
        """
        获取当前仓库状态
        
        Returns:
            np.ndarray: 仓库状态数组，包含仓库库存和交货周期
        """
        # 返回仓库库存（归一化）和交货周期
        warehouse_inventory_normalized = self.warehouse_inventory.copy() / self.config.warehouse_capacity
        warehouse_inventory_normalized = np.clip(warehouse_inventory_normalized, 0.0, 1.0)
        
        # 交货周期（归一化）
        lead_time_normalized = np.array([self.lead_time_mean / 14.0], dtype=np.float32)  # 假设最大交货周期为14周
        
        # 返回连接后的数组
        return np.concatenate([warehouse_inventory_normalized, lead_time_normalized])
    
    def _get_state(self) -> np.ndarray:
        """
        获取当前状态（支持TFT分位数预测和WMA预测）
        
        状态空间组成（特征解耦）：
        1. 需求侧特征：下一周的预测 (3 × num_skus,)
           - TFT分位数预测：expected_demand, uncertainty, skewness
           - WMA预测：expected_demand, 0, 0（WMA没有不确定性估计）
        2. 供给侧特征（PPO内部状态）：
           - 当前库存 (num_skus,)
           - 在途库存 (num_skus,)
           - 距上次下单周数 (num_skus,)
           - 过去3周实际销量 (3 × num_skus,)
           - 过去3周缺货量 (3 × num_skus,)
        3. 时间特征：星期几的sin/cos编码 + 距离下次补货倒计时 (3,)
        
        归一化策略：
        - 手动归一化：将不同量级的特征缩放到相似范围
        - 建议使用VecNormalize进行二次归一化（Z-score）
        
        Returns:
            np.ndarray: 状态向量 (11 × num_skus + 3,)
        """
        # ========== 1. 需求侧特征 ==========
        # 根据预测方法提取需求特征
        if self.forecast_method == 'tft_quantile':
            # -------- TFT分位数预测 --------
            # 只看下一周（第t+1周）的预测
            t = self.current_time
            next_week = min(t + 1, self.demand_forecasts_quantiles.shape[0] - 1)
            
            # 提取分位数 (num_skus,)
            q10 = self.demand_forecasts_quantiles[next_week, :, self.q10_idx]
            q50 = self.demand_forecasts_quantiles[next_week, :, self.q50_idx]
            q90 = self.demand_forecasts_quantiles[next_week, :, self.q90_idx]
            
            # 特征工程提取
            expected_demand = q50  # 预期需求（中位数）
            uncertainty = q90 - q10  # 需求不确定性
            skewness = (q90 - q50) - (q50 - q10)  # 分布偏度
            
        elif self.forecast_method == 'real_data':
            # -------- 真实数据 --------
            # 🔥 使用真实的销售数据作为需求特征
            t = self.current_time
            
            # 使用当前周的真实销售数据作为预期需求
            if t < self.real_sales_data.shape[0]:
                expected_demand = self.real_sales_data[t, 0, :].astype(np.float32)  # (num_skus,)
            else:
                # 如果超出范围，使用最后一周的数据
                expected_demand = self.real_sales_data[-1, 0, :].astype(np.float32)
            
            # 真实数据没有不确定性估计，设为0
            uncertainty = np.zeros(self.num_skus, dtype=np.float32)
            skewness = np.zeros(self.num_skus, dtype=np.float32)
            
        else:  # forecast_method == 'wma'
            # -------- WMA预测 --------
            t = self.current_time
            
            # 使用过去4周的历史销量计算WMA预测
            if t >= 4:
                # 提取过去4周的销量 (4, num_skus)
                past_sales = self.historical_sales_data[t-4:t, :]
            else:
                # 如果历史数据不足4周，用可用数据
                past_sales = self.historical_sales_data[:t+1, :]
                if len(past_sales) < 4:
                    # 用0填充到4周
                    padding = np.zeros((4 - len(past_sales), self.num_skus), dtype=np.float32)
                    past_sales = np.concatenate([padding, past_sales], axis=0)
            
            # 计算WMA预测（对每个SKU）
            expected_demand = np.zeros(self.num_skus, dtype=np.float32)
            for sku_idx in range(self.num_skus):
                # 使用WeightedMovingAverageForecaster预测
                forecaster = self.wma_forecasters[sku_idx]
                forecaster.fit(past_sales[:, sku_idx])
                wma_pred = forecaster.predict(horizon=1)[0]  # 预测下一周
                expected_demand[sku_idx] = wma_pred
            
            # WMA没有不确定性估计，设为0
            uncertainty = np.zeros(self.num_skus, dtype=np.float32)
            skewness = np.zeros(self.num_skus, dtype=np.float32)
        
        # 拼接特征: (3 × num_skus,)
        demand_features = np.concatenate([expected_demand, uncertainty, skewness])
        
        # 手动归一化：除以最大可能需求（防止量级爆炸）
        max_possible_demand = self._max_weekly_demand if hasattr(self, '_max_weekly_demand') and self._max_weekly_demand > 0 else 50.0
        demand_features = demand_features / max_possible_demand
        demand_features = np.clip(demand_features, 0.0, 10.0)  # 裁剪到 [0, 10]
        
        # ========== 2. 供给侧特征（PPO内部状态） ==========
        # 特征1：当前库存 (num_skus,)
        inventory_state = self.store_inventory.copy() / self.config.store_capacity
        inventory_state = np.clip(inventory_state, 0.0, 1.0)
        
        # 特征2：在途库存 (num_skus,)
        in_transit = np.sum(self.on_order, axis=1)
        in_transit_normalized = in_transit / self.config.store_capacity
        in_transit_normalized = np.clip(in_transit_normalized, 0.0, 1.0)
        
        # 特征3：距上次下单周数 (num_skus,)
        if hasattr(self, 'weeks_since_last_order'):
            days_since_order = self.weeks_since_last_order.copy().astype(np.float32)
            # 归一化：除以最大可能值（例如30周）
            days_since_order = days_since_order / 30.0
            days_since_order = np.clip(days_since_order, 0.0, 1.0)
        else:
            days_since_order = np.zeros(self.num_skus, dtype=np.float32)
        
        # 特征4：过去3周实际销量 (3 × num_skus,)
        if hasattr(self, 'sales_history_weekly'):
            # 取最近3周的数据（最后3行）
            recent_sales = self.sales_history_weekly[-min(3, self.sales_history_weekly.shape[0]):, :]
            # 如果不足3周，用0填充
            if recent_sales.shape[0] < 3:
                padding = np.zeros((3 - recent_sales.shape[0], self.num_skus), dtype=np.float32)
                recent_sales = np.concatenate([padding, recent_sales], axis=0)
            # 展平为1维数组
            recent_sales_flat = recent_sales.flatten()
            # 归一化：除以最大可能销量
            recent_sales_flat = recent_sales_flat / max_possible_demand
            recent_sales_flat = np.clip(recent_sales_flat, 0.0, 10.0)
        else:
            recent_sales_flat = np.zeros(3 * self.num_skus, dtype=np.float32)
        
        # 特征5：过去3周缺货量 (3 × num_skus,)
        if hasattr(self, 'stockout_history_weekly'):
            # 取最近3周的数据
            recent_stockouts = self.stockout_history_weekly[:min(3, self.stockout_history_weekly.shape[0]), :]
            # 如果不足3周，用0填充
            if recent_stockouts.shape[0] < 3:
                padding = np.zeros((3 - recent_stockouts.shape[0], self.num_skus), dtype=np.float32)
                recent_stockouts = np.concatenate([padding, recent_stockouts], axis=0)
            # 展平为1维数组
            recent_stockouts_flat = recent_stockouts.flatten()
            # 归一化：除以最大可能缺货量
            recent_stockouts_flat = recent_stockouts_flat / max_possible_demand
            recent_stockouts_flat = np.clip(recent_stockouts_flat, 0.0, 10.0)
        else:
            recent_stockouts_flat = np.zeros(3 * self.num_skus, dtype=np.float32)
        
        # ========== 3. 时间特征 ==========
        # 核心特征1：周期性时间编码 (Cyclical Time Encoding)
        week_of_year = self.current_time % 7
        sin_week = np.array([np.sin(2 * np.pi * week_of_year / 7)], dtype=np.float32)
        cos_week = np.array([np.cos(2 * np.pi * week_of_year / 7)], dtype=np.float32)
        
        # 核心特征2：距离下次补货的倒计时 (Days until next replenishment)
        weeks_to_next_order = (7 - week_of_year) % 7
        norm_weeks_to_order = np.array([weeks_to_next_order / 7.0], dtype=np.float32)
        
        time_features = np.concatenate([sin_week, cos_week, norm_weeks_to_order])
        
        # ========== 4. 拼接所有特征（使用预分配 buffer） ==========
        # 🔧 修复：使用预分配的 _state_buffer，避免重复拼接
        idx = 0
        
        # 1. 下一周TFT分位数特征 (3 × num_skus,)
        end_idx = idx + demand_features.shape[0]
        self._state_buffer[idx:end_idx] = demand_features
        idx = end_idx
        
        # 2. 当前库存 (num_skus,)
        end_idx = idx + inventory_state.shape[0]
        self._state_buffer[idx:end_idx] = inventory_state
        idx = end_idx
        
        # 3. 在途库存 (num_skus,)
        end_idx = idx + in_transit_normalized.shape[0]
        self._state_buffer[idx:end_idx] = in_transit_normalized
        idx = end_idx
        
        # 4. 距上次下单周数 (num_skus,)
        end_idx = idx + days_since_order.shape[0]
        self._state_buffer[idx:end_idx] = days_since_order
        idx = end_idx
        
        # 5. 过去3周销量 (3 × num_skus,)
        end_idx = idx + recent_sales_flat.shape[0]
        self._state_buffer[idx:end_idx] = recent_sales_flat
        idx = end_idx
        
        # 6. 过去3周缺货量 (3 × num_skus,)
        end_idx = idx + recent_stockouts_flat.shape[0]
        self._state_buffer[idx:end_idx] = recent_stockouts_flat
        idx = end_idx
        
        # 7. 时间特征 (3,)
        end_idx = idx + time_features.shape[0]
        self._state_buffer[idx:end_idx] = time_features
        idx = end_idx
        
        # 使用 buffer 作为 state（返回副本，防止引用污染）
        state = self._state_buffer.copy()
        
        # ========== 5. 归一化（Z-score） ==========
        # 计算滚动均值和标准差（使用RunningStatistics）
        if not hasattr(self, 'state_running_mean'):
            # 初始化滚动统计量
            self.state_running_mean = state.copy()  # 第一次使用当前状态初始化
            self.state_running_std = np.ones_like(state)
            self.state_count = 1
            # 🔧 修复：第一次调用时，不进行Z-score归一化，直接返回state
            return state.astype(np.float32)
        else:
            # 更新滚动统计量（指数移动平均）
            alpha = 0.01  # 学习率
            self.state_running_mean = (1 - alpha) * self.state_running_mean + alpha * state
            self.state_running_std = (1 - alpha) * self.state_running_std + alpha * np.abs(state - self.state_running_mean)
            self.state_count += 1
        
        # Z-score归一化
        state_normalized = (state - self.state_running_mean) / (self.state_running_std + 1e-8)
        
        # 裁剪到合理范围（防止异常值）
        state_normalized = np.clip(state_normalized, -10.0, 10.0)
        
        return state_normalized.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """获取额外信息（含 OFR 基于补货前仓库状态计算）"""
        # OFR 基于实际发货比例 (delivered / requested)
        # 使用补货前的仓库快照计算
        requested = self._step_requested
        delivered = self._step_delivered
        # 计算 OFR：有请求的 SKU 中，实际发货比例
        active_mask = requested > self.min_reorder_threshold
        if np.any(active_mask):
            ofr_step = float(np.mean(delivered[active_mask] / (requested[active_mask] + 1e-6)))
            ofr_step = min(ofr_step, 1.0)  # 上限截断
        else:
            ofr_step = 0.0  # 没有请求则视为不满足（惩罚不订购行为）
        
        orders_placed = int(np.sum(active_mask))
        orders_filled = int(np.sum(
            active_mask & (delivered >= requested * 0.95)  # 95%以上算满足
        ))
        
        return {
            'time': self.current_time,
            'total_inventory': float(np.sum(self.store_inventory)),
            'total_stockout': float(self.total_stockout),
            'total_holding_cost': float(self.total_holding_cost),
            'total_expiry_loss': float(self.total_expiry_loss),
            'episode_reward': float(self.total_reward),
            'stockout_rate': float(np.mean(self.stockout_history)),
            'turnover_rate': float(np.mean(self.turnover_history)),
            'total_sales_amount': float(self.total_sales_amount),
            'orders_placed': orders_placed,
            'orders_filled': orders_filled,
            'ofr_step': ofr_step,
            'warehouse_total': float(np.sum(self.warehouse_inventory)),
            # 🩹 修复：添加 total_demand 字段，用于计算满足率
            'total_demand': float(self.cumulative_demand),
            'total_filled': float(self.cumulative_filled),
            'episode_fill_rate': float(self.current_episode_service_level),
        }
    
    def get_today_demand(self) -> np.ndarray:
        """
        获取当前步骤的确定性需求（用于随机评估框架）
        
        返回:
            current_demand: 当前需求数组 (num_skus,)
        """
        return self.current_demand.copy()
    
    def set_today_demand(self, new_demand: np.ndarray):
        """
        设置当前步骤的需求（用于随机评估框架）
        
        参数:
            new_demand: 新的需求数组 (num_skus,)
        """
        if len(new_demand) != self.num_skus:
            raise ValueError(f"需求数组长度不匹配: {len(new_demand)} != {self.num_skus}")
        
        # 更新当前需求
        self.current_demand = np.array(new_demand, dtype=np.float32)
        
        # 记录原始需求（用于后续分析）
        if not hasattr(self, 'original_demand'):
            self.original_demand = {}
        self.original_demand[self.current_time] = self.current_demand.copy()
    
    def render(self, mode: str = 'human'):
        """渲染环境（可视化接口）"""
        if mode == 'human':
            info = self._get_info()
            logger.info(f"时间步 {info['time']}: "
                  f"库存={info['total_inventory']:.0f}, "
                  f"缺货={info['total_stockout']:.0f}, "
                  f"奖励={info['episode_reward']:.2f}")
    
    def close(self):
        """关闭环境"""
        pass
