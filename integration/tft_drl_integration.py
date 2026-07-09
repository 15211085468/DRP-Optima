"""
TFT-DRL端到端集成模块 - integration/tft_drl_integration.py
连接时序预测和强化学习优化

核心设计原则（来自用户方案）：
1. 特征空间解耦：TFT负责"看周"（预测外部需求），PPO负责"看地"（感知内部库存并做决策）
2. TFT推理包装器：屏蔽训练时和推理时的数据格式差异
3. PPO环境动态拼接：将TFT预测结果与库存状态拼接成State
4. 归一化：对State进行Z-score归一化，处理预测值和库存值的量级差异
5. TFT预测缓存：每个时间步调用一次TFT预测，缓存结果
6. 冷启动问题：用0或历史平均值Padding
"""

import numpy as np
import torch
import traceback
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass, field

# 使用绝对导入路径，避免 models/forecasting/__init__.py 中的 try-except 将 TFTForecaster 设置为 None
from models.forecasting.tft_model import TFTForecaster, TFTModelConfig
# 注释：models.rl 中没有 PPOAgent，暂时注释
# from models.rl import PPOAgent
from models.optimization import SupplyChainEnv, ConstraintHandler
from config.exceptions import DataError
from utils.logger import get_logger
from stable_baselines3 import PPO as PPOAgent  # 使用 stable-baselines3 的 PPO
logger = get_logger(__name__)


@dataclass
class TFTInferenceWrapper:
    """
    TFT推理包装器
    
    核心功能：
    1. 屏蔽TFT训练时和推理时的数据格式差异
    2. 从环境状态中提取历史数据（补齐窗口）
    3. 补全TFT训练时必需但实时数据没有的"静态列"和"时间列"
    4. 注入实时特征（如当周的价格、促销）
    5. 转换为TFT要求的格式并预测
    6. 缓存预测结果（每周调用一次）
    
    设计原则：
    - TFT输入：只包含需求侧特征（历史销量、时间特征、促销活动）
    - 不包含库存信息（预测的是"潜在真实需求"，而非"受库存限制的观测销量"）
    """
    
    tft_forecaster: TFTForecaster
    max_encoder_length: int = 30  # 历史窗口长度（默认30周）
    forecast_horizon: int = 7  # 预测视界（默认N周，由环境配置决定）
    use_uncertainty: bool = True  # 是否使用不确定性估计
    uncertainty_quantile: float = 0.9  # 使用的分位数（保守预测）
    
    # 预测缓存（每周调用一次，缓存结果）
    forecast_cache: Dict[int, np.ndarray] = field(default_factory=dict)  # key: week_idx, value: (forecast_horizon, num_skus) array
    last_cache_week: int = -1  # 上次缓存的周索引
    
    # 静态特征映射（从TFT训练时保存）
    static_mappings: Optional[Dict] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保TFT模型已训练
        if not self.tft_forecaster._is_trained:
            logger.warning("TFT model not trained! Predictions may be inaccurate.")
        
        # 从TFT模型配置中获取max_encoder_length
        if hasattr(self.tft_forecaster, 'model_config'):
            self.max_encoder_length = self.tft_forecaster.model_config.max_encoder_length
    
    def predict_weekly(self, 
                      current_time: int,
                      env_state: Dict[str, Any],
                      num_skus: int) -> np.ndarray:
        """
        预测未来一周的需求（每周调用一次，缓存结果）
        
        Args:
            current_time: 当前时间步（周）
            env_state: 环境状态（包含历史销量、库存等）
            num_skus: SKU数量
            
        Returns:
            np.ndarray: 未来N周的预测需求，形状 (forecast_horizon, num_skus)
        """
        # 计算当前是第几周
        week_idx = current_time // 7
        
        # 如果缓存中有，直接返回
        if week_idx in self.forecast_cache:
            logger.debug(f"TFT预测缓存命中: week={week_idx}")
            return self.forecast_cache[week_idx]
        
        logger.info(f"TFT预测缓存未命中，执行预测: week={week_idx}, current_time={current_time}")
        
        # 1. 从环境状态中提取历史数据（补齐窗口）
        history_df = self._extract_history_data(env_state, num_skus, current_time)
        
        # 2. 补全TFT训练时必需的特征列
        history_df = self._augment_features(history_df, current_time, num_skus)
        
        # 3. 调用TFT模型进行预测
        if self.use_uncertainty and self.tft_forecaster.model_config.loss_type == "quantile":
            # 使用分位数预测（不确定性估计）
            predictions = self.tft_forecaster.predict_with_uncertainty(history_df, n_samples=100)
            
            # 使用保守预测（上分位数）
            if self.uncertainty_quantile <= 0.5:
                demand_forecast = predictions['q10']  # 乐观预测
            else:
                demand_forecast = predictions['q90']  # 保守预测
        else:
            # 使用简单预测
            predictions = self.tft_forecaster.predict(df=history_df)
            
            # predictions可能是tensor或numpy array
            if isinstance(predictions, torch.Tensor):
                demand_forecast = predictions.cpu().numpy()
            elif isinstance(predictions, np.ndarray):
                demand_forecast = predictions
            else:
                # 如果返回的是列表，取第一个元素
                if isinstance(predictions, list) and len(predictions) > 0:
                    demand_forecast = predictions[0]
                else:
                    raise DataError(f"Unexpected prediction type: {type(predictions)}")
        
        # 【调试】打印预测统计信息
        logger.info(f"[DEBUG] TFT预测统计: shape={demand_forecast.shape}, mean={np.mean(demand_forecast):.2f}, min={np.min(demand_forecast):.2f}, max={np.max(demand_forecast):.2f}")
        
        # 【新增】检查预测值是否全为0
        if np.all(demand_forecast == 0):
            logger.warning(f"[WARNING] TFT预测值全为0！可能是输入数据问题或模型未训练好")
            logger.warning(f"  - 输入数据 shape: {history_df.shape}")
            logger.warning(f"  - 输入数据 sale_qty mean: {history_df['sale_qty'].mean():.2f}")
        
        # 4. 确保预测结果形状正确 (forecast_horizon, num_skus)
        if len(demand_forecast.shape) == 1:
            # 如果是一维数组，假设是单个时间步的预测
            demand_forecast = demand_forecast.reshape(1, -1)
        
        if demand_forecast.shape[0] < self.forecast_horizon:
            # 如果预测长度不足，用最后一个值填充
            last_row = demand_forecast[-1:, :]
            padding = np.repeat(last_row, self.forecast_horizon - demand_forecast.shape[0], axis=0)
            demand_forecast = np.concatenate([demand_forecast, padding], axis=0)
        elif demand_forecast.shape[0] > self.forecast_horizon:
            # 如果预测长度超出，截断
            demand_forecast = demand_forecast[:self.forecast_horizon, :]
        
        # 5. 缓存预测结果
        self.forecast_cache[week_idx] = demand_forecast
        self.last_cache_week = week_idx
        
        # 6. 清理旧缓存（只保留最近4周的缓存）
        weeks_to_keep = [week_idx - i for i in range(4) if week_idx - i >= 0]
        self.forecast_cache = {k: v for k, v in self.forecast_cache.items() if k in weeks_to_keep}
        
        logger.info(f"✓ TFT预测完成: shape={demand_forecast.shape}, mean={np.mean(demand_forecast):.1f}")
        
        return demand_forecast.astype(np.float32)
    
    def _extract_history_data(self, 
                            env_state: Dict[str, Any], 
                            num_skus: int,
                            current_time: int) -> pd.DataFrame:
        """
        从环境状态中提取历史数据（补齐窗口）
        
        核心逻辑：
        - TFT需要max_encoder_length周历史数据作为编码器输入
        - 从env_state中提取历史销量数据
        - 如果历史数据不足，用0填充
        
        Args:
            env_state: 环境状态（包含历史销量、门店代码、商品代码、价格等）
            num_skus: SKU数量
            current_time: 当前时间步
            
        Returns:
            pd.DataFrame: 历史数据，包含TFT所需的特征列（与训练时格式一致）
        """
        # 【修复】从env_state中提取历史销量数据
        # 注意：env_state 可能是 numpy 数组或字典
        sales_history = None
        
        if isinstance(env_state, dict):
            # 如果 env_state 是字典，尝试获取 sales_history
            sales_history = env_state.get('sales_history', None)
        
        if sales_history is None:
            # 【修复】如果仍然为 None，使用模拟的历史数据（基于当前需求）
            # 这是一个临时方案，实际应该从环境的历史记录中获取
            logger.warning("No sales history found in env_state, using simulated history based on current demand")
            
            # 获取当前需求作为历史数据的参考
            current_demand = env_state.get('current_demand', None) if isinstance(env_state, dict) else None
            
            if current_demand is None:
                # 如果无法获取当前需求，使用随机数据
                sales_history = np.random.uniform(10, 50, size=(self.max_encoder_length, num_skus)).astype(np.float32)
            else:
                # 使用当前需求 ± 随机扰动作为历史数据
                sales_history = np.tile(current_demand.copy(), (self.max_encoder_length, 1))
                noise = np.random.uniform(0.8, 1.2, size=sales_history.shape)
                sales_history = sales_history * noise
            
            logger.info(f"  - 模拟历史数据 shape: {sales_history.shape}, mean: {np.mean(sales_history):.1f}")
        
        # 确保历史数据长度足够
        if len(sales_history.shape) == 1:
            sales_history = sales_history.reshape(1, -1)
        
        if sales_history.shape[0] < self.max_encoder_length:
            # 用0填充
            padding = np.zeros((self.max_encoder_length - sales_history.shape[0], num_skus))
            sales_history = np.concatenate([padding, sales_history], axis=0)
        
        # 只取最近max_encoder_length周的数据
        sales_history = sales_history[-self.max_encoder_length:, :]
        
        # 从env_state中获取其他信息
        shop_codes = env_state.get('shop_codes', ['SHOP_001'] * num_skus)
        goods_codes = env_state.get('goods_codes', [f'GOODS_{i:03d}' for i in range(num_skus)])
        prices = env_state.get('prices', np.ones(num_skus) * 10.0)  # 默认价格=10
        
        # 构造DataFrame（长格式：每行是一个(time_idx, sku_idx)对）
        rows = []
        for week_idx in range(self.max_encoder_length):
            for sku_idx in range(num_skus):
                row = {
                    'time_idx': week_idx,
                    'shop_code': shop_codes[sku_idx],
                    'goods_code': goods_codes[sku_idx],
                    'sale_qty': sales_history[week_idx, sku_idx],
                    'sale_amt': sales_history[week_idx, sku_idx] * prices[sku_idx],
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 添加group_id列（训练时使用）
        df['group_id'] = df['shop_code'] + '_' + df['goods_code']
        
        return df
    
    def _augment_features(self, 
                         df: pd.DataFrame, 
                         current_time: int,
                         num_skus: int) -> pd.DataFrame:
        """
        补全TFT训练时必需的特征列，并确保格式与训练时一致
        
        核心逻辑：
        - 从tft_forecaster.training_dataset中获取训练时的列名和数据类型
        - 确保DataFrame包含这些列，并且数据类型正确
        - 如果缺少某些列，用默认值填充
        
        Args:
            df: 输入数据（从_extract_history_data()返回）
            current_time: 当前时间步
            num_skus: SKU数量
            
        Returns:
            pd.DataFrame: 增强后的数据（格式与训练时一致）
        """
        # 获取training_dataset
        training_dataset = self.tft_forecaster.training_dataset
        
        if training_dataset is not None:
            # 从training_dataset中获取训练时的列名
            expected_columns = []
            expected_columns.append(training_dataset.target)
            expected_columns.extend(training_dataset.group_ids)
            expected_columns.extend(training_dataset.static_categoricals)
            expected_columns.extend(training_dataset.static_reals)
            expected_columns.extend(training_dataset.time_varying_known_reals)
            expected_columns.extend(training_dataset.time_varying_unknown_reals)
            expected_columns.append(training_dataset.time_idx)
            
            # 去除重复列名
            expected_columns = list(set(expected_columns))
            
            # 确保所有expected_columns都在df中
            for col in expected_columns:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' not found in DataFrame, adding default values")
                    df[col] = 0  # 添加默认值
            
            # 确保数据类型正确
            # 分类变量应该是string类型
            for col in training_dataset.static_categoricals:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            # 连续变量应该是float类型
            continuous_cols = training_dataset.static_reals + training_dataset.time_varying_known_reals + training_dataset.time_varying_unknown_reals
            for col in continuous_cols:
                if col in df.columns and col != training_dataset.target:
                    df[col] = df[col].astype(float)
        else:
            logger.warning("training_dataset not available, using default format")
            # 使用默认格式（假设列名是 shop_code, goods_code, sale_qty, sale_amt, group_id, time_idx）
            expected_columns = ['time_idx', 'shop_code', 'goods_code', 'sale_qty', 'sale_amt', 'group_id']
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0
        
        return df
    
    def clear_cache(self):
        """清理预测缓存"""
        self.forecast_cache.clear()
        self.last_cache_week = -1
        logger.info("TFT预测缓存已清理")


@dataclass
class IntegrationConfig:
    """集成配置"""
    forecast_horizon: int = 7  # 预测视界(天)
    update_frequency: int = 7  # 模型更新频率
    use_uncertainty: bool = True
    uncertainty_quantile: float = 0.9  # 使用上分位数作为保守预测
    confidence_threshold: float = 0.5  # 置信度阈值


class TFTDRLIntegration:
    """TFT-DRL端到端集成器"""
    
    def __init__(self,
                 tft_forecaster: TFTForecaster,
                 ppo_agent: PPOAgent,
                 supply_chain_env: SupplyChainEnv,
                 config: Optional[IntegrationConfig] = None):
        """
        初始化集成器
        
        Args:
            tft_forecaster: TFT预测器
            ppo_agent: PPO智能体
            supply_chain_env: 供应链环境
            config: 集成配置
        """
        self.tft = tft_forecaster
        self.ppo = ppo_agent
        self.env = supply_chain_env
        self.config = config or IntegrationConfig()
        
        # 约束处理器（从环境配置获取门店容量）
        # 【修复】使用 get_wrapper_attr() 来访问被包装环境的属性
        store_capacity = 500  # 默认值
        if hasattr(supply_chain_env, 'get_wrapper_attr'):
            # 如果环境被 VecEnv 包装，使用 get_wrapper_attr()
            try:
                env_config = supply_chain_env.get_wrapper_attr('config')
                store_capacity = getattr(env_config, 'store_capacity', 500)
            except:
                store_capacity = 500
        else:
            # 如果环境未被包装，直接访问
            store_capacity = getattr(supply_chain_env.config, 'store_capacity', 500)
        self.constraint_handler = ConstraintHandler(store_capacity=store_capacity)
        
        # 预测缓存
        self.forecast_cache = {}
        self.last_update_step = 0
        
        # 引用TFT模型配置用于数据获取
        self.model_config = tft_forecaster.model_config
        
        # 统计信息
        self.integration_stats = {
            'total_predictions': 0,
            'total_actions': 0,
            'forecast_errors': [],
            'reward_history': []
        }
    
    def predict_demand(self, 
                      df: pd.DataFrame,
                      use_conservative: bool = True) -> np.ndarray:
        """
        预测需求
        
        Args:
            df: 历史数据
            use_conservative: 是否使用保守预测
        
        Returns:
            np.ndarray: 预测需求
        """
        # 输入验证（fail-fast）
        if df is None:
            raise DataError("Input DataFrame is None")
        
        if df.empty:
            raise DataError("History data cannot be empty")
        
        # 【修复】自动创建 time_idx 列（如果不存在）
        if 'time_idx' not in df.columns:
            logger.warning("time_idx column not found, creating from ds or using default")
            
            if 'ds' in df.columns:
                # 使用 ds 列创建 time_idx（每个 group 内连续）
                dataset_group_ids = getattr(self.tft.model_config, 'dataset_group_ids', None)
                if dataset_group_ids and all(col in df.columns for col in dataset_group_ids):
                    # 每个 group 内按 ds 排序，创建连续 time_idx
                    df = df.sort_values(dataset_group_ids + ['ds']).copy()
                    df['time_idx'] = df.groupby(dataset_group_ids).cumcount()
                    logger.info(f"  ✓ time_idx created from ds column (grouped by {dataset_group_ids})")
                else:
                    # 不使用分组，直接创建 time_idx
                    df = df.sort_values('ds').copy() if 'ds' in df.columns else df
                    df['time_idx'] = range(len(df))
                    logger.info("  ✓ time_idx created (default: 0, 1, 2, ...)")
            else:
                # 没有 ds 列，使用默认 time_idx
                df['time_idx'] = range(len(df))
                logger.info("  ✓ time_idx created (default: 0, 1, 2, ...)")
        
        # 【修复】自动转换分类变量为字符串类型（TFT 要求）
        dataset_group_ids = getattr(self.tft.model_config, 'dataset_group_ids', None)
        static_cats = getattr(self.tft.model_config, 'dataset_static_categoricals', None)
        
        # 转换 group_ids 列
        if dataset_group_ids:
            for col in dataset_group_ids:
                if col in df.columns and df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    df[col] = df[col].astype(str)
                    logger.info(f"  ✓ Converted {col} to string type (for TFT)")
        
        # 转换 static_categoricals 列
        if static_cats:
            for col in static_cats:
                if col in df.columns and df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                    df[col] = df[col].astype(str)
                    logger.info(f"  ✓ Converted {col} to string type (for TFT)")
        
        # 检查必要的列是否存在
        if len(df.columns) == 0:
            raise DataError("Input DataFrame has no columns")
        
        # 【修复】使用统一数据通道方案，不要求 group_id 列
        # TFT 训练时使用的是 group_ids 参数，不会创建 group_id 列
        required_columns = ['time_idx', 'sale_qty']
        
        # 检查是否有 TFT 模型配置中的分组列
        dataset_group_ids = getattr(self.tft.model_config, 'dataset_group_ids', None)
        if dataset_group_ids:
            # 如果配置了分组列，也添加到必需列中
            for col in dataset_group_ids:
                if col not in required_columns:
                    required_columns.append(col)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataError(f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}")
        
        # 检查 time_idx 是否连续
        if 'time_idx' in df.columns:
            time_idx_min = df['time_idx'].min()
            time_idx_max = df['time_idx'].max()
            expected_range = range(time_idx_min, time_idx_max + 1)
            actual_values = set(df['time_idx'].unique())
            expected_values = set(expected_range)
            
            if actual_values != expected_values:
                missing_time = expected_values - actual_values
                logger.warning(f"time_idx has gaps! Missing {len(missing_time)} values. First 10 missing: {sorted(missing_time)[:10]}")
        
        # 【修复】使用 dataset_group_ids 检查每个 group 的数据量
        # 如果数据不足，跳过该 group（记录警告，不报错）
        valid_groups = []
        invalid_groups = []
        
        if dataset_group_ids and all(col in df.columns for col in dataset_group_ids):
            min_rows_needed = getattr(self.tft.model_config, 'max_encoder_length', 30)
            # 使用 group_ids 进行分组
            for group_vals, group_df in df.groupby(dataset_group_ids):
                if len(group_df) < min_rows_needed:
                    logger.warning(f"Group {group_vals} has only {len(group_df)} rows, need at least {min_rows_needed} - SKIPPING")
                    invalid_groups.append(group_vals)
                else:
                    valid_groups.append(group_vals)
            
            if len(invalid_groups) > 0:
                logger.warning(f"  ⚠ {len(invalid_groups)} groups have insufficient data and will be skipped")
                # 只保留有效 groups 的数据
                df = df[df.set_index(dataset_group_ids).index.isin([g for g in valid_groups])].reset_index()
                logger.info(f"  ✓ Using {len(valid_groups)} valid groups for prediction")
        elif 'group_id' in df.columns:
            # 向后兼容：如果 df 有 group_id 列，使用它
            min_rows_needed = getattr(self.tft.model_config, 'max_encoder_length', 30)
            for group_id, group_df in df.groupby('group_id'):
                if len(group_df) < min_rows_needed:
                    logger.warning(f"Group {group_id} has only {len(group_df)} rows, need at least {min_rows_needed} - SKIPPING")
                    invalid_groups.append(group_id)
                else:
                    valid_groups.append(group_id)
            
            if len(invalid_groups) > 0:
                logger.warning(f"  ⚠ {len(invalid_groups)} groups have insufficient data and will be skipped")
                df = df[df['group_id'].isin(valid_groups)].reset_index(drop=True)
                logger.info(f"  ✓ Using {len(valid_groups)} valid groups for prediction")
        
        # 检查是否有 NaN 或 Inf
        if df[required_columns].isnull().any().any():
            logger.error("DataFrame contains NaN values!")
            raise DataError("Input DataFrame contains NaN values. Please clean the data.")
        
        # 【修复】使用正确的列名记录日志
        if dataset_group_ids and all(col in df.columns for col in dataset_group_ids):
            n_groups = df.groupby(dataset_group_ids).ngroups
        elif 'group_id' in df.columns:
            n_groups = df['group_id'].nunique()
        else:
            n_groups = 1  # 如果没有分组列，假设只有 1 个 group
        
        logger.info(f"✓ Data validation passed: {df.shape[0]} rows, {df.shape[1]} columns, {n_groups} groups")
        
        # 获取num_skus（动态获取，避免硬编码）
        num_skus = getattr(self.env, 'num_skus', None)
        if num_skus is None:
            # 如果环境中没有 num_skus 属性，尝试从配置中获取
            if hasattr(self.env, 'config') and hasattr(self.env.config, 'num_skus'):
                num_skus = self.env.config.num_skus
            else:
                raise ValueError(
                    "无法获取 num_skus！请确保 SupplyChainEnv 已正确设置 num_skus 属性，"
                    "或通过 --n-skus 参数指定 SKU 数量。"
                )
        
        try:
            # 【修复】先调用TFT模型进行预测
            logger.info(f"开始TFT预测: df.shape={df.shape}")
            
            # 使用TFTForecaster进行预测
            if self.tft._is_trained:
                # 调用预测方法
                predictions = self.tft.predict(df=df)
                
                # 处理预测结果
                if isinstance(predictions, torch.Tensor):
                    demand = predictions.cpu().numpy()
                elif isinstance(predictions, np.ndarray):
                    demand = predictions
                else:
                    # 如果返回的是列表，取第一个元素
                    if isinstance(predictions, list) and len(predictions) > 0:
                        demand = predictions[0]
                    else:
                        raise DataError(f"Unexpected prediction type: {type(predictions)}")
                
                logger.info(f"✓ TFT预测完成: demand.shape={demand.shape}")
            else:
                # 如果TFT模型未训练，使用简单预测（历史平均值）
                logger.warning("TFT model not trained, using historical average")
                if 'sale_qty' in df.columns:
                    demand = df.groupby('group_id')['sale_qty'].mean().values
                else:
                    demand = np.ones(num_skus) * 10.0  # 默认预测=10
            
            # 【修复】现在demand已经定义，可以进行后处理
            if not isinstance(demand, np.ndarray):
                demand = np.array(demand, dtype=np.float32)
            
            demand = np.maximum(demand, 0).flatten()
            
            # 预测结果合理性校验
            if np.any(np.isnan(demand)) or np.any(np.isinf(demand)):
                logger.error("TFT prediction contains NaN or Inf")
                raise DataError("Prediction output contains NaN or Inf. Suggestion: clean input data.")
            
            # 基于历史销售数据的合理上限
            if 'sale_qty' in df.columns:
                max_historical = df['sale_qty'].max() * 3
            else:
                max_historical = 1000.0
            demand = np.clip(demand, 0, max_historical)
            
            # 确保维度匹配
            if len(demand) < num_skus:
                demand = np.pad(demand, (0, num_skus - len(demand)), mode='edge')
            elif len(demand) > num_skus:
                demand = demand[:num_skus]
                
        except DataError as e:
            logger.error(f"TFT prediction failed (DataError): {e}")
            # 【修复】不重新抛出异常，而是回退到简单预测
            logger.warning("Falling back to simple prediction (historical average)")
            if 'sale_qty' in df.columns:
                demand = df.groupby('group_id')['sale_qty'].mean().values
            else:
                demand = np.ones(num_skus) * 10.0
        except Exception as e:
            logger.error(f"TFT prediction failed (unexpected error): {e}")
            logger.error(traceback.format_exc())
            # 【修复】回退到随机预测
            logger.warning("Falling back to random prediction")
            demand = np.random.poisson(10, size=num_skus).astype(np.float32)
        
        # 【修复】保存 DataFrame 供 prepare_forecasts_for_ppo() 使用
        self._last_predict_df = df.copy()
        
        self.integration_stats['total_predictions'] += 1
        return demand.astype(np.float32)
    
    def generate_state_from_forecast(self,
                                    demand_forecast: np.ndarray,
                                    inventory_state: np.ndarray,
                                    warehouse_state: np.ndarray,
                                    performance_history: np.ndarray) -> np.ndarray:
        """
        从预测生成状态
        
        Args:
            demand_forecast: 需求预测
            inventory_state: 库存状态
            warehouse_state: 仓库状态
            performance_history: 历史绩效
        
        Returns:
            np.ndarray: 状态向量
        """
        # 拼接状态组件
        state = np.concatenate([
            inventory_state,
            demand_forecast,
            warehouse_state[:1] if len(warehouse_state) > 0 else np.array([3.0]),  # 交货周期
            performance_history
        ])
        
        return state.astype(np.float32)
    
    def select_action_with_constraints(self,
                                       state: np.ndarray,
                                       deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        带约束的动作选择
        
        Args:
            state: 当前状态
            deterministic: 确定性策略
        
        Returns:
            Tuple: (action, info)
        """
        # 获取PPO策略动作
        action, log_prob = self.ppo.select_action(state, deterministic=deterministic)
        
        # 获取当前库存状态（从状态中提取）
        inventory = state[:self.env.num_skus]
        warehouse = self.env.warehouse_inventory
        
        # 约束检查和调整
        constraint_result = self.constraint_handler.check_and_adjust(
            action=action,
            inventory=inventory,
            warehouse_inventory=warehouse,
            min_batch_size=10.0,
            max_order=1000.0
        )
        
        info = {
            'original_action': action,
            'adjusted_action': constraint_result.adjusted_action,
            'constraint_violations': constraint_result.violations,
            'constraint_penalty': constraint_result.penalty,
            'log_prob': log_prob,
            'was_adjusted': not constraint_result.is_valid
        }
        
        self.integration_stats['total_actions'] += 1
        
        return constraint_result.adjusted_action, info
    
    def step_with_integration(self,
                             df: pd.DataFrame,
                             deterministic: bool = True) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步完整的 TFT预测 -> PPO决策 -> 环境步进 闭环。
        
        【修复】添加 TFT 预测防御：如果 TFT 预测失败，使用默认需求预测
        """
        # 输入验证
        if df.empty:
            raise DataError("Input DataFrame is empty in step_with_integration")
        
        # 1. TFT 预测需求 (带缓存，只在需要时预测)
        demand_forecast = None
        
        if self.last_update_step >= self.config.update_frequency:
            # 需要更新预测
            try:
                demand_forecast = self.predict_demand(df, use_conservative=True)
                self.last_update_step = 0
                self.forecast_cache['latest_demand'] = demand_forecast  # 缓存预测结果
                logger.info(f"TFT预测完成（更新）: demand.shape={demand_forecast.shape}")
            except Exception as e:
                logger.error(f"TFT预测失败: {e}")
                logger.warning("将使用默认需求预测（全0）")
                # 使用默认需求预测（全0）
                if hasattr(self.env, 'num_skus'):
                    demand_forecast = np.zeros(self.env.num_skus, dtype=np.float32)
                else:
                    demand_forecast = np.zeros(50, dtype=np.float32)  # 默认值
                self.forecast_cache['latest_demand'] = demand_forecast
        else:
            # 使用缓存的预测
            if 'latest_demand' in self.forecast_cache and self.forecast_cache['latest_demand'] is not None:
                demand_forecast = self.forecast_cache['latest_demand']
                logger.info(f"TFT预测完成（缓存）: demand.shape={demand_forecast.shape}")
            else:
                # 第一次，需要预测
                try:
                    demand_forecast = self.predict_demand(df, use_conservative=True)
                    self.forecast_cache['latest_demand'] = demand_forecast
                    logger.info(f"TFT预测完成（首次）: demand.shape={demand_forecast.shape}")
                except Exception as e:
                    logger.error(f"TFT预测失败: {e}")
                    logger.warning("将使用默认需求预测（全0）")
                    # 使用默认需求预测（全0）
                    if hasattr(self.env, 'num_skus'):
                        demand_forecast = np.zeros(self.env.num_skus, dtype=np.float32)
                    else:
                        demand_forecast = np.zeros(50, dtype=np.float32)  # 默认值
                    self.forecast_cache['latest_demand'] = demand_forecast
        
        # 【防御性检查】确保 demand_forecast 不为 None
        if demand_forecast is None:
            logger.error(" demand_forecast 为 None！使用默认预测（全0）")
            if hasattr(self.env, 'num_skus'):
                demand_forecast = np.zeros(self.env.num_skus, dtype=np.float32)
            else:
                demand_forecast = np.zeros(50, dtype=np.float32)  # 默认值
        
        # 2. 获取当前环境状态 (✅ 修复点：调用环境标准接口，而非手动拼凑)
        # 【防御性编程】检查环境是否有 _get_state() 或 _get_obs() 方法
        if hasattr(self.env, '_get_state'):
            current_state = self.env._get_state()
        elif hasattr(self.env, '_get_obs'):
            current_state = self.env._get_obs()
        else:
            raise AttributeError("环境缺少 _get_state() 或 _get_obs() 方法，无法获取当前状态！")
        
        # 【防御性校验】确保状态维度和类型与 PPO 期望的完全一致
        current_state = np.array(current_state, dtype=np.float32).flatten()
        expected_shape = self.env.observation_space.shape
        
        if current_state.shape != expected_shape:
            raise ValueError(
                f"状态维度不匹配！环境 _get_state() 返回了 {current_state.shape}，"
                f"但 observation_space 期望的是 {expected_shape}。"
                f"请检查环境的状态构造逻辑。"
            )
        
        logger.info(f"环境状态获取完成: state.shape={current_state.shape}")
        
        # 3. PPO 决策 (带约束)
        action, info = self.select_action_with_constraints(current_state, deterministic=deterministic)
        
        # 4. 环境步进
        next_state, reward, terminated, truncated, step_info = self.env.step(action)
        done = terminated or truncated
        
        # 5. 更新统计
        self.integration_stats['reward_history'].append(reward)
        if len(self.integration_stats['reward_history']) > 1000:
            self.integration_stats['reward_history'] = self.integration_stats['reward_history'][-1000:]
        
        # 6. 更新步数计数器
        self.last_update_step += 1
        
        # 合并 info
        step_info.update(info)
        
        # 【防御性编程】确保返回的 next_state 也是 float32
        next_state = np.array(next_state, dtype=np.float32)
        
        return next_state, float(reward), done, step_info
    
    @classmethod
    def load_tft_quantile_model(cls, 
                                checkpoint_path: str = "records/models/tft_model_quantile.ckpt",
                                model_config: Optional[Dict] = None) -> TFTForecaster:
        """
        加载训练好的TFT分位数模型
        
        Args:
            checkpoint_path: 模型检查点路径
            model_config: 模型配置（可选，如果不提供则从检查点自动恢复）
        
        Returns:
            TFTForecaster: 加载好的TFT预测器
        """
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading TFT quantile model from {checkpoint_path}")
        
        # 创建TFTForecaster实例（使用默认配置，load_model会从检查点恢复配置）
        forecaster = TFTForecaster()
        
        # 加载模型
        forecaster.load_model(checkpoint_path)
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  - Loss type: {forecaster.model_config.loss_type}")
        logger.info(f"  - Quantiles: {forecaster.model_config.quantiles}")
        
        return forecaster
    
    def prepare_forecasts_for_ppo(self,
                                   historical_data: pd.DataFrame,
                                   num_steps: int = 100) -> pd.DataFrame:
        """
        为PPO训练准备需求预测数据
        
        Args:
            historical_data: 历史销售数据
            num_steps: 预测步数（PPO训练步数）
        
        Returns:
            pd.DataFrame: 包含预测需求的DataFrame，用于PPO训练
        """
        logger.info(f"Preparing forecasts for PPO training ({num_steps} steps)...")
        
        # 【修复】获取真实的 SKU 分组标识
        group_ids = self.tft.model_config.dataset_group_ids
        sku_id_list = None
        
        # 先做一次预测，获取 SKU ID 列表
        try:
            sample_demand = self.predict_demand(historical_data, use_conservative=True)
            if hasattr(self, '_last_predict_df') and self._last_predict_df is not None:
                df_sample = self._last_predict_df
                if group_ids and all(col in df_sample.columns for col in group_ids):
                    # 获取唯一的分组组合
                    if len(group_ids) == 1:
                        unique_groups = df_sample[group_ids[0]].unique().tolist()
                        sku_id_list = [str(g) for g in unique_groups]
                    else:
                        # 【修复】多个分组列时，只使用最后一个作为 SKU ID（通常是 goods_code）
                        unique_groups = df_sample.groupby(group_ids).size().index.tolist()
                        # unique_groups 是元组列表，如 [('SHOP_001', 'SKU_000'), ...]
                        # 取最后一个元素作为 SKU ID
                        sku_id_list = [str(g[-1]) for g in unique_groups]
                    logger.info(f"  ✓ SKU ID list created: {len(sku_id_list)} SKUs")
                    logger.info(f"    Sample SKU IDs: {sku_id_list[:5]}")
        except Exception as e:
            logger.warning(f"  ⚠ Could not create SKU ID list: {e}")
        
        forecasts = []
        
        for step in range(num_steps):
            # 预测需求
            demand = self.predict_demand(historical_data, use_conservative=True)
            
            # 【修复】使用真实的 SKU ID
            if sku_id_list and len(demand) == len(sku_id_list):
                sku_ids = sku_id_list
            elif hasattr(self, '_last_predict_df') and self._last_predict_df is not None:
                df_sample = self._last_predict_df
                if group_ids and all(col in df_sample.columns for col in group_ids):
                    if len(group_ids) == 1:
                        unique_groups = df_sample[group_ids[0]].unique().tolist()
                        sku_ids = [str(g) for g in unique_groups]
                    else:
                        # 【修复】多个分组列时，只使用最后一个作为 SKU ID
                        unique_groups = df_sample.groupby(group_ids).size().index.tolist()
                        # 取最后一个元素作为 SKU ID
                        sku_ids = [str(g[-1]) for g in unique_groups]
                else:
                    sku_ids = [str(i) for i in range(len(demand))]
            else:
                sku_ids = [str(i) for i in range(len(demand))]
            
            # 保存预测结果
            forecast_df = pd.DataFrame({
                'step': [step] * len(demand),
                'sku_id': sku_ids,  # ✅ 使用真实的 SKU ID
                'demand_median': demand,
                'demand_q10': demand * 0.8,  # 简化：使用分位数比例
                'demand_q90': demand * 1.2,
            })
            forecasts.append(forecast_df)
            
            if (step + 1) % 10 == 0:
                logger.info(f"  Generated forecasts for {step + 1}/{num_steps} steps")
        
        result_df = pd.concat(forecasts, ignore_index=True)
        logger.info(f"✓ Generated {len(result_df)} forecast records for PPO training")
        logger.info(f"  Unique SKU IDs in forecasts: {result_df['sku_id'].nunique()}")
        
        return result_df


def create_tft_drl_integration(checkpoint_path: str = "records/models/tft_model_quantile.ckpt",
                                ppo_config: Optional[Dict] = None,
                                env_config: Optional[Dict] = None,
                                num_skus: Optional[int] = None) -> TFTDRLIntegration:
    """
    创建TFT-DRL集成器的便捷函数
    
    Args:
        checkpoint_path: TFT模型检查点路径
        ppo_config: PPO配置
        env_config: 环境配置
        num_skus: SKU数量（如果提供，则覆盖 env_config 中的设置）
    
    Returns:
        TFTDRLIntegration: 配置好的集成器
    """
    from models.rl import PPOAgent
    from models.optimization import SupplyChainEnv
    from config import PPOConfig, EnvConfig  # 【修复】添加 EnvConfig 导入
    if num_skus is not None:
        if env_config is None:
            env_config = {}
        env_config['num_skus'] = num_skus
        logger.info(f"使用指定的 num_skus={num_skus}")
    
    # 1. 加载TFT模型
    logger.info("=" * 60)
    logger.info("Step 1: Loading TFT quantile model...")
    tft_forecaster = TFTDRLIntegration.load_tft_quantile_model(checkpoint_path)
    
    # 2. 创建供应链环境（需要先创建环境，才能知道 state_dim 和 action_dim）
    logger.info("\nStep 2: Creating supply chain environment...")
    if env_config is None:
        default_num_skus = ppo_config.get('action_dim', 50) if ppo_config else 50
        # 【修复】只使用 EnvConfig 接受的参数
        env_config = {
            'lead_time': 7,
            'order_cost': 10.0,
            'holding_cost_rate': 0.01,
            'warehouse_capacity': 100.0,
            'store_capacity': 500.0,
        }
        # num_skus 是 SupplyChainEnv 的参数，不是 EnvConfig 的参数
        num_skus = default_num_skus
    
    # 【修复】将字典转换为 EnvConfig 对象，并显式传入 num_skus
    if isinstance(env_config, dict):
        # 移除 num_skus（它不是 EnvConfig 的参数，而是 SupplyChainEnv 的参数）
        env_config_copy = env_config.copy()
        if 'num_skus' in env_config_copy:
            del env_config_copy['num_skus']
        env_config_obj = EnvConfig(**env_config_copy)
    else:
        env_config_obj = env_config
    
    # 显式获取 num_skus（从参数或 env_config 字典）
    if num_skus is None:
        num_skus = env_config.get('num_skus', 50) if isinstance(env_config, dict) else 50
    
    supply_chain_env = SupplyChainEnv(
        num_skus=num_skus,  # 【关键】显式传入 num_skus
        env_config=env_config_obj  # 【关键】传入 EnvConfig 对象，而非字典
    )
    
    # 【修复】在添加 Monitor 之前，先从原始环境中获取 state_dim 和 action_dim
    state_dim = supply_chain_env.state_dim
    action_dim = supply_chain_env.action_space.shape[0]
    logger.info(f"  - state_dim: {state_dim}, action_dim: {action_dim}")
    
    # 【修复】添加 Monitor 包装器（SB3 强制最佳实践）
    # Monitor 会拦截 done=True，并计算 episode 的总 reward 和 length，写入 logger
    # 【关键】Monitor 应该在最内层（直接包装原始环境）
    from stable_baselines3.common.monitor import Monitor
    import os
    import tempfile
    
    # 创建临时目录用于存储 Monitor 日志（如果不关心日志文件，可以指向临时目录）
    monitor_log_dir = tempfile.mkdtemp()
    supply_chain_env = Monitor(supply_chain_env, filename=os.path.join(monitor_log_dir, "monitor.csv"))
    logger.info(f"  ✓ Monitor 已添加（日志目录: {monitor_log_dir}）")
    
    # 【修复】将环境包装为 DummyVecEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: supply_chain_env])
    
    # 【关键修复】加入 VecNormalize 归一化 Observation 和 Reward
    # 【注意】VecNormalize 应该在 Monitor 外面（这样 Monitor 才能正确记录原始 Reward）
    from stable_baselines3.common.vec_env import VecNormalize
    
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,       # 归一化状态 (对高维状态极其重要！)
        norm_reward=True,    # 归一化 Reward (解决 Reward 尺度问题)
        clip_reward=10.0,    # 截断极端异常的 Reward
        gamma=0.99,         # 折扣因子（与 PPO 保持一致）
        epsilon=1e-8         # 数值稳定性
    )
    
    logger.info(f"  ✓ VecNormalize 已添加：norm_obs=True, norm_reward=True")
    
    # 3. 创建PPO智能体
    logger.info("\nStep 3: Creating PPO agent...")
    # 动态获取 num_skus，避免硬编码
    if ppo_config is None:
        ppo_config = {
            'learning_rate': 3e-4,
            'n_steps': 4096,  # 【修复】增大到 4096，确保每次更新包含更多完整 episode
            'batch_size': 64,
            'n_epochs': 10,
            'ent_coef': 0.0001,  # 【紧急修复】从 0.001 降到 0.0001，加速 Entropy 下降
            'max_grad_norm': 0.5,
            'vf_coef': 0.5,
            'policy_kwargs': dict(
                net_arch=[dict(pi=[256, 128], vf=[256, 128])],
                ortho_init=True,  # 如果是连续动作，确保使用正交初始化
            ),
        }
    
    # 【修复】将字典转换为 PPOConfig 对象
    from config import PPOConfig
    valid_keys = PPOConfig.__dataclass_fields__.keys()
    ppo_config_filtered = {k: v for k, v in ppo_config.items() if k in valid_keys}
    ppo_config_obj = PPOConfig(**ppo_config_filtered)
    
    ppo_agent = PPOAgent(state_dim, action_dim, ppo_config=ppo_config_obj)
    
    # 【修复】初始化 PPO 模型（延迟创建）
    # 将环境包装为 DummyVecEnv，然后设置给 PPO 智能体
    ppo_agent.set_env(vec_env)
    
    # 【健康检查】确保 PPO 的底层 model 不是 None
    if hasattr(ppo_agent, 'model') and ppo_agent.model is None:
        raise RuntimeError("PPO Agent 初始化失败！ppo_agent.model 为 None。请检查 PPOAgent 类的初始化逻辑。")
    
    logger.info(f"  ✓ PPO 模型已初始化（观测空间: {supply_chain_env.observation_space.shape[0]}）")
    
    # 【维度验证】打印关键维度信息，确保万无一失
    logger.info(f"  环境 Obs Space: {supply_chain_env.observation_space.shape}")
    if hasattr(ppo_agent, 'model') and ppo_agent.model is not None:
        logger.info(f"  PPO Model Obs Space: {ppo_agent.model.observation_space.shape}")
        # 验证维度是否一致
        if supply_chain_env.observation_space.shape != ppo_agent.model.observation_space.shape:
            raise ValueError(
                f"维度不匹配！环境 Obs Space: {supply_chain_env.observation_space.shape}, "
                f"PPO Model Obs Space: {ppo_agent.model.observation_space.shape}"
            )
        logger.info(f"  ✓ 维度验证通过！环境 Obs Space 和 PPO Model Obs Space 完全一致")
    else:
        logger.warning(f"  ⚠ PPO model 未初始化，无法进行维度验证")
    
    # 5. 创建集成器
    logger.info("\nStep 4: Creating TFT-DRL integration...")
    integration = TFTDRLIntegration(
        tft_forecaster=tft_forecaster,
        ppo_agent=ppo_agent,
        supply_chain_env=supply_chain_env,
        config=IntegrationConfig(use_uncertainty=True, uncertainty_quantile=0.9)
    )
    
    logger.info("✓ TFT-DRL integration created successfully!")
    logger.info("=" * 60)
    
    return integration
