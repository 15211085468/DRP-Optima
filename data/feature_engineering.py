"""
特征工程模块 - data/feature_engineering.py
负责生成滞后特征、滚动统计特征和增长率特征
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from config import DataConfig
from utils.logger import get_logger
logger = get_logger(__name__)


class FeatureEngineer:
    """特征工程处理器"""
    
    def __init__(self, data_config: Optional[DataConfig] = None):
        """
        初始化特征工程器
        
        Args:
            data_config: 数据配置
        """
        self.config = data_config or DataConfig()
    
    def create_lag_features(self, df: pd.DataFrame, 
                           group_cols: List[str] = None,
                           target_col: str = "sale_qty",
                           lag_windows: List[int] = None) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 输入数据（已按shop_code, goods_code, time_idx排序）
            group_cols: 分组列，默认为['shop_code', 'goods_code']
            target_col: 目标变量列名
            lag_windows: 滞后期列表，默认为[1, 2, 4]
        
        Returns:
            pd.DataFrame: 添加滞后特征后的数据
        """
        if group_cols is None:
            group_cols = ['shop_code', 'goods_code']
        
        if lag_windows is None:
            lag_windows = self.config.lag_windows
        
        df = df.copy()
        
        for lag in lag_windows:
            lag_col = f"lag_{lag}_qty"
            df[lag_col] = df.groupby(group_cols)[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               group_cols: List[str] = None,
                               target_col: str = "sale_qty",
                               windows: List[int] = None,
                               stats: List[str] = None) -> pd.DataFrame:
        """
        创建滚动统计特征（双重保险防止数据泄露）
        
        设计说明：
        - 使用 min_periods=1 允许数据不足时计算近似值，避免大量 NaN 影响模型训练
        - 通过 shift(1) 确保只使用历史数据，防止未来信息泄露
        - 这意味着：前 (window-1) 行可能有值，而非全部为 NaN
        
        如果需要严格窗口长度（不足时为 NaN），请修改代码设置 min_periods=window
        但请注意这会导致大量缺失值，可能影响模型性能。
        
        Args:
            df: 输入数据
            group_cols: 分组列
            target_col: 目标变量
            windows: 滚动窗口列表，默认为[4, 12, 26, 52]周
            stats: 统计量列表，默认为['mean', 'std', 'max']
        
        Returns:
            pd.DataFrame: 添加滚动特征后的数据
        """
        if group_cols is None:
            group_cols = ['shop_code', 'goods_code']
        
        if windows is None:
            windows = self.config.rolling_windows
        
        if stats is None:
            stats = self.config.rolling_stats
        
        df = df.copy()
        
        # 双重保险：先按组 shift，确保只使用历史数据
        df['_shifted'] = df.groupby(group_cols)[target_col].shift(1)
        
        for window in windows:
            for stat in stats:
                col_name = f"rolling_{window}w_{stat}"
                # 使用 rolling 计算滚动统计
                # 注意：_shifted 已经通过 shift(1) 确保了不包含未来数据
                roll = df.groupby(group_cols)['_shifted'].rolling(
                    window=window, min_periods=1
                ).agg(stat)
                
                # 重置索引以匹配原始数据
                df[col_name] = roll.reset_index(level=group_cols, drop=True).values
        
        # 删除临时列
        df.drop(columns=['_shifted'], inplace=True)
        
        return df
    
    def create_growth_rate_features(self, df: pd.DataFrame,
                                   group_cols: List[str] = None,
                                   target_col: str = "sale_qty",
                                   rate_col: str = "qty_wow",
                                   max_cap: float = 5.0,
                                   min_floor: float = 0.5) -> pd.DataFrame:
        """
        创建周环比增长率特征

        Args:
            df: 输入数据
            group_cols: 分组列
            target_col: 目标变量
            rate_col: 增长率列名
            max_cap: 增长率上限（500%，即增长5倍）
            min_floor: 增长率下限绝对值（裁剪下限为 -min_floor，即最大下降50%）

        Returns:
            pd.DataFrame: 添加增长率特征后的数据
        """
        if group_cols is None:
            group_cols = ['shop_code', 'goods_code']

        df = df.copy()

        # 计算周环比: (本周 - 上周) / 上周
        shifted = df.groupby(group_cols)[target_col].shift(1)
        df[rate_col] = (df[target_col] - shifted) / (shifted + 1e-6)

        # 限制增长率范围，防止极端值
        # 下限: -min_floor（最大下降 min_floor*100%），上限: max_cap（最大增长 max_cap*100%）
        df[rate_col] = df[rate_col].clip(lower=-min_floor, upper=max_cap)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame,
                            date_col: str = "date") -> pd.DataFrame:
        """
        创建时间相关特征
        
        Args:
            df: 输入数据
            date_col: 日期列名
        
        Returns:
            pd.DataFrame: 添加时间特征后的数据
        """
        df = df.copy()
        
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            
            # 年/月/周
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['week'] = df[date_col].dt.isocalendar().week.astype(int)
            df['day_of_week'] = df[date_col].dt.dayofweek
            
            # 季度
            df['quarter'] = df[date_col].dt.quarter
            
            # 是否年末/季末
            df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
            df['is_year_end'] = df[date_col].dt.is_year_end.astype(int)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame,
                           group_cols: List[str] = None,
                           target_col: str = "sale_qty",
                           date_col: str = "date") -> pd.DataFrame:
        """
        创建所有特征（便捷方法）
        
        Args:
            df: 输入数据
            group_cols: 分组列
            target_col: 目标变量
            date_col: 日期列
        
        Returns:
            pd.DataFrame: 包含所有特征的数据
        """
        # 1. 滞后特征
        df = self.create_lag_features(df, group_cols, target_col)
        
        # 2. 滚动统计特征
        df = self.create_rolling_features(df, group_cols, target_col)
        
        # 3. 增长率特征
        df = self.create_growth_rate_features(df, group_cols, target_col)
        
        # 4. 时间特征
        df = self.create_time_features(df, date_col)
        
        logger.info(f"特征工程完成: {df.shape[1]} 列")
        return df
    
    def get_feature_list(self, include_lag: bool = True,
                        include_rolling: bool = True,
                        include_growth: bool = True,
                        include_time: bool = True) -> List[str]:
        """
        获取特征列名列表
        
        Args:
            include_*: 是否包含各类型特征
        
        Returns:
            List[str]: 特征列名列表
        """
        features = []
        
        if include_lag:
            for lag in self.config.lag_windows:
                features.append(f"lag_{lag}_qty")
        
        if include_rolling:
            for window in self.config.rolling_windows:
                for stat in self.config.rolling_stats:
                    features.append(f"rolling_{window}w_{stat}")
        
        if include_growth:
            features.append(self.config.growth_rate_col)
        
        if include_time:
            features.extend(['year', 'month', 'week', 'day_of_week', 'quarter'])
        
        return features
    
    def calculate_sku_level_stats(self, df: pd.DataFrame,
                                  group_cols: List[str] = None) -> pd.DataFrame:
        """
        计算SKU级别统计量
        
        Args:
            df: 输入数据
            group_cols: 分组列
        
        Returns:
            pd.DataFrame: SKU统计信息
        """
        if group_cols is None:
            group_cols = ['shop_code', 'goods_code']
        
        stats = df.groupby(group_cols).agg({
            'sale_qty': ['mean', 'std', 'sum', 'count'],
            'sale_amt': ['mean', 'sum']
        }).reset_index()
        
        # 展平多级索引列名
        stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                        for col in stats.columns]
        
        return stats
    
    def calculate_stockout_rate(self, df: pd.DataFrame,
                               threshold: float = 0.0) -> pd.DataFrame:
        """
        计算缺货率（销售量为0的比例）
        
        Args:
            df: 输入数据
            threshold: 缺货阈值
        
        Returns:
            pd.DataFrame: 缺货率统计
        """
        group_cols = ['shop_code', 'goods_code']
        
        stockout = df.groupby(group_cols).apply(
            lambda x: (x['sale_qty'] <= threshold).mean()
        ).reset_index(name='stockout_rate')
        
        return stockout
