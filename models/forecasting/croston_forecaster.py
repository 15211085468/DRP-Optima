"""
Croston 方法预测器
用于间歇性需求预测（Intermittent Demand Forecasting）
适合药店零售数据（大量零值）
"""

from .base_forecaster import BaseForecaster
from abc import ABC, abstractmethod

import numpy as np
from typing import Optional
import warnings


class CrostonForecaster(BaseForecaster):
    """
    Croston 方法预测器
    
    Croston 方法用于预测间歇性需求（Intermittent Demand），
    它将需求量和需求间隔分开预测。
    
    公式：
    - 需求量 q(t) = q(t-1) + alpha * (d(t) - q(t-1))
    - 间隔 p(t) = p(t-1) + alpha * (days_between_demands - p(t-1))
    - 预测值 = q(t) / p(t)
    
    参考：
    - Croston, J. D. (1972). "Forecasting and stock control for intermittent demands."
    - Syntetos, A., & Boylan, J. E. (2005). "The accuracy of intermittent demand estimates."
    """

    def __init__(self, name: str = "Croston", alpha: float = 0.1):
        """
        初始化 Croston 预测器
        
        Args:
            name: 模型名称
            alpha: 指数平滑参数 (0 < alpha < 1)
        """
        super().__init__()
        self._name = name
        self.alpha = alpha
        self._fitted = False
        self._q = None  # 需求量
        self._p = None  # 间隔
        self._last_series = None

    @property
    def name(self) -> str:
        """模型名称"""
        return self._name

    def fit(self, series: np.ndarray):
        """
        用 Croston 方法拟合序列
        
        Args:
            series: 时间序列数据 (1D array)
        """
        series = np.asarray(series, dtype=np.float64)
        self._last_series = series.copy()
        
        if len(series) < 2:
            self._fitted = False
            return
        
        # 找出非零需求的索引
        non_zero_mask = series > 0
        non_zero_demands = series[non_zero_mask]
        
        if len(non_zero_demands) == 0:
            # 所有值都是零，无法预测
            self._fitted = False
            return
        
        if len(non_zero_demands) == 1:
            # 只有一个非零值
            self._q = non_zero_demands[0]
            self._p = len(series)  # 间隔为整个序列长度
            self._fitted = True
            return
        
        # 计算需求量（非零值的平均值，用指数平滑）
        self._q = non_zero_demands[0]
        for i in range(1, len(non_zero_demands)):
            self._q = self.alpha * non_zero_demands[i] + (1 - self.alpha) * self._q
        
        # 计算间隔（非零值之间的平均间隔）
        non_zero_indices = np.where(non_zero_mask)[0]
        intervals = np.diff(non_zero_indices)
        
        if len(intervals) > 0:
            self._p = intervals[0]
            for i in range(1, len(intervals)):
                self._p = self.alpha * intervals[i] + (1 - self.alpha) * self._p
        else:
            self._p = len(series)
        
        self._fitted = True

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        预测未来 horizon 个时间点
        
        Args:
            horizon: 预测步数
            
        Returns:
            预测值数组 (horizon,)
        """
        if not self._fitted:
            # 未拟合，返回零
            return np.zeros(horizon)
        
        if self._p == 0:
            # 避免除零
            pred_value = 0
        else:
            pred_value = self._q / self._p
        
        return np.full(horizon, pred_value)

    def fit_predict(self, series: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        拟合并预测（单次调用）
        
        Args:
            series: 时间序列数据
            horizon: 预测步数
            
        Returns:
            预测值数组
        """
        self.fit(series)
        return self.predict(horizon)

    def get_params(self) -> dict:
        """返回模型参数"""
        return {
            "alpha": self.alpha,
            "q": self._q,
            "p": self._p,
            "fitted": self._fitted,
        }
