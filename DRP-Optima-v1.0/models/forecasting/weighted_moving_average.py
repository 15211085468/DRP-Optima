"""
加权移动平均预测器（公司现有算法）

公式：WEEK-4 * 0.1 + WEEK-3 * 0.2 + WEEK-2 * 0.3 + WEEK-1 * 0.4
"""

import numpy as np
import warnings
from typing import Optional


class WeightedMovingAverageForecaster:
    """
    加权移动平均预测器
    
    使用最近 4 周的销量，按指数递增权重进行加权预测：
    pred = week_4 * 0.1 + week_3 * 0.2 + week_2 * 0.3 + week_1 * 0.4
    
    这是公司现有的预测算法，作为基准模型之一。
    """
    
    def __init__(self, name: str = "WMA", weights: Optional[list] = None):
        """
        初始化加权移动平均预测器
        
        Args:
            name: 模型名称
            weights: 4 个权重值，默认为 [0.1, 0.2, 0.3, 0.4]
        """
        self._name = name
        self.weights = weights if weights is not None else [0.1, 0.2, 0.3, 0.4]
        self._last_series = None
        self._fitted = False
        
        # 验证权重
        if len(self.weights) != 4:
            raise ValueError("weights 必须有 4 个值（对应最近 4 周）")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            warnings.warn(f"权重之和不为 1.0（当前={sum(self.weights):.4f}），将自动归一化")
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
    
    @property
    def name(self) -> str:
        """返回模型名称"""
        return self._name
    
    def fit(self, series: np.ndarray) -> None:
        """
        拟合模型（实际上只是保存历史序列）
        
        Args:
            series: 历史销量序列
        """
        series = np.asarray(series, dtype=np.float64)
        self._last_series = series.copy()
        self._fitted = True
    
    def predict(self, horizon: int = 4) -> np.ndarray:
        """
        预测未来 horizon 周销量
        
        使用最近 4 周的加权平均值作为未来每周的预测值。
        对于多步预测，所有步都使用相同的加权平均值（即假设未来每周销量相同）。
        
        Args:
            horizon: 预测步数（周数）
        
        Returns:
            预测值数组，形状为 (horizon,)
        """
        if not self._fitted or self._last_series is None:
            # 未拟合，使用 LastValue（最后一周销量）
            return np.zeros(horizon)
        
        series = self._last_series
        
        # 至少需要 4 个历史值
        if len(series) < 4:
            warnings.warn(f"序列长度 {len(series)} < 4，无法使用完整权重，将使用可用的最近值")
            # 使用可用的最近值的平均值
            recent = series[-min(4, len(series)):]
            pred_value = float(np.mean(recent))
        else:
            # 取最近 4 周的值
            week_1 = series[-1]  # 最近一周
            week_2 = series[-2]
            week_3 = series[-3]
            week_4 = series[-4]
            
            # 加权计算
            pred_value = (
                week_4 * self.weights[0] +
                week_3 * self.weights[1] +
                week_2 * self.weights[2] +
                week_1 * self.weights[3]
            )
        
        # 所有预测步都使用相同的值
        return np.full(horizon, pred_value)
    
    def fit_predict(self, series: np.ndarray, horizon: int = 4) -> np.ndarray:
        """
        拟合模型并预测
        
        Args:
            series: 历史销量序列
            horizon: 预测步数
            
        Returns:
            预测值数组
        """
        self.fit(series)
        return self.predict(horizon)
    
    def __str__(self) -> str:
        return f"{self._name}(weights={self.weights})"
    
    def __repr__(self) -> str:
        return self.__str__()
