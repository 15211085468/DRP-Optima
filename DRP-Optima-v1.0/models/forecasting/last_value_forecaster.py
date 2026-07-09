"""
Last-Value (T-1 Naive) 基线模型 - models/forecasting/last_value_forecaster.py

最简 Baseline：用最近一期观测值作为所有未来步的预测。
无需任何依赖，纯 NumPy 实现。
"""

import numpy as np
from .base_forecaster import BaseForecaster


class LastValueForecaster(BaseForecaster):
    """
    T-1 Naive 预测器

    预测规则：predict(h) = series[-1] 对所有 h 步均相同。
    实现简单，作为最低性能 baseline，任何有效模型都应优于此。
    """

    def __init__(self):
        self._last_value: float = 0.0
        self._fitted: bool = False

    @property
    def name(self) -> str:
        return "LastValue (T-1 Naive)"

    def fit(self, series: np.ndarray) -> None:
        """记录序列末尾值"""
        series = np.asarray(series, dtype=np.float64)
        if len(series) == 0:
            raise ValueError("series 不能为空")
        self._last_value = float(series[-1])
        self._fitted = True

    def predict(self, horizon: int = 1) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        return np.full(horizon, max(0.0, self._last_value), dtype=np.float64)
