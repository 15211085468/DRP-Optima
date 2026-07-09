"""
指数平滑预测器（Exponential Smoothing / Holt-Winters）
--------------------------------------------------------

与 ARIMA / LSTM 接口保持一致：
    model = ExpSmoothingForecaster(name="ExpSmooth")
    pred  = model.fit_predict(hist, horizon=4)

依赖：statsmodels（已随 ARIMAForecaster 安装）
"""

from .base_forecaster import BaseForecaster
import numpy as np
from typing import Optional
import logging
import warnings

logger = logging.getLogger(__name__)


class ExpSmoothingForecaster(BaseForecaster):
    """
    指数平滑预测器，支持趋势 + 季节性（Holt-Winters）。
    
    Parameters
    ----------
    name : str
        模型显示名称
    trend : str or None
        趋势类型：'add' / 'mul' / None
    seasonal : str or None
        季节性类型：'add' / 'mul' / None
    seasonal_periods : int or None
        季节性周期（周数据建议 52）
    damping_slope : float or None
        阻尼系数（None 表示不阻尼）
    """

    def __init__(
        self,
        name: str = "ExpSmooth",
        trend: str | None = "add",
        seasonal: str | None = "add",
        seasonal_periods: int | None = 52,
        damping_slope: float | None = None,
    ):
        super().__init__()
        self._name = name
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damping_slope = damping_slope
        self._fitted = False
        self._model = None

    @property
    def name(self) -> str:
        """模型名称（用于展示/保存）"""
        return self._name

    def fit(self, series: np.ndarray):
        """
        在单条时序上训练指数平滑模型。

        Parameters
        ----------
        series : np.ndarray, shape (T,)
            周度销量序列，float64，允许含 0，不允许 NaN。
        """
        series = np.asarray(series, dtype=np.float64)
        self._last_series = series.copy()
        
        if len(series) < 3:
            warnings.warn(f"序列长度 {len(series)} 太短，无法拟合指数平滑")
            self._fitted = False
            return
        
        # 根据序列长度决定是否使用季节性
        use_seasonal = self.seasonal
        use_seasonal_periods = self.seasonal_periods
        
        if use_seasonal and use_seasonal_periods and len(series) < use_seasonal_periods * 2:
            # 序列太短，无法估计季节性，禁用季节性
            use_seasonal = None
            use_seasonal_periods = None
            warnings.warn(f"序列长度 {len(series)} 太短，禁用季节性")
        
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=use_seasonal,
                seasonal_periods=use_seasonal_periods,
                initialization_method="estimated",
            )
            self._fitted_model = model.fit(damping_slope=self.damping_slope)
            self._fitted = True
            
        except Exception as e:
            warnings.warn(f"指数平滑拟合失败: {e}")
            self._fitted = False

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        预测未来 horizon 步。

        Parameters
        ----------
        horizon : int
            预测步数（周）

        Returns
        -------
        np.ndarray, shape (horizon,)
            预测值，已 clip 至 >= 0。
        """
        if not self._fitted or self._fitted_model is None:
            # 未拟合，用最后一个值
            if hasattr(self, '_last_series') and self._last_series is not None and len(self._last_series) > 0:
                last_val = self._last_series[-1]
                return np.full(horizon, last_val)
            else:
                return np.zeros(horizon)
        
        try:
            pred = self._fitted_model.forecast(horizon)
            pred = np.maximum(pred, 0)  # 确保非负
            return pred
            
        except Exception as e:
            warnings.warn(f"指数平滑预测失败: {e}")
            if hasattr(self, '_last_series') and self._last_series is not None and len(self._last_series) > 0:
                return np.full(horizon, self._last_series[-1])
            else:
                return np.zeros(horizon)

    def fit_predict(self, series: np.ndarray, horizon: int = 1) -> np.ndarray:
        """便捷方法：先 fit 再 predict"""
        self.fit(series)
        return self.predict(horizon)
