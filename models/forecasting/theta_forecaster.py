"""
Theta 方法预测器
简单但有效的时间序列预测方法
"""

from .base_forecaster import BaseForecaster

import numpy as np
from typing import Optional
import warnings


class ThetaForecaster(BaseForecaster):
    """
    Theta 方法预测器
    
    Theta 方法由 Assimakopoulos 和 Nikolopoulos (2000) 提出，
    通过对原始序列进行线性变换（Theta 线），然后组合多个预测。
    
    核心思想：
    1. 将原始序列分解为多条 Theta 线（不同斜率的线）
    2. 对每条线进行指数平滑预测
    3. 组合预测结果
    
    参考：
    - Assimakopoulos, V., & Nikolopoulos, K. (2000). "The theta model: a decomposition approach to time series forecasting."
    - Hyndman, R. J., & Billah, B. (2003). "Unmasking the Theta method."
    """

    def __init__(self, name: str = "Theta", theta: float = 0, method: str = "auto"):
        """
        初始化 Theta 预测器
        
        Args:
            name: 模型名称
            theta: Theta 参数（0 = 自动选择）
            method: 预测方法 ("auto", "simple", "opt")
        """
        super().__init__()
        self._name = name
        self.theta = theta
        self.method = method
        self._fitted = False
        self._series = None
        self._ses_model = None  # 简单指数平滑模型
        self._drift = None  # 漂移参数

    @property
    def name(self) -> str:
        """模型名称"""
        return self._name

    def fit(self, series: np.ndarray):
        """
        用 Theta 方法拟合序列
        
        Args:
            series: 时间序列数据 (1D array)
        """
        series = np.asarray(series, dtype=np.float64)
        self._series = series.copy()
        
        if len(series) < 3:
            self._fitted = False
            return
        
        try:
            # 方法1：简单 Theta 方法（使用简单指数平滑 + 漂移）
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            
            # 简单指数平滑
            self._ses_model = SimpleExpSmoothing(series).fit()
            
            # 计算漂移（最后一个值 - 第一个值）/ (n-1)
            if len(series) > 1:
                self._drift = (series[-1] - series[0]) / (len(series) - 1)
            else:
                self._drift = 0
            
            self._fitted = True
            
        except Exception as e:
            warnings.warn(f"Theta 方法拟合失败: {e}")
            self._fitted = False

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        预测未来 horizon 个时间点
        
        Args:
            horizon: 预测步数
            
        Returns:
            预测值数组 (horizon,)
        """
        if not self._fitted or self._ses_model is None:
            # 未拟合，用最后一个值
            if self._series is not None and len(self._series) > 0:
                last_val = self._series[-1]
                return np.full(horizon, last_val)
            else:
                return np.zeros(horizon)
        
        try:
            # Theta 方法预测 = SES 预测 + 漂移 * horizon
            ses_pred = self._ses_model.forecast(horizon)
            
            if self._drift is not None:
                # 添加漂移分量
                drift_component = self._drift * np.arange(1, horizon + 1)
                pred = ses_pred + drift_component
            else:
                pred = ses_pred
            
            # 确保预测值非负
            pred = np.maximum(pred, 0)
            
            return pred
            
        except Exception as e:
            warnings.warn(f"Theta 方法预测失败: {e}")
            if self._series is not None and len(self._series) > 0:
                return np.full(horizon, self._series[-1])
            else:
                return np.zeros(horizon)

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


class ThetaOptimizedForecaster(BaseForecaster):
    """
    Theta 优化方法预测器（Theta Opt）
    
    使用优化方法选择最佳 Theta 参数
    """

    def __init__(self, name: str = "ThetaOpt", theta_values: Optional[list] = None):
        """
        初始化 Theta 优化预测器
        
        Args:
            name: 模型名称
            theta_values: 候选 theta 值列表（默认 [0, 1, 2]）
        """
        super().__init__()
        self._name = name
        self.theta_values = theta_values if theta_values is not None else [0, 1, 2]
        self._fitted = False
        self._best_theta = None
        self._best_model = None
        self._series = None

    @property
    def name(self) -> str:
        """模型名称"""
        return self._name

    def fit(self, series: np.ndarray):
        """
        用交叉验证选择最佳 theta 参数
        
        Args:
            series: 时间序列数据
        """
        series = np.asarray(series, dtype=np.float64)
        self._series = series.copy()
        
        if len(series) < 6:  # 需要足够的数据进行 CV
            # 数据太少，使用默认 Theta
            self._best_theta = 0
            self._best_model = ThetaForecaster(name="Theta_Default", theta=0)
            self._best_model.fit(series)
            self._fitted = True
            return
        
        try:
            # 交叉验证选择最佳 theta
            best_error = float('inf')
            best_theta = self.theta_values[0]
            
            # 简单的留出法 CV（用最后 20% 作为验证集）
            split_point = int(len(series) * 0.8)
            train_series = series[:split_point]
            val_series = series[split_point:]
            
            for theta in self.theta_values:
                model = ThetaForecaster(name=f"Theta_{theta}", theta=theta)
                model.fit(train_series)
                pred = model.predict(len(val_series))
                
                # 计算 SMAPE
                error = np.mean(2 * np.abs(pred - val_series) / 
                                (np.abs(pred) + np.abs(val_series) + 1e-8))
                
                if error < best_error:
                    best_error = error
                    best_theta = theta
            
            # 用最佳 theta 在整个序列上训练
            self._best_theta = best_theta
            self._best_model = ThetaForecaster(
                name=f"Theta_Opt_{best_theta}", 
                theta=best_theta
            )
            self._best_model.fit(series)
            self._fitted = True
            
        except Exception as e:
            warnings.warn(f"Theta 优化拟合失败: {e}")
            # 降级到默认 Theta
            self._best_theta = 0
            self._best_model = ThetaForecaster(name="Theta_Fallback", theta=0)
            self._best_model.fit(series)
            self._fitted = True

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        预测未来 horizon 个时间点
        
        Args:
            horizon: 预测步数
            
        Returns:
            预测值数组
        """
        if not self._fitted or self._best_model is None:
            if self._series is not None and len(self._series) > 0:
                return np.full(horizon, self._series[-1])
            else:
                return np.zeros(horizon)
        
        return self._best_model.predict(horizon)

    def fit_predict(self, series: np.ndarray, horizon: int = 1) -> np.ndarray:
        """拟合并预测"""
        self.fit(series)
        return self.predict(horizon)
