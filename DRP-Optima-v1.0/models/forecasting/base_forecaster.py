"""
对比模型基类 - models/forecasting/base_forecaster.py

定义所有对比模型（ARIMA / NHiTS / NBeats / LastValue）的统一接口。
TFT 作为主模型不继承此类，但 baseline_comparison.py 中会用相同 API 调用。

接口约定
---------
fit(series: np.ndarray) -> None
    训练模型。series shape: (T,) 单变量时序（周度销量，已去趋势/不去趋势均可）。

predict(horizon: int) -> np.ndarray
    返回未来 horizon 步的点预测，shape: (horizon,)，值域 >= 0。

name: str
    模型名称，用于结果展示和文件命名。
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """所有对比预测器的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """模型名称（用于展示/保存）"""
        ...

    @abstractmethod
    def fit(self, series: np.ndarray) -> None:
        """
        在单条时序上训练/拟合模型。

        Parameters
        ----------
        series : np.ndarray, shape (T,)
            周度销量序列，float32/float64，允许含 0，不允许 NaN。
        """
        ...

    @abstractmethod
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
        ...

    def fit_predict(self, series: np.ndarray, horizon: int = 1) -> np.ndarray:
        """便捷方法：先 fit 再 predict"""
        self.fit(series)
        return self.predict(horizon)
