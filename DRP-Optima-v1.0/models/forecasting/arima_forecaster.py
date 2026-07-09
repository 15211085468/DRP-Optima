"""
ARIMA 对比模型 - models/forecasting/arima_forecaster.py

基于 statsmodels 的 ARIMA 实现，支持：
  - 手动指定阶数 (p, d, q)
  - 自动阶数搜索（AIC 准则，auto_order=True）

依赖：statsmodels >= 0.14

注意事项
--------
- 数据规模大时（>500 条 SKU），自动阶数搜索会显著增加运行时间。
  建议在 baseline_comparison.py 中用 ``auto_order=False`` + 固定阶数 (1,1,1)。
- ARIMA 对零膨胀序列（大量 0 销量）鲁棒性差，此时预测偏差较大属正常现象。
"""

import warnings
import numpy as np
from .base_forecaster import BaseForecaster

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA(p, d, q) 预测器

    Parameters
    ----------
    order : tuple (p, d, q), default (1, 1, 1)
        ARIMA 阶数。
    auto_order : bool, default False
        是否自动搜索最优阶数（AIC 准则）。开启后会覆盖 order 参数。
        候选 p, q ∈ {0,1,2}，d 由 ADF 检验决定（0 或 1）。
    max_p : int, default 2
        auto_order 时 p 的最大候选值。
    max_q : int, default 2
        auto_order 时 q 的最大候选值。
    """

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        auto_order: bool = False,
        max_p: int = 2,
        max_q: int = 2,
    ):
        if not _STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels 未安装，请运行：pip install statsmodels>=0.14"
            )
        # name 固定，不受拟合影响
        self._name = "ARIMA"
        self.order = order
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_q = max_q

        self._model_fit = None
        self._fitted_order: tuple = order
        self._fitted: bool = False

    @property
    def name(self) -> str:
        """固定返回 'ARIMA'，不因拟合而改变。"""
        return self._name

    @property
    def fitted_order(self) -> tuple:
        """拟合后返回实际使用的 (p, d, q)，未拟合返回初始 order。"""
        return self._fitted_order

    # ------------------------------------------------------------------
    # 辅助：ADF 检验确定差分阶数
    # ------------------------------------------------------------------
    @staticmethod
    def _adf_diff_order(series: np.ndarray) -> int:
        """ADF 检验，返回 0（平稳）或 1（需一阶差分）"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = adfuller(series, autolag="AIC")
            p_val = result[1]
            return 0 if p_val < 0.05 else 1
        except Exception:
            return 1

    # ------------------------------------------------------------------
    # 辅助：AIC 自动阶数搜索
    # ------------------------------------------------------------------
    def _auto_select_order(self, series: np.ndarray) -> tuple:
        """在候选范围内穷举，返回 AIC 最小的 (p, d, q)"""
        d = self._adf_diff_order(series)
        best_aic = np.inf
        best_order = (1, d, 1)

        for p in range(0, self.max_p + 1):
            for q in range(0, self.max_q + 1):
                if p == 0 and q == 0:
                    continue  # 纯常数模型，跳过
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = ARIMA(series, order=(p, d, q))
                        res = m.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

        return best_order

    # ------------------------------------------------------------------
    # fit / predict / fit_predict
    # ------------------------------------------------------------------
    def fit(self, series: np.ndarray) -> None:
        """
        在单条时序上训练 ARIMA 模型。

        Parameters
        ----------
        series : np.ndarray, shape (T,)
            周度销量序列。
        """
        series = np.asarray(series, dtype=np.float64)
        if len(series) < 4:
            raise ValueError(f"ARIMA 训练序列太短（{len(series)} 步），至少需要 4 步")

        # 防止全零序列导致数值问题：加微小扰动
        if np.allclose(series, 0.0):
            series = series + 1e-6

        # 选择阶数（自动或固定）
        if self.auto_order:
            self._fitted_order = self._auto_select_order(series)
        else:
            self._fitted_order = self.order

        # 拟合
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=self._fitted_order)
            self._model_fit = model.fit()

        self._fitted = True

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
        if not self._fitted:
            raise RuntimeError("请先调用 fit() 或 fit_predict()")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._model_fit.forecast(steps=horizon)
        preds = np.asarray(forecast, dtype=np.float64)
        return np.clip(preds, 0.0, None)

    def fit_predict(self, series: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        便捷方法：拟合并预测，但不改变 name 属性。

        覆盖基类方法，确保 name 在调用前后保持一致。
        """
        series = np.asarray(series, dtype=np.float64)
        if len(series) < 4:
            raise ValueError(f"ARIMA 序列太短（{len(series)} 步），至少需要 4 步")

        if np.allclose(series, 0.0):
            series = series + 1e-6

        # 选择阶数
        if self.auto_order:
            order = self._auto_select_order(series)
        else:
            order = self.order

        # 拟合并预测
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(series, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=horizon)

        # 保存拟合状态（供后续 predict() 调用）
        self._fitted_order = order
        self._model_fit = model_fit
        self._fitted = True

        preds = np.asarray(forecast, dtype=np.float64)
        return np.clip(preds, 0.0, None)
