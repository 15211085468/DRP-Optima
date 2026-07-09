"""
models/forecasting子模块 - 时序预测模型

主模型
------
- TFTForecaster          : Temporal Fusion Transformer（主预测模型）
- HyperparameterOptimizer : Optuna 超参优化器

对比模型（即插即用，继承 BaseForecaster）
------------------------------------------
- BaseForecaster     : 统一接口基类
- ARIMAForecaster    : ARIMA(p,d,q)，基于 statsmodels
- ExpSmoothingForecaster : 指数平滑（趋势 + 季节性）
- LSTMForecaster        : LSTM 序列预测（PyTorch）
- LastValueForecaster: T-1 Naive Baseline
- CrostonForecaster  : Croston 方法（间歇性需求预测）
- ThetaForecaster     : Theta 方法（简单有效）
- ThetaOptimizedForecaster: Theta 优化方法（CV 选择最佳 theta）
- NHiTSForecaster   : NHiTS 神经网络（需要 pytorch-forecasting）
- NBeatsForecaster  : NBeats 神经网络（需要 pytorch-forecasting，可选）
- WeightedMovingAverageForecaster: 加权移动平均（公司现有算法）
- LightGBMForecaster        : LightGBM 梯度提升树基准模型
"""

try:
    from .tft_model import TFTForecaster
    from .hyperparameter_optimizer import HyperparameterOptimizer
except Exception:
    # torch 未安装时，TFT 相关模型不可用（不影响其他对比模型）
    TFTForecaster = None
    HyperparameterOptimizer = None

# 对比模型（无 torch 依赖）
from .base_forecaster import BaseForecaster
from .last_value_forecaster import LastValueForecaster
from .arima_forecaster import ARIMAForecaster
from .exponential_smoothing_forecaster import ExpSmoothingForecaster
from .croston_forecaster import CrostonForecaster
from .theta_forecaster import ThetaForecaster, ThetaOptimizedForecaster

# 加权移动平均（公司现有算法）
from .weighted_moving_average import WeightedMovingAverageForecaster

# LightGBM 基准模型（可选，需要 lightgbm）
try:
    from .lightgbm_forecaster import LightGBMForecaster
except Exception:
    LightGBMForecaster = None

# LSTM 可选（需要 torch）
try:
    from .lstm_forecaster import LSTMForecaster
except Exception:
    LSTMForecaster = None

# NHiTS/NBeats 可选（需要 pytorch-forecasting + torch）
try:
    from .neural_forecasters import NHiTSForecaster
except Exception:
    NHiTSForecaster = None

try:
    from .neural_forecasters import NBeatsForecaster
except Exception:
    NBeatsForecaster = None

__all__ = [
    # 主模型
    "TFTForecaster",
    "HyperparameterOptimizer",
    # 对比模型
    "BaseForecaster",
    "LastValueForecaster",
    "ARIMAForecaster",
    "ExpSmoothingForecaster",
    "CrostonForecaster",
    "ThetaForecaster",
    "ThetaOptimizedForecaster",
    "WeightedMovingAverageForecaster",
    "LightGBMForecaster",
    "LSTMForecaster",
    "NHiTSForecaster",
    "NBeatsForecaster",
]
