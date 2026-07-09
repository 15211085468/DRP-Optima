"""
NHiTS 和 NBeats 神经网络预测器
使用 pytorch-forecasting 库实现
"""

from .base_forecaster import BaseForecaster

try:
    from pytorch_forecasting import NHiTS, NBeats
    from pytorch_forecasting.data import TimeSeriesDataSet
    from pytorch_forecasting.metrics import SMAPE
    import torch
    import lightning.pytorch as pl
    _HAS_PTF = True
except ImportError:
    _HAS_PTF = False

import numpy as np
import pandas as pd
import warnings
import logging

logger = logging.getLogger(__name__)


class NHiTSForecaster(BaseForecaster):
    """
    NHiTS (Neural Hierarchical Interpolation for Time Series) 预测器
    适合周期性强的零售销量预测
    """

    def __init__(
        self,
        name: str = "NHiTS",
        max_encoder_length: int = 52,
        min_encoder_length: int = 12,
        max_prediction_length: int = 4,
        hidden_size: int = 64,
        num_blocks: int = 2,
        lr: float = 1e-3,
        max_epochs: int = 30,
    ):
        super().__init__()
        self._name = name
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.lr = lr
        self.max_epochs = max_epochs
        self._fitted = False
        self._model = None
        self._training_dataset = None
        self._last_series = None

    @property
    def name(self) -> str:
        """模型名称"""
        return self._name

    def fit(self, series: np.ndarray):
        """用给定序列训练 NHiTS 模型"""
        if not _HAS_PTF:
            raise RuntimeError("pytorch-forecasting 未安装，无法使用 NHiTS")
        
        series = np.asarray(series, dtype=np.float64)
        self._last_series = series.copy()
        
        # 计算有效的编码器长度
        available_len = len(series) - self.max_prediction_length
        if available_len < self.min_encoder_length:
            warnings.warn(f"序列长度 {len(series)} 太短，跳过 NHiTS 训练")
            self._fitted = False
            return
        
        effective_encoder = min(available_len, self.max_encoder_length)
        
        # 构造训练数据（单个时间序列）
        df = pd.DataFrame({
            "time_idx": np.arange(len(series)),
            "target": series,
            "group": 0,  # 单变量序列用固定 group=0（整数）
        })
        
        try:
            print(f"[NHiTS] 开始训练，序列长度={len(series)}, effective_encoder={effective_encoder}")
            # TimeSeriesDataSet 需要 group_ids，但 NHiTS 模型只接受 target
            # 解决方案：设置 group_ids=[]（空列表）
            self._training_dataset = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="target",
                group_ids=["group"],  # 设置 group_ids
                max_encoder_length=effective_encoder,
                min_encoder_length=effective_encoder,
                max_prediction_length=self.max_prediction_length,
            )
            
            train_loader = self._training_dataset.to_dataloader(
                train=True, batch_size=32, shuffle=True
            )
            
            # NHiTS 参数：n_blocks 必须是列表！
            # 例如：n_blocks=[1, 1, 1] 表示 3 个 stack，每个 1 个 block
            self._model = NHiTS.from_dataset(
                self._training_dataset,
                # NHiTS 特定参数
                n_blocks=[self.num_blocks] * 3,  # 3 个 stack，每个 stack 有 num_blocks 个 block 
                n_layers=2,  # 每个 block 的层数（默认 2）
                hidden_size=self.hidden_size,
                learning_rate=self.lr,
                log_interval=-1,
            )
            
            # 使用 lightning Trainer 训练
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(self._model, train_loader)
            
            self._fitted = True
            print(f"[NHiTS] 训练成功")
            
        except Exception as e:
            warnings.warn(f"NHiTS 训练失败: {e}")
            print(f"[NHiTS] 训练失败: {e}")
            self._fitted = False

    def predict(self, horizon: int = 4) -> np.ndarray:
        """预测未来 horizon 个时间点"""
        if not self._fitted or self._model is None:
            # 回落到 naive 预测
            return self._naive_forecast(horizon)
        
        try:
            # 构造预测输入
            if self._last_series is None or len(self._last_series) == 0:
                return np.zeros(horizon)
            
            # 使用最后一个窗口进行预测
            test_data = pd.DataFrame({
                "time_idx": np.arange(len(self._last_series)),
                "target": self._last_series,
                # NHiTS 不需要 group 列
            })
            
            test_dataset = TimeSeriesDataSet.from_dataset(
                self._training_dataset,
                test_data,
                predict=True,
                stop_normalization=True,
            )
            test_loader = test_dataset.to_dataloader(
                train=False, batch_size=1, shuffle=False
            )
            
            # 预测
            self._model.eval()
            predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    x, _ = batch
                    out = self._model(x)
                    predictions.append(out.numpy())
            
            if predictions:
                pred = np.concatenate(predictions, axis=0)
                return pred.flatten()[:horizon]
            else:
                return self._naive_forecast(horizon)
                
        except Exception as e:
            warnings.warn(f"NHiTS 预测失败: {e}")
            return self._naive_forecast(horizon)

    def fit_predict(self, series: np.ndarray, horizon: int = 4) -> np.ndarray:
        """训练并预测（单次调用）"""
        self.fit(series)
        return self.predict(horizon)

    def _naive_forecast(self, horizon: int) -> np.ndarray:
        """Naive 预测：用最后一个值"""
        if self._last_series is not None and len(self._last_series) > 0:
            last_val = self._last_series[-1]
            return np.full(horizon, last_val)
        else:
            return np.zeros(horizon)


class NBeatsForecaster(BaseForecaster):
    """
    NBeats (Neural Basis Expansion Analysis) 预测器
    适合单变量时间序列预测
    """

    def __init__(
        self,
        name: str = "NBeats",
        max_encoder_length: int = 52,
        min_encoder_length: int = 12,
        max_prediction_length: int = 4,
        hidden_size: int = 64,
        num_blocks: int = 2,
        lr: float = 1e-3,
        max_epochs: int = 30,
    ):
        super().__init__()
        self._name = name
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.max_prediction_length = max_prediction_length
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.lr = lr
        self.max_epochs = max_epochs
        self._fitted = False
        self._model = None
        self._training_dataset = None
        self._last_series = None

    @property
    def name(self) -> str:
        """模型名称"""
        return self._name

    def fit(self, series: np.ndarray):
        """训练 NBeats 模型"""
        if not _HAS_PTF:
            raise RuntimeError("pytorch-forecasting 未安装，无法使用 NBeats")
        
        series = np.asarray(series, dtype=np.float64)
        self._last_series = series.copy()
        
        available_len = len(series) - self.max_prediction_length
        if available_len < self.min_encoder_length:
            warnings.warn(f"序列长度 {len(series)} 太短，跳过 NBeats 训练")
            self._fitted = False
            return
        
        effective_encoder = min(available_len, self.max_encoder_length)
        
        df = pd.DataFrame({
            "time_idx": np.arange(len(series)),
            "target": series,
            "group": 0,  # 单变量序列用固定 group=0（整数）
        })
        
        try:
            print(f"[NBeats] 开始训练，序列长度={len(series)}, effective_encoder={effective_encoder}")
            # TimeSeriesDataSet 必须设置 group_ids
            # NBeats 要求 target 在 time_varying_unknown_reals 中
            self._training_dataset = TimeSeriesDataSet(
                df,
                time_idx="time_idx",
                target="target",
                group_ids=["group"],  # 必须有
                time_varying_unknown_reals=["target"],  # NBeats 要求
                max_encoder_length=effective_encoder,
                min_encoder_length=effective_encoder,
                max_prediction_length=self.max_prediction_length,
            )
            
            train_loader = self._training_dataset.to_dataloader(
                train=True, batch_size=32, shuffle=True
            )
            
            # NBeats 参数（使用默认值，让模型自动选择架构）
            self._model = NBeats.from_dataset(
                self._training_dataset,
                learning_rate=self.lr,
                log_interval=-1,
            )
            
            trainer = pl.Trainer(
                max_epochs=self.max_epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(self._model, train_loader)
            
            self._fitted = True
            print(f"[NBeats] 训练成功")
            
        except Exception as e:
            warnings.warn(f"NBeats 训练失败: {e}")
            print(f"[NBeats] 训练失败: {e}")
            self._fitted = False

    def predict(self, horizon: int = 4) -> np.ndarray:
        """预测未来 horizon 个时间点"""
        if not self._fitted or self._model is None:
            return self._naive_forecast(horizon)
        
        try:
            if self._last_series is None or len(self._last_series) == 0:
                return np.zeros(horizon)
            
            # 构造预测输入：用整个序列作为历史数据
            test_data = pd.DataFrame({
                "time_idx": np.arange(len(self._last_series)),
                "target": self._last_series,
                "group": 0,
            })
            
            # 使用训练数据集的配置创建测试数据集
            test_dataset = TimeSeriesDataSet.from_dataset(
                self._training_dataset,
                test_data,
                predict=True,
            )
            test_loader = test_dataset.to_dataloader(
                train=False, batch_size=1, shuffle=False
            )
            
            # 使用 pytorch_forecasting 的 predict 方法
            self._model.eval()
            predictions = self._model.predict(test_loader)
            
            # 提取预测结果（正确方式）
            if predictions is not None:
                # predictions 是张量，形状为 (batch, horizon, ...)
                pred = predictions.numpy().flatten()
                return pred[:horizon]
            else:
                return self._naive_forecast(horizon)
                
        except Exception as e:
            warnings.warn(f"NBeats 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return self._naive_forecast(horizon)

    def fit_predict(self, series: np.ndarray, horizon: int = 4) -> np.ndarray:
        """训练并预测"""
        self.fit(series)
        return self.predict(horizon)

    def _naive_forecast(self, horizon: int) -> np.ndarray:
        """Naive 预测"""
        if self._last_series is not None and len(self._last_series) > 0:
            last_val = self._last_series[-1]
            return np.full(horizon, last_val)
        else:
            return np.zeros(horizon)
