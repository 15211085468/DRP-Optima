"""
LSTM 预测器
--------------

简单的序列到序列 LSTM 预测器，用于基线对比。

接口与其他 forecaster 保持一致：
    model = LSTMForecaster(name="LSTM")
    pred  = model.fit_predict(hist, horizon=4)
"""

from __future__ import annotations

import logging
import numpy as np
import warnings

from .base_forecaster import BaseForecaster

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True

    class _LSTMModel(nn.Module):
        """简单的 LSTM 序列预测模型。"""

        def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, output_len: int = 4):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_len = output_len
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, output_len)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size)
            out, (h_n, c_n) = self.lstm(x)
            # 取最后一个时间步的输出
            out_last = out[:, -1, :]          # (batch, hidden_size)
            pred = self.fc(out_last)            # (batch, output_len)
            return pred

except ImportError:
    _TORCH_OK = False
    _LSTMModel = None


class LSTMForecaster(BaseForecaster):
    """
    简单的序列到序列 LSTM 预测器，用于基线对比。

    接口与其他 forecaster 保持一致：
        model = LSTMForecaster(name="LSTM")
        pred  = model.fit_predict(hist, horizon=4)
    """

    def __init__(
        self,
        name: str = "LSTM",
        input_len: int = 12,
        hidden_size: int = 32,
        num_layers: int = 1,
        num_epochs: int = 30,
        lr: float = 1e-3,
    ):
        super().__init__()
        self._name = name
        self.input_len = input_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.lr = lr
        self._fitted = False
        self._model = None
        self._last_series = None
        self._train_min = None
        self._train_max = None

    @property
    def name(self) -> str:
        """模型名称（用于展示/保存）"""
        return self._name

    def fit(self, series: np.ndarray):
        """
        在单条时序上训练 LSTM 模型。

        Parameters
        ----------
        series : np.ndarray, shape (T,)
            周度销量序列，float64，允许含 0，不允许 NaN。
        """
        if not _TORCH_OK:
            raise RuntimeError("pytorch 未安装，无法使用 LSTMForecaster")
        
        series = np.asarray(series, dtype=np.float64)
        self._last_series = series.copy()
        
        # 检查训练样本数
        n_samples = len(series) - self.input_len
        if n_samples < 20:
            warnings.warn(f"训练样本太少 ({n_samples})，跳过 LSTM 训练")
            self._fitted = False
            return
        
        try:
            # 归一化：[0, 1]
            self._train_min = float(series.min())
            self._train_max = float(series.max())
            if self._train_max - self._train_min < 1e-8:
                # 常数序列
                self._fitted = False
                return
            series_norm = (series - self._train_min) / (self._train_max - self._train_min)
            
            # 准备训练数据
            X = []
            y = []
            
            for i in range(len(series_norm) - self.input_len):
                X.append(series_norm[i:i+self.input_len])
                y.append(series_norm[i+self.input_len])
            
            X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, input_len, 1)
            y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)    # (N, 1)
            
            # 创建模型（输出长度=1，单次预测1步）
            model = _LSTMModel(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_len=1,
            )
            
            # 训练
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(self.num_epochs):
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            
            self._model = model
            self._fitted = True
            
        except Exception as e:
            warnings.warn(f"LSTM 训练失败: {e}")
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
        if not self._fitted or self._model is None:
            # 未拟合，用最后一个值
            if self._last_series is not None and len(self._last_series) > 0:
                last_val = self._last_series[-1]
                return np.full(horizon, last_val)
            else:
                return np.zeros(horizon)
        
        try:
            self._model.eval()
            with torch.no_grad():
                # 递归预测
                preds = []
                # 归一化初始输入
                current_input_norm = (self._last_series[-self.input_len:] - self._train_min) / (self._train_max - self._train_min)
                current_input = torch.tensor(
                    current_input_norm, 
                    dtype=torch.float32
                ).unsqueeze(0).unsqueeze(-1)  # (1, input_len, 1)
                
                for i in range(horizon):
                    pred_norm = self._model(current_input)
                    pred_value_norm = pred_norm.item()
                    # 反归一化
                    pred_value = pred_value_norm * (self._train_max - self._train_min) + self._train_min
                    preds.append(pred_value)
                    
                    # 更新输入：移除第一个，添加归一化的预测值
                    pred_norm_clipped = np.clip(pred_value_norm, 0, 1)
                    current_input = torch.cat([
                        current_input[:, 1:, :],
                        torch.tensor([[[pred_norm_clipped]]], dtype=torch.float32)
                    ], dim=1)
                
                preds = np.array(preds)
                preds = np.maximum(preds, 0)  # 确保非负
                return preds
                
        except Exception as e:
            warnings.warn(f"LSTM 预测失败: {e}")
            if self._last_series is not None and len(self._last_series) > 0:
                return np.full(horizon, self._last_series[-1])
            else:
                return np.zeros(horizon)

    def fit_predict(self, series: np.ndarray, horizon: int = 1) -> np.ndarray:
        """便捷方法：先 fit 再 predict"""
        self.fit(series)
        return self.predict(horizon)
