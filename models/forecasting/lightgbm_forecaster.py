"""
LightGBM 基准预测器

使用 LightGBM 回归模型，基于滞后特征进行时间序列预测。
滞后特征包括：最近 N 周的销量、移动平均、趋势等。
"""

import numpy as np
import warnings
from typing import Optional, List


class LightGBMForecaster:
    """
    LightGBM 基准预测器
    
    使用 LightGBM 回归模型进行时间序列预测。
    特征工程：
    - 最近 K 周的销量（滞后特征）
    - 最近 K 周的移动平均
    - 最近 K 周的滚动标准差
    - 周序号（周期性）
    """
    
    def __init__(self, name: str = "LightGBM", 
                 n_lags: int = 4,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 num_leaves: int = 31,
                 random_state: int = 42):
        """
        初始化 LightGBM 预测器
        
        Args:
            name: 模型名称
            n_lags: 使用的滞后周数
            n_estimators: 树的数量
            max_depth: 最大深度
            learning_rate: 学习率
            num_leaves: 叶子数量
            random_state: 随机种子
        """
        self._name = name
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        
        self._model = None
        self._last_series = None
        self._fitted = False
        self._feature_names = None
    
    @property
    def name(self) -> str:
        """返回模型名称"""
        return self._name
    
    def _create_features(self, series: np.ndarray) -> np.ndarray:
        """
        从时间序列创建特征矩阵
        
        Args:
            series: 时间序列，形状为 (n_samples,)
            
        Returns:
            X: 特征矩阵，形状为 (n_samples - n_lags, n_features)
        """
        n = len(series)
        if n <= self.n_lags:
            return np.array([]).reshape(0, 0)
        
        features = []
        for i in range(self.n_lags, n):
            row = []
            # 1. 滞后特征：最近 n_lags 周的销量
            for lag in range(1, self.n_lags + 1):
                row.append(series[i - lag])
            
            # 2. 移动平均特征
            if i >= 4:
                row.append(np.mean(series[i-4:i]))
            else:
                row.append(np.mean(series[:i]))
                
            # 3. 滚动标准差（波动性）
            if i >= 4:
                row.append(np.std(series[i-4:i]))
            else:
                row.append(np.std(series[:i]) if i > 1 else 0.0)
                
            # 4. 趋势特征（最近 2 周 vs 前 2 周）
            if i >= 4:
                trend = np.mean(series[i-2:i]) - np.mean(series[i-4:i-2])
                row.append(trend)
            else:
                row.append(0.0)
                
            features.append(row)
            
        # 特征名称
        self._feature_names = [f'lag_{lag}' for lag in range(1, self.n_lags + 1)] + \
                            ['ma_4', 'std_4', 'trend']
            
        return np.array(features)
    
    def fit(self, series: np.ndarray) -> None:
        """
        拟合 LightGBM 模型
        
        Args:
            series: 历史销量序列
        """
        try:
            import lightgbm as lgb
        except ImportError:
            warnings.warn("lightgbm 未安装，请运行: pip install lightgbm")
            self._fitted = False
            return
            
        series = np.asarray(series, dtype=np.float64)
        self._last_series = series.copy()
        
        if len(series) <= self.n_lags + 1:
            warnings.warn(f"序列长度 {len(series)} 太短，无法训练 LightGBM")
            self._fitted = False
            return
            
        # 创建特征和目标
        X = self._create_features(series)
        if X.shape[0] == 0:
            warnings.warn("无法创建有效特征，LightGBM 训练失败")
            self._fitted = False
            return
            
        # 目标：下一周的销量
        y = series[self.n_lags:]
        
        # 确保 X 和 y 长度一致
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        if len(X) < 10:
            warnings.warn(f"训练样本数 {len(X)} 太少，LightGBM 训练可能不稳定")
            # 仍继续训练，但发出警告
            
        # 创建 LightGBM 数据集
        train_data = lgb.Dataset(X, label=y, feature_name=self._feature_names)
        
        # LightGBM 参数
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'verbosity': -1,
            'n_jobs': -1,
        }
        
        # 训练模型
        try:
            self._model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=None,
            )
            self._fitted = True
        except Exception as e:
            warnings.warn(f"LightGBM 训练失败: {e}")
            self._fitted = False
    
    def predict(self, horizon: int = 4) -> np.ndarray:
        """
        预测未来 horizon 周销量
        
        LightGBM 是单步预测模型，对于多步预测：
        - 第 1 步：使用历史序列的最后 n_lags 个值作为特征
        - 第 2 步及以后：使用预测值填充滞后特征（递归预测）
        
        Args:
            horizon: 预测步数（周数）
            
        Returns:
            预测值数组，形状为 (horizon,)
        """
        if not self._fitted or self._model is None:
            # 未拟合，返回 LastValue
            if self._last_series is not None:
                return np.full(horizon, self._last_series[-1])
            return np.zeros(horizon)
            
        # 递归预测
        series_extended = self._last_series.copy()
        predictions = []
        
        for h in range(horizon):
            # 创建当前特征（使用最近 n_lags 个值）
            if len(series_extended) < self.n_lags:
                # 不够，用 0 填充
                lags = [0.0] * (self.n_lags - len(series_extended)) + list(series_extended)
            else:
                lags = series_extended[-self.n_lags:].tolist()
                
            # 构建特征向量
            features = list(lags)
            
            # 移动平均
            if len(series_extended) >= 4:
                features.append(np.mean(series_extended[-4:]))
            else:
                features.append(np.mean(series_extended))
                
            # 滚动标准差
            if len(series_extended) >= 4:
                features.append(np.std(series_extended[-4:]))
            else:
                features.append(np.std(series_extended) if len(series_extended) > 1 else 0.0)
                
            # 趋势
            if len(series_extended) >= 4:
                trend = np.mean(series_extended[-2:]) - np.mean(series_extended[-4:-2])
                features.append(trend)
            else:
                features.append(0.0)
                
            X_pred = np.array(features).reshape(1, -1)
            
            # 预测
            pred = self._model.predict(X_pred)[0]
            predictions.append(pred)
            
            # 将预测值添加到序列中（用于下一步预测）
            series_extended = np.append(series_extended, pred)
            
        return np.array(predictions)
    
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
        return f"{self._name}(lags={self.n_lags}, trees={self.n_estimators})"
    
    def __repr__(self) -> str:
        return self.__str__()
