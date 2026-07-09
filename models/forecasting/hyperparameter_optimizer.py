"""
超参数优化器 - models/forecasting/hyperparameter_optimizer.py
基于Optuna的TFT超参数自动搜索
"""

import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE, QuantileLoss
from typing import Dict, List, Optional, Tuple, Any
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 定义简化的 OptunaConfig 类（避免导入错误）
from dataclasses import dataclass

@dataclass
class OptunaConfig:
    """Optuna 超参数优化配置类（简化版）"""
    n_trials: int = 20
    timeout: Optional[int] = None  # 超时时间（秒）
    gradient_clip_range: Tuple[float, float] = (0.1, 1.0)
    hidden_size_range: Tuple[int, int] = (16, 256)
    attention_head_range: Tuple[int, int] = (1, 8)
    learning_rate_range: Tuple[float, float] = (1e-4, 1e-2)
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    study_direction: str = "minimize"  # 最小化验证损失

# 现在导入其他配置类
from config import TFTModelConfig, HardwareConfig
from config.exceptions import UnsupportedLossError


class HyperparameterOptimizer:
    """基于Optuna的超参数优化器"""
    
    def __init__(self, 
                 optuna_config: Optional[OptunaConfig] = None,
                 model_config: Optional[TFTModelConfig] = None):
        """
        初始化超参数优化器
        
        Args:
            optuna_config: Optuna配置
            model_config: 模型配置
        """
        self.optuna_config = optuna_config or OptunaConfig()
        self.model_config = model_config or TFTModelConfig()
        self.hardware_config = HardwareConfig()
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None
    
    def objective(self, 
                 trial: optuna.Trial,
                 train_dataloaders,
                 val_dataloaders,
                 log_dir: str = "optuna_tft_smape") -> float:
        """
        Optuna优化目标函数
        
        Args:
            trial: Optuna试验对象
            train_dataloaders: 训练数据加载器
            val_dataloaders: 验证数据加载器
            log_dir: 日志目录
        
        Returns:
            float: 验证损失
        """
        cfg = self.optuna_config
        
        # 定义搜索空间
        gradient_clip_val = trial.suggest_float(
            "gradient_clip_val",
            cfg.gradient_clip_range[0],
            cfg.gradient_clip_range[1],
            log=True
        )
        
        hidden_size = trial.suggest_int(
            "hidden_size",
            cfg.hidden_size_range[0],
            cfg.hidden_size_range[1]
        )
        
        attention_head_size = trial.suggest_int(
            "attention_head_size",
            cfg.attention_head_range[0],
            cfg.attention_head_range[1]
        )
        
        learning_rate = trial.suggest_float(
            "learning_rate",
            cfg.learning_rate_range[0],
            cfg.learning_rate_range[1],
            log=True
        )
        
        dropout = trial.suggest_float(
            "dropout",
            cfg.dropout_range[0],
            cfg.dropout_range[1]
        )
        
        hidden_continuous_size = trial.suggest_int(
            "hidden_continuous_size",
            cfg.hidden_size_range[0],
            cfg.hidden_size_range[1]
        )
        
        # 根据配置选择损失函数
        model_config = self.model_config
        if model_config.loss_type == "quantile":
            loss = QuantileLoss(quantiles=model_config.quantiles)
        elif model_config.loss_type == "smape":
            loss = SMAPE()
        else:
            raise UnsupportedLossError(f"Unsupported loss type: {model_config.loss_type}")
        
        # 创建模型
        model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=loss,
            optimizer="Adam",  # 稳定选择
            reduce_on_plateau_patience=4
        )
        
        # 从配置中获取设备设置
        hw_cfg = self.hardware_config
        accel = "gpu" if hw_cfg.device == "cuda" else "auto"
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=cfg.search_max_epochs,
            accelerator=accel,
            gradient_clip_val=gradient_clip_val,
            limit_train_batches=cfg.search_limit_train_batches,
            callbacks=[
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", 
                    patience=3, 
                    verbose=False,
                    mode="min"
                )
            ],
            logger=pl.loggers.TensorBoardLogger(log_dir),
            enable_checkpointing=False,
            log_every_n_steps=5
        )
        
        # 训练
        trainer.fit(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders
        )
        
        # 返回验证损失
        return trainer.callback_metrics["val_loss"].item()
    
    def optimize(self,
                train_dataloaders,
                val_dataloaders,
                training_dataset: TimeSeriesDataSet,
                study_name: str = "tft_optimization",
                storage: Optional[str] = None,
                n_trials: Optional[int] = None,
                timeout: Optional[int] = None,
                n_jobs: int = 1) -> Dict[str, Any]:
        """
        运行超参数优化
        
        Args:
            train_dataloaders: 训练数据加载器
            val_dataloaders: 验证数据加载器
            training_dataset: 训练数据集
            study_name: 研究名称
            storage: 存储后端
            n_trials: 试验数量
            timeout: 超时时间（秒）
            n_jobs: 并行数
        
        Returns:
            Dict: 最佳超参数
        """
        if n_trials is None:
            n_trials = self.optuna_config.n_trials
        if timeout is None:
            timeout = self.optuna_config.timeout
        if storage is None:
            storage = f"sqlite:///{study_name}.db"
        
        self.training_dataset = training_dataset
        
        # 创建目标函数
        objective_fn = partial(
            self.objective,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            log_dir=self.optuna_config.study_save_dir
        )
        
        # 创建学习器
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        self.study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage,
            pruner=pruner,
            load_if_exists=True
        )
        
        # 运行优化
        self.study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # 保存最佳参数
        self.best_params = self.study.best_params
        print(f"最佳参数: {self.best_params}")
        print(f"最佳验证损失: {self.study.best_value:.4f}")
        
        return self.best_params
    
    def get_best_params(self) -> Dict[str, Any]:
        """获取最佳参数"""
        return self.best_params or {}
    
    def get_study_stats(self) -> Dict[str, Any]:
        """
        获取研究统计信息
        
        Returns:
            Dict: 统计信息
        """
        if self.study is None:
            return {}
        
        return {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial': self.study.best_trial.number,
            'n_complete_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        }
    
    def get_param_importance(self) -> Dict[str, float]:
        """
        获取参数重要性
        
        Returns:
            Dict: 参数名 -> 重要性
        """
        if self.study is None:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception as e:
            print(f"计算参数重要性失败: {e}")
            return {}
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        绘制优化历史
        
        Args:
            save_path: 保存路径
        """
        if self.study is None:
            return
        
        fig = optuna.visualization.plot_optimization_history(self.study)
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    
    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        绘制参数重要性
        
        Args:
            save_path: 保存路径
        """
        if self.study is None:
            return
        
        fig = optuna.visualization.plot_param_importances(self.study)
        
        if save_path:
            fig.write_image(save_path)
        
        return fig
    
    def suggest_params_for_training(self) -> Dict[str, Any]:
        """
        获取用于训练的参数
        
        Returns:
            Dict: 训练参数
        """
        best = self.get_best_params()
        
        # 构建完整的模型参数
        training_params = {
            'learning_rate': best.get('learning_rate', self.model_config.learning_rate),
            'hidden_size': best.get('hidden_size', self.model_config.hidden_size),
            'attention_head_size': best.get('attention_head_size', self.model_config.attention_head_size),
            'dropout': best.get('dropout', self.model_config.dropout),
            'hidden_continuous_size': best.get('hidden_continuous_size', self.model_config.hidden_continuous_size),
            'gradient_clip_val': best.get('gradient_clip_val', self.model_config.gradient_clip_val)
        }
        
        return training_params
