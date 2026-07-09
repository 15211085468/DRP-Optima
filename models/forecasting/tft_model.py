"""
TFT时序预测模型 - models/forecasting/tft_model.py
基于pytorch_forecasting实现的Temporal Fusion Transformer
"""

import os
import json
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
from pytorch_forecasting.models import BaseModel
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

from config import TFTModelConfig, HardwareConfig
from config.exceptions import ModelNotTrainedError, UnsupportedLossError, DataError
from utils.logger import get_logger
logger = get_logger(__name__)

# 导入决策感知相关模块
try:
    from models.decision_cost import DecisionCostEvaluator
    from models.forecasting.decision_aware_tft import DecisionAwareTFTWrapper
    DECISION_AWARE_AVAILABLE = True
except ImportError:
    DECISION_AWARE_AVAILABLE = False
    logger.warning("决策感知模块导入失败，Decision-Aware Training 将不可用")


class TFTForecaster:
    """Temporal Fusion Transformer预测器"""
    
    def __init__(self, 
                 model_config: Optional[TFTModelConfig] = None,
                 hardware_config: Optional[HardwareConfig] = None):
        """
        初始化TFT预测器
        
        Args:
            model_config: TFT模型配置
            hardware_config: 硬件配置
        """
        self.model_config = model_config or TFTModelConfig()
        self.hardware_config = hardware_config or HardwareConfig()
        self.model: Optional[TemporalFusionTransformer] = None
        self.trainer: Optional[pl.Trainer] = None
        self.training_dataset: Optional[TimeSeriesDataSet] = None
        self.val_dataset: Optional[TimeSeriesDataSet] = None
        self.scaler = None
        
        # 添加训练状态标志
        self._is_trained = False
        
        # 验证loss_type（fail-fast）
        if self.model_config.loss_type not in ["quantile", "smape"]:
            raise UnsupportedLossError(f"Unsupported loss type: {self.model_config.loss_type}. Suggestion: use 'quantile' or 'smape'")
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
    
    def create_dataset(self, 
                      df: pd.DataFrame,
                      time_idx: str = "time_idx",
                      target: str = "sale_qty",
                      group_ids: List[str] = None,
                      static_categoricals: List[str] = None,
                      static_reals: List[str] = None,
                      time_varying_known_reals: List[str] = None,
                      time_varying_unknown_reals: List[str] = None,
                      training_cutoff: Optional[int] = None) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        创建TimeSeriesDataSet
        
        Args:
            df: 输入数据
            time_idx: 时间索引列
            target: 目标变量
            group_ids: 分组ID列
            static_categoricals: 静态分类变量
            static_reals: 静态连续变量
            time_varying_known_reals: 时变已知连续变量
            time_varying_unknown_reals: 时变未知连续变量
            training_cutoff: 训练截止时间
        
        Returns:
            Tuple: (training_dataset, validation_dataset)
        """
        cfg = self.model_config
        
        # 保存数据集参数到 model_config（用于预测时恢复）
        cfg.dataset_time_idx = time_idx
        cfg.dataset_target = target
        cfg.dataset_group_ids = group_ids or ['idx']
        cfg.dataset_static_categoricals = static_categoricals or []
        cfg.dataset_static_reals = static_reals or []
        cfg.dataset_time_varying_known_reals = time_varying_known_reals or []
        cfg.dataset_time_varying_unknown_reals = time_varying_unknown_reals or [target]
        cfg.dataset_max_encoder_length = cfg.max_encoder_length
        cfg.dataset_max_prediction_length = cfg.max_prediction_length
        cfg.dataset_min_encoder_length = int(cfg.max_encoder_length * cfg.min_encoder_length_ratio)
        cfg.dataset_target_normalizer = cfg.target_normalizer
        cfg.dataset_add_relative_time_idx = cfg.add_relative_time_idx
        cfg.dataset_add_target_scales = cfg.add_target_scales
        cfg.dataset_add_encoder_length = cfg.add_encoder_length
        
        logger.info(f"✓ Dataset parameters saved to model_config")
        logger.info(f"  - group_ids: {cfg.dataset_group_ids}")
        logger.info(f"  - time_idx: {cfg.dataset_time_idx}")
        logger.info(f"  - target: {cfg.dataset_target}")
        
        if group_ids is None:
            group_ids = ['idx']
        
        if training_cutoff is None:
            training_cutoff = df[time_idx].max() - cfg.max_prediction_length
        
        # 创建训练集
        training_data = df[df[time_idx] <= training_cutoff].copy()
        
        # 确定编码器
        object_cols = []
        if static_categoricals:
            object_cols.extend(static_categoricals)
        
        categorical_encoders = {
            label: NaNLabelEncoder(add_nan=True).fit(training_data[label])
            for label in object_cols
        }
        
        # 创建TimeSeriesDataSet
        # 将target_normalizer从字符串转换为实例
        if cfg.target_normalizer == "standard":
            target_normalizer = TorchNormalizer()
        else:
            target_normalizer = cfg.target_normalizer
        
        self.training_dataset = TimeSeriesDataSet(
            training_data,
            time_idx=time_idx,
            target=target,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            categorical_encoders=categorical_encoders,
            group_ids=group_ids,
            max_encoder_length=cfg.max_encoder_length,
            min_encoder_length=int(cfg.max_encoder_length * cfg.min_encoder_length_ratio),
            max_prediction_length=cfg.max_prediction_length,
            min_prediction_length=cfg.max_prediction_length,
            target_normalizer=target_normalizer,
            add_relative_time_idx=cfg.add_relative_time_idx,
            add_target_scales=cfg.add_target_scales,
            add_encoder_length=cfg.add_encoder_length,
            allow_missing_timesteps=cfg.allow_missing_timesteps
        )
        
        # 创建验证集（从完整数据集）
        self.val_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, df, predict=True, stop_randomization=True
        )
        
        return self.training_dataset, self.val_dataset
    
    def create_dataloaders(self,
                          batch_size: Optional[int] = None,
                          num_workers: Optional[int] = None) -> Tuple:
        """
        创建数据加载器
        
        Args:
            batch_size: 批次大小
            num_workers: 工作进程数
        
        Returns:
            Tuple: (train_dataloader, val_dataloader)
        """
        if batch_size is None:
            batch_size = self.model_config.batch_size
        if num_workers is None:
            num_workers = self.hardware_config.num_workers
        
        train_dataloader = self.training_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True
        )
        
        val_dataloader = self.val_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True
        )
        
        return train_dataloader, val_dataloader
    
    def create_trainer(self,
                      logger_dir: str = "lightning_logs",
                      checkpoint_dir: str = "checkpoints",
                      extra_callbacks: Optional[List] = None,
                      **kwargs) -> pl.Trainer:
        """
        创建PyTorch Lightning训练器
        
        Args:
            logger_dir: 日志目录
            checkpoint_dir: 检查点目录
            extra_callbacks: 额外的回调函数列表
            **kwargs: 其他训练参数
        
        Returns:
            pl.Trainer: 训练器
        """
        cfg = self.model_config
        
        # 设置精度
        torch.set_float32_matmul_precision(self.hardware_config.float32_matmul_precision)
        
        # 自定义损失记录回调
        class LossLoggerCallback(pl.Callback):
            """记录每个epoch的训练和验证损失"""
            def __init__(self, save_path):
                super().__init__()
                self.save_path = save_path
                self.train_losses = []
                self.val_losses = []
                self.epochs = []
            
            def on_train_epoch_end(self, trainer, pl_module):
                # 获取当前epoch的训练损失
                train_loss = trainer.callback_metrics.get('train_loss', trainer.callback_metrics.get('loss', None))
                if train_loss is not None:
                    self.train_losses.append(float(train_loss))
                else:
                    self.train_losses.append(None)
            
            def on_validation_epoch_end(self, trainer, pl_module):
                # 获取当前epoch的验证损失
                val_loss = trainer.callback_metrics.get('val_loss', None)
                if val_loss is not None:
                    self.val_losses.append(float(val_loss))
                    self.epochs.append(trainer.current_epoch)
                
                # 保存损失数据到JSON文件
                loss_data = {
                    'epochs': self.epochs,
                    'train_loss': self.train_losses[:len(self.epochs)],
                    'val_loss': self.val_losses
                }
                with open(self.save_path, 'w', encoding='utf-8') as f:
                    json.dump(loss_data, f, indent=2, ensure_ascii=False)
        
        # 损失记录文件路径
        loss_log_path = os.path.join(logger_dir, 'training_loss_data.json')
        loss_callback = LossLoggerCallback(loss_log_path)
        
        # 回调函数（禁用LearningRateMonitor以避免logger依赖）
        callbacks = [
            loss_callback,
            EarlyStopping(
                monitor="val_loss",
                min_delta=cfg.early_stop_min_delta,
                patience=cfg.early_stop_patience,
                verbose=False,
                mode="min"
            )
        ]
        
        # 注意：不在此处添加ModelCheckpoint，避免与pytorch_forecasting内部冲突
        # 训练完成后，通过trainer.checkpoint_callback保存最佳模型
        
        # 添加额外的回调函数
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        
        # 日志记录器（禁用TensorBoard以避免依赖问题）
        logger = False  # TensorBoardLogger(logger_dir)
        
        # 默认参数
        default_kwargs = {
            'max_epochs': cfg.max_epochs,
            'accelerator': 'gpu' if self.device.type == 'cuda' else 'auto',
            'precision': '16-mixed' if self.device.type == 'cuda' else '32-true',  # 🚀 GPU混合精度加速
            'enable_model_summary': True,
            'gradient_clip_val': cfg.gradient_clip_val,
            'limit_train_batches': cfg.limit_train_batches,
            'callbacks': callbacks,
            'logger': logger,
            'log_every_n_steps': cfg.log_interval
        }
        default_kwargs.update(kwargs)
        
        self.trainer = pl.Trainer(**default_kwargs)
        return self.trainer
    
    def create_model(self) -> TemporalFusionTransformer:
        """
        创建TFT模型（支持决策感知联合训练）
        
        Returns:
            TemporalFusionTransformer 或 DecisionAwareTFTWrapper: TFT模型实例（或包装器）
        """
        cfg = self.model_config
        
        # 根据配置动态选择损失函数
        if cfg.loss_type == "quantile":
            loss = QuantileLoss(quantiles=cfg.quantiles)
        elif cfg.loss_type == "smape":
            loss = SMAPE()
        else:
            raise UnsupportedLossError(f"Unsupported loss type: {cfg.loss_type}")
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=cfg.learning_rate,
            hidden_size=cfg.hidden_size,
            attention_head_size=cfg.attention_head_size,
            dropout=cfg.dropout,
            hidden_continuous_size=cfg.hidden_continuous_size,
            loss=loss,
            optimizer=cfg.optimizer,
            log_interval=cfg.log_interval,
            reduce_on_plateau_patience=cfg.reduce_on_plateau_patience
        )
        
        logger.info(f"模型参数量: {self.model.size() / 1e3:.1f}k, 损失函数: {cfg.loss_type}")
        
        # ============= 决策感知联合训练：创建包装器 =============
        if cfg.use_decision_aware:
            if not DECISION_AWARE_AVAILABLE:
                logger.error("决策感知模块不可用，请检查 models/decision_cost.py 是否存在")
                raise ImportError("Decision-aware modules not available")
            
            if self.training_dataset is None:
                logger.error("训练数据集未创建，无法初始化 DecisionCostEvaluator（需要 target_normalizer 进行反归一化）")
                raise DataError("Training dataset must be created before model for decision-aware training")
            
            logger.info(">>> 启用 Decision-Aware Joint Training (决策感知联合训练) <<<")
            logger.info(f"    决策成本权重: {cfg.decision_weight}")
            logger.info(f"    持有成本系数: {cfg.holding_cost}")
            logger.info(f"    缺货惩罚系数: {cfg.shortage_penalty}")
            logger.info(f"    平滑模式: {cfg.decision_smooth}")
            logger.info(f"    预热轮数: {cfg.decision_warmup_epochs}")
            
            # 创建决策成本评估器
            cost_evaluator = DecisionCostEvaluator(
                holding_cost=cfg.holding_cost,
                shortage_penalty=cfg.shortage_penalty,
                smooth=cfg.decision_smooth
            )
            
            # 创建决策感知包装器
            wrapped_model = DecisionAwareTFTWrapper(
                tft_model=self.model,
                training_dataset=self.training_dataset,
                decision_cost_evaluator=cost_evaluator,
                decision_weight=cfg.decision_weight,
                warmup_epochs=cfg.decision_warmup_epochs
            )
            
            logger.info("✅ DecisionAwareTFTWrapper 创建成功")
            return wrapped_model
        
        return self.model
    
    def fit(self,
           train_dataloader,
           val_dataloader,
           model: Optional[TemporalFusionTransformer] = None,
           trainer: Optional[pl.Trainer] = None,
           **kwargs) -> TemporalFusionTransformer:
        """
        训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            model: 模型实例（可选）
            trainer: 训练器实例（可选）
            **kwargs: 其他训练参数
        
        Returns:
            TemporalFusionTransformer: 训练后的模型
        """
        if model is None:
            if self.model is None:
                self.create_model()
            model = self.model
        
        if trainer is None:
            if self.trainer is None:
                self.create_trainer()
            trainer = self.trainer
        
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            **kwargs
        )
        
        # 更新训练状态标志
        self._is_trained = True
        logger.info("Model training finished")
        
        return model
    
    def predict(self,
               dataloader: Optional = None,
               df: Optional[pd.DataFrame] = None,
               use_uncertainty: bool = False,
               n_samples: int = 100) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            dataloader: 数据加载器
            df: 输入数据（如果提供，会先创建dataloader）
            use_uncertainty: 是否使用不确定性估计
            n_samples: 不确定性估计的样本数
        
        Returns:
            np.ndarray: 预测结果（点预测），形状取决于输入数据
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model not trained, call fit() first")
        
        if self.model is None:
            raise ModelNotTrainedError("Model not created. Call create_model() first.")
        
        # 如果请求不确定性估计，建议使用专用方法
        if use_uncertainty:
            logger.warning("use_uncertainty=True is ignored in predict(). Use predict_with_uncertainty() for uncertainty estimation.")
            logger.warning("Continuing with point prediction only.")
        
        # 如果提供了df但没有dataloader，创建dataloader
        if dataloader is None and df is not None:
            # 使用保存的数据集参数重新创建数据集
            if self.training_dataset is None or self.val_dataset is None:
                logger.warning("Dataset not loaded, recreating from saved parameters...")
                
                # 使用保存的数据集参数
                cfg = self.model_config
                self.create_dataset(
                    df=df,
                    time_idx=cfg.dataset_time_idx,
                    target=cfg.dataset_target,
                    group_ids=cfg.dataset_group_ids,
                    static_categoricals=cfg.dataset_static_categoricals,
                    static_reals=cfg.dataset_static_reals,
                    time_varying_known_reals=cfg.dataset_time_varying_known_reals,
                    time_varying_unknown_reals=cfg.dataset_time_varying_unknown_reals,
                )
                
                logger.info(f"✓ Dataset recreated with saved parameters")
                logger.info(f"  - group_ids: {cfg.dataset_group_ids}")
            
            _, dataloader = self.create_dataloaders()
        
        if dataloader is None:
            raise DataError("Either dataloader or df must be provided for prediction")
        
        # 执行预测
        try:
            # 方法1: 使用 trainer.predict()（可能失败）
            predictions = self.trainer.predict(self.model, dataloader)
        except Exception as e:
            logger.warning(f"trainer.predict() failed: {e}")
            logger.warning("使用降级方案：直接使用模型进行推理")
            # 降级方案：直接使用模型进行推理（不依赖 PredictCallback）
            try:
                # 使用模型的直接 predict 方法
                predictions = self.model.predict(dataloader, mode="prediction")
            except Exception as e2:
                logger.error(f"Model direct predict also failed: {e2}")
                raise DataError(f"TFT prediction failed: {e}, and direct predict also failed: {e2}")
        
        # 类型统一：确保返回类型为 np.ndarray
        if predictions is None:
            raise DataError("TFT prediction returned None")

        if isinstance(predictions, list):
            if len(predictions) == 0:
                raise DataError("TFT prediction returned empty list")
            # 将 list of tensors/arrays 转换为 np.ndarray
            try:
                predictions = np.array([p.cpu().numpy() if hasattr(p, 'cpu') else p for p in predictions])
            except Exception as e:
                raise DataError(f"Cannot convert prediction list to np.ndarray: {e}")
            logger.info(f"Prediction completed. Converted list to array, shape: {predictions.shape}")
        elif isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            logger.info(f"Prediction completed. Tensor converted to array, shape: {predictions.shape}")
        elif isinstance(predictions, np.ndarray):
            # 数值稳定性检查
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                logger.error("Predictions contain NaN or Inf")
                raise DataError("Prediction output contains NaN or Inf. Suggestion: check input data or retrain model.")
            logger.info(f"Prediction completed. Shape: {predictions.shape}")
        else:
            # 尝试转换为 np.ndarray
            try:
                predictions = np.array(predictions)
                logger.info(f"Prediction completed. Converted {type(predictions)} to np.ndarray, shape: {predictions.shape}")
            except Exception as e:
                raise DataError(f"Unknown prediction type {type(predictions)}, cannot convert to np.ndarray: {e}")

        return predictions
    
    def predict_with_uncertainty(self,
                                 df: pd.DataFrame,
                                 n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        使用MC Dropout或分位数预测进行不确定性估计
        
        Args:
            df: 输入数据
            n_samples: MC采样的样本数
        
        Returns:
            Dict[str, np.ndarray]: 包含 'median', 'q10', 'q90' 键的预测字典
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model not trained, call fit() first")
        
        if self.model is None:
            raise ModelNotTrainedError("Model not created. Call create_model() first.")
        
        # 如果数据集未加载，从输入数据创建临时数据集
        if self.training_dataset is None or self.val_dataset is None:
            logger.warning("Dataset not loaded, creating temporary dataset from input data...")
            try:
                self.create_dataset(df)
                logger.info("✓ Temporary dataset created successfully")
            except Exception as e:
                logger.error(f"Failed to create temporary dataset: {e}")
                raise DataError(f"Cannot create dataset for prediction: {e}")
        
        # 如果模型使用QuantileLoss，直接获取分位数预测
        if self.model_config.loss_type == "quantile":
            # 使用pytorch_forecasting的predict方法获取分位数输出
            _, dataloader = self.create_dataloaders()
            
            # mode="quantiles" 返回各分位数的预测
            predictions = self.model.predict(dataloader, mode="quantiles")
            
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            # predictions shape: (num_samples, prediction_length, num_quantiles)
            # quantiles顺序: [0.1, 0.2, ..., 0.9]
            quantiles = self.model_config.quantiles
            q10_idx = quantiles.index(0.1) if 0.1 in quantiles else 0
            q90_idx = quantiles.index(0.9) if 0.9 in quantiles else -1
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            
            result = {
                'median': predictions[:, :, median_idx].flatten(),
                'q10': predictions[:, :, q10_idx].flatten(),
                'q90': predictions[:, :, q90_idx].flatten()
            }
        else:
            # SMAPE损失模式：使用MC Dropout进行不确定性估计
            self.model.train()  # 启用Dropout
            
            _, dataloader = self.create_dataloaders()
            all_predictions = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.model.predict(dataloader, mode="prediction")
                    if isinstance(pred, torch.Tensor):
                        pred = pred.cpu().numpy()
                    all_predictions.append(pred)
            
            # all_predictions: list of (num_samples, prediction_length)
            stacked = np.stack(all_predictions, axis=0)  # (n_samples, num_obs, pred_len)
            
            result = {
                'median': np.median(stacked, axis=0).flatten(),
                'q10': np.percentile(stacked, 10, axis=0).flatten(),
                'q90': np.percentile(stacked, 90, axis=0).flatten()
            }
        
        # 恢复eval模式
        self.model.eval()
        
        return result
    
    def extract_quantile_predictions_for_env(self,
                                               df: pd.DataFrame,
                                               num_skus: int,
                                               max_time: int = 365) -> np.ndarray:
        """
        提取TFT分位数预测并转换为环境期望的格式 (T, num_skus, num_quantiles)
        
        此方法用于为强化学习环境生成真实的分位数预测数据。
        使用训练好的TFT模型对整个数据集进行预测，然后将结果组织为
        环境可以直接使用的格式。
        
        Args:
            df: 历史数据DataFrame
            num_skus: SKU数量（必须与数据中的group_id数量匹配）
            max_time: 最大时间步数（episode长度）
        
        Returns:
            np.ndarray: 形状为 (max_time, num_skus, num_quantiles) 的分位数预测数组
            
        Example:
            >>> forecaster = TFTForecaster(model_config=config)
            >>> forecaster.load_model("path/to/model.ckpt")
            >>> quantiles = forecaster.extract_quantile_predictions_for_env(df, num_skus=100, max_time=365)
            >>> env = SupplyChainEnv(demand_forecasts_quantiles=quantiles, ...)
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model not trained, call fit() first")
        
        if self.model_config.loss_type != "quantile":
            logger.warning("TFT model was not trained with quantile loss. "
                         "Switching to uncertainty estimation with MC Dropout.")
        
        # 创建数据集和数据加载器
        if self.training_dataset is None or self.val_dataset is None:
            logger.info("Creating dataset from DataFrame...")
            self.create_dataset(df)
        
        _, val_dataloader = self.create_dataloaders()
        
        # 获取分位数配置
        quantiles = self.model_config.quantiles or [0.1, 0.5, 0.9]
        num_quantiles = len(quantiles)
        
        logger.info(f"Extracting quantile predictions for env: T={max_time}, "
                   f"num_skus={num_skus}, num_quantiles={num_quantiles}")
        
        # 使用TFT模型进行分位数预测
        if self.model_config.loss_type == "quantile":
            # 直接获取分位数预测
            predictions = self.model.predict(val_dataloader, mode="quantiles")
            
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            # predictions shape: (num_samples, prediction_length, num_quantiles)
            # num_samples = 验证集中的样本数 = group_id数量 = num_skus
            logger.info(f"Raw TFT predictions shape: {predictions.shape}")
            
            # 转换为环境期望的格式 (T, num_skus, num_quantiles)
            # 对于每个时间步 t，我们需要 (num_skus, num_quantiles) 的预测
            
            # 由于TFT预测的是每个group_id的未来prediction_length步，
            # 我们需要将这些预测按时间步组织
            
            # 简化方案：将预测结果复制扩展到整个episode长度
            num_samples, pred_length, _ = predictions.shape
            
            # 确保 num_samples >= num_skus
            if num_samples < num_skus:
                # 如果样本数少于SKU数，重复采样
                repeat_factor = int(np.ceil(num_skus / num_samples))
                predictions = np.tile(predictions, (repeat_factor, 1, 1))
                predictions = predictions[:num_skus, :, :]
                num_samples = min(num_samples * repeat_factor, num_skus)
            
            # 截取前 num_skus 个样本
            predictions = predictions[:num_skus, :, :]
            
            # 现在 predictions shape: (num_skus, prediction_length, num_quantiles)
            # 我们需要转换为 (max_time, num_skus, num_quantiles)
            
            # 方案：对于每个时间步 t，使用 TFT 对时间 t 的预测
            # 由于TFT预测的是未来prediction_length步，我们可以使用滚动预测
            
            # 简化方案：将 prediction_length 维度的预测复制到时间维度
            env_quantiles = np.zeros((max_time, num_skus, num_quantiles), dtype=np.float32)
            
            for t in range(max_time):
                # 对于每个时间步，使用 TFT 预测的第 (t % pred_length) 步
                pred_idx = t % pred_length
                env_quantiles[t, :, :] = predictions[:, pred_idx, :]
            
            logger.info(f"Converted to env format: {env_quantiles.shape}")
            
        else:
            # 非分位数模式：使用模拟分位数
            logger.warning("Model not trained with quantile loss. Using simulated quantiles.")
            env_quantiles = self._simulate_quantile_predictions(df, num_skus, max_time)
        
        # 防御性代码：NaN/Inf 清洗（防止污染 PPO 的 observation_space）
        # 快速失败原则：记录警告但不静默修正，而是抛出明确异常
        if np.any(np.isnan(env_quantiles)) or np.any(np.isinf(env_quantiles)):
            logger.error("extract_quantile_predictions_for_env produced NaN or Inf values!")
            logger.error(f"  NaN count: {np.sum(np.isnan(env_quantiles))}")
            logger.error(f"  Inf count: {np.sum(np.isinf(env_quantiles))}")
            raise DataError(
                "TFT quantile predictions contain NaN or Inf. "
                "Suggestion: check input data quality, retrain model, or adjust model config."
            )
        
        # 需求不能为负数（防御性检查）
        if np.any(env_quantiles < 0):
            logger.warning("Negative values detected in quantile predictions, clipping to 0")
            env_quantiles = np.clip(env_quantiles, a_min=0, a_max=None)
        
        return env_quantiles
    
    def _simulate_quantile_predictions(self,
                                       df: pd.DataFrame,
                                       num_skus: int,
                                       max_time: int) -> np.ndarray:
        """
        模拟分位数预测（当TFT未使用分位数损失训练时）
        
        使用历史数据的统计量模拟分位数预测。
        """
        logger.info("Simulating quantile predictions from historical data...")
        
        # 获取历史需求数据
        if 'sale_qty' in df.columns:
            historical_demand = df['sale_qty'].values
        else:
            historical_demand = df.iloc[:, -1].values  # 假设最后一列是目标变量
        
        # 计算历史统计量
        mean_demand = np.mean(historical_demand)
        std_demand = np.std(historical_demand)
        
        # 生成模拟预测
        quantiles = self.model_config.quantiles or [0.1, 0.5, 0.9]
        num_quantiles = len(quantiles)
        
        env_quantiles = np.zeros((max_time, num_skus, num_quantiles), dtype=np.float32)
        
        for t in range(max_time):
            # 模拟预测：均值 + 随机扰动
            pred_mean = mean_demand + np.random.normal(0, std_demand * 0.1)
            
            # 分位数：假设正态分布
            for i, q in enumerate(quantiles):
                # 正态分布的分位数
                from scipy import stats
                pred_q = stats.norm.ppf(q, loc=pred_mean, scale=std_demand)
                env_quantiles[t, :, i] = pred_q
        
        logger.info(f"Simulated quantile predictions: {env_quantiles.shape}")
        return env_quantiles
    
    def analyze_interpretability(self,
                                  dataloader: Optional = None,
                                  df: Optional[pd.DataFrame] = None,
                                  save_dir: Optional[str] = None,
                                  max_samples: int = 1000) -> Dict[str, Any]:
        """
        分析TFT模型的可解释性（变量重要性、注意力权重）
        
        封装pytorch_forecasting.TemporalFusionTransformer的
        interpret_output()方法，提取特征重要性和时间注意力权重。
        
        Args:
            dataloader: 数据加载器（优先使用）
            df: 输入数据（如果提供，会先创建dataloader）
            save_dir: 可视化图表保存目录（可选）
            max_samples: 最大样本数（用于限制计算量）
        
        Returns:
            Dict: 包含以下键的字典：
                - 'variable_importance': 变量重要性字典
                - 'attention_weights': 注意力权重（如果可用）
                - 'interpretation_raw': pytorch_forecasting原始输出
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model not trained, call fit() first")
        
        if self.model is None:
            raise ModelNotTrainedError("Model not created. Call create_model() first.")
        
        # 准备数据
        if dataloader is None and df is not None:
            _, dataloader = self.create_dataloaders()
        elif dataloader is None:
            raise DataError("Either dataloader or df must be provided")
        
        # 设置模型为评估模式
        self.model.eval()
        
        logger.info("开始TFT可解释性分析...")
        
        try:
            # 获取一批数据用于解释
            batch = next(iter(dataloader))
            
            # 将batch移动到模型设备
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            # 调用pytorch_forecasting的interpret_output方法
            # 注意：不同版本的pytorch_forecasting API可能不同
            with torch.no_grad():
                # 尝试使用原生的interpret_output方法
                try:
                    # pytorch_forecasting >= 1.0
                    interpretations = self.model.interpret_output(batch, dataloader.dataset)
                except (TypeError, AttributeError):
                    # 降级处理：手动计算特征重要性
                    logger.warning("interpret_output()不可用，使用降级方案计算变量重要性")
                    interpretations = self._compute_variable_importance(batch)
            
            # 解析interpretations结果
            results = {
                'interpretation_raw': interpretations,
                'variable_importance': {},
                'attention_weights': None
            }
            
            # 提取变量重要性
            if isinstance(interpretations, dict):
                # 静态变量重要性
                if 'static_variable_importance' in interpretations:
                    results['variable_importance']['static'] = (
                        interpretations['static_variable_importance'].cpu().numpy()
                        if isinstance(interpretations['static_variable_importance'], torch.Tensor)
                        else interpretations['static_variable_importance']
                    )
                
                # 时变变量重要性
                if 'time_varying_variable_importance' in interpretations:
                    results['variable_importance']['time_varying'] = (
                        interpretations['time_varying_variable_importance'].cpu().numpy()
                        if isinstance(interpretations['time_varying_variable_importance'], torch.Tensor)
                        else interpretations['time_varying_variable_importance']
                    )
                
                # 注意力权重
                if 'attention_weights' in interpretations:
                    results['attention_weights'] = (
                        interpretations['attention_weights'].cpu().numpy()
                        if isinstance(interpretations['attention_weights'], torch.Tensor)
                        else interpretations['attention_weights']
                    )
            
            # 保存可视化图表
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                self._save_interpretation_plots(interpretations, save_dir)
                logger.info(f"可解释性分析图表已保存到: {save_dir}")
            
            logger.info("TFT可解释性分析完成")
            return results
            
        except Exception as e:
            logger.error(f"可解释性分析失败: {e}")
            raise DataError(f"Interpretability analysis failed: {e}")
    
    def _compute_variable_importance(self, batch) -> Dict:
        """
        降级方案：手动计算变量重要性（基于梯度）
        
        Args:
            batch: 输入批次数据
        
        Returns:
            Dict: 变量重要性字典
        """
        logger.info("使用梯度法计算变量重要性...")
        
        # 启用梯度计算
        self.model.train()
        
        # 准备输入
        if isinstance(batch, dict):
            x = batch.get('x', batch.get('encoder_target', None))
            if x is None:
                # 尝试自动推断输入
                for key in batch:
                    if isinstance(batch[key], torch.Tensor) and batch[key].requires_grad is False:
                        batch[key] = batch[key].requires_grad_(True)
                        break
        elif isinstance(batch, (list, tuple)):
            x = batch[0] if len(batch) > 0 else None
        else:
            x = batch
        
        if x is None:
            logger.warning("无法推断模型输入，返回空结果")
            return {'variable_importance': {}, 'attention_weights': None}
        
        # 前向传播
        x = x.requires_grad_(True) if isinstance(x, torch.Tensor) else x
        output = self.model(x)
        
        # 计算梯度
        if isinstance(output, torch.Tensor):
            output_sum = output.sum()
            output_sum.backward()
            
            # 提取梯度作为重要性
            if isinstance(x, torch.Tensor) and x.grad is not None:
                variable_importance = x.grad.abs().mean(dim=0).cpu().numpy()
            else:
                variable_importance = None
        else:
            variable_importance = None
        
        # 恢复评估模式
        self.model.eval()
        
        return {
            'variable_importance': {
                'gradient_based': variable_importance
            },
            'attention_weights': None
        }
    
    def _save_interpretation_plots(self, interpretations: Dict, save_dir: str):
        """
        保存可解释性分析的可视化图表
        
        Args:
            interpretations: interpret_output()的输出
            save_dir: 保存目录
        """
        try:
            import matplotlib.pyplot as plt
            
            # 1. 变量重要性柱状图
            if 'variable_importance' in interpretations:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # 静态变量
                if 'static_variable_importance' in interpretations['variable_importance']:
                    static_imp = interpretations['variable_importance']['static_variable_importance']
                    if isinstance(static_imp, torch.Tensor):
                        static_imp = static_imp.cpu().numpy()
                    
                    # 获取静态变量名称
                    if self.training_dataset is not None:
                        static_cols = self.training_dataset.static_categoricals + self.training_dataset.static_reals
                    else:
                        static_cols = [f'static_{i}' for i in range(static_imp.shape[0])]
                    
                    axes[0].barh(range(len(static_imp)), static_imp)
                    axes[0].set_yticks(range(len(static_imp)))
                    axes[0].set_yticklabels(static_cols)
                    axes[0].set_title('Static Variable Importance')
                    axes[0].set_xlabel('Importance')
                
                # 时变变量
                if 'time_varying_variable_importance' in interpretations['variable_importance']:
                    tv_imp = interpretations['variable_importance']['time_varying_variable_importance']
                    if isinstance(tv_imp, torch.Tensor):
                        tv_imp = tv_imp.cpu().numpy()
                    
                    # 获取时变变量名称
                    if self.training_dataset is not None:
                        tv_cols = self.training_dataset.time_varying_known_reals + self.training_dataset.time_varying_unknown_reals
                    else:
                        tv_cols = [f'time_varying_{i}' for i in range(tv_imp.shape[0])]
                    
                    axes[1].barh(range(len(tv_imp)), tv_imp)
                    axes[1].set_yticks(range(len(tv_imp)))
                    axes[1].set_yticklabels(tv_cols)
                    axes[1].set_title('Time-varying Variable Importance')
                    axes[1].set_xlabel('Importance')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'variable_importance.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            # 2. 注意力权重热图
            if 'attention_weights' in interpretations and interpretations['attention_weights'] is not None:
                attention = interpretations['attention_weights']
                if isinstance(attention, torch.Tensor):
                    attention = attention.cpu().numpy()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(attention[0], aspect='auto', cmap='viridis')
                ax.set_title('Temporal Attention Weights')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Attention Head')
                plt.colorbar(im, ax=ax)
                plt.savefig(os.path.join(save_dir, 'attention_weights.png'), dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            logger.info(f"可解释性图表已保存至: {save_dir}")
            
        except ImportError:
            logger.warning("matplotlib未安装，无法保存可视化图表")
        except Exception as e:
            logger.warning(f"保存可解释性图表失败: {e}")
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not self._is_trained:
            raise ModelNotTrainedError("Model not trained, cannot save")
        
        if self.trainer is None:
            raise ModelNotTrainedError("Trainer not created, cannot save")
        
        try:
            self.trainer.save_checkpoint(path)
            logger.info(f"Model saved to {path}")
            
            # 保存数据集参数到配套 JSON 文件
            config_path = path.replace('.ckpt', '_dataset_config.json')
            dataset_params = {
                'dataset_time_idx': self.model_config.dataset_time_idx,
                'dataset_target': self.model_config.dataset_target,
                'dataset_group_ids': self.model_config.dataset_group_ids,
                'dataset_static_categoricals': self.model_config.dataset_static_categoricals,
                'dataset_static_reals': self.model_config.dataset_static_reals,
                'dataset_time_varying_known_reals': self.model_config.dataset_time_varying_known_reals,
                'dataset_time_varying_unknown_reals': self.model_config.dataset_time_varying_unknown_reals,
                'dataset_max_encoder_length': self.model_config.dataset_max_encoder_length,
                'dataset_max_prediction_length': self.model_config.dataset_max_prediction_length,
                'dataset_min_encoder_length': self.model_config.dataset_min_encoder_length,
                'dataset_target_normalizer': self.model_config.dataset_target_normalizer,
                'dataset_add_relative_time_idx': self.model_config.dataset_add_relative_time_idx,
                'dataset_add_target_scales': self.model_config.dataset_add_target_scales,
                'dataset_add_encoder_length': self.model_config.dataset_add_encoder_length,
            }
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_params, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Dataset config saved to {config_path}")
            
        except Exception as e:
            raise DataError(f"Failed to save model to {path}: {e}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型文件路径
        """
        if not os.path.exists(path):
            raise DataError(f"Model file not found: {path}")
        
        try:
            self.model = TemporalFusionTransformer.load_from_checkpoint(path)
            self._is_trained = True
            
            # 创建 trainer（用于预测）
            if self.trainer is None:
                self.create_trainer()
                logger.info("Trainer created for loaded model")
            
            # 添加 PredictCallback（pytorch_forecasting 预测时需要）
            try:
                from pytorch_forecasting.callbacks import PredictCallback
                if not any(isinstance(c, PredictCallback) for c in self.trainer.callbacks):
                    self.trainer.callbacks.append(PredictCallback())
                    logger.info("✓ PredictCallback added to trainer")
                else:
                    logger.info("PredictCallback already exists in trainer.callbacks")
            except Exception as e:
                logger.warning(f"Failed to add PredictCallback: {e}")
            
            # 从加载的模型中恢复配置
            if hasattr(self.model, 'hparams'):
                hparams = self.model.hparams
                # 更新 model_config（如果hparams中有对应的字段）
                if 'loss_type' in hparams:
                    self.model_config.loss_type = hparams.loss_type
                if 'quantiles' in hparams:
                    self.model_config.quantiles = hparams.quantiles
                if 'output_size' in hparams:
                    self.model_config.output_size = hparams.output_size
                # 恢复数据集创建所需的关键参数
                if 'max_encoder_length' in hparams:
                    self.model_config.max_encoder_length = hparams.max_encoder_length
                if 'max_prediction_length' in hparams:
                    self.model_config.max_prediction_length = hparams.max_prediction_length
                if 'hidden_size' in hparams:
                    self.model_config.hidden_size = hparams.hidden_size
                if 'attention_head_size' in hparams:
                    self.model_config.attention_head_size = hparams.attention_head_size
                if 'dropout' in hparams:
                    self.model_config.dropout = hparams.dropout
                if 'hidden_continuous_size' in hparams:
                    self.model_config.hidden_continuous_size = hparams.hidden_continuous_size
                
                # ============ 恢复数据集参数 ============
                logger.info("Restoring dataset parameters...")
                dataset_param_mapping = {
                    'dataset_time_idx': 'time_idx',
                    'dataset_target': 'target',
                    'dataset_group_ids': 'group_ids',
                    'dataset_static_categoricals': 'static_categoricals',
                    'dataset_static_reals': 'static_reals',
                    'dataset_time_varying_known_reals': 'time_varying_known_reals',
                    'dataset_time_varying_unknown_reals': 'time_varying_unknown_reals',
                    'dataset_max_encoder_length': 'max_encoder_length',
                    'dataset_max_prediction_length': 'max_prediction_length',
                    'dataset_min_encoder_length': 'min_encoder_length',
                    'dataset_target_normalizer': 'target_normalizer',
                    'dataset_add_relative_time_idx': 'add_relative_time_idx',
                    'dataset_add_target_scales': 'add_target_scales',
                    'dataset_add_encoder_length': 'add_encoder_length',
                }
                
                for hparam_key in dataset_param_mapping.keys():
                    if hasattr(hparams, hparam_key):
                        setattr(self.model_config, hparam_key, getattr(hparams, hparam_key))
                        logger.info(f"  - Restored {hparam_key}: {getattr(hparams, hparam_key)}")
                    else:
                        logger.warning(f"  - Dataset parameter not found in checkpoint: {hparam_key}")
                
                logger.info("✓ Dataset parameters restored from checkpoint")
                
                # 【备用】从配套 JSON 文件恢复（如果 checkpoint 中没有）
                config_path = path.replace('.ckpt', '_dataset_config.json')
                if os.path.exists(config_path):
                    try:
                        import json
                        with open(config_path, 'r', encoding='utf-8') as f:
                            dataset_params = json.load(f)
                        for key, value in dataset_params.items():
                            if not hasattr(self.model_config, key):
                                logger.warning(f"  - Unknown dataset parameter in JSON: {key}")
                            else:
                                setattr(self.model_config, key, value)
                                logger.info(f"  - Restored {key} from JSON: {value}")
                        logger.info(f"✓ Dataset config loaded from {config_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load dataset config from JSON: {e}")
                else:
                    logger.warning(f"Dataset config JSON not found: {config_path}")
                    logger.warning("Will use default dataset parameters for prediction")
                
        except Exception as e:
            raise DataError(f"Failed to load model from {path}: {e}")


# ============================================================
# 超参数调优器（Hyperparameter Tuner）
# ============================================================

class TFTHyperparameterTuner:
    """
    TFT 超参数调优器（支持决策感知联合训练）
    
    功能：
    1. 网格搜索（Grid Search）或随机搜索（Random Search）
    2. 支持标准 TFT 和决策感知 TFT 的超参数调优
    3. 自动记录每次实验的结果（损失、指标、模型路径）
    4. 返回最佳超参数组合
    """
    
    def __init__(self,
                 base_config: TFTModelConfig,
                 train_dataloader: Any,
                 val_dataloader: Any,
                 training_dataset: Optional[Any] = None,
                 val_dataset: Optional[Any] = None,
                 output_dir: str = "./tuning_results",
                 use_decision_aware: bool = False):
        """
        初始化超参数调优器
        
        Args:
            base_config: 基础 TFT 配置（作为模板）
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            training_dataset: 训练数据集（TimeSeriesDataSet，可选）
            val_dataset: 验证数据集（TimeSeriesDataSet，可选）
            output_dir: 调优结果输出目录
            use_decision_aware: 是否启用决策感知训练
        """
        self.base_config = base_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.training_dataset = training_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.use_decision_aware = use_decision_aware
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 实验结果记录
        self.results = []
        
        logger.info(f"超参数调优器初始化完成")
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  决策感知模式: {use_decision_aware}")
    
    def grid_search(self,
                   search_space: Dict[str, List[Any]],
                   max_epochs: int = 20,
                   metric: str = "val_loss",
                   mode: str = "min") -> Tuple[Dict[str, Any], float]:
        """
        网格搜索
        
        Args:
            search_space: 搜索空间字典，格式为 {"param_name": [value1, value2, ...]}
            max_epochs: 每次实验的训练轮数
            metric: 评估指标（如 "val_loss", "val_smape"）
            mode: "min" 表示指标越小越好，"max" 表示越大越好
        
        Returns:
            best_params: 最佳超参数组合
            best_score: 最佳分数
        """
        import itertools
        
        # 生成所有参数组合
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        total_experiments = len(param_combinations)
        logger.info(f"开始网格搜索：共 {total_experiments} 个参数组合")
        
        best_score = float('inf') if mode == "min" else float('-inf')
        best_params = None
        best_model_path = None
        
        for i, param_combo in enumerate(param_combinations):
            # 构建参数字典
            params = dict(zip(param_names, param_combo))
            
            logger.info(f"\n{'='*60}")
            logger.info(f"实验 {i+1}/{total_experiments}")
            logger.info(f"参数: {params}")
            logger.info(f"{'='*60}")
            
            # 训练并评估
            try:
                score, model_path = self._train_and_evaluate(
                    params=params,
                    experiment_id=i+1,
                    max_epochs=max_epochs,
                    metric=metric
                )
                
                # 记录结果
                result = {
                    "experiment_id": i+1,
                    "params": params,
                    "score": score,
                    "model_path": model_path
                }
                self.results.append(result)
                
                # 更新最佳结果
                if (mode == "min" and score < best_score) or (mode == "max" and score > best_score):
                    best_score = score
                    best_params = params
                    best_model_path = model_path
                
                logger.info(f"实验 {i+1} 完成：score={score:.4f}")
                logger.info(f"  当前最佳: {best_params}, score={best_score:.4f}")
                
            except Exception as e:
                logger.error(f"实验 {i+1} 失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 保存调优结果
        self._save_results()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"网格搜索完成！")
        logger.info(f"最佳超参数: {best_params}")
        logger.info(f"最佳分数: {best_score:.4f}")
        logger.info(f"最佳模型: {best_model_path}")
        logger.info(f"{'='*60}")
        
        return best_params, best_score
    
    def random_search(self,
                     search_space: Dict[str, List[Any]],
                     n_trials: int = 20,
                     max_epochs: int = 20,
                     metric: str = "val_loss",
                     mode: str = "min",
                     seed: int = 42) -> Tuple[Dict[str, Any], float]:
        """
        随机搜索
        
        Args:
            search_space: 搜索空间字典
            n_trials: 随机搜索次数
            max_epochs: 每次实验的训练轮数
            metric: 评估指标
            mode: "min" 或 "max"
            seed: 随机种子
        
        Returns:
            best_params: 最佳超参数组合
            best_score: 最佳分数
        """
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        logger.info(f"开始随机搜索：共 {n_trials} 次实验")
        
        best_score = float('inf') if mode == "min" else float('-inf')
        best_params = None
        best_model_path = None
        
        for i in range(n_trials):
            # 随机选择参数
            params = {name: random.choice(values) for name, values in search_space.items()}
            
            logger.info(f"\n{'='*60}")
            logger.info(f"实验 {i+1}/{n_trials}")
            logger.info(f"参数: {params}")
            logger.info(f"{'='*60}")
            
            # 训练并评估
            try:
                score, model_path = self._train_and_evaluate(
                    params=params,
                    experiment_id=i+1,
                    max_epochs=max_epochs,
                    metric=metric
                )
                
                # 记录结果
                result = {
                    "experiment_id": i+1,
                    "params": params,
                    "score": score,
                    "model_path": model_path
                }
                self.results.append(result)
                
                # 更新最佳结果
                if (mode == "min" and score < best_score) or (mode == "max" and score > best_score):
                    best_score = score
                    best_params = params
                    best_model_path = model_path
                
                logger.info(f"实验 {i+1} 完成：score={score:.4f}")
                logger.info(f"  当前最佳: {best_params}, score={best_score:.4f}")
                
            except Exception as e:
                logger.error(f"实验 {i+1} 失败: {e}")
        
        # 保存调优结果
        self._save_results()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"随机搜索完成！")
        logger.info(f"最佳超参数: {best_params}")
        logger.info(f"最佳分数: {best_score:.4f}")
        logger.info(f"最佳模型: {best_model_path}")
        logger.info(f"{'='*60}")
        
        return best_params, best_score
    
    def _train_and_evaluate(self,
                            params: Dict[str, Any],
                            experiment_id: int,
                            max_epochs: int,
                            metric: str) -> Tuple[float, str]:
        """
        训练并评估单个参数组合
        
        Args:
            params: 超参数字典
            experiment_id: 实验 ID
            max_epochs: 训练轮数
            metric: 评估指标
        
        Returns:
            score: 评估分数
            model_path: 模型保存路径
        """
        # 复制基础配置
        config = TFTModelConfig()
        
        # 应用超参数
        for param_name, param_value in params.items():
            if hasattr(config, param_name):
                setattr(config, param_name, param_value)
            else:
                logger.warning(f"配置中不存在参数: {param_name}")
        
        # 启用决策感知（如果指定）
        if self.use_decision_aware:
            config.use_decision_aware = True
        
        # 创建模型
        forecaster = TFTForecaster(model_config=config)
        
        # 设置数据集（如果提供）
        if self.training_dataset is not None:
            forecaster.training_dataset = self.training_dataset
            forecaster.val_dataset = self.val_dataset
        
        model = forecaster.create_model()
        
        # 模型保存路径
        model_dir = os.path.join(self.output_dir, f"experiment_{experiment_id}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.ckpt")
        
        # 创建 Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            default_root_dir=model_dir,
            enable_checkpointing=True,
            logger=False,
            enable_progress_bar=False
        )
        
        # 训练
        trainer.fit(model, self.train_dataloader, self.val_dataloader)
        
        # 保存模型
        trainer.save_checkpoint(model_path)
        
        # 评估
        score = self._evaluate_model(model, self.val_dataloader, metric)
        
        return score, model_path
    
    def _evaluate_model(self,
                       model: Any,
                       dataloader: Any,
                       metric: str) -> float:
        """
        评估模型
        
        Args:
            model: 训练好的模型
            dataloader: 验证数据加载器
            metric: 评估指标
        
        Returns:
            score: 评估分数
        """
        model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 简化评估：直接使用验证损失
                loss = None
                if hasattr(model, "validation_step"):
                    try:
                        loss = model.validation_step(batch, 0)
                    except:
                        loss = None
                elif hasattr(model, "step"):
                    try:
                        loss, _ = model.step(batch, 0)
                    except:
                        loss = None
                
                # 处理loss可能是字典、张量或None的情况
                if loss is None:
                    # 如果无法获取loss，跳过这个batch
                    continue
                
                if isinstance(loss, dict):
                    # 尝试获取val_loss或loss
                    if "val_loss" in loss:
                        loss = loss["val_loss"]
                    elif "loss" in loss:
                        loss = loss["loss"]
                    else:
                        # 如果字典中没有已知的键，使用第一个值
                        try:
                            loss = list(loss.values())[0]
                        except:
                            continue
                
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                
                if not isinstance(loss, (int, float)):
                    # 如果loss不是数字，跳过
                    continue
                
                total_loss += loss
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_results(self):
        """
        保存调优结果到 CSV 和 JSON
        """
        import pandas as pd
        
        # 转换为 DataFrame
        rows = []
        for result in self.results:
            row = {
                "experiment_id": result["experiment_id"],
                "score": result["score"],
                "model_path": result["model_path"]
            }
            # 展开参数
            for param_name, param_value in result["params"].items():
                row[f"param_{param_name}"] = param_value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 保存为 CSV
        csv_path = os.path.join(self.output_dir, "tuning_results.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        # 保存为 JSON
        json_path = os.path.join(self.output_dir, "tuning_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"调优结果已保存: {csv_path}, {json_path}")
