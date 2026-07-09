"""
训练器模块 - training/trainer.py
管理TFT和PPO的完整训练流程
"""

import os
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym  # 导入gymnasium用于RL环境
import lightning.pytorch as pl  # 导入Lightning用于回调

# 【新增】导入 VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pytorch_forecasting import TimeSeriesDataSet

from config.data_config import TFTModelConfig, PPOConfig, EnvConfig, HardwareConfig
from data import DataLoader, FeatureEngineer, DataPreprocessor
from models.forecasting.tft_model import TFTForecaster
from models.forecasting.hyperparameter_optimizer import HyperparameterOptimizer
from models.rl import PPOAgent
from models.optimization import SupplyChainEnv, ConstraintHandler
from integration import TFTDRLIntegration, MultiObjectiveOptimizer
from integration.tft_drl_integration import IntegrationConfig
from config.exceptions import DataError
from utils.visualization_collector import VisualizationCollector
from training.visualization_callbacks import TFTVisualizationCallback, PPOVisualizationCallback
from utils.logger import get_logger
logger = get_logger(__name__)


@dataclass
class TrainingStats:
    """训练统计"""
    epoch: int
    tft_loss: float
    ppo_reward: float
    stockout_rate: float
    holding_cost: float
    total_time: float


class Trainer:
    """统一训练器"""
    
    def __init__(self, 
                 config=None,
                 output_dir: str = "./records/output"):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            output_dir: 输出目录
        """
        self.config = config or get_config()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        self.data_loader = DataLoader(
            self.config.paths,
            self.config.data
        )
        self.feature_engineer = FeatureEngineer(self.config.data)
        self.preprocessor = DataPreprocessor()
        
        # 模型
        self.tft_forecaster: Optional[TFTForecaster] = None
        self.ppo_agent: Optional[PPOAgent] = None
        self.supply_chain_env: Optional[SupplyChainEnv] = None
        self.integration: Optional[TFTDRLIntegration] = None
        
        # 训练历史
        self.tft_history: List[Dict] = []
        self.ppo_history: List[Dict] = []
        self.integration_history: List[Dict] = []
        
        # 最佳模型
        self.best_tft_loss = float('inf')
        self.best_ppo_reward = float('-inf')
        
        # 可视化数据收集器
        self.viz_collector = VisualizationCollector(
            output_dir=os.path.join(output_dir, 'visualization_data')
        )
    
    def load_data(self, num_stores: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        加载并预处理数据
        
        Args:
            num_stores: 限制门店数量
        
        Returns:
            Tuple: (预处理后的数据, 列配置)
        """
        logger.info("=" * 50)
        logger.info("加载数据...")
        
        # 加载数据
        sales_df = self.data_loader.load_main_sales_data()
        shop_df = self.data_loader.load_shop_info()
        goods_df = self.data_loader.load_goods_info()
        
        # 限制门店数量
        if num_stores:
            stores = self.data_loader.get_unique_stores(sales_df)[:num_stores]
            sales_df = self.data_loader.filter_by_stores(sales_df, stores)
        
        # 加载列配置
        column_info = self.data_loader.load_column_config()
        
        logger.info(f"数据加载完成: {len(sales_df):,} 行")
        
        return sales_df, column_info
    
    def prepare_tft_data(self,
                        df: pd.DataFrame,
                        column_info: Dict) -> Tuple:
        """
        准备TFT训练数据
        
        Args:
            df: 销售数据
            column_info: 列配置
        
        Returns:
            Tuple: (train_dataloader, val_dataloader, test_dataloader)
        """
        logger.info("=" * 50)
        logger.info("准备TFT数据...")
        
        # 创建TFT预测器
        # 修复：使用 self.config.model 而非 self.config.tft_model
        self.tft_forecaster = TFTForecaster(
            self.config.model,
            self.config.hardware
        )
        
        # 准备数据
        # 手动添加time_idx（基于日期，如果不存在）
        if 'time_idx' not in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['shop_code', 'goods_code', 'date'])
            df['time_idx'] = df.groupby(['shop_code', 'goods_code']).cumcount()
            logger.info(f"time_idx 已自动添加，范围: {df['time_idx'].min()} - {df['time_idx'].max()}")
        
        df = self.preprocessor.prepare_tft_data(df)

        # 填充 lag/rolling/growth 等衍生特征中的 NaN
        # pytorch_forecasting 不允许 time_varying_unknown_reals 中存在 NaN
        derived_cols = [c for c in df.columns
                        if c.startswith(('lag_', 'rolling_', 'qty_wow'))]
        for col in derived_cols:
            if df[col].isna().any():
                # 按组前向/后向填充，然后填充剩余 NaN 为 0
                df[col] = df.groupby(['shop_code', 'goods_code'])[col] \
                           .transform(lambda s: s.ffill().bfill())
                df[col] = df[col].fillna(0)
        if derived_cols:
            logger.info(f"已填充衍生特征 NaN: {derived_cols}")

        # 计算训练截止时间
        training_cutoff = df["time_idx"].max() - self.config.data.test_weeks
        
        # 创建数据集
        self.tft_forecaster.create_dataset(
            df,
            time_idx="time_idx",
            target="sale_qty",
            group_ids=['shop_code', 'goods_code'],
            static_categoricals=column_info['static_cate'],
            static_reals=column_info['static_reals'],
            time_varying_known_reals=column_info['time_varying_known_reals'],
            time_varying_unknown_reals=column_info['time_varying_unknown_reals'],
            training_cutoff=training_cutoff
        )
        
        # 创建数据加载器
        train_dataloader, val_dataloader = self.tft_forecaster.create_dataloaders()
        
        logger.info(f"TFT数据准备完成")
        
        return train_dataloader, val_dataloader
    
    def train_tft(self,
                 train_dataloader,
                 val_dataloader,
                 optimize_hyperparams: bool = False,
                 n_trials: int = 50) -> Dict:
        """
        训练TFT模型
        
        Args:
            train_dataloader: 训练数据
            val_dataloader: 验证数据
            optimize_hyperparams: 是否优化超参数
            n_trials: Optuna试验次数
        
        Returns:
            Dict: 训练结果
        """
        logger.info("=" * 50)
        logger.info("训练TFT模型...")
        
        start_time = time.time()
        
        # 超参数优化
        if optimize_hyperparams:
            logger.info("运行超参数优化...")
            optimizer = HyperparameterOptimizer(
                self.config.optuna,
                self.config.tft_model
            )
            
            best_params = optimizer.optimize(
                train_dataloader,
                val_dataloader,
                self.tft_forecaster.training_dataset,
                n_trials=n_trials
            )
            
            # 应用最佳参数
            training_params = optimizer.suggest_params_for_training()
            self.config.tft_model.learning_rate = training_params['learning_rate']
            self.config.tft_model.hidden_size = training_params['hidden_size']
            self.config.tft_model.attention_head_size = training_params['attention_head_size']
            self.config.tft_model.dropout = training_params['dropout']
            self.config.tft_model.hidden_continuous_size = training_params['hidden_continuous_size']
            self.config.tft_model.gradient_clip_val = training_params['gradient_clip_val']
        
        # 创建模型
        self.tft_forecaster.create_model()
        
        # 准备回调 - pytorch_forecasting会自动添加ModelCheckpoint
        # 为了只保存最佳模型，我们需要修改checkpoint_dir中的已有检查点
        extra_cbs = []
        logger.info(f"  跳过手动添加ModelCheckpoint（避免与pytorch_forecasting冲突）")
        logger.info(f"  提示：训练完成后将自动保存最佳模型到 models/")
        
        # 创建训练器
        self.tft_forecaster.create_trainer(
            logger_dir=os.path.join(self.output_dir, "tft_logs"),
            checkpoint_dir=os.path.join(self.output_dir, "tft_checkpoints"),
            extra_callbacks=extra_cbs
        )
        
        # 训练
        self.tft_forecaster.fit(
            train_dataloader,
            val_dataloader
        )
        
        # 训练后获取最佳损失并保存模型
        try:
            # 方法1：从trainer的best_model_score获取最佳损失
            if hasattr(self.tft_forecaster, 'trainer') and self.tft_forecaster.trainer is not None:
                trainer = self.tft_forecaster.trainer
                
                # 尝试获取best_model_score
                if hasattr(trainer, 'best_model_score') and trainer.best_model_score is not None:
                    self.best_tft_loss = float(trainer.best_model_score)
                    logger.info(f"    最佳损失已从trainer.best_model_score获取: {self.best_tft_loss:.4f}")
                
                # 方法2：从callback_metrics获取
                elif hasattr(trainer, 'callback_metrics') and len(trainer.callback_metrics) > 0:
                    val_loss = trainer.callback_metrics.get('val_loss')
                    if val_loss is not None:
                        self.best_tft_loss = float(val_loss)
                        logger.info(f"    最佳损失已从trainer.callback_metrics获取: {self.best_tft_loss:.4f}")
                
                # 方法3：从logged_metrics获取
                elif hasattr(trainer, 'logged_metrics') and len(trainer.logged_metrics) > 0:
                    val_loss = trainer.logged_metrics.get('val_loss')
                    if val_loss is not None:
                        self.best_tft_loss = float(val_loss)
                        logger.info(f"    最佳损失已从trainer.logged_metrics获取: {self.best_tft_loss:.4f}")
                
                else:
                    logger.warning(f"    无法获取最佳损失，保持默认值inf")
            
            # 如果仍然是inf，尝试使用最后一次的val_loss
            if self.best_tft_loss == float('inf'):
                logger.warning(f"    最佳损失仍为inf，可能是训练配置问题")
                
        except Exception as e:
            logger.warning(f"无法获取最佳损失: {e}，保持默认值")
        
        train_time = time.time() - start_time
        logger.info(f"TFT训练完成，耗时: {train_time/3600:.2f} 小时")
        
        # 保存最佳模型
        best_model_path = os.path.join(self.output_dir, "..", "models", "tft_best_smape.ckpt")
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        
        try:
            # 方法1：尝试从trainer的checkpoint_callback获取最佳模型路径
            if hasattr(self.tft_forecaster, 'trainer') and self.tft_forecaster.trainer:
                trainer = self.tft_forecaster.trainer
                if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
                    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
                    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                        import shutil
                        shutil.copy2(best_checkpoint_path, best_model_path)
                        self.best_tft_loss = float(trainer.checkpoint_callback.best_model_score)
                        logger.info(f"最佳模型已保存: {best_model_path} (val_loss={self.best_tft_loss:.4f})")
                    else:
                        # 方法2：直接保存当前模型
                        self.tft_forecaster.save_model(best_model_path)
                        logger.info(f"当前模型已保存: {best_model_path}")
                else:
                    # 方法2：直接保存当前模型
                    self.tft_forecaster.save_model(best_model_path)
                    logger.info(f"当前模型已保存: {best_model_path}")
            else:
                logger.warning("无法获取trainer，模型未保存")
                
        except Exception as e:
            logger.warning(f"保存最佳模型失败: {e}")
        
        return {
            'train_time': train_time,
            'best_loss': self.best_tft_loss,
            'best_model_path': best_model_path
        }
    
    def _generate_demand_forecasts(self, df: pd.DataFrame, test_weeks: int) -> np.ndarray:
        """
        使用TFT生成需求预测（仅使用训练数据，避免数据泄露）
        
        Args:
            df: 完整数据
            test_weeks: 测试周数
        Returns:
            np.ndarray: 需求预测数组
        """
        # 确保df有time_idx列
        if "time_idx" not in df.columns:
            logger.warning("df缺少time_idx列，自动添加...")
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['shop_code', 'goods_code', 'date'])
            df['time_idx'] = df.groupby(['shop_code', 'goods_code']).cumcount()
            logger.info(f"time_idx 已自动添加，范围: {df['time_idx'].min()} - {df['time_idx'].max()}")

        # 填充衍生特征中的 NaN（与 prepare_tft_data 保持一致）
        derived_cols = [c for c in df.columns
                        if c.startswith(('lag_', 'rolling_', 'qty_wow'))]
        for col in derived_cols:
            if df[col].isna().any():
                df[col] = df.groupby(['shop_code', 'goods_code'])[col] \
                           .transform(lambda s: s.ffill().bfill())
                df[col] = df[col].fillna(0)
        if derived_cols:
            logger.info(f"预测前已填充衍生特征 NaN: {len(derived_cols)} 列")

        training_cutoff = df["time_idx"].max() - test_weeks
        train_data = df[df["time_idx"] <= training_cutoff].copy()
        
        try:
            # 检查training_dataset是否存在
            if not hasattr(self.tft_forecaster, 'training_dataset') or self.tft_forecaster.training_dataset is None:
                raise ValueError("TFT模型未正确训练，training_dataset不存在")
            
            # 创建预测数据集
            pred_dataset = TimeSeriesDataSet.from_dataset(
                self.tft_forecaster.training_dataset,
                train_data,
                predict=True,
                stop_randomization=True
            )
            
            logger.info(f"  预测数据集长度: {len(pred_dataset)}")
            logger.info(f"  训练数据形状: {train_data.shape}")
            logger.info(f"  SKU数量: {len(train_data['goods_code'].unique())}")
            
            # 使用pytorch_forecasting模型自带的predict方法
            # 该方法可以直接对TimeSeriesDataSet进行预测
            predictions = self.tft_forecaster.model.predict(
                pred_dataset,
                mode="prediction"
            )
            
            logger.info(f"  原始预测结果类型: {type(predictions)}")
            logger.info(f"  原始预测结果形状: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")
            
            # 将预测结果转换为numpy数组
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
            
            logger.info(f"  预测结果维度: {predictions.shape}")
            
            # 重塑预测结果
            # predictions.shape = (num_samples, prediction_length)
            # 需要转换为 (test_weeks, num_skus)
            num_skus = len(train_data['goods_code'].unique())
            prediction_length = predictions.shape[1] if predictions.ndim == 2 else 1
            
            logger.info(f"  prediction_length: {prediction_length}, num_skus: {num_skus}")
            
            # 将预测结果按SKU聚合
            # 使用group_ids映射聚合预测结果，而非假设固定顺序
            if predictions.ndim == 2:
                num_stores = len(train_data['shop_code'].unique())
                num_samples = predictions.shape[0]

                if num_samples == num_stores * num_skus:
                    # 按 (num_stores, num_skus, prediction_length) 重塑
                    predictions_reshaped = predictions.reshape(num_stores, num_skus, prediction_length)
                    # 对所有门店取平均值
                    demand_forecasts = predictions_reshaped.mean(axis=0)  # shape: (num_skus, prediction_length)
                else:
                    # 无法按门店-SKU矩阵重塑，按SKU分组聚合
                    logger.warning(f"预测样本数({num_samples}) != 门店数×SKU数({num_stores}×{num_skus}={num_stores * num_skus})，"
                                   f"使用均值扩展策略")
                    demand_forecasts = predictions.mean(axis=0, keepdims=True)  # shape: (1, prediction_length)
                    demand_forecasts = np.repeat(demand_forecasts, num_skus, axis=0)  # shape: (num_skus, prediction_length)
            else:
                # 1维数组
                demand_forecasts = predictions.reshape(1, -1)  # shape: (1, prediction_length)
                demand_forecasts = np.repeat(demand_forecasts, num_skus, axis=0)  # shape: (num_skus, prediction_length)
            
            logger.info(f"  聚合后预测形状: {demand_forecasts.shape}")
            
            # 如果prediction_length < test_weeks，用最后一期预测填充
            if demand_forecasts.shape[1] < test_weeks:
                pad = test_weeks - demand_forecasts.shape[1]
                last_prediction = demand_forecasts[:, -1:]  # shape: (num_skus, 1)
                pad_array = np.repeat(last_prediction, pad, axis=1)  # shape: (num_skus, pad)
                demand_forecasts = np.concatenate([demand_forecasts, pad_array], axis=1)  # shape: (num_skus, test_weeks)
            
            # 转置为 (test_weeks, num_skus)
            demand_forecasts = demand_forecasts.T
            
            logger.info(f"  最终预测形状: {demand_forecasts.shape}")
            
        except DataError:
            # DataError直接向上传播，不使用随机预测兜底
            raise
        except Exception as e:
            logger.error(f"⚠️ TFT 预测失败: {e}")
            logger.warning("开始降级预测...")
            
            # 降级策略 1：简单移动平均（过去 4 周均值）
            logger.info("  尝试降级策略 1：简单移动平均预测")
            try:
                # 按商品计算过去 4 周销量均值
                train_data = df[df["time_idx"] <= training_cutoff].copy()
                train_data['week'] = train_data['time_idx'] // 7  # 假设 7 天 = 1 周
                weekly_sales = train_data.groupby(['goods_code', 'week'])['sale_qty'].sum().reset_index()
                
                # 计算过去 4 周均值
                fallback = weekly_sales.groupby('goods_code')['sale_qty'].tail(4).groupby(weekly_sales.groupby('goods_code')['sale_qty'].cumcount() // 4).mean()
                fallback = fallback.reindex(train_data['goods_code'].unique()).fillna(1.0).values
                
                # 扩展为 (test_weeks, num_skus)
                demand_forecasts = np.tile(fallback, (test_weeks, 1))
                
                logger.info(f"  ✓ 降级策略 1 成功: 形状={demand_forecasts.shape}")
                return demand_forecasts
            except Exception as e1:
                logger.warning(f"  降级策略 1 失败: {e1}")
            
            # 降级策略 2：历史均值
            logger.info("  尝试降级策略 2：历史均值预测")
            try:
                train_data = df[df["time_idx"] <= training_cutoff].copy()
                fallback = train_data.groupby('goods_code')['sale_qty'].mean().fillna(1.0)
                fallback = fallback.reindex(train_data['goods_code'].unique()).fillna(1.0).values
                
                # 扩展为 (test_weeks, num_skus)
                demand_forecasts = np.tile(fallback, (test_weeks, 1))
                
                logger.info(f"  ✓ 降级策略 2 成功: 形状={demand_forecasts.shape}")
                return demand_forecasts
            except Exception as e2:
                logger.warning(f"  降级策略 2 失败: {e2}")
            
            # 降级策略 3：零预测（避免训练中断）
            logger.warning("  使用降级策略 3：零预测（保守估计）")
            num_skus = len(df['goods_code'].unique())
            demand_forecasts = np.ones((test_weeks, num_skus), dtype=np.float32) * 0.1  # 很小的正数
            
            logger.info(f"  ✓ 降级策略 3 成功: 形状={demand_forecasts.shape}")
            return demand_forecasts
        
        # 确保是numpy数组
        if not isinstance(demand_forecasts, np.ndarray):
            demand_forecasts = np.array(demand_forecasts)
        
        # 如果预测长度不足 RL 环境步数，用最后一期预测填充
        env_steps = test_weeks
        if len(demand_forecasts) < env_steps:
            pad = env_steps - len(demand_forecasts)
            demand_forecasts = np.concatenate([demand_forecasts, demand_forecasts[-1:].repeat(pad, axis=0)])
        
        return demand_forecasts
    
    def prepare_rl_env(self,
                      demand_forecasts: Optional[np.ndarray] = None,
                      demand_forecasts_quantiles: Optional[np.ndarray] = None,
                      use_vec_normalize: bool = True) -> 'DummyVecEnv':
        """准备RL环境（委托给 integration/env_factory）"""
        from integration.env_factory import create_rl_env
        
        # 确定 num_skus
        if demand_forecasts is not None:
            num_skus = demand_forecasts.shape[1]
        elif demand_forecasts_quantiles is not None:
            num_skus = demand_forecasts_quantiles.shape[1]
        else:
            num_skus = self.config.data.num_skus
        
        _, vec_env = create_rl_env(
            demand_forecasts=demand_forecasts,
            demand_forecasts_quantiles=demand_forecasts_quantiles,
            num_skus=num_skus,
            num_stores=self.config.data.num_train_stores,
            env_config=self.config.env,
            use_vec_normalize=use_vec_normalize,
            normalize_gamma=self.config.ppo.gamma,
        )
        
        self.vec_env = vec_env
        return vec_env

    def train_ppo(self,
                 env: Optional[DummyVecEnv] = None,
                 total_timesteps: int = 18250,
                 eval_freq: int = 1000,
                 save_vec_normalize: bool = True) -> Dict:
        """
        训练PPO模型
        
        Args:
            env: 供应链环境（可以是 VecNormalize 包装后的）
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            save_vec_normalize: 是否保存 VecNormalize 参数（默认True）
        
        Returns:
            Dict: 训练结果
        """
        logger.info("=" * 50)
        logger.info("训练PPO模型...")
        
        start_time = time.time()
        
        # 【修复】如果没有传入 env，则使用 self.vec_env（已包装 VecNormalize）
        if env is None:
            if hasattr(self, 'vec_env'):
                env = self.vec_env
                logger.info(f"  ✓ 使用 VecNormalize 包装后的环境")
            else:
                raise ValueError("环境未准备，请先调用 prepare_rl_env()")
        
        # 创建PPO智能体
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            ppo_config=self.config.ppo
        )
        
        # 设置环境
        self.ppo_agent.set_env(env)
        
        # 准备可视化回调
        ppo_viz_cb = PPOVisualizationCallback(self.viz_collector)
        
        # 训练
        self.ppo_agent.learn(
            env=env,
            total_timesteps=total_timesteps,
            eval_env=env,
            eval_freq=eval_freq,
            callback=ppo_viz_cb
        )
        
        train_time = time.time() - start_time
        logger.info(f"PPO训练完成，耗时: {train_time/3600:.2f} 小时")
        
        # 【新增】保存 VecNormalize 参数
        if save_vec_normalize and hasattr(self, 'vec_env'):
            if isinstance(self.vec_env, VecNormalize):
                vec_normalize_path = os.path.join(self.output_dir, 'vec_normalize.pkl')
                self.vec_env.save(vec_normalize_path)
                logger.info(f"  ✓ VecNormalize 参数已保存: {vec_normalize_path}")
        
        return {
            'train_time': train_time,
            'total_steps': total_timesteps
        }
    
    def train_end_to_end(self,
                        df: pd.DataFrame,
                        column_info: Dict,
                        ppo_timesteps: int = 18250) -> Dict:
        """
        端到端训练（先TFT后PPO）
        
        修复说明：使用TFT的真实预测而非随机数来驱动RL环境
        
        Args:
            df: 数据
            column_info: 列配置
            ppo_timesteps: PPO训练步数
        
        Returns:
            Dict: 训练结果
        """
        logger.info("=" * 60)
        logger.info(f"开始端到端训练 | SKU: {self.config.data.num_skus} | PPO步数: {ppo_timesteps}")
        logger.info("=" * 60)
        
        # 使用全部数据，移除演示限制
        logger.info(f"使用全部数据: {self.config.data.num_skus} SKU, {self.config.data.num_train_stores} 门店")
        
        # ========== 步骤1: TFT训练 ==========
        logger.info("\n[步骤1] TFT模型训练")
        train_dl, val_dl = self.prepare_tft_data(df, column_info)
        tft_result = self.train_tft(train_dl, val_dl)
        logger.info(f"  ✓ TFT训练完成: best_loss={tft_result['best_loss']:.4f}")
        
        # ========== 步骤2: 生成需求预测（周粒度，直接用于PPO环境） ==========
        logger.info("\n[步骤2] 生成TFT需求预测（周粒度）")
        # 注意：项目已确定使用周粒度，不再转换为日粒度
        # 原 _convert_weekly_to_daily() 已删除（2026-06-17）
        demand_forecasts = self._generate_demand_forecasts(df, self.config.data.test_weeks)
        logger.info(f"  ✓ 需求预测生成（周粒度）: shape={demand_forecasts.shape}")
        
        # ========== 步骤2.5: 生成TFT分位数预测 ==========
        logger.info("\n[步骤2.5] 生成TFT分位数预测（用于优化三：特征降维）")
        try:
            demand_forecasts_quantiles = self.tft_forecaster.extract_quantile_predictions_for_env(
                df=df,
                num_skus=self.config.data.num_skus,
                max_time=demand_forecasts.shape[0]  # 与需求预测相同的长度
            )
            logger.info(f"  ✓ TFT分位数预测生成: shape={demand_forecasts_quantiles.shape}")
            logger.info(f"  分位数数量: {demand_forecasts_quantiles.shape[2]}")
        except Exception as e:
            logger.warning(f"  ⚠ TFT分位数预测生成失败: {e}")
            logger.warning(f"  将使用模拟分位数（环境将自动生成）")
            demand_forecasts_quantiles = None
        
        # ========== 步骤3: 准备RL环境 ==========
        logger.info("\n[步骤3] 准备PPO环境")
        env = self.prepare_rl_env(demand_forecasts, demand_forecasts_quantiles)
        logger.info(f"  ✓ 环境创建: {env.observation_space.shape}")
        
        # ========== 步骤4: PPO训练 ==========
        logger.info("\n[步骤4] PPO模型训练")
        ppo_result = self.train_ppo(env, total_timesteps=ppo_timesteps)
        logger.info(f"  ✓ PPO训练完成: steps={ppo_result['total_steps']}")
        
        # ========== 步骤5: 构造返回结果 ==========
        results = {
            'tft_result': tft_result,
            'ppo_result': ppo_result,
            'status': 'success'
        }
        
        logger.info("=" * 60)
        logger.info(f"✓ 端到端训练完成")
        logger.info("=" * 60)
        
