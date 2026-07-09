"""
可视化回调模块 - training/visualization_callbacks.py
包含TFT和PPO训练的数据收集回调
"""

import numpy as np
from typing import Optional, Any
import pytorch_lightning as pl
from stable_baselines3.common.callbacks import BaseCallback

from utils.visualization_collector import VisualizationCollector


class TFTVisualizationCallback(pl.Callback):
    """TFT训练可视化数据收集回调"""
    
    def __init__(self, collector: VisualizationCollector):
        """
        初始化回调
        
        Args:
            collector: 可视化数据收集器
        """
        super().__init__()
        self.collector = collector
        self.last_logged_metrics = {}
        
    def state_dict(self):
        """返回回调状态（兼容PyTorch Lightning）"""
        return {'collector_data': self.collector.get_summary()}
    
    def load_state_dict(self, state_dict):
        """加载回调状态（兼容PyTorch Lightning）"""
        pass  # 不需要恢复状态
        
    def on_train_epoch_end(self, trainer, pl_module):
        """每个epoch结束时收集数据"""
        # 获取当前epoch
        epoch = trainer.current_epoch
        
        # 获取logged_metrics
        logged_metrics = trainer.logged_metrics
        
        # 获取训练损失
        train_loss = logged_metrics.get('train_loss', None)
        if train_loss is not None:
            train_loss = train_loss.item() if hasattr(train_loss, 'item') else float(train_loss)
        
        # 获取验证损失
        val_loss = logged_metrics.get('val_loss', None)
        if val_loss is not None:
            val_loss = val_loss.item() if hasattr(val_loss, 'item') else float(val_loss)
        
        # 获取学习率
        learning_rate = logged_metrics.get('learning_rate', None)
        if learning_rate is not None:
            learning_rate = learning_rate.item() if hasattr(learning_rate, 'item') else float(learning_rate)
        
        # 收集数据
        if train_loss is not None:
            self.collector.collect_tft_metrics(epoch, train_loss, val_loss, learning_rate)
            
    def on_train_end(self, trainer, pl_module):
        """训练结束时保存数据"""
        self.collector.save_tft_data()


class PPOVisualizationCallback(BaseCallback):
    """PPO训练可视化数据收集回调"""
    
    def __init__(self, collector: VisualizationCollector, verbose: int = 0):
        """
        初始化回调
        
        Args:
            collector: 可视化数据收集器
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.collector = collector
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """每个step结束时调用"""
        # 获取当前timestep
        timestep = self.n_calls
        
        # 从PPO model获取训练信息
        if self.model is not None:
            # 修复问题7：添加 hasattr 检查（兼容不同版本的stable-baselines3）
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                recent_info = self.model.ep_info_buffer[-1]
                episode_reward = recent_info.get('r', 0.0)
                episode_length = recent_info.get('l', 0)
                
                # 如果是新的episode，收集数据
                if len(self.episode_rewards) == 0 or episode_length > 0:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # 收集数据
                    self.collector.collect_ppo_metrics(
                        timestep=timestep,
                        episode_reward=episode_reward,
                        episode_length=episode_length
                    )
                    
        return True
        
    def on_training_end(self) -> None:
        """训练结束时保存数据"""
        self.collector.save_ppo_data()
