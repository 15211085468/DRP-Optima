"""
自定义回调函数 - training/callbacks.py
"""

import numpy as np
from typing import Dict, List, Optional, Callable
import json
import os
from pathlib import Path


class CustomCallbacks:
    """自定义回调集合"""
    
    def __init__(self):
        """初始化回调集合"""
        self.callbacks = []
        self.state = {
            'epoch': 0,
            'step': 0,
            'best_loss': float('inf'),
            'best_reward': float('-inf')
        }
    
    def add_callback(self, callback: 'DRPBaseCallback'):
        """添加回调"""
        self.callbacks.append(callback)
    
    def on_epoch_start(self, epoch: int):
        """每个epoch开始时调用"""
        self.state['epoch'] = epoch
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_start'):
                callback.on_epoch_start(epoch)
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """每个epoch结束时调用"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, metrics)
    
    def on_step(self, step: int, logs: Dict):
        """每步时调用"""
        self.state['step'] = step
        for callback in self.callbacks:
            if hasattr(callback, 'on_step'):
                callback.on_step(step, logs)


class DRPBaseCallback:
    """基础回调类"""
    
    def __init__(self):
        pass
    
    def on_epoch_start(self, epoch: int):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        pass
    
    def on_step(self, step: int, logs: Dict):
        pass


class EarlyStoppingCallback(DRPBaseCallback):
    """早停回调"""
    
    def __init__(self,
                 monitor: str = 'val_loss',
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = 'min'):
        """
        Args:
            monitor: 监控指标
            patience: 早停耐心值
            min_delta: 最小改善量
            mode: 'min' 或 'max'
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """检查是否应该早停"""
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            return
        
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                print(f"\n早停于 epoch {epoch}")
                return True  # 返回True表示应该停止
        
        return False


class ModelCheckpointCallback(DRPBaseCallback):
    """模型检查点回调"""
    
    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_frequency: int = 1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """保存检查点"""
        should_save = False
        
        if self.save_best_only:
            current_value = metrics.get(self.monitor)
            
            if current_value is not None:
                if self.mode == 'min':
                    should_save = current_value < self.best_value
                else:
                    should_save = current_value > self.best_value
                
                if should_save:
                    self.best_value = current_value
        else:
            should_save = epoch % self.save_frequency == 0
        
        if should_save:
            # 实际保存逻辑
            save_path = self.filepath.format(epoch=epoch, **metrics)
            print(f"保存检查点到: {save_path}")


class LearningRateSchedulerCallback(DRPBaseCallback):
    """学习率调度回调"""
    
    def __init__(self,
                 schedule_fn: Callable[[int], float],
                 warmup_epochs: int = 0):
        super().__init__()
        self.schedule_fn = schedule_fn
        self.warmup_epochs = warmup_epochs
    
    def on_epoch_start(self, epoch: int):
        """计算学习率"""
        if epoch < self.warmup_epochs:
            lr = (epoch + 1) / self.warmup_epochs
        else:
            lr = self.schedule_fn(epoch - self.warmup_epochs)
        
        return lr


class TensorBoardLoggerCallback(DRPBaseCallback):
    """TensorBoard日志回调"""
    
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self.metrics_history = []
        
        os.makedirs(log_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """记录指标"""
        self.metrics_history.append({
            'epoch': epoch,
            **metrics
        })
        
        # 可以在这里写入TensorBoard事件文件
        log_file = os.path.join(self.log_dir, f"epoch_{epoch}.json")
        with open(log_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def on_step(self, step: int, logs: Dict):
        """记录步级指标"""
        pass


class ProgressCallback(DRPBaseCallback):
    """进度显示回调"""
    
    def __init__(self, total_epochs: int, print_frequency: int = 1):
        super().__init__()
        self.total_epochs = total_epochs
        self.print_frequency = print_frequency
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """显示进度"""
        if epoch % self.print_frequency == 0:
            metric_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ])
            progress = (epoch + 1) / self.total_epochs * 100
            print(f"Epoch {epoch+1}/{self.total_epochs} [{progress:.1f}%] | {metric_str}")


class MetricsHistoryCallback(DRPBaseCallback):
    """指标历史回调"""
    
    def __init__(self):
        super().__init__()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_reward': [],
            'val_reward': []
        }
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """记录指标"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_loss') -> int:
        """获取最佳epoch"""
        if metric not in self.history or not self.history[metric]:
            return 0
        
        values = self.history[metric]
        
        if 'loss' in metric:
            return np.argmin(values)
        else:
            return np.argmax(values)
    
    def save_history(self, path: str):
        """保存历史"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, path: str):
        """加载历史"""
        with open(path, 'r') as f:
            self.history = json.load(f)


class WandbLoggerCallback(DRPBaseCallback):
    """Weights & Biases日志回调（可选）"""
    
    def __init__(self, 
                 project_name: str,
                 run_name: Optional[str] = None,
                 config: Optional[Dict] = None):
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True
        except ImportError:
            print("wandb未安装，跳过W&B日志")
            self.enabled = False
            return
        
        super().__init__()
        
        if self.enabled:
            self.run = self.wandb.init(
                project=project_name,
                name=run_name,
                config=config
            )
    
    def on_epoch_end(self, epoch: int, metrics: Dict):
        """记录到W&B"""
        if self.enabled:
            self.wandb.log(metrics, step=epoch)
    
    def finish(self):
        """结束W&B运行"""
        if self.enabled:
            self.wandb.finish()
