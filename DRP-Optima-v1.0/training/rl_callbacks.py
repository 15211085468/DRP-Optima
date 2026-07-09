"""
RL 训练回调函数 - training/rl_callbacks.py
从 train_ppo_base.py 迁移而来
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class LinearScheduleCallback(BaseCallback):
    """
    线性衰减学习率和熵系数的Callback
    """
    def __init__(self, lr_start: float, lr_end: float, 
                 ent_start: float, ent_end: float, 
                 total_timesteps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.total_timesteps = total_timesteps
        
    def _on_step(self) -> bool:
        """每个step调用"""
        try:
            # 计算当前进度
            progress = self.num_timesteps / self.total_timesteps
            
            # 线性衰减学习率（确保不为负）
            current_lr = self.lr_start + progress * (self.lr_end - self.lr_start)
            current_lr = max(current_lr, 1e-8)
            self.model.learning_rate = current_lr
            
            # 线性衰减熵系数（确保不为负）
            current_ent_coef = self.ent_start + progress * (self.ent_end - self.ent_start)
            current_ent_coef = max(current_ent_coef, 1e-8)
            self.model.ent_coef = current_ent_coef
            
        except Exception as e:
            logger.error(f"LinearScheduleCallback error: {e}")
        
        return True


class ProgressLogCallback(BaseCallback):
    """
    定期打印训练进度的Callback（每10,000步）
    """
    def __init__(self, log_interval_steps: int = 10_000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_interval_steps = log_interval_steps
        self.last_log_step = 0
        
    def _on_step(self) -> bool:
        """每个step调用"""
        if self.num_timesteps - self.last_log_step >= self.log_interval_steps:
            # 获取当前学习率和熵系数
            lr = self.model.learning_rate
            ent_coef = self.model.ent_coef
            
            # 获取最近的平均奖励（从ep_info_buffer）
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                avg_reward = np.mean(recent_rewards)
                avg_ep_len = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
            else:
                avg_reward = 0.0
                avg_ep_len = 0.0
            
            logger.info(f"  Step {self.num_timesteps:>7,} | Avg Reward: {avg_reward:>8.2f} | Avg Ep Len: {avg_ep_len:>6.0f} | LR: {lr:.2e} | Ent: {ent_coef:.4f}")
            self.last_log_step = self.num_timesteps
        
        return True


# ========== 以下回调从 models/rl/ppo_agent.py 迁移而来 ==========

class CustomCallback(BaseCallback):
    """自定义训练回调"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_reward = 0
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """每步回调"""
        self.episode_reward += self.locals['rewards'][0]
        
        if self.locals['dones'][0]:
            self.episode_count += 1
            if self.verbose > 0 and self.episode_count % 10 == 0:
                print(f"Episode {self.episode_count}: Reward = {self.episode_reward:.2f}")
            self.episode_reward = 0
        
        return True



class AdaptiveLRScheduler(BaseCallback):
    """PPO学习率自适应调度器 - 根据奖励停滞情况自动降低学习率"""
    
    def __init__(self, patience: int = 50, lr_reduction: float = 0.5, verbose: int = 0):
        """
        初始化学习率调度器
        
        Args:
            patience: 奖励停滞多少步后降低学习率
            lr_reduction: 学习率降低因子（0.5表示减半）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.best_reward = -float('inf')
        self.stagnant_steps = 0
        self.patience = patience
        self.lr_reduction = lr_reduction
        self.last_lr = None
    
    def _on_training_start(self) -> None:
        """训练开始时初始化学习率"""
        self.last_lr = self.model.learning_rate
        if self.verbose > 0:
            print(f"初始学习率: {self.last_lr:.2e}")
    
    def _on_step(self) -> bool:
        """每一步检查奖励变化，必要时调整学习率"""
        if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
            # 获取最近10个episode的奖励
            recent_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[-10:]]
            if recent_rewards:
                avg_reward = np.mean(recent_rewards)
                
                # 检查奖励是否有显著提升
                if avg_reward > self.best_reward * 1.01:  # 1%提升阈值
                    self.best_reward = avg_reward
                    self.stagnant_steps = 0
                else:
                    self.stagnant_steps += 1
                
                # 持续停滞时降低学习率
                if self.stagnant_steps >= self.patience:
                    new_lr = self.last_lr * self.lr_reduction
                    self.model.learning_rate = new_lr
                    self.last_lr = new_lr
                    self.stagnant_steps = 0
                    
                    if self.verbose > 0:
                        print(f"步骤 {self.num_timesteps}: 奖励停滞 {self.patience} 次，"
                              f"学习率降至 {new_lr:.2e}, 当前平均奖励: {avg_reward:.2f}")
        return True
    
    
    # 优化五：PPO训练健康监控与动态早停回调

class PPOHealthMonitorCallback(BaseCallback):
    """
    监控 PPO 训练健康度，实现动态早停
    
    监控指标：
    1. approx_kl（近似KL散度）— 检测策略更新过大
    2. entropy（策略熵）— 检测探索能力坍塌
    3. 奖励收敛 — 实现早停
    """
    
    def __init__(self, 
                 patience: int = 15, 
                 kl_threshold: float = 0.05,
                 entropy_threshold: float = 0.1,
                 check_freq: int = 2048,
                 verbose: int = 0):
        """
        初始化健康监控回调
        
        Args:
            patience: 奖励无提升多少评估周期后早停
            kl_threshold: KL散度阈值（超过则终止训练）
            entropy_threshold: 熵阈值（低于则终止训练）
            check_freq: 检查频率（每多少步检查一次）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.patience = patience
        self.kl_threshold = kl_threshold
        self.entropy_threshold = entropy_threshold
        self.check_freq = check_freq
        self.no_improve_count = 0
        self.best_reward = -np.inf
        self.last_kl = 0.0
        self.last_entropy = 0.0
        
    def _on_step(self) -> bool:
        """每步回调（按 check_freq 频率检查）"""
        if self.n_calls % self.check_freq != 0:
            return True
        
        # 1. 获取训练日志（KL散度和熵）
        # SB3 日志键名：train/approx_kl, train/entropy
        approx_kl = self.model.logger.name_to_value.get("train/approx_kl", 0.0)
        entropy = self.model.logger.name_to_value.get("train/entropy", 0.0)
        
        # 保存最近值（用于日志）
        self.last_kl = approx_kl
        self.last_entropy = entropy
        
        # 2. 健康度检查：KL散度爆炸 → 策略更新步长过大
        if isinstance(approx_kl, (int, float)) and approx_kl > self.kl_threshold:
            print(f"[HealthMonitor] ⚠️ KL散度过高: {approx_kl:.4f} > {self.kl_threshold:.4f}，"
                  f"策略可能崩溃。提前终止训练。")
            return False  # 终止训练
            
        # 3. 探索度检查：熵过低 → 策略变成确定性，失去探索能力
        if isinstance(entropy, (int, float)) and entropy < self.entropy_threshold:
            print(f"[HealthMonitor] ⚠️ 策略熵过低: {entropy:.4f} < {self.entropy_threshold:.4f}，"
                  f"策略坍塌。提前终止训练。")
            return False  # 终止训练
            
        # 4. 早停逻辑（基于最近评估奖励）
        # 获取最近评估奖励（从 model.ep_info_buffer）
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep['r'] for ep in list(self.model.ep_info_buffer)[-10:]]
            if recent_rewards:
                avg_reward = np.mean(recent_rewards)
                
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.no_improve_count = 0
                    # 保存当前最优模型
                    self.model.save("best_ppo_policy")
                    if self.verbose > 0:
                        print(f"[HealthMonitor] 新最优模型保存：平均奖励={avg_reward:.2f}")
                else:
                    self.no_improve_count += 1
                    
                if self.no_improve_count >= self.patience:
                    print(f"[HealthMonitor] ℹ️ 奖励已 {self.patience} 次评估无提升，"
                          f"触发早停。最优奖励: {self.best_reward:.2f}")
                    return False  # 终止训练
                    
        # 5. 周期性健康报告
        if self.verbose > 0 and self.n_calls % (self.check_freq * 5) == 0:
            print(f"[HealthMonitor] 训练健康报告 — "
                  f"KL: {self.last_kl:.4f}, 熵: {self.last_entropy:.4f}, "
                  f"最优奖励: {self.best_reward:.2f}")
        
        return True  # 继续训练

