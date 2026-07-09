"""
可视化数据收集器 - utils/visualization_collector.py
用于收集训练过程数据，供论文作图使用
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime


class VisualizationCollector:
    """可视化数据收集器"""
    
    def __init__(self, output_dir: str = "./records/output/visualization_data"):
        """
        初始化收集器
        
        Args:
            output_dir: 可视化数据输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # TFT训练数据
        self.tft_train_loss = []
        self.tft_val_loss = []
        self.tft_learning_rate = []
        self.tft_epoch = []
        
        # PPO训练数据
        self.ppo_episode_rewards = []
        self.ppo_episode_lengths = []
        self.ppo_policy_loss = []
        self.ppo_value_loss = []
        self.ppo_entropy = []
        self.ppo_kl_divergence = []
        self.ppo_timesteps = []
        
        # 集成系统数据
        self.integration_rewards = []
        self.integration_demands = []
        self.integration_actions = []
        self.integration_stockout_rates = []
        self.integration_steps = []
        
        # 时间戳
        self.start_time = datetime.now()
        
    def collect_tft_metrics(self, epoch: int, train_loss: float, 
                           val_loss: Optional[float] = None, 
                           learning_rate: Optional[float] = None):
        """
        收集TFT训练指标
        
        Args:
            epoch: 训练轮次
            train_loss: 训练损失
            val_loss: 验证损失
            learning_rate: 学习率
        """
        self.tft_epoch.append(epoch)
        self.tft_train_loss.append(train_loss)
        self.tft_val_loss.append(val_loss)
        self.tft_learning_rate.append(learning_rate)
        
    def collect_ppo_metrics(self, timestep: int, episode_reward: float,
                            episode_length: int, policy_loss: Optional[float] = None,
                            value_loss: Optional[float] = None,
                            entropy: Optional[float] = None,
                            kl_divergence: Optional[float] = None):
        """
        收集PPO训练指标
        
        Args:
            timestep: 时间步
            episode_reward: 回合奖励
            episode_length: 回合长度
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy: 熵
            kl_divergence: KL散度
        """
        self.ppo_timesteps.append(timestep)
        self.ppo_episode_rewards.append(episode_reward)
        self.ppo_episode_lengths.append(episode_length)
        self.ppo_policy_loss.append(policy_loss)
        self.ppo_value_loss.append(value_loss)
        self.ppo_entropy.append(entropy)
        self.ppo_kl_divergence.append(kl_divergence)
        
    def collect_integration_metrics(self, step: int, reward: float,
                                  demand: np.ndarray, action: np.ndarray,
                                  stockout_rate: float):
        """
        收集集成系统指标
        
        Args:
            step: 时间步
            reward: 奖励
            demand: 需求预测
            action: 动作
            stockout_rate: 缺货率
        """
        self.integration_steps.append(step)
        self.integration_rewards.append(reward)
        self.integration_demands.append(demand.copy())
        self.integration_actions.append(action.copy())
        self.integration_stockout_rates.append(stockout_rate)
        
    def save_tft_data(self):
        """保存TFT训练数据"""
        if not self.tft_epoch:
            return
            
        df = pd.DataFrame({
            'epoch': self.tft_epoch,
            'train_loss': self.tft_train_loss,
            'val_loss': self.tft_val_loss,
            'learning_rate': self.tft_learning_rate
        })
        
        output_path = os.path.join(self.output_dir, 'tft_training_metrics.csv')
        df.to_csv(output_path, index=False)
        print(f"TFT训练数据已保存: {output_path}")
        
    def save_ppo_data(self):
        """保存PPO训练数据"""
        if not self.ppo_timesteps:
            return
            
        df = pd.DataFrame({
            'timestep': self.ppo_timesteps,
            'episode_reward': self.ppo_episode_rewards,
            'episode_length': self.ppo_episode_lengths,
            'policy_loss': self.ppo_policy_loss,
            'value_loss': self.ppo_value_loss,
            'entropy': self.ppo_entropy,
            'kl_divergence': self.ppo_kl_divergence
        })
        
        output_path = os.path.join(self.output_dir, 'ppo_training_metrics.csv')
        df.to_csv(output_path, index=False)
        print(f"PPO训练数据已保存: {output_path}")
        
    def save_integration_data(self):
        """保存集成系统数据"""
        if not self.integration_steps:
            return
            
        # 保存汇总数据
        df_summary = pd.DataFrame({
            'step': self.integration_steps,
            'reward': self.integration_rewards,
            'stockout_rate': self.integration_stockout_rates
        })
        
        output_path_summary = os.path.join(self.output_dir, 'integration_summary.csv')
        df_summary.to_csv(output_path_summary, index=False)
        print(f"集成系统汇总数据已保存: {output_path_summary}")
        
        # 保存详细数据（需求预测和动作）
        if self.integration_demands:
            demands_array = np.array(self.integration_demands)
            actions_array = np.array(self.integration_actions)
            
            # 保存为npz格式（压缩的numpy数组）
            output_path_detail = os.path.join(self.output_dir, 'integration_details.npz')
            np.savez_compressed(
                output_path_detail,
                demands=demands_array,
                actions=actions_array,
                steps=np.array(self.integration_steps)
            )
            print(f"集成系统详细数据已保存: {output_path_detail}")
            
    def save_all_data(self):
        """保存所有数据"""
        self.save_tft_data()
        self.save_ppo_data()
        self.save_integration_data()
        
        # 保存元数据
        metadata = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'tft_epochs': len(self.tft_epoch),
            'ppo_timesteps': len(self.ppo_timesteps),
            'integration_steps': len(self.integration_steps)
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"元数据已保存: {metadata_path}")
        
    def get_tft_loss_curve_data(self) -> Dict:
        """获取TFT损失曲线数据（用于作图）"""
        return {
            'epochs': self.tft_epoch,
            'train_loss': self.tft_train_loss,
            'val_loss': self.tft_val_loss
        }
        
    def get_ppo_reward_curve_data(self) -> Dict:
        """获取PPO奖励曲线数据（用于作图）"""
        return {
            'timesteps': self.ppo_timesteps,
            'episode_rewards': self.ppo_episode_rewards
        }
        
    def get_integration_performance_data(self) -> Dict:
        """获取集成系统性能数据（用于作图）"""
        return {
            'steps': self.integration_steps,
            'rewards': self.integration_rewards,
            'stockout_rates': self.integration_stockout_rates
        }
        

def create_visualization_plots(data_dir: str = "./records/output/visualization_data",
                              output_dir: str = "./records/output/visualization_plots"):
    """
    根据收集的数据创建可视化图表（用于论文）
    
    Args:
        data_dir: 数据目录
        output_dir: 图表输出目录
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置论文风格的图表
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. TFT损失曲线
    tft_data_path = os.path.join(data_dir, 'tft_training_metrics.csv')
    if os.path.exists(tft_data_path):
        df_tft = pd.read_csv(tft_data_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(df_tft['epoch'], df_tft['train_loss'], label='训练损失', linewidth=2)
        if df_tft['val_loss'].notna().any():
            axes[0].plot(df_tft['epoch'], df_tft['val_loss'], label='验证损失', linewidth=2)
        axes[0].set_xlabel('训练轮次')
        axes[0].set_ylabel('损失')
        axes[0].set_title('TFT训练损失曲线')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 学习率曲线
        if df_tft['learning_rate'].notna().any():
            axes[1].plot(df_tft['epoch'], df_tft['learning_rate'], linewidth=2, color='orange')
            axes[1].set_xlabel('训练轮次')
            axes[1].set_ylabel('学习率')
            axes[1].set_title('TFT学习率变化曲线')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tft_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"TFT训练曲线已保存: {output_dir}/tft_training_curves.png")
        
    # 2. PPO奖励曲线
    ppo_data_path = os.path.join(data_dir, 'ppo_training_metrics.csv')
    if os.path.exists(ppo_data_path):
        df_ppo = pd.read_csv(ppo_data_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 奖励曲线
        axes[0, 0].plot(df_ppo['timestep'], df_ppo['episode_reward'], linewidth=2)
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('回合奖励')
        axes[0, 0].set_title('PPO训练奖励曲线')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 策略损失
        if df_ppo['policy_loss'].notna().any():
            axes[0, 1].plot(df_ppo['timestep'], df_ppo['policy_loss'], linewidth=2, color='red')
            axes[0, 1].set_xlabel('时间步')
            axes[0, 1].set_ylabel('策略损失')
            axes[0, 1].set_title('PPO策略损失曲线')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 价值损失
        if df_ppo['value_loss'].notna().any():
            axes[1, 0].plot(df_ppo['timestep'], df_ppo['value_loss'], linewidth=2, color='green')
            axes[1, 0].set_xlabel('时间步')
            axes[1, 0].set_ylabel('价值损失')
            axes[1, 0].set_title('PPO价值损失曲线')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 熵
        if df_ppo['entropy'].notna().any():
            axes[1, 1].plot(df_ppo['timestep'], df_ppo['entropy'], linewidth=2, color='purple')
            axes[1, 1].set_xlabel('时间步')
            axes[1, 1].set_ylabel('熵')
            axes[1, 1].set_title('PPO策略熵曲线')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ppo_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PPO训练曲线已保存: {output_dir}/ppo_training_curves.png")
        
    # 3. 集成系统性能曲线
    integration_data_path = os.path.join(data_dir, 'integration_summary.csv')
    if os.path.exists(integration_data_path):
        df_int = pd.read_csv(integration_data_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 奖励曲线
        axes[0].plot(df_int['step'], df_int['reward'], linewidth=2, color='blue')
        axes[0].set_xlabel('时间步')
        axes[0].set_ylabel('奖励')
        axes[0].set_title('集成系统奖励曲线')
        axes[0].grid(True, alpha=0.3)
        
        # 缺货率曲线
        axes[1].plot(df_int['step'], df_int['stockout_rate'], linewidth=2, color='red')
        axes[1].set_xlabel('时间步')
        axes[1].set_ylabel('缺货率')
        axes[1].set_title('集成系统缺货率曲线')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'integration_performance_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"集成系统性能曲线已保存: {output_dir}/integration_performance_curves.png")
        
    print(f"\n所有图表已保存到: {output_dir}")
