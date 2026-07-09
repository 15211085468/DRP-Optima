"""
V10.3 自定义回调（Callbacks）

包含：
1. SyncedEvalCallbackV9 - 同步 VecNormalize 统计量
2. CurriculumTruncationCallbackV9 - Curriculum Learning 控制
3. BusinessMetricsCallbackV9 - CSV 日志（无 TensorBoard）
"""

import os
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from utils.logger import get_logger

logger = get_logger(__name__)


class SyncedEvalCallbackV9(EvalCallback):
    """
    同步评估回调（V9 版本）
    
    在评估前同步 VecNormalize 的统计量（obs_rms, ret_rms）
    确保评估使用最新的归一化参数
    """
    
    def __init__(
        self,
        eval_env: VecEnv,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
    ):
        """
        初始化同步评估回调
        
        Args:
            eval_env: 评估环境
            callback_on_new_best: 新最佳模型时的回调
            callback_after_eval: 评估后的回调
            n_eval_episodes: 评估回合数
            eval_freq: 评估频率（步数）
            log_path: 日志路径
            best_model_save_path: 最佳模型保存路径
            deterministic: 是否使用确定性动作
            render: 是否渲染
            verbose: 日志级别
        """
        # 🔧 修复：确保 eval_env 是 VecNormalize
        from stable_baselines3.common.vec_env import VecNormalize as VecNormalizeClass
        if not isinstance(eval_env, VecNormalizeClass):
            logger.warning(f"[WARN] eval_env is not VecNormalize, wrapping it now...")
            eval_env = VecNormalizeClass(
                eval_env,
                norm_obs=True,
                norm_reward=False,
                training=False,
                clip_obs=10.0,
            )
        
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )
        
        self.eval_env = eval_env
    
    def _on_step(self) -> bool:
        """
        每一步执行的逻辑
        
        Returns:
            是否继续训练
        """
        # 🔧 动态修复：确保 eval_env 是 VecNormalize
        from stable_baselines3.common.vec_env import VecNormalize as VecNormalizeClass
        if not isinstance(self.eval_env, VecNormalizeClass):
            logger.warning(f"[WARN] eval_env is not VecNormalize in _on_step(), wrapping it now...")
            self.eval_env = VecNormalizeClass(
                self.eval_env,
                norm_obs=True,
                norm_reward=False,
                training=False,
                clip_obs=10.0,
            )
        
        # 🔧 关键修复：在评估前同步 VecNormalize 统计量
        self._sync_vec_normalize()
        
        # 调用父类的 _on_step()
        return super()._on_step()
    
    def _sync_vec_normalize(self):
        """
        同步 VecNormalize 统计量（已禁用）
        
        将训练环境的 obs_rms 和 ret_rms 同步到评估环境
        """
        # 🔧 已禁用：sync_vec_envs_normalization() 有问题
        # 不直接调用，避免 AssertionError
        pass


class CurriculumTruncationCallbackV9(BaseCallback):
    """
    Curriculum Learning 回调（V9 版本）
    
    控制硬截断的启用时机：
    - 前 N 步：软惩罚阶段
    - N 步后：硬截断阶段
    """
    
    def __init__(
        self,
        curriculum_soft_epochs: int = 50_000,
        verbose: int = 0,
    ):
        """
        初始化 Curriculum Learning 回调
        
        Args:
            curriculum_soft_epochs: 软惩罚阶段步数
            verbose: 日志级别
        """
        super().__init__(verbose=verbose)
        self.curriculum_soft_epochs = curriculum_soft_epochs
        self.truncation_enabled = False
    
    def _on_step(self) -> bool:
        """
        每一步执行的逻辑
        
        Returns:
            是否继续训练
        """
        # 检查是否进入硬截断阶段
        if (
            not self.truncation_enabled
            and self.num_timesteps >= self.curriculum_soft_epochs
        ):
            self.truncation_enabled = True
            
            # 更新环境配置
            if hasattr(self.training_env, "envs"):
                for env in self.training_env.envs:
                    if hasattr(env, "env"):
                        env.env.env_config.enable_hard_truncation = True
            
            if self.verbose >= 1:
                logger.info(
                    f"Curriculum Learning: 进入硬截断阶段 "
                    f"(step={self.num_timesteps})"
                )
        
        return True


class BusinessMetricsCallbackV9(BaseCallback):
    """
    业务指标回调（V9 版本）
    
    记录业务指标到 CSV 文件（无 TensorBoard）
    """
    
    def __init__(
        self,
        log_path: str = "logs/business_metrics.csv",
        log_freq: int = 1000,
        verbose: int = 0,
    ):
        """
        初始化业务指标回调
        
        Args:
            log_path: CSV 日志路径
            log_freq: 日志频率（步数）
            verbose: 日志级别
        """
        super().__init__(verbose=verbose)
        self.log_path = log_path
        self.log_freq = log_freq
        
        # 创建日志目录
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 初始化 CSV 文件
        self._init_csv()
    
    def _init_csv(self):
        """初始化 CSV 文件（写入表头）"""
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep",
                "episode",
                "avg_fill_rate",
                "avg_stockout_rate",
                "avg_reward",
                "avg_stock_sum",
                "avg_pending_sum",
            ])
    
    def _on_step(self) -> bool:
        """
        每一步执行的逻辑
        
        Returns:
            是否继续训练
        """
        # 按频率记录
        if self.num_timesteps % self.log_freq != 0:
            return True
        
        # 🔧 关键：使用 self.locals（SB3 BaseCallback 的属性名）
        # BaseCallback 在 on_step() 被调用前会设置 self.locals
        locals_dict = self.locals
        
        # 提取指标
        metrics = self._extract_metrics(locals_dict)
        
        # 写入 CSV
        if metrics:
            self._write_metrics(metrics)
        
        return True
    
    def _extract_metrics(self, locals_dict: Dict) -> Optional[Dict]:
        """
        从训练循环中提取指标 [已修复]
        
        Args:
            locals_dict: 训练循环的局部变量
        
        Returns:
            指标字典，如果提取失败则返回 None
        """
        try:
            # 🔧 修复：从多个来源尝试提取指标
            
            # 方法 1：从 self.training_env 获取（推荐）
            env = self.training_env
            
            if env is None:
                logger.warning("   ⚠️ training_env 为 None")
                return None
            
            # 获取 unwrapped 环境
            unwrapped = env.unwrapped
            
            # 如果是 SubprocVecEnv，获取第一个子环境
            if hasattr(unwrapped, 'envs'):
                # SubprocVecEnv
                try:
                    # 获取第一个子环境的 unwrapped
                    child_env = unwrapped.envs[0]
                    if hasattr(child_env, 'unwrapped'):
                        child_unwrapped = child_env.unwrapped
                    else:
                        child_unwrapped = child_env
                    
                    # 从子环境获取指标
                    return self._extract_from_env(child_unwrapped)
                except Exception as e:
                    logger.warning(f"   ⚠️ 从 SubprocVecEnv 提取失败: {e}")
                    return None
            else:
                # 单个环境
                return self._extract_from_env(unwrapped)
        
        except Exception as e:
            logger.warning(f"提取指标失败: {e}")
            import traceback
            logger.warning(f"追溯: {traceback.format_exc()}")
            return None
    
    def _extract_from_env(self, env) -> Optional[Dict]:
        """
        从环境实例中提取指标 [辅助方法]
        
        Args:
            env: 环境实例（已 unwrapped）
        
        Returns:
            指标字典
        """
        # 检查环境类型
        if not hasattr(env, 'episode_reward'):
            logger.warning(f"   ⚠️ 环境缺少 episode_reward 属性: {type(env)}")
            return None
        
        # 检查是否有完整的 episode
        if env.episode_steps <= 0:
            return None  # 当前 episode 还未开始或已结束
        
        # 提取指标
        metrics = {
            "timestep": self.num_timesteps,
            "episode": getattr(env, 'n_episodes', 0),
            "avg_fill_rate": getattr(env, 'last_fill_rate', 0.0),
            "avg_stockout_rate": getattr(env, 'last_stockout_rate', 0.0),
            "avg_reward": env.episode_reward / max(env.episode_steps, 1),
            "avg_stock_sum": float(np.sum(env.current_stock)) if hasattr(env, 'current_stock') else 0.0,
            "avg_pending_sum": float(np.sum(env.pending_order)) if hasattr(env, 'pending_order') else 0.0,
        }
        
        logger.info(f"   ✓ 指标已提取: episode={metrics['episode']}, reward={metrics['avg_reward']:.2f}")
        
        return metrics
    
    def _write_metrics(self, metrics: Dict):
        """
        写入指标到 CSV
        
        Args:
            metrics: 指标字典
        """
        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics["timestep"],
                    metrics["episode"],
                    f"{metrics['avg_fill_rate']:.4f}",
                    f"{metrics['avg_stockout_rate']:.4f}",
                    f"{metrics['avg_reward']:.4f}",
                    f"{metrics['avg_stock_sum']:.2f}",
                    f"{metrics['avg_pending_sum']:.2f}",
                ])
        
        except Exception as e:
            logger.warning(f"写入指标失败: {e}")


def create_v9_callbacks(
    eval_env: VecEnv,
    log_dir: str = "logs",
    eval_freq: int = 50_000,
    n_eval_episodes: int = 5,
    curriculum_soft_epochs: int = 50_000,
    business_metrics_freq: int = 1000,
) -> List[BaseCallback]:
    """
    创建 V9 回调列表（工厂函数）
    
    Args:
        eval_env: 评估环境
        log_dir: 日志目录
        eval_freq: 评估频率
        n_eval_episodes: 评估回合数
        curriculum_soft_epochs: Curriculum 软惩罚步数
        business_metrics_freq: 业务指标记录频率
    
    Returns:
        回调列表
    """
    callbacks = []
    
    # 1. SyncedEvalCallbackV9 【已禁用，避免 sync_vec_envs_normalization() 错误】
    # eval_callback = SyncedEvalCallbackV9(
    #     eval_env=eval_env,
    #     n_eval_episodes=n_eval_episodes,
    #     eval_freq=eval_freq,
    #     log_path=os.path.join(log_dir, "eval"),
    #     best_model_save_path=os.path.join(log_dir, "best_model"),
    #     verbose=1,
    # )
    # callbacks.append(eval_callback)
    logger.info("  ⚠️ SyncedEvalCallbackV9 已禁用（避免同步错误）")
    
    # 2. CurriculumTruncationCallbackV9
    curriculum_callback = CurriculumTruncationCallbackV9(
        curriculum_soft_epochs=curriculum_soft_epochs,
        verbose=1,
    )
    callbacks.append(curriculum_callback)
    
    # 3. BusinessMetricsCallbackV9
    business_callback = BusinessMetricsCallbackV9(
        log_path=os.path.join(log_dir, "business_metrics.csv"),
        log_freq=business_metrics_freq,
        verbose=1,
    )
    callbacks.append(business_callback)
    
    logger.info(f"已创建 {len(callbacks)} 个 V9 回调")
    
    return callbacks
