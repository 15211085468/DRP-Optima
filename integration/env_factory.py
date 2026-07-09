"""
RL 环境工厂模块 - integration/env_factory.py
提供创建和包装 RL 环境的统一函数。

迁移来源：training/trainer.py::prepare_rl_env()
迁移时间：2026-06-17
"""

import numpy as np
from typing import Optional, Tuple

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from config import EnvConfig
from models.optimization import SupplyChainEnv
from utils.logger import get_logger

logger = get_logger(__name__)


def create_rl_env(
    demand_forecasts: Optional[np.ndarray] = None,
    demand_forecasts_quantiles: Optional[np.ndarray] = None,
    num_skus: Optional[int] = None,
    num_stores: int = 1,
    env_config: Optional[EnvConfig] = None,
    use_vec_normalize: bool = True,
    normalize_gamma: float = 0.99,
    monitor_log_dir: Optional[str] = None,
) -> Tuple[SupplyChainEnv, VecNormalize]:
    """
    创建并包装 RL 环境（统一入口）
    
    此函数替代 training/trainer.py 中的 prepare_rl_env() 方法，
    将环境创建逻辑统一到 integration 模块。
    
    Args:
        demand_forecasts: 需求预测数据，形状 (T, num_skus)
        demand_forecasts_quantiles: TFT分位数预测，形状 (T, num_skus, num_quantiles)
        num_skus: SKU数量（如果为None，则从demand_forecasts推断）
        num_stores: 门店数量
        env_config: 环境配置（EnvConfig 对象）
        use_vec_normalize: 是否使用 VecNormalize 包装环境
        normalize_gamma: VecNormalize 的折扣因子
        monitor_log_dir: Monitor 日志目录（如果为None，则使用临时目录）
    
    Returns:
        Tuple: (supply_chain_env, vec_env)
        - supple_chain_env: 原始环境（未包装）
        - vec_env: 包装后的向量环境（已添加 VecNormalize）
    """
    logger.info("=" * 50)
    logger.info("创建RL环境（integration/env_factory）...")
    
    # 1. 确定 num_skus
    if num_skus is None:
        if demand_forecasts is not None:
            num_skus = demand_forecasts.shape[1]
        elif demand_forecasts_quantiles is not None:
            num_skus = demand_forecasts_quantiles.shape[1]
        else:
            num_skus = 50  # 默认值
        logger.info(f"  自动确定 num_skus: {num_skus}")
    
    # 2. 创建 SupplyChainEnv
    env_config_obj = env_config or EnvConfig()
    
    supply_chain_env = SupplyChainEnv(
        num_skus=num_skus,
        num_stores=num_stores,
        demand_forecasts=demand_forecasts,
        demand_forecasts_quantiles=demand_forecasts_quantiles,
        env_config=env_config_obj,
    )
    
    logger.info(f"  ✓ SupplyChainEnv 创建完成: obs_space={supply_chain_env.observation_space.shape}, "
                f"action_space={supply_chain_env.action_space.shape}")
    
    # 3. 添加 Monitor 包装器（SB3 强制最佳实践）
    if monitor_log_dir is None:
        import tempfile
        monitor_log_dir = tempfile.mkdtemp()
    
    from stable_baselines3.common.monitor import Monitor
    import os
    
    monitor_path = os.path.join(monitor_log_dir, "monitor.csv")
    supply_chain_env = Monitor(supply_chain_env, filename=monitor_path)
    logger.info(f"  ✓ Monitor 已添加（日志: {monitor_path}）")
    
    # 4. 包装成 DummyVecEnv
    vec_env = DummyVecEnv([lambda: supply_chain_env])
    logger.info(f"  ✓ 环境已包装成 DummyVecEnv")
    
    # 5. 添加 VecNormalize 包装器
    if use_vec_normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=normalize_gamma,
        )
        logger.info(f"  ✓ VecNormalize 已启用（norm_obs=True, norm_reward=True）")
    
    logger.info(f"✓ RL环境创建完成")
    logger.info("=" * 50)
    
    return supply_chain_env, vec_env


def create_tft_drl_integration_env(
    tft_forecaster,
    ppo_config: Optional[dict] = None,
    env_config: Optional[EnvConfig] = None,
    num_skus: Optional[int] = None,
    use_vec_normalize: bool = True,
) -> "TFTDRLIntegration":
    """
    创建完整的 TFT-DRL 集成环境（便捷函数）
    
    此函数封装了：
    1. 创建 TFT 推理包装器
    2. 创建供应链环境
    3. 创建 PPO 智能体
    4. 创建 TFTDRLIntegration 实例
    
    警告：此函数会创建 PPO 智能体，需要确保 stable_baselines3 已安装。
    """
    # 延迟导入，避免循环导入
    from .tft_drl_integration import TFTDRLIntegration, IntegrationConfig
    
    logger.info("=" * 60)
    logger.info("创建 TFT-DRL 集成环境...")
    
    # 1. 创建供应链环境
    _, vec_env = create_rl_env(
        demand_forecasts=None,  # 稍后由 TFT 预测
        num_skus=num_skus,
        env_config=env_config,
        use_vec_normalize=use_vec_normalize,
    )
    
    # 2. 创建 PPO 智能体
    from models.rl import PPOAgent
    from config import PPOConfig
    
    state_dim = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_space.shape[0]
    
    ppo_config_obj = PPOConfig()
    if ppo_config:
        # 过滤有效键
        valid_keys = PPOConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in ppo_config.items() if k in valid_keys}
        for k, v in filtered.items():
            setattr(ppo_config_obj, k, v)
    
    ppo_agent = PPOAgent(state_dim, action_dim, ppo_config=ppo_config_obj)
    ppo_agent.set_env(vec_env)
    
    # 3. 创建 TFTDRLIntegration 实例
    integration = TFTDRLIntegration(
        tft_forecaster=tft_forecaster,
        ppo_agent=ppo_agent,
        supply_chain_env=vec_env,
        config=IntegrationConfig(),
    )
    
    logger.info("✓ TFT-DRL 集成环境创建完成")
    logger.info("=" * 60)
    
    return integration
