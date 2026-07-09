"""
DRP-Optima 配置模块
包含环境配置、PPO配置、TFT配置等
"""

from config.data_config import (
    PathConfig,
    DataConfig,
    EnvConfig,
    PPOConfig,
    TFTModelConfig,
    HardwareConfig
)

__all__ = [
    'PathConfig',
    'DataConfig',
    'EnvConfig',
    'PPOConfig',
    'TFTModelConfig',
    'HardwareConfig'
]


def get_config(config_name: str = 'default'):
    """
    获取配置（兼容旧代码）
    
    这是一个占位函数，用于兼容旧代码中的 get_config() 调用。
    新代码应该使用 config.loader.load_config() 或直接实例化配置类。
    
    参数:
        config_name: 配置名称（当前未使用，保留用于兼容性）
    
    返回:
        默认配置对象（EnvConfig）
    """
    return EnvConfig()
