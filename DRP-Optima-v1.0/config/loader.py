"""
配置加载器 (Config Loader)
支持 YAML 预设文件加载和 CLI 参数覆盖

使用方法：
    from config.loader import load_config
    
    # 加载 baseline 预设
    configs = load_config(preset_name='baseline')
    env_config = configs['env']
    ppo_config = configs['ppo']
    
    # 加载 proposed 预设，并用 CLI 参数覆盖
    configs = load_config(
        preset_name='proposed',
        cli_overrides={'env.shortage_penalty_per_unit': 8.0}
    )
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from dataclasses import asdict
from typing import Dict, Any, Optional

from config.data_config import EnvConfig, PPOConfig


def load_config(preset_name: str = None, cli_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    加载配置：Dataclass默认值 <- YAML预设 <- CLI参数覆盖
    
    优先级（从低到高）：
    1. Dataclass 默认值（在 EnvConfig 和 PPOConfig 中定义）
    2. YAML 预设文件（config/presets/{preset_name}.yaml）
    3. CLI 参数覆盖（cli_overrides 字典）
    
    参数:
        preset_name: 预设名称（'baseline' 或 'proposed'）
        cli_overrides: CLI 参数覆盖（例如 {'env.shortage_penalty_per_unit': 8.0}）
    
    返回:
        configs: 配置字典，包含 'env' 和 'ppo' 两个键
    """
    # 1. 获取默认值（从 Dataclass）
    env_config = EnvConfig()
    ppo_config = PPOConfig()
    
    # 2. 加载 YAML 预设（如果指定了）
    if preset_name:
        yaml_path = f"config/presets/{preset_name}.yaml"
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"找不到预设文件: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # 覆盖 env 配置
        for k, v in yaml_data.get('env', {}).items():
            if hasattr(env_config, k):
                # 类型转换（确保类型正确）
                target_type = type(getattr(env_config, k))
                if target_type == bool and isinstance(v, str):
                    # 处理布尔值（YAML 中的 true/false 可能被解析为字符串）
                    v = v.lower() == 'true'
                elif target_type in [int, float] and isinstance(v, str):
                    v = target_type(v)
                setattr(env_config, k, v)
            else:
                print(f"⚠️  警告：EnvConfig 中没有属性 '{k}'，已忽略")
        
        # 覆盖 ppo 配置
        for k, v in yaml_data.get('ppo', {}).items():
            if hasattr(ppo_config, k):
                # 类型转换
                target_type = type(getattr(ppo_config, k))
                if target_type in [int, float] and isinstance(v, str):
                    v = target_type(v)
                setattr(ppo_config, k, v)
            else:
                print(f"[WARNING] 警告：PPOConfig 中没有属性 '{k}'，已忽略")
        
        print(f"[OK] YAML 预设加载成功：{preset_name}")
    
    # 3. CLI 参数覆盖（处理类似 --env.shortage_penalty 8.0 的参数）
    if cli_overrides:
        for k, v in cli_overrides.items():
            if k.startswith('env.') and hasattr(env_config, k[4:]):
                attr_name = k[4:]
                target_type = type(getattr(env_config, attr_name))
                setattr(env_config, attr_name, target_type(v))
                print(f"  ✓ CLI 覆盖：env.{attr_name} = {target_type(v)}")
            elif k.startswith('ppo.') and hasattr(ppo_config, k[4:]):
                attr_name = k[4:]
                target_type = type(getattr(ppo_config, attr_name))
                setattr(ppo_config, attr_name, target_type(v))
                print(f"  ✓ CLI 覆盖：ppo.{attr_name} = {target_type(v)}")
            else:
                print(f"[WARNING] 警告：未知的配置键 '{k}'，已忽略")
    
    return {'env': env_config, 'ppo': ppo_config}


def save_config_to_yaml(configs: Dict[str, Any], yaml_path: str):
    """
    将配置保存到 YAML 文件
    
    参数:
        configs: 配置字典（包含 'env' 和 'ppo' 两个键）
        yaml_path: YAML 文件路径
    """
    data = {
        'env': {k: v for k, v in asdict(configs['env']).items() if not k.startswith('_')},
        'ppo': {k: v for k, v in asdict(configs['ppo']).items() if not k.startswith('_')}
    }
    
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"[OK] 配置已保存到：{yaml_path}")


if __name__ == '__main__':
    # 测试代码
    print("=" * 80)
    print("测试配置加载器")
    print("=" * 80)
    
    # 测试1：加载 baseline 预设
    print("\n测试1：加载 baseline 预设")
    configs = load_config(preset_name='baseline')
    print(f"  - shortage_penalty_per_unit: {configs['env'].shortage_penalty_per_unit}")
    print(f"  - enable_abc_reward: {configs['env'].enable_abc_reward}")
    print(f"  - total_timesteps: {configs['ppo'].total_timesteps}")
    
    # 测试2：加载 proposed 预设
    print("\n测试2：加载 proposed 预设")
    configs = load_config(preset_name='proposed')
    print(f"  - shortage_penalty_per_unit: {configs['env'].shortage_penalty_per_unit}")
    print(f"  - enable_abc_reward: {configs['env'].enable_abc_reward}")
    print(f"  - total_timesteps: {configs['ppo'].total_timesteps}")
    
    # 测试3：CLI 参数覆盖
    print("\n测试3：CLI 参数覆盖")
    configs = load_config(
        preset_name='proposed',
        cli_overrides={'env.shortage_penalty_per_unit': 8.0, 'ppo.total_timesteps': 1000000}
    )
    print(f"  - shortage_penalty_per_unit: {configs['env'].shortage_penalty_per_unit}")
    print(f"  - total_timesteps: {configs['ppo'].total_timesteps}")
    
    print("\n[OK] 所有测试通过！")
