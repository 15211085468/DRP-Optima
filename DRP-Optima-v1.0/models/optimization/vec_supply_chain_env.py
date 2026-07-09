"""
向量化供应链环境 - models/optimization/vec_supply_chain_env.py
使用多线程并行运行多个SupplyChainEnv实例，支持批量步进
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import gymnasium as gym
from gymnasium.vector import VectorEnv

from config import EnvConfig
from models.optimization.supply_chain_env import SupplyChainEnv


class VecSupplyChainEnv(VectorEnv):
    """向量化供应链环境（并行运行多个实例）"""
    
    def __init__(self,
                 num_envs: int = 4,
                 num_skus: int = 478,
                 num_stores: int = 100,
                 demand_forecasts: Optional[np.ndarray] = None,
                 env_config: Optional[EnvConfig] = None):
        """
        初始化向量化环境
        
        Args:
            num_envs: 并行环境数量
            num_skus: SKU数量
            num_stores: 门店数量
            demand_forecasts: 需求预测数据
            env_config: 环境配置
        """
        self.num_envs = num_envs
        self.num_skus = num_skus
        
        # 创建多个环境实例
        self.envs = []
        for i in range(num_envs):
            env = SupplyChainEnv(
                num_skus=num_skus,
                num_stores=num_stores,
                demand_forecasts=demand_forecasts,
                env_config=env_config
            )
            self.envs.append(env)
        
        # 使用第一个环境定义空间和配置
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space
        
        # 初始化VectorEnv
        super().__init__(num_envs, self.single_observation_space, self.single_action_space)
        
        # 并行执行器
        self.executor = ThreadPoolExecutor(max_workers=num_envs)
        
        # 跟踪哪些环境已结束
        self.episode_ended = [False] * num_envs
    
    def reset(self, 
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置所有并行环境
        
        Args:
            seed: 随机种子（可选）
            options: 重置选项（可选）
        
        Returns:
            Tuple: (states, infos)
        """
        if seed is not None:
            for i, env in enumerate(self.envs):
                env.reset(seed=seed+i, options=options)
        
        # 并行重置所有环境
        futures = [self.executor.submit(env.reset, seed, options) for env in self.envs]
        
        states = []
        infos = []
        for future in as_completed(futures):
            idx = futures.index(future)
            state, info = future.result()
            states.append(state)
            infos.append(info)
        
        # 重置结束标志
        self.episode_ended = [False] * self.num_envs
        
        return np.array(states), infos
    
    def step(self, 
             actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        并行步进所有环境
        
        Args:
            actions: 批量动作 (num_envs, action_dim)
        
        Returns:
            Tuple: (states, rewards, terminateds, truncateds, infos)
        """
        # 并行执行step
        futures = []
        for i, env in enumerate(self.envs):
            if not self.episode_ended[i]:
                future = self.executor.submit(env.step, actions[i])
                futures.append((i, future))
        
        # 收集结果
        states = np.zeros((self.num_envs, self.single_observation_space.shape[0]))
        rewards = np.zeros(self.num_envs)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        
        for i, future in futures:
            state, reward, terminated, truncated, info = future.result()
            states[i] = state
            rewards[i] = reward
            terminateds[i] = terminated
            truncateds[i] = truncated
            infos[i] = info
            
            if terminated or truncated:
                self.episode_ended[i] = True
        
        return states, rewards, terminateds, truncateds, infos
    
    def close(self):
        """关闭所有环境和执行器"""
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=True)
    
    def get_attr(self, attr_name: str) -> List[Any]:
        """获取所有环境的属性"""
        return [getattr(env, attr_name) for env in self.envs]
    
    def set_attr(self, attr_name: str, values: List[Any]):
        """设置所有环境的属性"""
        for env, value in zip(self.envs, values):
            setattr(env, attr_name, value)
    
    def call_method(self, method_name: str, *args, **kwargs) -> List[Any]:
        """调用所有环境的方法"""
        return [getattr(env, method_name)(*args, **kwargs) for env in self.envs]


def make_vec_env(num_envs: int = 4, **kwargs) -> VecSupplyChainEnv:
    """
    创建向量化环境的便捷函数
    
    Args:
        num_envs: 并行环境数量
        **kwargs: SupplyChainEnv的初始化参数
    
    Returns:
        VecSupplyChainEnv: 向量化环境
    """
    return VecSupplyChainEnv(num_envs=num_envs, **kwargs)


if __name__ == "__main__":
    # 测试向量化环境
    vec_env = make_vec_env(num_envs=4, num_skus=10, num_stores=5)
    
    states, infos = vec_env.reset()
    print(f"初始状态形状: {states.shape}")
    
    for step in range(10):
        actions = np.random.uniform(0, 100, (vec_env.num_envs, vec_env.single_action_space.shape[0]))
        states, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        
        print(f"Step {step}: 奖励={rewards}, 结束={terminateds}")
        
        if all(terminateds) or all(truncateds):
            break
    
    vec_env.close()
    print("向量化环境测试完成")
