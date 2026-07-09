"""
自定义供应链环境（封装 TFT 预测）
用于在训练 PPO 时自动调用 TFT 预测
"""
import sys
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.optimization.supply_chain_env import SupplyChainEnv
from models.forecasting.tft_model import TFTForecaster
from integration.tft_drl_integration import TFTDRLIntegration, IntegrationConfig

logger = logging.getLogger(__name__)


class SupplyChainEnvWithTFT(SupplyChainEnv):
    """
    封装了 TFT 预测的供应链环境
    
    在每次 step() 调用前，自动使用 TFT 预测需求，
    并将预测结果作为状态的一部分（或用于决策）
    """
    
    def __init__(self, 
                 num_skus: int = 5,
                 tft_checkpoint_path: str = "records/models/tft_model_quantile.ckpt",
                 history_df: Optional[pd.DataFrame] = None,
                 **kwargs):
        """
        初始化环境
        
        Args:
            num_skus: SKU 数量
            tft_checkpoint_path: TFT 模型检查点路径
            history_df: 历史数据 DataFrame（用于 TFT 预测）
            **kwargs: 传递给 SupplyChainEnv 的其他参数
        """
        super().__init__(num_skus=num_skus, **kwargs)
        
        # 初始化 TFT 预测器
        self.tft_forecaster = TFTForecaster()
        # 加载预训练模型
        if os.path.exists(tft_checkpoint_path):
            self.tft_forecaster.load_model(tft_checkpoint_path)
            logger.info(f"✅ TFT 预测器加载成功: {tft_checkpoint_path}")
        else:
            logger.warning(f"⚠️ TFT 模型文件不存在: {tft_checkpoint_path}，将使用随机初始化的模型")
        
        # 历史数据（用于 TFT 预测）
        self.history_df = history_df
        self.current_step_idx = 0
        
        # TFT 预测缓存
        self.tft_prediction_cache = None
        self.cache_update_freq = 5  # 每 5 步更新一次 TFT 预测
        self.step_counter = 0
        
        # 训练指标记录
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'tft_prediction_errors': [],  # TFT 预测误差（如果有真实值）
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
        }
        
        # 重新定义 observation_space（包含 TFT 预测特征）
        # 先进行一次 TFT 预测，获取特征维度
        self._update_tft_prediction()
        if self.tft_prediction_cache is not None:
            tft_feature_dim = self.tft_prediction_cache.size
        else:
            tft_feature_dim = 0
        
        # 新的观测空间维度 = 原维度 + TFT 特征维度
        original_obs_dim = self.observation_space.shape[0]
        new_obs_dim = original_obs_dim + tft_feature_dim
        
        # 重新定义 observation_space
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(new_obs_dim,),
            dtype=np.float32
        )
        logger.info(f"✅ Observation space 已更新: {original_obs_dim} -> {new_obs_dim}")
    
    def set_history_df(self, df: pd.DataFrame):
        """
        设置历史数据（用于 TFT 预测）
        
        Args:
            df: 历史数据 DataFrame
        """
        self.history_df = df.copy()
        self.tft_prediction_cache = None  # 清除缓存
        logger.info(f"✅ 历史数据已设置: {df.shape}")
    
    def _get_state(self) -> np.ndarray:
        """
        重写 _get_state 方法，包含 TFT 预测结果
        
        Returns:
            np.ndarray: 包含 TFT 预测的状态向量
        """
        # 获取基础状态（来自父类）
        base_state = super()._get_state()
        
        # 更新 TFT 预测缓存（如果需要）
        self.step_counter += 1
        if self.tft_prediction_cache is None or self.step_counter % self.cache_update_freq == 0:
            self._update_tft_prediction()
        
        # 将 TFT 预测结果添加到状态中
        if self.tft_prediction_cache is not None:
            # 展平预测结果
            tft_features = self.tft_prediction_cache.flatten()
            
            # 拼接基础状态和 TFT 预测特征
            state = np.concatenate([base_state, tft_features])
        else:
            # 如果没有 TFT 预测，使用零填充（或重复基础状态）
            # 这里需要根据实际情况调整
            state = base_state
        
        return state.astype(np.float32)
    
    def _update_tft_prediction(self):
        """
        更新 TFT 预测缓存
        """
        if self.history_df is None or self.history_df.empty:
            logger.warning("⚠️ 历史数据为空，无法更新 TFT 预测")
            return
        
        try:
            # 调用 TFT 预测
            # 注意：这里需要根据 TFTForecaster 的实际 API 来调整
            demand_forecast = self.tft_forecaster.predict(
                df=self.history_df,
                horizon=1  # 预测未来 1 步
            )
            
            self.tft_prediction_cache = demand_forecast
            logger.info(f"✅ TFT 预测更新成功: {demand_forecast.shape}")
            
        except Exception as e:
            logger.error(f"❌ TFT 预测失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_state_with_tft(self) -> np.ndarray:
        """
        获取包含 TFT 预测的状态
        
        Returns:
            np.ndarray: 包含 TFT 预测的状态向量
        """
        # 获取基础状态（来自父类）
        base_state = super()._get_state()
        
        # 更新 TFT 预测缓存（如果需要）
        self.step_counter += 1
        if self.tft_prediction_cache is None or self.step_counter % self.cache_update_freq == 0:
            self._update_tft_prediction()
        
        # 将 TFT 预测结果添加到状态中
        if self.tft_prediction_cache is not None:
            # 展平预测结果
            tft_features = self.tft_prediction_cache.flatten()
            
            # 拼接基础状态和 TFT 预测特征
            # 注意：这里需要确保维度匹配
            state = np.concatenate([base_state, tft_features])
        else:
            # 如果没有 TFT 预测，使用零填充
            state = base_state
        
        return state.astype(np.float32)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Returns:
            Tuple: (state, info)
        """
        # 调用父类 reset
        state, info = super().reset(**kwargs)
        
        # 重置步数计数器
        self.step_counter = 0
        
        # 更新 TFT 预测
        self._update_tft_prediction()
        
        # 返回包含 TFT 预测的状态
        # 注意：这里需要返回与 observation_space 匹配的状态
        # 如果不匹配，需要调整 observation_space
        return state, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步环境
        
        Args:
            action: 动作
        
        Returns:
            Tuple: (next_state, reward, terminated, truncated, info)
        """
        # 调用父类 step
        next_state, reward, terminated, truncated, info = super().step(action)
        
        # 更新历史数据（如果有新的真实值）
        # 这里需要根据实际数据流来调整
        
        # 更新 TFT 预测缓存（如果需要）
        if self.step_counter % self.cache_update_freq == 0:
            self._update_tft_prediction()
        
        return next_state, reward, terminated, truncated, info
    
    def record_training_metrics(self, 
                               episode_reward: float, 
                               episode_length: int,
                               policy_loss: Optional[float] = None,
                               value_loss: Optional[float] = None,
                               entropy_loss: Optional[float] = None):
        """
        记录训练指标
        
        Args:
            episode_reward: Episode 奖励
            episode_length: Episode 长度
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy_loss: 熵损失
        """
        self.training_metrics['episode_rewards'].append(episode_reward)
        self.training_metrics['episode_lengths'].append(episode_length)
        
        if policy_loss is not None:
            self.training_metrics['policy_losses'].append(policy_loss)
        if value_loss is not None:
            self.training_metrics['value_losses'].append(value_loss)
        if entropy_loss is not None:
            self.training_metrics['entropy_losses'].append(entropy_loss)
    
    def get_training_metrics(self) -> Dict:
        """
        获取训练指标
        
        Returns:
            Dict: 训练指标字典
        """
        return self.training_metrics.copy()
    
    def save_training_metrics(self, path: str):
        """
        保存训练指标到文件
        
        Args:
            path: 保存路径
        """
        import json
        
        # 转换为可序列化的格式
        metrics_serializable = {}
        for key, value in self.training_metrics.items():
            metrics_serializable[key] = value
        
        with open(path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"✅ 训练指标已保存: {path}")


def create_training_env(num_skus: int = 5, 
                       history_df: Optional[pd.DataFrame] = None) -> SupplyChainEnvWithTFT:
    """
    创建训练环境
    
    Args:
        num_skus: SKU 数量
        history_df: 历史数据
    
    Returns:
        SupplyChainEnvWithTFT: 训练环境
    """
    env = SupplyChainEnvWithTFT(
        num_skus=num_skus,
        tft_checkpoint_path="records/models/tft_model_quantile.ckpt",
        history_df=history_df
    )
    
    return env


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建示例数据
    print("创建示例数据...")
    data = []
    for sku in range(5):
        for day in range(100):
            data.append({
                'time_idx': day,
                'shop_code': 'SHOP_001',
                'goods_code': f'SKU_{sku:03d}',
                'sale_qty': np.random.randint(10, 100),
            })
    df = pd.DataFrame(data)
    
    # 创建环境
    print("创建环境...")
    env = create_training_env(num_skus=5, history_df=df)
    
    # 重置环境
    print("重置环境...")
    state, info = env.reset()
    print(f"初始状态维度: {state.shape}")
    
    # 执行几步
    print("执行几步...")
    for step in range(10):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: reward={reward:.4f}, done={terminated or truncated}")
        
        if terminated or truncated:
            break
    
    print("✅ 测试完成！")
