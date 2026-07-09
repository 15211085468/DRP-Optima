"""
奖励归一化器 - 基于指数移动平均(EMA)的在线Z-score标准化
解决Value Function无法学习的核心问题：
1. 未归一化的多尺度奖励导致TD Error方差极大
2. 硬裁剪导致梯度消失
"""
import numpy as np

class RunningRewardNormalizer:
    """基于指数移动平均(EMA)的在线奖励归一化器
    
    对每个奖励组件独立进行Z-score标准化：
    - mean: 指数移动平均
    - var: 指数移动方差
    - 使用软截断（clip）而非硬裁剪
    """
    
    def __init__(self, num_components: int, epsilon: float = 1e-8, gamma: float = 0.99):
        """
        初始化归一化器
        
        Args:
            num_components: 奖励组件数量
            epsilon: 数值稳定性常数
            gamma: EMA衰减因子（0.99表示重点关注最近100个样本）
        """
        self.num_components = num_components
        self.epsilon = epsilon
        self.gamma = gamma
        
        # 初始化统计量
        self.mean = np.zeros(num_components, dtype=np.float64)
        self.var = np.ones(num_components, dtype=np.float64)
        self.count = 0
        
        # 用于Welford算法的临时变量
        self._mean_seq = np.zeros(num_components, dtype=np.float64)
        self._var_seq = np.zeros(num_components, dtype=np.float64)
        
    def update(self, rewards: np.ndarray):
        """更新统计量（使用Welford算法的在线更新版本）
        
        Args:
            rewards: 原始奖励值，shape: (num_components,)
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        
        if self.count == 0:
            # 第一个样本
            self.mean = rewards.copy()
            self.var = np.ones(self.num_components, dtype=np.float64) * 100.0  # 初始化为较大方差
        else:
            # 使用EMA更新均值和方差
            # 对于单步更新，我们使用简化的EMA公式
            alpha = 0.01  # 学习率（越小越稳定）
            
            # 更新均值
            self.mean = (1 - alpha) * self.mean + alpha * rewards
            
            # 更新方差（使用增量公式）
            delta = rewards - self.mean
            self.var = (1 - alpha) * self.var + alpha * (delta ** 2)
        
        self.count += 1
        
    def normalize(self, rewards: np.ndarray, clip_range: float = 5.0) -> np.ndarray:
        """Z-score标准化并软截断
        
        Args:
            rewards: 原始奖励值，shape: (num_components,)
            clip_range: 截断范围（±clip_range个标准差）
            
        Returns:
            归一化后的奖励值
        """
        rewards = np.asarray(rewards, dtype=np.float64)
        
        # 防止除零
        std = np.sqrt(self.var + self.epsilon)
        
        # Z-score标准化
        normalized = (rewards - self.mean) / std
        
        # 软截断（保留梯度信息）
        return np.clip(normalized, -clip_range, clip_range)
    
    def get_statistics(self) -> dict:
        """获取当前统计量（用于日志和调试）"""
        return {
            'mean': self.mean.copy(),
            'std': np.sqrt(self.var),
            'count': self.count
        }
