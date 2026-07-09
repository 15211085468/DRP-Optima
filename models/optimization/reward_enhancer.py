"""
满足率增强器 - 专门优化订单满足率至95%+

核心策略：
1. 非线性满足率奖励（越过95%阈值才有大奖）
2. 累积满足率奖励（鼓励长期高满足率）
3. ABC差异化满足率目标（A类98%，B类95%，C类90%）
4. 缺货预警惩罚（提前惩罚可能导致缺货的行为）
"""

import numpy as np
from typing import Dict, Any

class ServiceLevelEnhancer:
    """满足率增强器"""
    
    def __init__(self, config):
        self.config = config
        
        # 满足率目标（可配置）
        self.target_service_level = getattr(config, 'min_service_level', 0.95)
        
        # ABC差异化目标
        self.abc_targets = {
            'A': 0.98,  # A类：98%满足率
            'B': 0.95,  # B类：95%满足率
            'C': 0.90,  # C类：90%满足率
        }
        
        # 累积满足率追踪
        self.episode_fill_rates = []
        self.window_size = 10  # 滑动窗口
        
    def calculate_enhanced_service_bonus(
        self, 
        current_fill_rate: float,
        abc_categories: np.ndarray,
        unmet_demand: np.ndarray,
        current_inventory: np.ndarray,
        demand: np.ndarray
    ) -> float:
        """
        计算增强版满足率奖励
        
        策略：
        1. 非线性奖励：满足率越高，奖励增长越快
        2. ABC差异化：A类缺货惩罚更重
        3. 预测性惩罚：库存不足时提前惩罚
        """
        
        # ========== 1. 基础满足率奖励（非线性） ==========
        if current_fill_rate >= self.target_service_level:
            # 超越目标：指数奖励
            base_bonus = 10.0 * (current_fill_rate - self.target_service_level + 1) ** 2
        elif current_fill_rate >= 0.90:
            # 接近目标：线性奖励
            base_bonus = 3.0 * (current_fill_rate - 0.90)
        elif current_fill_rate >= 0.80:
            # 低于目标：小幅惩罚
            base_bonus = -1.0 * (0.90 - current_fill_rate)
        else:
            # 远低于目标：重度惩罚
            base_bonus = -5.0 * (0.80 - current_fill_rate) ** 2
        
        # ========== 2. ABC差异化奖励 ==========
        abc_bonus = 0.0
        for i, category in enumerate(abc_categories):
            target = self.abc_targets.get(category, 0.95)
            
            # 计算该SKU的满足率
            if demand[i] > 1e-5:
                sku_fill_rate = min(1.0, current_inventory[i] / demand[i])
            else:
                sku_fill_rate = 1.0
            
            # ABC差异化惩罚/奖励
            if category == 'A':
                if sku_fill_rate < target:
                    abc_bonus -= 5.0 * (target - sku_fill_rate)  # A类缺货重罚
            elif category == 'C':
                if sku_fill_rate > 0.95:
                    abc_bonus += 2.0  # C类过度库存奖励（鼓励释放库存）
        
        # ========== 3. 累积满足率奖励 ==========
        self.episode_fill_rates.append(current_fill_rate)
        if len(self.episode_fill_rates) > self.window_size:
            self.episode_fill_rates.pop(0)
        
        avg_fill_rate = np.mean(self.episode_fill_rates)
        
        if len(self.episode_fill_rates) >= self.window_size:
            if avg_fill_rate >= self.target_service_level:
                cumulative_bonus = 5.0  # 持续高满足率奖励
            else:
                cumulative_bonus = -3.0 * (self.target_service_level - avg_fill_rate)
        else:
            cumulative_bonus = 0.0
        
        # ========== 4. 预测性缺货惩罚 ==========
        predictive_penalty = 0.0
        for i in range(len(demand)):
            # 如果当前库存 < 未来2周需求，提前惩罚
            if current_inventory[i] < demand[i] * 2:
                predictive_penalty -= 0.5 * (demand[i] * 2 - current_inventory[i]) / (demand[i] + 1e-5)
        
        # ========== 总奖励 ==========
        total_bonus = base_bonus + abc_bonus + cumulative_bonus + predictive_penalty
        
        return float(total_bonus)
    
    def reset(self):
        """重置追踪器"""
        self.episode_fill_rates = []


def integrate_service_level_enhancer(env_instance):
    """
    将满足率增强器集成到环境中
    
    使用方法：
    ```python
    from models.optimization.reward_enhancer import integrate_service_level_enhancer
    
    env = SupplyChainEnv(config=env_config)
    integrate_service_level_enhancer(env)
    ```
    """
    enhancer = ServiceLevelEnhancer(env_instance.config)
    env_instance.service_level_enhancer = enhancer
    
    # 修改 _calculate_reward 方法（通过猴子补丁）
    original_calculate_reward = env_instance._calculate_reward
    
    def enhanced_calculate_reward(info=None):
        # 调用原始奖励计算
        reward = original_calculate_reward(info)
        
        # 添加增强版满足率奖励
        if hasattr(env_instance, '_last_step_actual_sales') and hasattr(env_instance, 'current_demand'):
            current_fill_rate = np.sum(env_instance._last_step_actual_sales) / (np.sum(env_instance.current_demand) + 1e-5)
            
            # 获取ABC分类
            abc_categories = getattr(env_instance, 'abc_categories', ['B'] * env_instance.num_skus)
            
            # 计算增强奖励
            enhanced_bonus = enhancer.calculate_enhanced_service_bonus(
                current_fill_rate=current_fill_rate,
                abc_categories=abc_categories,
                unmet_demand=np.maximum(env_instance.current_demand - env_instance._last_step_actual_sales, 0),
                current_inventory=env_instance.store_inventory,
                demand=env_instance.current_demand
            )
            
            # 加入总奖励
            reward = reward + enhanced_bonus
            
            # 记录
            if info is not None:
                info['enhanced_service_bonus'] = float(enhanced_bonus)
        
        return reward
    
    env_instance._calculate_reward = enhanced_calculate_reward
    
    return env_instance
