"""
TFT结构化重组Wrapper（LT=1终结版）

[终结版] 周决策 TFT 适配器 (Weekly TFT-RL Adapter)

业务设定对齐：Lead Time = 1 (本周下单，下周到货)
策略：采用方案B，完美保留环境 16*N+3 维度，无信息损耗。

特征物理意义 (LT=1 专属)：
- in_transit (在途库存): 核心防呆特征！防止 Agent 忽略上周订单导致重复订货(牛鞭效应)。
- forecast (未来7周): 覆盖 LT(1) + Review(1) + 安全缓冲，用于跨期规划。
- sales/stockouts (过去3周): 捕捉短期需求动量与缺货惯性，修正 TFT 预测偏差。
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TFTStructureWrapper(gym.Wrapper):
    """
    [终结版] 周决策 TFT 适配器 (Weekly TFT-RL Adapter)
    
    业务设定对齐：Lead Time = 1 (本周下单，下周到货)
    策略：采用方案B，完美保留环境 16*N+3 维度，无信息损耗。
    
    特征物理意义 (LT=1 专属)：
    in_transit (在途库存): 核心防呆特征！防止 Agent 忽略上周订单导致重复订货(牛鞭效应)。
    forecast (未来7周): 覆盖 LT(1) + Review(1) + 安全缓冲，用于跨期规划。
    sales/stockouts (过去3周): 捕捉短期需求动量与缺货惯性，修正 TFT 预测偏差。
    """
    
    def __init__(self, env, num_skus=10, forecast_horizon=7, lookback_horizon=3):
        """
        初始化Wrapper
        
        Args:
            env: 基础环境（返回扁平数组）
            num_skus: SKU数量
            forecast_horizon: 预测视界（未来几周）
            lookback_horizon: 回溯视界（过去几周）
        """
        super().__init__(env)
        self.num_skus = num_skus
        self.fh = forecast_horizon
        self.lb = lookback_horizon
        
        # --- 维度定义 (严格对齐环境 _get_state 拼接顺序) ---
        # Local 分支：当前库存(1) + 在途库存(1) + 距下单周数(1) + 全局时间(3) = 6
        self.local_dim = 3 + 3  
        # TFT 分支：未来预测(7) + 历史销量(3) + 历史缺货(3) = 13
        self.tft_dim = self.fh + (self.lb * 2) 

        # 重新定义 observation_space 为 Dict 空间
        self.observation_space = spaces.Dict({
            'local_features': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1, num_skus, self.local_dim), 
                dtype=np.float32
            ),
            'tft_predictions': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(1, num_skus, self.tft_dim), 
                dtype=np.float32
            )
        })
    
    def _restructure_weekly(self, flat_obs):
        """
        按 LT=1 供应链时序逻辑进行精准切片
        
        Args:
            flat_obs: 扁平数组，形状为(state_dim,)
        
        Returns:
            dict: {
                'local_features': (1, num_skus, local_dim),
                'tft_predictions': (1, num_skus, tft_dim)
            }
        """
        N = self.num_skus
        
        # 1. 动态切片 (严格 16*N+3 顺序)
        idx = 0
        # [未来视界] 未来 fh 周需求预测 (TFT 核心输出，用于跨期规划)
        forecast = flat_obs[idx:idx + self.fh * N]; idx += self.fh * N 
        # [当前状态] 周末盘点可用库存
        inventory = flat_obs[idx:idx + N]; idx += N          
        # [当前状态] 在途库存 (LT=1 核心特征：上周下单、本周即将入库的量，防牛鞭效应)
        in_transit = flat_obs[idx:idx + N]; idx += N         
        # [当前状态] 距上次下单周数 (反映 Review Period 节奏)
        weeks_since_order = flat_obs[idx:idx + N]; idx += N  
        # [历史视界] 过去 lb 周实际销量 (捕捉短期动量，修正预测偏差)
        sales = flat_obs[idx:idx + self.lb * N]; idx += self.lb * N 
        # [历史视界] 过去 lb 周缺货量 (捕捉缺货惯性，触发补偿性订货)
        stockouts = flat_obs[idx:idx + self.lb * N]; idx += self.lb * N 
        # [全局上下文] 时间特征 (Month, Week_of_Year 等)
        global_time = flat_obs[idx:idx + 3]                    

        # 2. 矩阵重塑 (Reshape to SKU-level)
        forecast = forecast.reshape(N, self.fh)
        inventory = inventory.reshape(N, 1)
        in_transit = in_transit.reshape(N, 1)
        weeks_since_order = weeks_since_order.reshape(N, 1)
        sales = sales.reshape(N, self.lb)
        stockouts = stockouts.reshape(N, self.lb)
        
        # 3. 组装 Local 分支 (当前状态 + 广播全局时间)
        local_base = np.concatenate([inventory, in_transit, weeks_since_order], axis=1)
        time_broadcast = np.broadcast_to(global_time, (N, 3)) 
        local_final = np.concatenate([local_base, time_broadcast], axis=1)

        # 4. 组装 TFT 分支 (未来预测 + 历史回溯)
        tft_final = np.concatenate([forecast, sales, stockouts], axis=1)

        # 5. 增加 Batch 维度 (1, N, dim) 并返回
        return {
            'local_features': np.expand_dims(local_final, axis=0).astype(np.float32),
            'tft_predictions': np.expand_dims(tft_final, axis=0).astype(np.float32)
        }
    
    def reset(self, **kwargs):
        """
        重置环境并返回重组后的观测
        
        Returns:
            obs_dict: 重组后的观测字典
            info: 额外信息
        """
        obs, info = self.env.reset(**kwargs)
        return self._restructure_weekly(obs), info
    
    def step(self, action):
        """
        执行动作并返回重组后的观测
        
        Args:
            action: 动作
        
        Returns:
            obs_dict: 重组后的观测字典
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._restructure_weekly(obs), reward, terminated, truncated, info
