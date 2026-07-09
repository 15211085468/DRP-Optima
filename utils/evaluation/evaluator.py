"""
模型评估器 - evaluation/evaluator.py
评估预测模型和RL策略的效果
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import MetricsConfig
from models.forecasting import TFTForecaster
from models.rl import PPOAgent
from models.optimization import SupplyChainEnv
from models.optimization.vec_supply_chain_env import VecSupplyChainEnv, make_vec_env
from utils.logger import get_logger
logger = get_logger(__name__)


@dataclass
class ForecastMetrics:
    """预测评估指标"""
    smape: float
    mae: float
    rmse: float
    mape: float
    bias: float


@dataclass
class RLMetrics:
    """强化学习评估指标"""
    cdr: float  # Cumulative Discounted Reward
    itr: float  # Inventory Turnover Ratio
    sor: float  # Stockout Rate
    hcr: float  # Holding Cost Ratio
    ofr: float  # Order Fill Rate


class Evaluator:
    """模型评估器"""
    
    def __init__(self, metrics_config: Optional[MetricsConfig] = None):
        """
        初始化评估器
        
        Args:
            metrics_config: 评估指标配置
        """
        self.config = metrics_config or MetricsConfig()
        self.results = {}
    
    def calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算SMAPE (Symmetric Mean Absolute Percentage Error)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            float: SMAPE值
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        denominator = np.where(denominator == 0, 1e-8, denominator)  # 避免除零
        return np.mean(np.abs(y_true - y_pred) / denominator)
    
    def calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算MAPE (Mean Absolute Percentage Error)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            float: MAPE值
        """
        denominator = np.where(y_true == 0, 1, y_true)  # 避免除零
        return np.mean(np.abs((y_true - y_pred) / denominator)) * 100
    
    def calculate_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算预测偏差
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            float: 偏差值
        """
        return np.mean(y_pred - y_true)
    
    def evaluate_forecast(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray) -> ForecastMetrics:
        """
        评估预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
        
        Returns:
            ForecastMetrics: 预测指标
        """
        # 确保是numpy数组
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        smape = self.calculate_smape(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = self.calculate_mape(y_true, y_pred)
        bias = self.calculate_bias(y_true, y_pred)
        
        return ForecastMetrics(
            smape=smape,
            mae=mae,
            rmse=rmse,
            mape=mape,
            bias=bias
        )
    
    def compare_models(self,
                      results: Dict[str, np.ndarray],
                      true_values: np.ndarray) -> pd.DataFrame:
        """
        比较多个模型的预测结果
        
        Args:
            results: 模型名 -> 预测值的字典
            true_values: 真实值
        
        Returns:
            pd.DataFrame: 比较结果表
        """
        comparison_data = []
        
        for model_name, predictions in results.items():
            metrics = self.evaluate_forecast(true_values, predictions)
            
            comparison_data.append({
                '模型': model_name,
                'SMAPE': metrics.smape,
                'MAE': metrics.mae,
                'RMSE': metrics.rmse,
                'MAPE(%)': metrics.mape,
                '偏差': metrics.bias
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('SMAPE')
        
        return df
    
    def evaluate_rl_policy(self,
                         env: SupplyChainEnv,
                         agent: PPOAgent,
                         n_episodes: int = 10) -> RLMetrics:
        """
        评估RL策略（修正指标计算）
        
        修正内容：
        - CDR: 使用 gamma=0.99 折扣累积
        - SOR: 简化为平均每步缺货率
        - OFR: 从环境 info 中获取实际订单跟踪
        
        Args:
            env: 供应链环境
            agent: PPO智能体
            n_episodes: 评估episode数
        
        Returns:
            RLMetrics: RL业务指标
        """
        gamma = 0.99  # 折扣因子
        episode_rewards = []
        episode_discounted_rewards = []
        episode_lengths = []
        total_stockouts = []
        total_inventory = []
        total_holding_cost = []
        total_sales = []  # 新增：收集销售额
        order_fills = []
        # 新增：逐步缺货率跟踪
        per_step_stockout_rates = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_discounted_reward = 0
            episode_length = 0
            episode_stockout = 0
            episode_inventory = []
            episode_holding = []
            episode_filled = 0
            episode_orders = 0
            episode_step_stockouts = []
            done = False
            
            while not done:
                # 使用确定性策略
                action, _ = agent.predict(state, deterministic=True)
                
                # 执行动作
                next_state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                # 折扣累积
                episode_discounted_reward += (gamma ** episode_length) * reward
                episode_length += 1
                
                # 收集指标
                current_stockout = info.get('stockout_rate', 0) * env.num_skus
                episode_stockout += current_stockout
                episode_step_stockouts.append(info.get('stockout_rate', 0))
                episode_inventory.append(info.get('total_inventory', 0))
                episode_holding.append(info.get('total_holding_cost', 0))
                
                # 计算订单满足率
                if 'orders_placed' in info:
                    episode_orders += info['orders_placed']
                if 'orders_filled' in info:
                    episode_filled += info['orders_filled']
                
                state = next_state
                done = terminated or truncated
            
            # 收集完成的episode数据
            episode_rewards.append(episode_reward)
            episode_discounted_rewards.append(episode_discounted_reward)
            episode_lengths.append(episode_length)
            total_stockouts.append(episode_stockout)
            total_inventory.append(np.mean(episode_inventory) if episode_inventory else 0)
            total_holding_cost.append(np.sum(episode_holding))
            per_step_stockout_rates.append(np.mean(episode_step_stockouts) if episode_step_stockouts else 0)
            
            # 新增：获取实际销售额
            if hasattr(env, 'total_sales_amount'):
                total_sales.append(env.total_sales_amount)
            else:
                total_sales.append(0)
            
            if episode_orders > 0:
                order_fills.append(episode_filled / episode_orders)
            else:
                order_fills.append(1.0)
        
        # 计算CDR (累计折扣奖励) — 修正：使用折扣后的值
        cdr = np.mean(episode_discounted_rewards)
        
        # 计算ITR (库存周转率) - 修正：使用实际销售额
        avg_inventory = np.mean(total_inventory) if total_inventory else 1
        total_sales_all = np.sum(total_sales)
        if avg_inventory > 0:
            itr = total_sales_all / (n_episodes * avg_inventory + 1e-6)
        else:
            itr = 0.0
        
        # 计算SOR (缺货率) - 修正：简化为平均每步缺货率
        sor = np.mean(per_step_stockout_rates) if per_step_stockout_rates else 0
        
        # 计算HCR (持有成本比率)
        hcr = np.mean(total_holding_cost) / (np.mean(episode_rewards) + 1e-6) if episode_rewards else 0
        
        # 计算OFR (订单满足率) — 从环境 info 获取真实值
        ofr = np.mean(order_fills) if order_fills else 0.0
        
        return RLMetrics(
            cdr=cdr,
            itr=itr,
            sor=sor,
            hcr=hcr,
            ofr=ofr
        )
    
    def evaluate_rl_policy_vec(self,
                             agent: PPOAgent,
                             num_envs: int = 4,
                             n_episodes: int = 10,
                             **env_kwargs) -> RLMetrics:
        """
        使用向量化环境评估RL策略（修正ITR计算：使用实际销售额）
        
        Args:
            agent: PPO智能体
            num_envs: 并行环境数量
            n_episodes: 评估episode数
            **env_kwargs: 传递给SupplyChainEnv的关键字参数
        
        Returns:
            RLMetrics: RL业务指标
        """
        # 创建向量化环境
        vec_env = make_vec_env(num_envs=num_envs, **env_kwargs)
        
        episode_rewards = []
        episode_discounted_rewards = []
        episode_lengths = []
        total_stockouts = []
        total_inventory = []
        total_holding_cost = []
        total_sales = []  # 新增：收集销售额
        order_fills = []
        per_step_stockout_rates = []
        
        # 批量重置
        states, infos = vec_env.reset()
        
        episode_rewards_curr = np.zeros(num_envs)
        episode_discounted_rewards_curr = np.zeros(num_envs)
        episode_lengths_curr = np.zeros(num_envs, dtype=int)
        episode_stockouts_curr = np.zeros(num_envs)
        episode_sales_curr = np.zeros(num_envs)  # 新增
        episode_orders_curr = np.zeros(num_envs)
        episode_filled_curr = np.zeros(num_envs)
        episode_step_stockouts_curr = [[] for _ in range(num_envs)]  # 新增
        
        episode_inventory_curr = [[] for _ in range(num_envs)]
        episode_holding_curr = [[] for _ in range(num_envs)]
        
        active_episodes = np.ones(num_envs, dtype=bool)
        completed_episodes = 0
        
        while completed_episodes < n_episodes:
            # 真正的批量预测：一次性预测所有活跃环境的动作
            batch_states = states[active_episodes]
            if len(batch_states) > 0:
                batch_actions, _ = agent.predict(batch_states, deterministic=True)
            else:
                break
            
            # 构造完整动作数组
            actions = np.zeros((num_envs, vec_env.single_action_space.shape[0]))
            actions[active_episodes] = batch_actions
            
            # 批量步进
            states, rewards, terminateds, truncateds, infos = vec_env.step(actions)
            
            # 更新活跃episode的统计数据
            for i in range(num_envs):
                if active_episodes[i]:
                    episode_rewards_curr[i] += rewards[i]
                    episode_discounted_rewards_curr[i] += (0.99 ** episode_lengths_curr[i]) * rewards[i]
                    episode_lengths_curr[i] += 1
                    
                    # 收集指标
                    current_stockout = infos[i].get('stockout_rate', 0)
                    episode_step_stockouts_curr[i].append(current_stockout)
                    
                    episode_inventory_curr[i].append(infos[i].get('total_inventory', 0))
                    episode_holding_curr[i].append(infos[i].get('total_holding_cost', 0))
                    
                    # 新增：收集实际销售额
                    if 'total_sales_amount' in infos[i]:
                        episode_sales_curr[i] = infos[i]['total_sales_amount']
                    
                    if 'orders_placed' in infos[i]:
                        episode_orders_curr[i] += infos[i]['orders_placed']
                    if 'orders_filled' in infos[i]:
                        episode_filled_curr[i] += infos[i]['orders_filled']
                    
                    # 检查episode是否结束
                    if terminateds[i] or truncateds[i]:
                        # 收集完成的episode数据
                        episode_rewards.append(episode_rewards_curr[i])
                        episode_discounted_rewards.append(episode_discounted_rewards_curr[i])
                        episode_lengths.append(episode_lengths_curr[i])
                        total_stockouts.append(episode_stockouts_curr[i])
                        total_inventory.append(np.mean(episode_inventory_curr[i]) if episode_inventory_curr[i] else 0)
                        total_holding_cost.append(np.sum(episode_holding_curr[i]))
                        total_sales.append(episode_sales_curr[i])
                        per_step_stockout_rates.append(
                            np.mean(episode_step_stockouts_curr[i]) if episode_step_stockouts_curr[i] else 0
                        )
                        
                        if episode_orders_curr[i] > 0:
                            order_fills.append(episode_filled_curr[i] / episode_orders_curr[i])
                        else:
                            order_fills.append(1.0)
                        
                        completed_episodes += 1
                        
                        # 重置该环境
                        vec_env.reset(options={'seed': np.random.randint(0, 100000)})
                        episode_rewards_curr[i] = 0
                        episode_discounted_rewards_curr[i] = 0
                        episode_lengths_curr[i] = 0
                        episode_stockouts_curr[i] = 0
                        episode_sales_curr[i] = 0
                        episode_orders_curr[i] = 0
                        episode_filled_curr[i] = 0
                        episode_step_stockouts_curr[i] = []
                        episode_inventory_curr[i] = []
                        episode_holding_curr[i] = []
        
        # 关闭向量化环境
        vec_env.close()
        
        # 计算指标
        # CDR: 使用折扣累积
        cdr = np.mean(episode_discounted_rewards)
        
        # 计算ITR (库存周转率) - 修正：使用实际销售额
        avg_inventory = np.mean(total_inventory) if total_inventory else 1
        total_sales_all = np.sum(total_sales)
        if avg_inventory > 0:
            itr = total_sales_all / (n_episodes * avg_inventory + 1e-6)
        else:
            itr = 0.0
        
        # SOR: 简化为平均每步缺货率
        sor = np.mean(per_step_stockout_rates) if per_step_stockout_rates else 0
        
        hcr = np.mean(total_holding_cost) / (np.mean(episode_rewards) + 1e-6) if episode_rewards else 0
        
        # 计算OFR (订单满足率) — 从环境 info 获取真实值
        ofr = np.mean(order_fills) if order_fills else 0.0
        
        return RLMetrics(
            cdr=cdr,
            itr=itr,
            sor=sor,
            hcr=hcr,
            ofr=ofr
        )
    
    def backtest(self,
                env: SupplyChainEnv,
                agent: PPOAgent,
                historical_data: pd.DataFrame,
                window_size: int = 30) -> pd.DataFrame:
        """
        回测RL策略
        
        Args:
            env: 供应链环境
            agent: PPO智能体
            historical_data: 历史数据
            window_size: 滑动窗口大小
        
        Returns:
            pd.DataFrame: 回测结果
        """
        results = []
        state, _ = env.reset()
        
        for step in range(len(historical_data)):
            # 选择动作
            action, _ = agent.predict(state, deterministic=True)
            
            # 执行
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 记录
            results.append({
                'step': step,
                'date': historical_data.iloc[step]['date'] if 'date' in historical_data.columns else step,
                'reward': reward,
                'total_reward': info.get('episode_reward', 0),
                'stockout': info.get('total_stockout', 0),
                'holding_cost': info.get('total_holding_cost', 0),
                'inventory': info.get('total_inventory', 0)
            })
            
            state = next_state
            
            if terminated or truncated:
                state, _ = env.reset()
        
        df = pd.DataFrame(results)
        
        # 计算滚动指标
        if len(df) >= window_size:
            df['rolling_reward'] = df['reward'].rolling(window=window_size).mean()
            df['rolling_stockout'] = df['stockout'].rolling(window=window_size).mean()
            df['rolling_holding'] = df['holding_cost'].rolling(window=window_size).mean()
        
        return df
    
    def generate_report(self,
                      forecast_metrics: Optional[ForecastMetrics] = None,
                      rl_metrics: Optional[RLMetrics] = None,
                      comparison_df: Optional[pd.DataFrame] = None) -> str:
        """
        生成评估报告
        
        Args:
            forecast_metrics: 预测指标
            rl_metrics: RL指标
            comparison_df: 模型比较表
        
        Returns:
            str: 报告文本
        """
        report = []
        report.append("=" * 60)
        report.append("模型评估报告")
        report.append("=" * 60)
        
        if forecast_metrics:
            report.append("\n【预测模型评估】")
            report.append(f"  SMAPE: {forecast_metrics.smape:.4f}")
            report.append(f"  MAE:   {forecast_metrics.mae:.4f}")
            report.append(f"  RMSE:  {forecast_metrics.rmse:.4f}")
            report.append(f"  MAPE:  {forecast_metrics.mape:.2f}%")
            report.append(f"  偏差:  {forecast_metrics.bias:.4f}")
        
        if rl_metrics:
            report.append("\n【RL策略评估】")
            report.append(f"  CDR (累计折扣奖励): {rl_metrics.cdr:.4f}")
            report.append(f"  ITR (库存周转率):  {rl_metrics.itr:.4f}")
            report.append(f"  SOR (缺货率):      {rl_metrics.sor:.4f}")
            report.append(f"  HCR (持有成本率):  {rl_metrics.hcr:.4f}")
            # OFR指标：若未实现则标注
            if rl_metrics.ofr == 0.0:
                ofr_str = "未实现"
            else:
                ofr_str = f"{rl_metrics.ofr:.4f}"
            report.append(f"  OFR (订单满足率):  {ofr_str}")
        
        if comparison_df is not None:
            report.append("\n【模型对比】")
            report.append(comparison_df.to_string(index=False))
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_forecast_comparison(self,
                               y_true: np.ndarray,
                               predictions: Dict[str, np.ndarray],
                               save_path: Optional[str] = None):
        """
        绘制预测对比图
        
        Args:
            y_true: 真实值
            predictions: 模型名 -> 预测值
            save_path: 保存路径
        """
        plt.figure(figsize=(14, 6))
        
        # 真实值
        plt.plot(y_true, label='真实值', linewidth=2, color='black')
        
        # 各模型预测
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            plt.plot(y_pred, label=model_name, 
                    linewidth=1.5, color=colors[i % len(colors)], alpha=0.7)
        
        plt.xlabel('时间步')
        plt.ylabel('销售量')
        plt.title('需求预测对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_rl_metrics(self,
                       metrics_history: List[RLMetrics],
                       save_path: Optional[str] = None):
        """
        绘制RL指标变化
        
        Args:
            metrics_history: RL指标历史
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs = range(len(metrics_history))
        
        metrics_names = ['cdr', 'itr', 'sor', 'hcr', 'ofr']
        titles = ['CDR (累计折扣奖励)', 'ITR (库存周转率)', 
                 'SOR (缺货率)', 'HCR (持有成本率)', 'OFR (订单满足率)']
        
        for i, (metric, title) in enumerate(zip(metrics_names, titles)):
            row, col = i // 3, i % 3
            values = [getattr(m, metric) for m in metrics_history]
            axes[row, col].plot(epochs, values)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Episode')
            axes[row, col].grid(True, alpha=0.3)
        
        # 最后一个子图显示奖励
        rewards = [m.cdr for m in metrics_history]
        axes[1, 2].plot(epochs, rewards, color='green')
        axes[1, 2].set_title('CDR 奖励曲线')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def save_results(self, path: str):
        """保存评估结果"""
        pd.DataFrame(self.results).to_csv(path, index=False)
        logger.info(f"结果已保存到: {path}")
    
    def load_results(self, path: str):
        """加载评估结果"""
        self.results = pd.read_csv(path).to_dict('list')
        logger.info(f"结果已从: {path} 加载")
