"""
多目标优化器 - integration/multi_objective_optimizer.py
处理供应链优化中的多目标平衡
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class OptimizationMethod(Enum):
    """优化方法"""
    WEIGHTED_SUM = "weighted_sum"
    Pareto = "pareto"
    NSGA2 = "nsga2"
    DYNAMIC_WEIGHTING = "dynamic_weighting"


@dataclass
class Objective:
    """优化目标定义"""
    name: str
    weight: float
    direction: str = "minimize"  # minimize 或 maximize
    threshold: Optional[float] = None
    penalty_fn: Optional[Callable] = None


@dataclass
class OptimizationResult:
    """优化结果"""
    solution: np.ndarray
    objective_values: Dict[str, float]
    total_reward: float
    pareto_rank: Optional[int] = None


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self,
                 objectives: List[Objective],
                 method: OptimizationMethod = OptimizationMethod.WEIGHTED_SUM):
        """
        初始化多目标优化器
        
        Args:
            objectives: 目标列表
            method: 优化方法
        """
        self.objectives = objectives
        self.method = method
        
        # 动态权重状态
        self.current_weights = {obj.name: obj.weight for obj in objectives}
        self.weight_history = []
        
        # 历史记录
        self.solution_history = []
    
    def compute_weighted_reward(self,
                              objective_values: Dict[str, float],
                              weights: Optional[Dict[str, float]] = None) -> float:
        """
        计算加权奖励
        
        Args:
            objective_values: 各目标值
            weights: 权重（可选）
        
        Returns:
            float: 加权奖励
        """
        if weights is None:
            weights = self.current_weights
        
        total_reward = 0.0
        
        for obj in self.objectives:
            value = objective_values.get(obj.name, 0)
            weight = weights.get(obj.name, 1)
            
            # 标准化方向
            if obj.direction == "maximize":
                # 最大化目标，值越大越好
                normalized_value = value
            else:
                # 最小化目标
                normalized_value = -value
            
            total_reward += weight * normalized_value
        
        return total_reward
    
    def update_dynamic_weights(self,
                              recent_rewards: List[float],
                              window_size: int = 100):
        """
        动态更新权重（基于最近表现）
        
        Args:
            recent_rewards: 最近奖励历史
            window_size: 窗口大小
        """
        if len(recent_rewards) < window_size:
            return
        
        recent = recent_rewards[-window_size:]
        
        # 计算各目标的边际贡献
        for obj in self.objectives:
            # 简化：使用历史权重变化趋势
            if len(self.weight_history) > 1:
                prev_weights = self.weight_history[-1]
                if obj.name in prev_weights:
                    prev_weight = prev_weights[obj.name]
                    # 小幅调整
                    change = np.random.uniform(-0.05, 0.05)
                    new_weight = max(0.1, prev_weight + change)
                    self.current_weights[obj.name] = new_weight
        
        # 归一化权重
        total_weight = sum(self.current_weights.values())
        self.current_weights = {
            k: v / total_weight for k, v in self.current_weights.items()
        }
        
        self.weight_history.append(self.current_weights.copy())
    
    def apply_penalty(self,
                     objective_values: Dict[str, float],
                     state: np.ndarray) -> Tuple[Dict[str, float], float]:
        """
        应用惩罚项
        
        Args:
            objective_values: 目标值
            state: 当前状态
        
        Returns:
            Tuple: (调整后的目标值, 总惩罚)
        """
        adjusted_values = objective_values.copy()
        total_penalty = 0.0
        
        for obj in self.objectives:
            if obj.penalty_fn is not None:
                penalty = obj.penalty_fn(state)
                adjusted_values[obj.name] += penalty
                total_penalty += penalty
        
        return adjusted_values, total_penalty
    
    def check_constraints(self,
                          solution: np.ndarray,
                          constraint_fn: Callable[[np.ndarray], Tuple[bool, List[str]]]) -> Tuple[bool, List[str]]:
        """
        检查约束
        
        Args:
            solution: 解
            constraint_fn: 约束检查函数
        
        Returns:
            Tuple: (是否满足约束, 违反的约束列表)
        """
        return constraint_fn(solution)
    
    def pareto_dominance(self,
                        solution1: Dict[str, float],
                        solution2: Dict[str, float]) -> int:
        """
        判断Pareto支配关系
        
        Args:
            solution1: 解1的目标值
            solution2: 解2的目标值
        
        Returns:
            int: 1表示solution1支配solution2, -1表示被支配, 0表示互不支配
        """
        dominates_1 = False
        dominates_2 = False
        
        for obj in self.objectives:
            val1 = solution1.get(obj.name, 0)
            val2 = solution2.get(obj.name, 0)
            
            if obj.direction == "maximize":
                better_1 = val1 > val2
                better_2 = val2 > val1
            else:
                better_1 = val1 < val2
                better_2 = val2 < val1
            
            if better_1:
                dominates_1 = True
            if better_2:
                dominates_2 = True
        
        if dominates_1 and not dominates_2:
            return 1
        elif dominates_2 and not dominates_1:
            return -1
        else:
            return 0
    
    def find_pareto_front(self,
                         solutions: List[Dict[str, float]]) -> List[int]:
        """
        寻找Pareto前沿
        
        Args:
            solutions: 解列表
        
        Returns:
            List[int]: 非支配解的索引
        """
        pareto_indices = []
        
        for i, sol_i in enumerate(solutions):
            is_dominated = False
            
            for j, sol_j in enumerate(solutions):
                if i != j:
                    if self.pareto_dominance(sol_j, sol_i) == 1:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def optimize(self,
                 state: np.ndarray,
                 objective_fn: Callable[[np.ndarray, str], float],
                 constraint_fn: Optional[Callable] = None,
                 n_iterations: int = 100) -> OptimizationResult:
        """
        执行优化
        
        Args:
            state: 当前状态
            objective_fn: 目标函数 (state, objective_name) -> value
            constraint_fn: 约束函数
            n_iterations: 迭代次数
        
        Returns:
            OptimizationResult: 最优解
        """
        if self.method == OptimizationMethod.WEIGHTED_SUM:
            return self._weighted_sum_optimization(state, objective_fn, constraint_fn, n_iterations)
        elif self.method == OptimizationMethod.Pareto:
            return self._pareto_optimization(state, objective_fn, constraint_fn, n_iterations)
        elif self.method == OptimizationMethod.DYNAMIC_WEIGHTING:
            return self._dynamic_weighting_optimization(state, objective_fn, constraint_fn, n_iterations)
        else:
            return self._weighted_sum_optimization(state, objective_fn, constraint_fn, n_iterations)
    
    def _weighted_sum_optimization(self,
                                  state: np.ndarray,
                                  objective_fn: Callable,
                                  constraint_fn: Optional[Callable],
                                  n_iterations: int) -> OptimizationResult:
        """加权求和优化"""
        best_solution = None
        best_reward = float('-inf')
        best_objectives = None
        
        for _ in range(n_iterations):
            # 随机生成候选解
            solution = np.random.uniform(0, 100, size=state.shape[0])
            
            # 检查约束
            if constraint_fn:
                is_valid, _ = constraint_fn(solution)
                if not is_valid:
                    continue
            
            # 计算各目标值
            obj_values = {
                obj.name: objective_fn(state, obj.name)
                for obj in self.objectives
            }
            
            # 计算加权奖励
            reward = self.compute_weighted_reward(obj_values)
            
            if reward > best_reward:
                best_reward = reward
                best_solution = solution
                best_objectives = obj_values
        
        return OptimizationResult(
            solution=best_solution,
            objective_values=best_objectives,
            total_reward=best_reward
        )
    
    def _pareto_optimization(self,
                           state: np.ndarray,
                           objective_fn: Callable,
                           constraint_fn: Optional[Callable],
                           n_iterations: int) -> OptimizationResult:
        """Pareto优化"""
        solutions = []
        
        for _ in range(n_iterations):
            solution = np.random.uniform(0, 100, size=state.shape[0])
            
            if constraint_fn:
                is_valid, _ = constraint_fn(solution)
                if not is_valid:
                    continue
            
            obj_values = {
                obj.name: objective_fn(state, obj.name)
                for obj in self.objectives
            }
            
            solutions.append({
                'solution': solution,
                'objectives': obj_values
            })
        
        # 找到Pareto前沿
        obj_dicts = [s['objectives'] for s in solutions]
        pareto_indices = self.find_pareto_front(obj_dicts)
        
        # 选择第一个Pareto解（或可以用其他选择策略）
        if pareto_indices:
            best = solutions[pareto_indices[0]]
            return OptimizationResult(
                solution=best['solution'],
                objective_values=best['objectives'],
                total_reward=self.compute_weighted_reward(best['objectives']),
                pareto_rank=0
            )
        
        # 如果没有找到Pareto解，返回最好的加权解
        return self._weighted_sum_optimization(state, objective_fn, constraint_fn, n_iterations)
    
    def _dynamic_weighting_optimization(self,
                                       state: np.ndarray,
                                       objective_fn: Callable,
                                       constraint_fn: Optional[Callable],
                                       n_iterations: int) -> OptimizationResult:
        """动态权重优化"""
        # 首先更新权重
        if len(self.weight_history) > 0:
            recent_rewards = [r for r in self.weight_history if isinstance(r, float)]
            if len(recent_rewards) > 100:
                self.update_dynamic_weights(recent_rewards)
        
        # 使用更新后的权重进行优化
        return self._weighted_sum_optimization(state, objective_fn, constraint_fn, n_iterations)
    
    def get_weight_distribution(self) -> Dict[str, float]:
        """获取当前权重分布"""
        return self.current_weights.copy()
    
    def reset_weights(self):
        """重置权重为初始值"""
        self.current_weights = {obj.name: obj.weight for obj in self.objectives}
        self.weight_history = []


class RewardShaper:
    """奖励塑形器"""
    
    def __init__(self, base_reward_fn: Callable):
        """
        初始化奖励塑形器
        
        Args:
            base_reward_fn: 基础奖励函数
        """
        self.base_reward_fn = base_reward_fn
        self.potential_history = []
    
    def shape_reward(self,
                    state: np.ndarray,
                    action: np.ndarray,
                    next_state: np.ndarray,
                    base_reward: float,
                    gamma: float = 0.99) -> float:
        """
        塑形奖励
        
        Args:
            state: 当前状态
            action: 动作
            next_state: 下一状态
            base_reward: 基础奖励
            gamma: 折扣因子
        
        Returns:
            float: 塑形后的奖励
        """
        # 计算势能函数
        potential = self._potential(state)
        next_potential = self._potential(next_state)
        
        # 计算势能差
        potential_diff = gamma * next_potential - potential
        
        # 添加塑形奖励
        shaped_reward = base_reward + potential_diff
        
        self.potential_history.append(potential)
        
        return shaped_reward
    
    def _potential(self, state: np.ndarray) -> float:
        """
        势能函数
        
        Args:
            state: 状态
        
        Returns:
            float: 势能值
        """
        # 简化：使用库存充足度作为势能
        inventory = state[:len(state)//4]  # 假设库存在前1/4
        avg_fill_rate = np.mean(inventory / 100)  # 假设满容为100
        return avg_fill_rate
