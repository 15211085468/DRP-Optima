"""
约束处理模块 - models/optimization/constraint_handler.py
处理供应链中的各种业务约束
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ConstraintResult:
    """约束检查结果"""
    is_valid: bool
    violations: List[str]
    penalty: float
    adjusted_action: Optional[np.ndarray] = None


class ConstraintHandler:
    """约束处理器"""

    def __init__(self, store_capacity: float = 500.0):
        """
        初始化约束处理器

        Args:
            store_capacity: 门店容量上限（从EnvConfig.store_capacity传入）
        """
        self.store_capacity = store_capacity
        self.violation_history = []
    
    def check_and_adjust(self, 
                        action: np.ndarray,
                        inventory: np.ndarray,
                        warehouse_inventory: np.ndarray,
                        min_batch_size: float = 10.0,
                        max_order: float = 1000.0) -> ConstraintResult:
        """
        检查并调整动作
        
        Args:
            action: 原始动作
            inventory: 当前库存
            warehouse_inventory: 仓库库存
            min_batch_size: 最小补货批量
            max_order: 最大订单量
        
        Returns:
            ConstraintResult: 约束检查结果
        """
        violations = []
        penalty = 0.0
        adjusted_action = action.copy()
        
        # 1. 仓库可用性约束
        warehouse_constraint = self._check_warehouse_availability(
            adjusted_action, warehouse_inventory
        )
        violations.extend(warehouse_constraint['violations'])
        penalty += warehouse_constraint['penalty']
        adjusted_action = warehouse_constraint['adjusted']
        
        # 2. 最小批量约束
        batch_constraint = self._check_minimum_batch(
            adjusted_action, min_batch_size
        )
        violations.extend(batch_constraint['violations'])
        penalty += batch_constraint['penalty']
        adjusted_action = batch_constraint['adjusted']
        
        # 3. 最大订单约束
        max_constraint = self._check_maximum_order(
            adjusted_action, max_order
        )
        violations.extend(max_constraint['violations'])
        penalty += max_constraint['penalty']
        adjusted_action = max_constraint['adjusted']
        
        # 4. 容量约束
        capacity_constraint = self._check_capacity(
            adjusted_action, inventory
        )
        violations.extend(capacity_constraint['violations'])
        penalty += capacity_constraint['penalty']
        adjusted_action = capacity_constraint['adjusted']
        
        is_valid = len(violations) == 0
        
        return ConstraintResult(
            is_valid=is_valid,
            violations=violations,
            penalty=penalty,
            adjusted_action=adjusted_action
        )
    
    def _check_warehouse_availability(self, 
                                     action: np.ndarray,
                                     warehouse_inventory: np.ndarray) -> Dict:
        """检查仓库可用性"""
        violations = []
        penalty = 0.0
        adjusted = action.copy()
        
        # 不能超过仓库可用库存
        exceed_mask = adjusted > warehouse_inventory
        if np.any(exceed_mask):
            violations.append("部分SKU超过仓库可用库存")
            penalty += np.sum((adjusted[exceed_mask] - warehouse_inventory[exceed_mask])) * 0.1
            
            # 调整为仓库可用量
            adjusted = np.minimum(adjusted, warehouse_inventory)
        
        return {'violations': violations, 'penalty': penalty, 'adjusted': adjusted}
    
    def _check_minimum_batch(self,
                            action: np.ndarray,
                            min_batch_size: float) -> Dict:
        """检查最小批量约束"""
        violations = []
        penalty = 0.0
        adjusted = action.copy()
        
        # 非零动作必须大于最小批量
        nonzero_mask = (adjusted > 0) & (adjusted < min_batch_size)
        if np.any(nonzero_mask):
            violations.append(f"存在小于最小批量({min_batch_size})的补货")
            penalty += np.sum(min_batch_size - adjusted[nonzero_mask]) * 0.5
            
            # 设为0或最小批量
            adjusted[nonzero_mask] = 0
        
        return {'violations': violations, 'penalty': penalty, 'adjusted': adjusted}
    
    def _check_maximum_order(self,
                            action: np.ndarray,
                            max_order: float) -> Dict:
        """检查最大订单约束"""
        violations = []
        penalty = 0.0
        adjusted = action.copy()
        
        exceed_mask = adjusted > max_order
        if np.any(exceed_mask):
            violations.append(f"存在超过最大订单量({max_order})的补货")
            penalty += np.sum(adjusted[exceed_mask] - max_order) * 0.2
            
            adjusted = np.clip(adjusted, 0, max_order)
        
        return {'violations': violations, 'penalty': penalty, 'adjusted': adjusted}
    
    def _check_capacity(self,
                       action: np.ndarray,
                       current_inventory: np.ndarray) -> Dict:
        """检查门店容量约束"""
        violations = []
        penalty = 0.0
        adjusted = action.copy()
        
        store_capacity = self.store_capacity  # 从构造函数传入，而非硬编码
        current_fill_ratio = current_inventory / store_capacity
        
        # 当库存已经很高时，限制补货
        high_inventory_mask = current_fill_ratio > 0.8
        if np.any(high_inventory_mask):
            violations.append("部分门店库存已接近容量上限")
            penalty += np.sum(current_fill_ratio[high_inventory_mask]) * 2
            
            # 减少补货量
            reduction = np.where(
                high_inventory_mask,
                adjusted * (1 - (current_fill_ratio - 0.8) * 2),
                adjusted
            )
            adjusted = np.maximum(reduction, 0)
        
        return {'violations': violations, 'penalty': penalty, 'adjusted': adjusted}
    
    def apply_action_mask(self,
                         action: np.ndarray,
                         min_threshold: float = 10.0) -> np.ndarray:
        """
        应用动作掩码
        
        Args:
            action: 原始动作
            min_threshold: 最小阈值
        
        Returns:
            np.ndarray: 掩码后的动作
        """
        masked = action.copy()
        
        # 强制执行：小于阈值的动作为0
        masked[masked < min_threshold] = 0
        
        return masked
    
    def add_custom_constraint(self,
                              name: str,
                              check_fn: Callable[[np.ndarray], Tuple[bool, float]],
                              priority: int = 0):
        """
        添加自定义约束
        
        Args:
            name: 约束名称
            check_fn: 检查函数，返回(is_valid, penalty)
            priority: 优先级
        """
        if not hasattr(self, 'custom_constraints'):
            self.custom_constraints = []
        
        self.custom_constraints.append({
            'name': name,
            'check_fn': check_fn,
            'priority': priority
        })
    
    def check_custom_constraints(self, action: np.ndarray) -> Tuple[List[str], float]:
        """检查自定义约束"""
        if not hasattr(self, 'custom_constraints'):
            return [], 0.0
        
        violations = []
        total_penalty = 0.0
        
        for constraint in sorted(self.custom_constraints, key=lambda x: x['priority']):
            is_valid, penalty = constraint['check_fn'](action)
            if not is_valid:
                violations.append(constraint['name'])
                total_penalty += penalty
        
        return violations, total_penalty
    
    def get_violation_summary(self) -> Dict[str, int]:
        """获取违规历史汇总"""
        if not self.violation_history:
            return {}
        
        summary = {}
        for record in self.violation_history:
            for violation in record['violations']:
                summary[violation] = summary.get(violation, 0) + 1
        
        return summary
    
    def record_violation(self, violations: List[str], penalty: float):
        """记录违规"""
        self.violation_history.append({
            'violations': violations,
            'penalty': penalty
        })


class ExpiryConstraintHandler(ConstraintHandler):
    """效期约束处理器"""
    
    def __init__(self, expiry_threshold_days: int = 90):
        super().__init__()
        self.expiry_threshold_days = expiry_threshold_days
    
    def check_expiry_constraint(self,
                               action: np.ndarray,
                               expiry_dates: np.ndarray) -> ConstraintResult:
        """
        检查效期约束
        
        Args:
            action: 补货动作
            expiry_dates: 效期日期（天数）
        
        Returns:
            ConstraintResult: 约束结果
        """
        violations = []
        penalty = 0.0
        adjusted = action.copy()
        
        current_day = 0  # 假设从第0天开始
        
        for i in range(len(action)):
            days_to_expiry = expiry_dates[i] - current_day
            
            if days_to_expiry < self.expiry_threshold_days:
                # 接近效期的商品限制补货
                reduction_ratio = days_to_expiry / self.expiry_threshold_days
                adjusted[i] *= reduction_ratio
                
                violations.append(f"SKU {i}: 效期不足{self.expiry_threshold_days}天")
                penalty += (1 - reduction_ratio) * action[i] * 0.5
        
        return ConstraintResult(
            is_valid=len(violations) == 0,
            violations=violations,
            penalty=penalty,
            adjusted_action=adjusted
        )


class SafetyStockConstraintHandler(ConstraintHandler):
    """安全库存约束处理器"""
    
    def __init__(self, safety_stock_levels: np.ndarray):
        super().__init__()
        self.safety_stock_levels = safety_stock_levels
    
    def check_safety_stock(self,
                          action: np.ndarray,
                          current_inventory: np.ndarray) -> ConstraintResult:
        """
        检查安全库存约束
        
        Args:
            action: 补货动作
            current_inventory: 当前库存
        
        Returns:
            ConstraintResult: 约束结果
        """
        violations = []
        penalty = 0.0
        adjusted = action.copy()
        
        # 检查是否低于安全库存
        below_safety = current_inventory < self.safety_stock_levels
        
        if np.any(below_safety):
            deficit = self.safety_stock_levels[below_safety] - current_inventory[below_safety]
            
            # 必须补货到安全库存以上
            adjusted[below_safety] = np.maximum(adjusted[below_safety], deficit)
            
            violations.append(f"{np.sum(below_safety)}个SKU低于安全库存")
        
        # 计算惩罚
        for i in range(len(action)):
            if current_inventory[i] < self.safety_stock_levels[i]:
                deficit = self.safety_stock_levels[i] - current_inventory[i]
                penalty += deficit * 0.2
        
        return ConstraintResult(
            is_valid=len(violations) == 0,
            violations=violations,
            penalty=penalty,
            adjusted_action=adjusted
        )
