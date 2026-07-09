"""
可微决策成本评估器 (Differentiable Decision Cost Evaluator)
基于报童模型(Newsvendor)的连续松弛，计算预测值带来的预期库存成本。
用于 Decision-Aware Joint Training (决策感知联合训练)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionCostEvaluator(nn.Module):
    """
    可微决策成本评估器
    
    基于报童模型(Newsvendor)的连续松弛，计算预测值带来的预期库存成本。
    使用 Softplus 近似 ReLU，保证在 0 处严格可导，允许梯度从决策成本
    反向传播回 TFT 预测值，从而优化对未来补货决策最有利的预测。
    
    核心公式：
        Cost = Ch * max(0, y_hat - y) + Cp * max(0, y - y_hat)
    其中：
        - Ch: 单位库存持有成本 (holding cost)
        - Cp: 单位缺货惩罚成本 (shortage penalty)
        - y_hat: TFT 预测值
        - y: 真实需求量
    """
    
    def __init__(self, 
                 holding_cost: float = 1.0, 
                 shortage_penalty: float = 5.0, 
                 smooth: bool = True,
                 temperature: float = 1.0):
        """
        初始化决策成本评估器
        
        Args:
            holding_cost: 单位库存持有成本 (Ch)，默认 1.0
            shortage_penalty: 单位缺货惩罚成本 (Cp)，默认 5.0
            smooth: 是否使用 Softplus 平滑 max(0, x)，保证严格可导，默认 True
            temperature: Softplus 的温度参数，越小越接近 ReLU，默认 1.0
        """
        super().__init__()
        self.holding_cost = holding_cost
        self.shortage_penalty = shortage_penalty
        self.smooth = smooth
        self.temperature = temperature
        
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算决策成本（可微）
        
        Args:
            predictions: TFT 的预测值，形状 (batch_size, seq_len) 或 (batch_size,)
            targets: 真实需求量，形状与 predictions 相同
            
        Returns:
            标量张量，表示平均决策成本
            
        Note:
            - 输入应该是**反归一化后的实际物理销量**，否则成本数值会失真
            - 如果使用归一化后的数据（如 0~1 之间），则 holding_cost 
              和 shortage_penalty 也需要做相应的等比例缩放
        """
        # 确保 predictions 和 targets 在同一设备上
        if predictions.device != targets.device:
            targets = targets.to(predictions.device)
        
        # 计算过剩库存和缺货量
        # over_inventory: 预测过多导致的库存积压
        # under_inventory: 预测过少导致的缺货
        over_inventory = predictions - targets
        under_inventory = targets - predictions
        
        if self.smooth:
            # 使用 Softplus 近似 ReLU，保证在 0 处严格可导
            # softplus(x) = 1/beta * log(1 + exp(beta * x))
            # 当 temperature -> 0 时，softplus(x) -> ReLU(x)
            over_inventory = F.softplus(over_inventory, beta=1.0/self.temperature)
            under_inventory = F.softplus(under_inventory, beta=1.0/self.temperature)
        else:
            # 使用标准 ReLU（在 0 处不可导，但 PyTorch 会自动处理为 0 梯度）
            over_inventory = torch.relu(over_inventory)
            under_inventory = torch.relu(under_inventory)
            
        # 计算总成本
        # 持有成本：积压的库存导致的成本
        holding_cost = self.holding_cost * over_inventory
        
        # 缺货惩罚：未能满足需求导致的成本
        shortage_cost = self.shortage_penalty * under_inventory
        
        # 总决策成本
        total_cost = holding_cost + shortage_cost
        
        # 返回平均成本（标量）
        return total_cost.mean()
    
    def compute_cost_breakdown(self, 
                               predictions: torch.Tensor, 
                               targets: torch.Tensor) -> dict:
        """
        计算决策成本的详细分解（用于日志和调试）
        
        Args:
            predictions: TFT 的预测值
            targets: 真实需求量
            
        Returns:
            包含各项成本指标的字典
        """
        with torch.no_grad():
            over_inventory = predictions - targets
            under_inventory = targets - predictions
            
            if self.smooth:
                over_inventory = F.softplus(over_inventory, beta=1.0/self.temperature)
                under_inventory = F.softplus(under_inventory, beta=1.0/self.temperature)
            else:
                over_inventory = torch.relu(over_inventory)
                under_inventory = torch.relu(under_inventory)
            
            holding_cost = self.holding_cost * over_inventory
            shortage_cost = self.shortage_penalty * under_inventory
            total_cost = holding_cost + shortage_cost
            
            return {
                'total_cost': total_cost.mean().item(),
                'holding_cost': holding_cost.mean().item(),
                'shortage_cost': shortage_cost.mean().item(),
                'avg_over_inventory': over_inventory.mean().item(),
                'avg_under_inventory': under_inventory.mean().item(),
            }


class DecisionAwareLoss(nn.Module):
    """
    决策感知联合损失函数
    
    组合传统预测损失（如 SMAPE、Quantile Loss）和决策成本，
    通过可调节的权重平衡两者。
    """
    
    def __init__(self,
                 prediction_loss_fn: nn.Module,
                 decision_cost_evaluator: DecisionCostEvaluator,
                 decision_weight: float = 0.1,
                 warmup_epochs: int = 0):
        """
        初始化决策感知损失函数
        
        Args:
            prediction_loss_fn: 传统预测损失函数（如 SMAPE、MSE）
            decision_cost_evaluator: 决策成本评估器
            decision_weight: 决策成本在总损失中的权重 (lambda)，默认 0.1
            warmup_epochs: 预热轮数，在此期间 decision_weight 从 0 线性增加到目标值
        """
        super().__init__()
        self.prediction_loss_fn = prediction_loss_fn
        self.decision_cost_evaluator = decision_cost_evaluator
        self.decision_weight = decision_weight
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """
        设置当前轮数（用于预热调度）
        """
        self.current_epoch = epoch
        
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算联合损失
        
        Args:
            predictions: 模型预测值
            targets: 真实值
            
        Returns:
            联合损失值
        """
        # 计算预测损失
        pred_loss = self.prediction_loss_fn(predictions, targets)
        
        # 计算决策成本
        decision_cost = self.decision_cost_evaluator(predictions, targets)
        
        # 预热调度：在 warmup_epochs 期间，decision_weight 从 0 线性增加
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            current_weight = self.decision_weight * (self.current_epoch / self.warmup_epochs)
        else:
            current_weight = self.decision_weight
        
        # 联合损失
        joint_loss = pred_loss + current_weight * decision_cost
        
        return joint_loss
    
    def get_loss_breakdown(self, 
                            predictions: torch.Tensor, 
                            targets: torch.Tensor) -> dict:
        """
        获取损失分解（用于日志）
        """
        with torch.no_grad():
            pred_loss = self.prediction_loss_fn(predictions, targets)
            decision_cost = self.decision_cost_evaluator(predictions, targets)
            
            if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
                current_weight = self.decision_weight * (self.current_epoch / self.warmup_epochs)
            else:
                current_weight = self.decision_weight
            
            joint_loss = pred_loss + current_weight * decision_cost
            
            return {
                'joint_loss': joint_loss.item(),
                'pred_loss': pred_loss.item(),
                'decision_cost': decision_cost.item(),
                'current_weight': current_weight,
            }
