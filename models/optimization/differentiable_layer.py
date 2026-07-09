"""
可微库存决策层 - models/optimization/differentiable_layer.py
数值安全的可微库存决策层，用于决策感知 TFT 训练。

核心特性：
1. 分段 Softplus 彻底杜绝 exp() 溢出
2. 成本缩放防止梯度爆炸
3. 支持全向量化计算（无 For 循环）
"""

import torch
import torch.nn as nn


class DifferentiableInventoryLayer(nn.Module):
    """
    可微库存决策层
    
    用于计算库存决策的可微成本，可作为 TFT 训练的损失函数组件。
    使用分段 Softplus (Smooth Max) 实现数值绝对安全的平滑最大值计算。
    
    属性：
        h (float): 持有成本系数
        p (float): 缺货惩罚系数
        beta (float): 平滑温度参数（越大越接近硬 max）
        cost_scale (float): 成本缩放因子（防止梯度爆炸）
        safe_threshold (float): 数值安全阈值（bx > safe_threshold 时直接用线性近似）
    """
    
    def __init__(self, holding_cost=1.0, shortage_penalty=10.0,
                 smooth_temp=5.0, cost_scale=1e-3, safe_threshold=20.0):
        """
        初始化可微库存决策层
        
        Args:
            holding_cost: 持有成本系数（每单位库存的日成本）
            shortage_penalty: 缺货惩罚系数（每单位缺货的惩罚）
            smooth_temp: 平滑温度参数，控制 Smooth Max 的平滑程度
            cost_scale: 成本缩放因子，防止梯度爆炸
            safe_threshold: 数值安全阈值，bx > safe_threshold 时直接用线性近似
        """
        super().__init__()
        self.h = holding_cost
        self.p = shortage_penalty
        self.beta = smooth_temp
        self.cost_scale = cost_scale
        self.safe_threshold = safe_threshold

    def _smooth_max(self, x):
        """
        数值绝对安全的 Smooth Max
        
        使用分段近似：
        - 当 bx <= safe_threshold 时：使用 log1p(exp(bx)) / beta（数值稳定）
        - 当 bx > safe_threshold 时：直接使用线性近似 x（避免 exp() 溢出）
        
        Args:
            x: 输入张量
            
        Returns:
            平滑最大值近似张量（与 x 同形状）
        """
        bx = self.beta * x
        safe_mask = bx > self.safe_threshold
        result = torch.zeros_like(x)
        # 安全区域：使用 log1p(exp(bx)) 避免 exp() 溢出
        if (~safe_mask).any():
            result[~safe_mask] = torch.log1p(torch.exp(bx[~safe_mask])) / self.beta
        # 溢出区域：使用线性近似（max(x, 0) ≈ x when x >> 0）
        if safe_mask.any():
            result[safe_mask] = x[safe_mask]
        return result

    def forward(self, y_pred, y_true, current_inventory, in_transit):
        """
        计算可微库存决策成本（简化接口）
        
        此方法是便捷接口，直接返回缩放后的平均库存成本。
        如需更细粒度的控制（如按 SKU 求和），请直接使用 _smooth_max 方法。
        
        Args:
            y_pred: 预测目标库存水平 (batch, n_skus)
            y_true: 实际需求量 (batch, n_skus)
            current_inventory: 当前库存 (batch, n_skus)
            in_transit: 在途库存 (batch, n_skus)
            
        Returns:
            缩放后的平均库存成本（标量）
        """
        target_inv = y_pred
        order_qty = self._smooth_max(target_inv - (current_inventory + in_transit))
        available = current_inventory + in_transit + order_qty
        
        end_inv = self._smooth_max(available - y_true)
        shortage = self._smooth_max(y_true - available)
        
        total_cost = (self.h * end_inv) + (self.p * shortage)
        return self.cost_scale * torch.mean(total_cost)

    def compute_decision_cost(self, y_pred, y_true, proxy_inventory, proxy_in_transit):
        """
        计算决策成本（与 Step2 DecisionAwareTFTWrapper 对齐的接口）
        
        此方法与 Step2 中的 training_step 计算逻辑完全一致。
        
        Args:
            y_pred: 预测目标库存水平 (batch, n_skus)
            y_true: 实际需求量 (batch, n_skus)
            proxy_inventory: 代理库存状态 (batch, n_skus)
            proxy_in_transit: 代理在途库存 (batch, n_skus)
            
        Returns:
            decision_cost: 决策成本（标量，已缩放）
        """
        target_inv = y_pred
        order_qty = self._smooth_max(target_inv - (proxy_inventory + proxy_in_transit))
        available = proxy_inventory + proxy_in_transit + order_qty
        
        end_inv = self._smooth_max(available - y_true)
        shortage = self._smooth_max(y_true - available)
        
        # 在 SKU 维度求和，Batch 维度求平均
        decision_cost = self.cost_scale * torch.mean(
            torch.sum(self.h * end_inv + self.p * shortage, dim=-1)
        )
        
        return decision_cost


def test_smooth_max():
    """测试 _smooth_max 的数值稳定性"""
    layer = DifferentiableInventoryLayer(safe_threshold=20.0)
    
    # 测试正常范围
    x_normal = torch.randn(100, 10)
    y_normal = layer._smooth_max(x_normal)
    assert not torch.isnan(y_normal).any(), "Normal range: NaN detected!"
    assert not torch.isinf(y_normal).any(), "Normal range: Inf detected!"
    
    # 测试极端值（可能溢出 exp()）
    x_extreme = torch.ones(100, 10) * 100.0  # bx = 500 >> safe_threshold
    y_extreme = layer._smooth_max(x_extreme)
    assert not torch.isnan(y_extreme).any(), "Extreme range: NaN detected!"
    assert not torch.isinf(y_extreme).any(), "Extreme range: Inf detected!"
    
    print("[PASS] _smooth_max numerical stability test passed")
    print(f"   - Normal range: [{x_normal.min():.2f}, {x_normal.max():.2f}] -> [{y_normal.min():.2f}, {y_normal.max():.2f}]")
    print(f"   - Extreme range: [{x_extreme.min():.2f}, {x_extreme.max():.2f}] -> [{y_extreme.min():.2f}, {y_extreme.max():.2f}]")
    
    return True


def test_differentiable_layer():
    """测试 DifferentiableInventoryLayer 的完整功能"""
    torch.manual_seed(42)
    
    # 创建层
    layer = DifferentiableInventoryLayer(
        holding_cost=1.0,
        shortage_penalty=10.0,
        cost_scale=1e-2
    )
    
    # 创建测试数据
    batch_size = 8
    n_skus = 5
    
    y_pred = torch.randn(batch_size, n_skus, requires_grad=True)
    y_true = torch.abs(torch.randn(batch_size, n_skus))
    current_inventory = torch.abs(torch.randn(batch_size, n_skus))
    in_transit = torch.abs(torch.randn(batch_size, n_skus))
    
    # 测试 forward
    cost = layer.forward(y_pred, y_true, current_inventory, in_transit)
    assert cost.shape == torch.Size([]), f"forward 输出形状错误: {cost.shape}"
    assert not torch.isnan(cost).any(), "forward 输出包含 NaN!"
    
    # 测试梯度回传
    cost.backward()
    assert y_pred.grad is not None, "梯度未回传到 y_pred!"
    assert not torch.isnan(y_pred.grad).any(), "梯度包含 NaN!"
    
    print("[PASS] DifferentiableInventoryLayer full functionality test passed")
    print(f"   - Input shape: y_pred={y_pred.shape}")
    print(f"   - Output cost: {cost.item():.6f}")
    print(f"   - Gradient mean: {y_pred.grad.mean().item():.6f}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DifferentiableInventoryLayer Unit Tests")
    print("=" * 60)
    
    try:
        test_smooth_max()
        test_differentiable_layer()
        
        print("\n" + "=" * 60)
        print("[PASS] All tests passed! DifferentiableInventoryLayer implemented correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
