"""
Decision-Aware TFT Wrapper (决策感知 TFT 包装器)

完全向量化的决策感知 TFT 实现。
直接继承 pytorch_forecasting.TemporalFusionTransformer，在 training_step 中
添加可微决策成本，使用 DifferentiableInventoryLayer 实现数值安全的正则化。

核心特性：
1. 完全向量化计算（无 For 循环）
2. 使用代理库存状态（encoder_target 历史均值）
3. 数值安全的 Smooth Max（防止 exp() 溢出）
4. 决策成本与预测损失联合优化
"""

import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

from models.optimization.differentiable_layer import DifferentiableInventoryLayer


class DecisionAwareTFTWrapper(TemporalFusionTransformer):
    """
    决策感知 TFT 包装器
    
    直接继承 TemporalFusionTransformer，在训练过程中联合优化
    预测精度与库存决策成本。
    
    属性：
        decision_weight (float): 决策成本权重 (λ)
        inventory_layer (DifferentiableInventoryLayer): 可微库存决策层
        prediction_loss_fn (callable): 预测损失函数
    """
    
    def __init__(self, *args, decision_weight=0.3, holding_cost=1.0,
                 shortage_penalty=10.0, **kwargs):
        """
        初始化决策感知 TFT 包装器
        
        Args:
            *args: 传递给 TemporalFusionTransformer 的位置参数
            decision_weight: 决策成本权重 (λ ∈ [0, 1])
            holding_cost: 持有成本系数
            shortage_penalty: 缺货惩罚系数
            **kwargs: 传递给 TemporalFusionTransformer 的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.decision_weight = decision_weight
        self.inventory_layer = DifferentiableInventoryLayer(
            holding_cost=holding_cost,
            shortage_penalty=shortage_penalty,
            cost_scale=1e-2
        )
        # 保存预测损失函数（从父类获取）
        self.prediction_loss_fn = self.loss

    def training_step(self, batch, batch_idx):
        """
        训练步骤：计算联合损失（预测损失 + 决策成本）
        
        完全向量化实现，无 For 循环。
        
        Args:
            batch: 训练批次数据（pytorch_forecasting 格式）
            batch_idx: 批次索引
            
        Returns:
            联合损失值
        """
        # 1. 前向传播
        network_outputs = self(batch)
        
        # 2. 提取预测值和真实值
        # pytorch_forecasting 输出格式：
        #   network_outputs["prediction"]: (batch, n_timesteps, n_skus)
        #   batch["decoder_target"]: (batch, n_timesteps, n_skus)
        y_pred = network_outputs["prediction"][:, -1, :]  # (batch, n_skus)
        y_true = batch["decoder_target"][:, -1, :]        # (batch, n_skus)

        # 3. 计算代理库存状态（使用 encoder_target 的历史均值）
        # encoder_target: (batch, encoder_length, n_skus)
        # 假设 encoder_length 是历史窗口长度
        proxy_inventory = batch["encoder_target"].mean(dim=1)  # (batch, n_skus)
        proxy_in_transit = torch.zeros_like(proxy_inventory)      # (batch, n_skus)

        # 4. 【全向量化】计算决策成本
        decision_cost = self.inventory_layer.compute_decision_cost(
            y_pred, y_true, proxy_inventory, proxy_in_transit
        )

        # 5. 计算预测损失
        pred_loss = self.prediction_loss_fn(network_outputs, batch)

        # 6. 联合损失
        total_loss = (1 - self.decision_weight) * pred_loss + self.decision_weight * decision_cost

        # 7. 记录损失分量（用于日志）
        self.log("train_pred_loss", pred_loss, prog_bar=True)
        self.log("train_decision_cost", decision_cost, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤：计算联合损失（不包含梯度传播）
        
        Args:
            batch: 验证批次数据
            batch_idx: 批次索引
            
        Returns:
            联合损失值
        """
        # 前向传播
        network_outputs = self(batch)
        
        # 提取预测值和真实值
        y_pred = network_outputs["prediction"][:, -1, :]
        y_true = batch["decoder_target"][:, -1, :]

        # 计算代理库存状态
        proxy_inventory = batch["encoder_target"].mean(dim=1)
        proxy_in_transit = torch.zeros_like(proxy_inventory)

        # 计算决策成本
        decision_cost = self.inventory_layer.compute_decision_cost(
            y_pred, y_true, proxy_inventory, proxy_in_transit
        )

        # 计算预测损失
        pred_loss = self.prediction_loss_fn(network_outputs, batch)

        # 联合损失
        total_loss = (1 - self.decision_weight) * pred_loss + self.decision_weight * decision_cost

        # 记录损失分量
        self.log("val_pred_loss", pred_loss, on_step=False, on_epoch=True)
        self.log("val_decision_cost", decision_cost, on_step=False, on_epoch=True)
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True)

        return total_loss


def test_decision_aware_tft_wrapper():
    """测试 DecisionAwareTFTWrapper 的维度对齐"""
    import torch
    
    # 注意：这是一个单元测试，需要 mock TemporalFusionTransformer
    # 由于 TFT 初始化复杂，这里只测试逻辑正确性
    
    print("=" * 60)
    print("DecisionAwareTFTWrapper 逻辑验证")
    print("=" * 60)
    
    # 模拟 network_outputs 和 batch
    batch_size = 8
    n_timesteps = 7
    n_skus = 5
    
    # 模拟 network_outputs
    network_outputs = {
        "prediction": torch.randn(batch_size, n_timesteps, n_skus)
    }
    
    # 模拟 batch
    batch = {
        "decoder_target": torch.abs(torch.randn(batch_size, n_timesteps, n_skus)),
        "encoder_target": torch.abs(torch.randn(batch_size, 14, n_skus))
    }
    
    # 提取 y_pred 和 y_true
    y_pred = network_outputs["prediction"][:, -1, :]
    y_true = batch["decoder_target"][:, -1, :]
    
    assert y_pred.shape == (batch_size, n_skus), f"y_pred 形状错误: {y_pred.shape}"
    assert y_true.shape == (batch_size, n_skus), f"y_true 形状错误: {y_true.shape}"
    
    # 计算代理库存状态
    proxy_inventory = batch["encoder_target"].mean(dim=1)
    proxy_in_transit = torch.zeros_like(proxy_inventory)
    
    assert proxy_inventory.shape == (batch_size, n_skus), f"proxy_inventory 形状错误: {proxy_inventory.shape}"
    
    # 使用 DifferentiableInventoryLayer 计算决策成本
    inventory_layer = DifferentiableInventoryLayer(
        holding_cost=1.0,
        shortage_penalty=10.0,
        cost_scale=1e-2
    )
    
    decision_cost = inventory_layer.compute_decision_cost(
        y_pred, y_true, proxy_inventory, proxy_in_transit
    )
    
    assert decision_cost.shape == torch.Size([]), f"decision_cost 形状错误: {decision_cost.shape}"
    assert not torch.isnan(decision_cost).any(), "decision_cost 包含 NaN!"
    
    print(f"[PASS] 维度对齐验证通过")
    print(f"   - y_pred: {y_pred.shape}")
    print(f"   - y_true: {y_true.shape}")
    print(f"   - proxy_inventory: {proxy_inventory.shape}")
    print(f"   - decision_cost: {decision_cost.item():.6f}")
    print()
    print("=" * 60)
    print("[PASS] DecisionAwareTFTWrapper 逻辑验证通过")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_decision_aware_tft_wrapper()

