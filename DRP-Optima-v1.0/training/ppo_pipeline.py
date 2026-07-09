"""
PPO 训练流程 - training/ppo_pipeline.py

提供供代码调用的 main() 函数，替代已删除的 train_ppo_base.py。
内部统一使用 training/trainer.py 的 Trainer 类。
"""

import os
import sys
import time
import numpy as np

from config.data_config import PPOConfig, EnvConfig, DataConfig
from training.trainer import Trainer
from models.optimization.supply_chain_env import SupplyChainEnv


def main(
    config=None,  # 组合配置对象（由 get_config() 返回）
    n_skus=10,
    n_stores=1,
    use_tft=False,
    tft_checkpoint_path="",
    total_timesteps=None,
    **kwargs,
):
    """
    供代码调用的 PPO 训练入口（替代已删除的 train_ppo_base.py）
    
    Args:
        config: 组合配置对象（由 get_config() 返回，包含 paths/data/ppo/env 属性）
                 如果为 None，自动调用 get_config() 获取
        n_skus: SKU 数量
        n_stores: 门店数量
        use_tft: 是否使用 TFT 预测
        tft_checkpoint_path: TFT 模型路径
        total_timesteps: 训练步数（覆盖 config.ppo.total_timesteps）
        **kwargs: 其他兼容参数
    
    Returns:
        dict: 训练结果
    """
    start_time = time.time()
    
    # 1. 获取配置
    if config is None:
        from config.data_config import get_config
        config = get_config()
    
    if total_timesteps is not None:
        config.ppo.total_timesteps = total_timesteps
    
    # 2. 创建 Trainer 实例
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "records", "output")
    output_dir = os.path.normpath(output_dir)
    
    print(f"[ppo_pipeline] 开始 PPO 训练（n_skus={n_skus}, n_stores={n_stores}）")
    
    try:
        trainer = Trainer(config=config, output_dir=output_dir)
        
        # 3. 加载数据
        print("[ppo_pipeline] 步骤1：加载数据...")
        data = trainer.load_data()
        
        # 4. 如果使用 TFT，生成预测
        demand_forecasts = None
        demand_forecasts_quantiles = None
        if use_tft and tft_checkpoint_path and os.path.exists(tft_checkpoint_path):
            print("[ppo_pipeline] 步骤2：生成 TFT 预测...")
            try:
                from training.data_utils import generate_daily_forecast_with_tft
                demand_forecasts, demand_forecasts_quantiles = generate_daily_forecast_with_tft(
                    data, tft_checkpoint_path, n_skus=n_skus
                )
            except Exception as e:
                print(f"[WARN] TFT 预测失败: {e}")
        
        # 5. 准备 RL 环境（不需要传入 data，Trainer 内部已有）
        print("[ppo_pipeline] 步骤3：准备 RL 环境...")
        env = trainer.prepare_rl_env(
            demand_forecasts=demand_forecasts,
            demand_forecasts_quantiles=demand_forecasts_quantiles,
        )
        
        # 6. 训练 PPO
        print(f"[ppo_pipeline] 步骤4：训练 PPO（total_timesteps={config.ppo.total_timesteps}）...")
        results = trainer.train_ppo(
            env=env,
            total_timesteps=config.ppo.total_timesteps,
            eval_freq=kwargs.get("eval_freq", 1000),
        )
        
        total_time = time.time() - start_time
        print(f"[ppo_pipeline] 训练完成！耗时: {total_time:.2f} 秒")
        print(f"  模型已保存: {results.get('model_path', 'N/A')}")
        
        return results
        
    except Exception as e:
        print(f"[ppo_pipeline] 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO 训练（代码调用/命令行）")
    parser.add_argument("--n-skus", type=int, default=10, help="SKU 数量")
    parser.add_argument("--n-stores", type=int, default=1, help="门店数量")
    parser.add_argument("--total-steps", type=int, default=100000, help="训练步数")
    parser.add_argument("--use-tft", action="store_true", help="使用 TFT 预测")
    parser.add_argument("--tft-path", type=str, default="", help="TFT 模型路径")
    
    args = parser.parse_args()
    
    main(
        n_skus=args.n_skus,
        n_stores=args.n_stores,
        total_timesteps=args.total_steps,
        use_tft=args.use_tft,
        tft_checkpoint_path=args.tft_path,
    )
