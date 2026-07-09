"""
MVP训练脚本 - 最小可行化版本
禁用所有高级功能，只保留核心PPO训练
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.global_supply_chain_env_v9 import GlobalSupplyChainEnvV9
from config.data_config import EnvConfigV9

# 配置日志（模块级别）
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def make_env(env_config, store_id, use_tft_predictions, rank, monitor_dir):
    """
    创建环境的工厂函数（用于SubprocVecEnv）
    """
    def _init():
        env = GlobalSupplyChainEnvV9(
            env_config=env_config,
            store_id=store_id,
            use_tft_predictions=use_tft_predictions,
            verbose=0,
        )
        # 包装Monitor（用于记录episode统计）
        env = Monitor(env, filename=os.path.join(monitor_dir, f"env_{rank}_monitor.csv"))
        return env
    return _init


def train_mvp(
    total_timesteps=5000000,
    n_envs=4,
    log_dir="output/logs/train_mvp",
    model_save_dir="output/checkpoints/train_mvp",
    use_tft_predictions=False,
    num_skus=88,
    store_id="140",
    use_dummy_vecenv=False,  # 是否使用DummyVecEnv（单进程，更稳定的冒烟测试）
):
    """
    MVP训练函数 - 最小可行化版本
    
    Args:
        total_timesteps: 总训练步数
        n_envs: 并行环境数量
        log_dir: 日志目录
        model_save_dir: 模型保存目录
        use_tft_predictions: 是否使用TFT预测
        num_skus: SKU数量
        store_id: 门店ID
        use_dummy_vecenv: 是否使用DummyVecEnv（单进程）
    """
    logger.info("=" * 80)
    logger.info("🚀 启动MVP训练（最小可行化版本）")
    logger.info("=" * 80)
    logger.info(f"  总步数: {total_timesteps:,}")
    logger.info(f"  并行环境: {n_envs}")
    logger.info(f"  TFT预测: {use_tft_predictions}")
    logger.info(f"  SKU数量: {num_skus}")
    logger.info(f"  门店ID: {store_id}")
    logger.info(f"  使用DummyVecEnv: {use_dummy_vecenv}")
    logger.info("=" * 80)
    
    # 0. 创建日志和monitor目录
    monitor_dir = os.path.join(log_dir, "monitor_logs")
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir, exist_ok=True)
    
    # 1. 创建配置
    env_config = EnvConfigV9(
        num_skus=num_skus,
        data_weeks=100,
        verbose=0,
    )
    
    # 2. 创建训练环境（❌ 禁用VecNormalize，✅ 启用Monitor）
    if use_dummy_vecenv:
        # 使用DummyVecEnv（单进程，更稳定）
        logger.info("🔧 使用DummyVecEnv（单进程，用于冒烟测试）")
        train_env = DummyVecEnv([
            lambda: Monitor(
                GlobalSupplyChainEnvV9(
                    env_config=env_config,
                    store_id=store_id,
                    use_tft_predictions=use_tft_predictions,
                    verbose=0,
                ),
                filename=os.path.join(monitor_dir, f"env_0_monitor.csv")
            )
            for _ in range(2)  # 只用2个环境
        ])
    else:
        # 使用SubprocVecEnv（多进程）
        logger.info("🔧 使用SubprocVecEnv（多进程）")
        train_env = SubprocVecEnv([
            make_env(env_config, store_id, use_tft_predictions, i, monitor_dir)
            for i in range(n_envs)
        ])
    
    logger.info(f"✅ 训练环境创建成功（无VecNormalize）")
    
    # 3. 创建PPO模型（✅ 标准PPO，非MaskablePPO）
    # 确定策略网络类型
    # 注意：如果obs是Dict格式，使用"MultiInputPolicy"；如果是Box格式，使用"MlpPolicy"
    # 根据环境代码，obs是Box格式（连续的），所以使用"MlpPolicy"
    policy_type = "MlpPolicy"
    
    model = PPO(
        policy_type,
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=log_dir,
    )
    
    logger.info(f"✅ PPO模型创建成功（策略: {policy_type}）")
    logger.info(f"   注意：已禁用Action Masking（使用标准PPO）")
    
    # 4. 创建回调（❌ 禁用Eval Callback）
    callbacks = []
    
    # 只保留Checkpoint回调（定期保存模型）
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),  # 每1万步保存一次
        save_path=os.path.join(model_save_dir, "ppo_model"),
        name_prefix="ppo_model",
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"✅ Checkpoint回调已创建（禁用Eval Callback）")
    
    # 5. 开始训练
    logger.info(f"\n🚀 开始训练（{total_timesteps:,}步）...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="PPO_V9_MVP",
            log_interval=10,
        )
        logger.info(f"✅ 训练完成！")
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 紧急保存模型
        emergency_path = os.path.join(model_save_dir, "emergency_save.zip")
        model.save(emergency_path)
        logger.info(f"💾 模型已紧急保存到: {emergency_path}")
        return
    
    # 6. 保存最终模型
    final_model_path = os.path.join(model_save_dir, "final_model.zip")
    model.save(final_model_path)
    logger.info(f"💾 最终模型已保存到: {final_model_path}")
    
    # 7. 关闭环境
    train_env.close()
    
    logger.info("=" * 80)
    logger.info("🎉 MVP训练完成！")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="MVP训练脚本（最小可行化版本）")
    
    parser.add_argument("--total-timesteps", type=int, default=5000000,
                        help="总训练步数（默认: 5,000,000）")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="并行环境数量（默认: 4）")
    parser.add_argument("--log-dir", type=str, default="output/logs/train_mvp",
                        help="日志目录（默认: output/logs/train_mvp）")
    parser.add_argument("--model-save-dir", type=str, default="output/checkpoints/train_mvp",
                        help="模型保存目录（默认: output/checkpoints/train_mvp）")
    parser.add_argument("--use-tft-predictions", action="store_true",
                        help="是否使用TFT预测（默认: False）")
    parser.add_argument("--num-skus", type=int, default=88,
                        help="SKU数量（默认: 88）")
    parser.add_argument("--store-id", type=str, default="140",
                        help="门店ID（默认: 140）")
    parser.add_argument("--smoke-test", action="store_true",
                        help="运行冒烟测试（128步，使用DummyVecEnv）")
    
    args = parser.parse_args()
    
    # 如果是冒烟测试模式
    if args.smoke_test:
        logger.info("🔍 冒烟测试模式（128步）")
        train_mvp(
            total_timesteps=128,
            n_envs=2,
            log_dir=args.log_dir + "_smoke",
            model_save_dir=args.model_save_dir + "_smoke",
            use_tft_predictions=False,
            num_skus=10,  # 只用10个SKU，加快测试
            store_id=args.store_id,
            use_dummy_vecenv=True,  # 使用DummyVecEnv
        )
    else:
        # 正式训练
        train_mvp(
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            log_dir=args.log_dir,
            model_save_dir=args.model_save_dir,
            use_tft_predictions=args.use_tft_predictions,
            num_skus=args.num_skus,
            store_id=args.store_id,
            use_dummy_vecenv=False,  # 使用SubprocVecEnv
        )


if __name__ == "__main__":
    main()
