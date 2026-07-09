"""
V10.3 训练脚本（Train V9）

符合 V10.3 生产方案完整规范：
- 使用 MaskablePPO（支持 Action Masking）
- SubprocVecEnv 多进程训练
- VecNormalize 观测归一化
- ActionMasker 动作掩码
- V9 回调（SyncedEvalCallbackV9 等）
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from config.data_config import EnvConfigV9, linear_schedule
from env.global_supply_chain_env_v9 import GlobalSupplyChainEnvV9
from data.global_data_loader import GlobalDataLoader
from utils.callbacks_v9 import create_v9_callbacks
from utils.logger import get_logger, configure_logging

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

logger = get_logger(__name__)


def make_env(
    env_config: EnvConfigV9,
    data_loader: GlobalDataLoader,
    store_ids: List[str],
    eval_mode: bool = False,
    rank: int = 0,
    seed: int = 42,
    log_dir: Optional[str] = None,
    use_tft_predictions: bool = False,
    tft_prediction_path: Optional[str] = None,
) -> callable:
    """
    环境工厂函数（用于 SubprocVecEnv）
    
    Args:
        env_config: 环境配置
        data_loader: 数据加载器
        store_ids: 门店 ID 列表
        eval_mode: 是否评估模式
        rank: 环境排名（用于多进程）
        seed: 随机种子
        log_dir: Monitor 日志目录（如果提供，则启用 Monitor）
        use_tft_predictions: 是否使用 TFT 预测
        tft_prediction_path: TFT 预测文件路径
    
    Returns:
        环境构造函数
    """
    def _init() -> GlobalSupplyChainEnvV9:
        # 循环分配门店（支持多环境）
        store_id = store_ids[rank % len(store_ids)]
        
        env = GlobalSupplyChainEnvV9(
            env_config=env_config,
            data_loader=data_loader,
            store_id=store_id,
            eval_mode=eval_mode,
            verbose=env_config.verbose,
            use_tft_predictions=use_tft_predictions,
            tft_prediction_path=tft_prediction_path,
        )
        
        # 🌟 包装 Monitor（记录 episode 奖励到 CSV）
        if log_dir is not None:
            monitor_path = os.path.join(log_dir, f"{'eval' if eval_mode else 'train'}_env_{rank}")
            env = Monitor(env, filename=monitor_path)
        
        return env
    
    return _init


def train_v9(
    env_config: Optional[EnvConfigV9] = None,
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    log_dir: str = "output/logs/train_v9",
    model_save_dir: str = "output/checkpoints/train_v9",
    seed: int = 42,
    verbose: int = 1,
    use_tft_predictions: bool = False,
    tft_prediction_path: Optional[str] = None,
):
    """
    训练 V9 模型
    
    Args:
        env_config: 环境配置
        total_timesteps: 总训练步数
        n_envs: 并行环境数
        log_dir: 日志目录
        model_save_dir: 模型保存目录
        seed: 随机种子
        verbose: 日志级别
    """
    # 1. 初始化配置
    env_config = env_config or EnvConfigV9(num_skus=88)  # 默认88个SKU
    env_config.verbose = verbose
    
    # 2. 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 3. 初始化数据加载器
    logger.info("初始化数据加载器...")
    data_loader = GlobalDataLoader()
    
    # 4. 创建 Monitor 日志目录
    monitor_log_dir = os.path.join(log_dir, "monitor_logs")
    os.makedirs(monitor_log_dir, exist_ok=True)
    logger.info(f"Monitor 日志目录: {monitor_log_dir}")
    
    # 5. 创建训练环境
    logger.info(f"创建 {n_envs} 个并行训练环境...")
    store_ids = ["140"] * n_envs  # 使用门店140
    
    train_env = SubprocVecEnv([
        make_env(
            env_config, data_loader, store_ids, 
            eval_mode=False, rank=i, seed=seed+i,
            log_dir=monitor_log_dir,  # ← 启用 Monitor
            use_tft_predictions=use_tft_predictions,
            tft_prediction_path=tft_prediction_path,
        )
        for i in range(n_envs)
    ])
    
    # 🔧 修复11：添加 VecNormalize（Observation 归一化）
    # 目的：对观察空间进行滚动归一化，帮助神经网络收敛
    # 注意：不归一化奖励（我们已经有 Soft Clipping）
    from stable_baselines3.common.vec_env import VecNormalize
    train_env = VecNormalize(
        train_env,
        norm_obs=True,   # 归一化观察空间
        norm_reward=False,  # 不归一化奖励
        clip_obs=10.0,  # 裁剪观察值（防止极端值）
    )
    logger.info("✅ VecNormalize 已启用（Observation 归一化）")
    
    # 6. 创建评估环境
    logger.info("创建评估环境...")
    eval_env = SubprocVecEnv([
        make_env(
            env_config, data_loader, store_ids, 
            eval_mode=True, rank=0, seed=seed+999,
            log_dir=monitor_log_dir,  # ← 启用 Monitor
            use_tft_predictions=use_tft_predictions,
            tft_prediction_path=tft_prediction_path,
        )
    ])
    
    # 7. 包装 ActionMasker（动作掩码）
    # 🌟 注意：MaskablePPO 会自动调用环境的 action_masks() 方法
    # 不需要显式包装 ActionMasker（这会导致类型错误）
    if env_config.enable_action_mask:
        logger.info("启用 Action Masking（MaskablePPO 自动处理）...")
        # 不需要包装 ActionMasker
        # train_env = ActionMasker(train_env, mask_fn)
        # eval_env = ActionMasker(eval_env, mask_fn)
    
    # 7. 包装 VecNormalize（观测归一化）
    logger.info("启用 VecNormalize...")
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
    )
    
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # 评估时不归一化奖励
        training=False,  # 冻结统计量
        clip_obs=10.0,
    )
    
    # 8. 创建回调
    logger.info("创建回调...")
    callbacks = create_v9_callbacks(
        eval_env=eval_env,
        log_dir=log_dir,
        eval_freq=env_config.log_interval,
        n_eval_episodes=5,
        curriculum_soft_epochs=env_config.curriculum_soft_epochs,
        business_metrics_freq=1000,
    )
    
    # 9. 创建 MaskablePPO 模型
    logger.info("创建 MaskablePPO 模型...")
    
    # 检查是否使用 MaskablePPO
    try:
        from sb3_contrib import MaskablePPO
        
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(3e-4, 1e-6),
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=verbose,
            tensorboard_log=log_dir,
        )
        
        logger.info("使用 MaskablePPO（支持 Action Masking）")
    
    except ImportError:
        logger.warning("sb3-contrib 未安装，使用标准 PPO（不支持 Action Masking）")
        
        from stable_baselines3 import PPO
        
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(3e-4, 1e-6),
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=verbose,
            tensorboard_log=log_dir,
        )
    
    # 10. 训练
    logger.info(f"开始训练（总步数: {total_timesteps}）...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="MaskablePPO_V9",
        )
        
        training_time = time.time() - start_time
        logger.info(f"训练完成！耗时: {training_time:.2f} 秒")
        
        # 11. 保存最终模型
        final_model_path = os.path.join(model_save_dir, "final_model.zip")
        model.save(final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")
        
        # 12. 保存 VecNormalize 统计量
        vec_normalize_path = os.path.join(model_save_dir, "vec_normalize.pkl")
        train_env.save(vec_normalize_path)
        logger.info(f"VecNormalize 统计量已保存: {vec_normalize_path}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        # 即使训练失败，也尝试保存模型
        try:
            emergency_save_path = os.path.join(model_save_dir, "emergency_save.zip")
            model.save(emergency_save_path)
            logger.info(f"紧急保存模型: {emergency_save_path}")
        except:
            logger.error("紧急保存失败！")
        raise
    
    finally:
        # 清理环境
        train_env.close()
        eval_env.close()
    
    return model, train_env, eval_env


def read_monitor_logs(log_dir: str, env_type: str = "train"):
    """
    读取 Monitor 日志并显示训练进度
    
    Args:
        log_dir: Monitor 日志目录
        env_type: 环境类型（"train" 或 "eval"）
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # 查找所有 Monitor 日志文件
    # SB3 Monitor 生成的文件名格式: {filename}.monitor.csv
    monitor_files = list(Path(log_dir).glob(f"*.monitor.csv"))
    
    if not monitor_files:
        print(f"❌ 未找到 Monitor 日志文件")
        print(f"   搜索路径: {log_dir}")
        print(f"   搜索模式: *.monitor.csv")
        return
    
    print(f"📊 找到 {len(monitor_files)} 个 Monitor 日志文件")
    print(f"   路径: {log_dir}")
    print()
    
    # 读取所有日志文件
    all_data = []
    for file in sorted(monitor_files):
        try:
            # Monitor CSV 文件有特殊的头部格式（以 # 开头）
            data = pd.read_csv(file, skiprows=1)  # 跳过头部注释行
            data['env_id'] = file.stem
            all_data.append(data)
        except Exception as e:
            print(f"⚠️  读取 {file.name} 失败: {e}")
    
    if not all_data:
        print("❌ 没有成功读取任何日志文件")
        return
    
    # 合并所有数据
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values('t')  # 按时间步排序
    
    # 打印统计信息
    print(f"📈 训练进度统计（{env_type} 环境）")
    print(f"   总 episode 数: {len(df)}")
    print(f"   总时间步: {df['t'].max():,}")
    print(f"   平均 episode 奖励: {df['r'].mean():.2f}")
    print(f"   平均 episode 长度: {df['l'].mean():.2f}")
    print(f"   最高 episode 奖励: {df['r'].max():.2f}")
    print(f"   最低 episode 奖励: {df['r'].min():.2f}")
    print()
    
    # 打印最近的 episode
    print(f"📋 最近 10 个 episode:")
    print(f"{'Episode':<10} {'时间步':<15} {'奖励':<15} {'长度':<10}")
    print("-" * 50)
    for idx, row in df.tail(10).iterrows():
        print(f"{idx:<10} {row['t']:<15,} {row['r']:<15.2f} {row['l']:<10}")
    print()
    
    # 绘制训练曲线
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 奖励曲线
        axes[0].plot(df['t'], df['r'], alpha=0.6, label='Episode Reward')
        # 移动平均
        window = min(100, len(df))
        df['r_ma'] = df['r'].rolling(window=window, min_periods=1).mean()
        axes[0].plot(df['t'], df['r_ma'], linewidth=2, label=f'Moving Average (window={window})')
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title(f'{env_type.capitalize()} Environment - Episode Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Episode 长度曲线
        axes[1].plot(df['t'], df['l'], alpha=0.6, label='Episode Length')
        df['l_ma'] = df['l'].rolling(window=window, min_periods=1).mean()
        axes[1].plot(df['t'], df['l_ma'], linewidth=2, label=f'Moving Average (window={window})')
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel('Episode Length')
        axes[1].set_title(f'{env_type.capitalize()} Environment - Episode Length')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(log_dir, f"{env_type}_training_curve.png")
        plt.savefig(plot_path, dpi=150)
        print(f"📊 训练曲线已保存: {plot_path}")
        
        plt.show()
        
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过绘图")
    except Exception as e:
        print(f"⚠️  绘图失败: {e}")


def main():
    """主函数（命令行入口）"""
    parser = argparse.ArgumentParser(description="训练 V9 模型")
    
    # 训练参数
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="总训练步数（默认: 1,000,000）",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="并行环境数（默认: 4）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="output/logs/train_v9",
        help="日志目录（默认: output/logs/train_v9）",
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        default="output/checkpoints/train_v9",
        help="模型保存目录（默认: output/checkpoints/train_v9）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="日志级别（默认: 1）",
    )
    parser.add_argument(
        "--use-tft-predictions",
        action="store_true",
        help="启用 TFT 预测（默认: False）",
    )
    parser.add_argument(
        "--tft-prediction-path",
        type=str,
        default="data/processed/tft_rolling_predictions.npy",
        help="TFT 预测文件路径（默认: data/processed/tft_rolling_predictions.npy）",
    )
    parser.add_argument(
        "--num-skus",
        type=int,
        default=88,
        help="SKU 数量（默认: 88）",
    )
    
    # Monitor 日志查看
    parser.add_argument(
        "--view-monitor",
        action="store_true",
        help="查看 Monitor 日志（不启动训练）",
    )
    parser.add_argument(
        "--monitor-env-type",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Monitor 环境类型（默认: train）",
    )
    
    args = parser.parse_args()
    
    # 查看 Monitor 日志模式
    if args.view_monitor:
        monitor_log_dir = os.path.join(args.log_dir, "monitor_logs")
        if not os.path.exists(monitor_log_dir):
            print(f"❌ Monitor 日志目录不存在: {monitor_log_dir}")
            print(f"   请先运行训练脚本生成 Monitor 日志")
            return
        
        read_monitor_logs(monitor_log_dir, env_type=args.monitor_env_type)
        return
    
    # 配置日志（同时输出到文件）
    log_file = os.path.join(args.log_dir, "training.log")
    configure_logging(verbose=args.verbose, log_file=log_file)
    
    # 开始训练
    train_v9(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        log_dir=args.log_dir,
        model_save_dir=args.model_save_dir,
        seed=args.seed,
        verbose=args.verbose,
        use_tft_predictions=args.use_tft_predictions,
        tft_prediction_path=args.tft_prediction_path,
    )


if __name__ == "__main__":
    main()
