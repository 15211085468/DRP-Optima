"""
数据加载工具函数 - training/data_utils.py
从 train_ppo_base.py 迁移而来
"""

import os
import time
import json
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from config import DataConfig, EnvConfig, PPOConfig
from models.optimization import SupplyChainEnv
from integration.tft_drl_integration import (
    TFTDRLIntegration,
    IntegrationConfig,
    create_tft_drl_integration
)
from utils.logger import get_logger

logger = get_logger(__name__)


def load_data(config: Optional[DataConfig] = None) -> pd.DataFrame:
    """
    加载CSV数据
    
    Args:
        config: 数据配置，如果为None则使用默认配置
    
    Returns:
        加载的DataFrame
    """
    if config is None:
        from config import data_config
        config = data_config
    
    logger.info(f"加载数据: {config.data_path}")
    start_time = time.time()
    
    try:
        df = pd.read_csv(config.data_path, nrows=config.nrows)
        load_time = time.time() - start_time
        logger.info(f"数据加载完成: {df.shape}")
        logger.info(f"数据加载耗时: {load_time:.1f} 秒")
        return df
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        logger.error(traceback.format_exc())
        raise


def load_inventory_data(inventory_path: str, sku_list: list, store_list: list, 
                       latest_only: bool = True) -> np.ndarray:
    """
    加载库存数据并使用最新快照初始化库存
    
    Args:
        inventory_path: 库存数据文件路径（shop_stock.csv）
        sku_list: SKU列表（用于匹配）
        store_list: 门店列表（用于匹配，当前仅使用第一个门店）
        latest_only: 是否仅使用最新日期的快照
    
    Returns:
        初始库存数组 (len(sku_list),)
    """
    logger.info(f"加载库存数据: {inventory_path}")
    start_time = time.time()
    
    try:
        df = pd.read_csv(inventory_path)
        load_time = time.time() - start_time
        logger.info(f"库存数据加载完成: {df.shape}, 耗时: {load_time:.1f} 秒")
        
        # 注意：跳过门店筛选（使用所有库存数据）
        logger.info("  跳过门店筛选（使用所有库存数据）")
        
        # 如果仅使用最新快照
        if latest_only:
            latest_date = df['dt'].max()
            df = df[df['dt'] == latest_date]
            logger.info(f"使用最新日期 {latest_date} 的库存快照: {len(df)} 条记录")
        
        # 创建初始库存数组
        initial_inventory = np.zeros(len(sku_list), dtype=np.float32)
        
        # 填充库存数据
        sku_to_idx = {sku: idx for idx, sku in enumerate(sku_list)}
        matched_count = 0
        for _, row in df.iterrows():
            sku = row['goods_code']
            if sku in sku_to_idx:
                idx = sku_to_idx[sku]
                initial_inventory[idx] = max(row['stock_qty'], 0)  # 确保非负
                matched_count += 1
        
        logger.info(f"库存数据匹配: {matched_count}/{len(sku_list)} 个SKU")
        logger.info(f"初始库存统计: mean={np.mean(initial_inventory):.2f}, "
                    f"std={np.std(initial_inventory):.2f}, "
                    f"zeros={np.sum(initial_inventory == 0)}")
        
        return initial_inventory
        
    except Exception as e:
        logger.error(f"库存数据加载失败: {e}")
        logger.error(traceback.format_exc())
        # 返回零数组作为后备
        return np.zeros(len(sku_list), dtype=np.float32)


def prepare_data(df: pd.DataFrame, config: Optional[DataConfig] = None, 
                 inventory_path: Optional[str] = None,
                 use_tft: bool = False,
                 tft_checkpoint_path: str = "records/models/tft_model_quantile.ckpt") -> Tuple[np.ndarray, list, list]:
    """
    准备训练和预测数据
    
    Args:
        df: 原始DataFrame（销售数据）
        config: 数据配置
        inventory_path: 库存数据文件路径（如果提供，则仅使用同时具有销售和库存数据的SKU）
        use_tft: 是否使用TFT模型进行需求预测
        tft_checkpoint_path: TFT模型检查点路径（仅当 use_tft=True 时使用）
    
    Returns:
        (日度预测数组, SKU列表, 门店列表)
    """
    if config is None:
        from config import data_config
        config = data_config
    
    # 获取SKU和门店列表
    sku_list = df['goods_code'].unique()[:config.sku_limit]
    store_list = df['shop_code'].unique()[:config.store_limit]
    
    logger.info(f"原始数据 SKU数量: {len(sku_list)}")
    logger.info(f"原始数据 门店数量: {len(store_list)}")
    
    # 如果提供了库存数据路径，则筛选同时具有销售和库存数据的SKU
    if inventory_path is not None:
        logger.info(f"筛选同时具有销售和库存数据的SKU: {inventory_path}")
        try:
            inventory_df = pd.read_csv(inventory_path)
            inventory_skus = set(inventory_df['goods_code'].unique())
            sales_skus = set(sku_list)
            
            # 找到共有的SKU
            common_skus = sales_skus.intersection(inventory_skus)
            logger.info(f"销售数据SKU数: {len(sales_skus)}")
            logger.info(f"库存数据SKU数: {len(inventory_skus)}")
            logger.info(f"共有SKU数: {len(common_skus)}")
            
            # 仅使用共有的SKU（保持原有顺序）
            sku_list = [sku for sku in sku_list if sku in common_skus]
            logger.info(f"筛选后 SKU数量: {len(sku_list)}")
            
            if len(sku_list) == 0:
                logger.error("错误：没有同时具有销售和库存数据的SKU！")
                raise ValueError("没有同时具有销售和库存数据的SKU")
                
        except Exception as e:
            logger.error(f"筛选共有SKU失败: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # 强制清理显存
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            except:
                pass
        
    
    # 生成日度预测
    if use_tft:
        logger.info("=" * 70)
        logger.info("  使用TFT分位数模型进行需求预测")
        logger.info("=" * 70)
        daily_forecast = generate_daily_forecast_with_tft(
            df, sku_list, store_list, 
            tft_checkpoint_path=tft_checkpoint_path,
            forecast_days=84
        )
    else:
        logger.info("=" * 70)
        logger.info("  使用简化方法进行需求预测")
        logger.info("=" * 70)
        daily_forecast = generate_daily_forecast(df, sku_list, store_list)
    
    return daily_forecast, sku_list, store_list


def generate_daily_forecast(df: pd.DataFrame, sku_list: list, store_list: list, 
                           forecast_days: int = 84) -> np.ndarray:
    """
    生成日度需求预测（简化版：使用统计量）
    
    Args:
        df: 原始DataFrame
        sku_list: SKU列表
        store_list: 门店列表
        forecast_days: 预测天数
    
    Returns:
        日度预测数组 (forecast_days, len(sku_list))
    """
    logger.info("生成日度预测（简化版）...")
    
    # 简化版：使用SKU均值 + 星期几系数 + 随机扰动
    sku_mean = df.groupby('goods_code')['sale_qty'].mean().reindex(sku_list).fillna(1.0)
    dow_coef = np.array([1.2, 1.1, 1.0, 1.0, 1.0, 0.9, 0.8])  # 周一~周日
    
    daily_forecast = np.zeros((forecast_days, len(sku_list)))
    for day in range(forecast_days):
        dow = day % 7
        daily_forecast[day, :] = sku_mean.values * dow_coef[dow] * np.random.uniform(0.8, 1.2, len(sku_list))
    
    logger.info(f"日度预测生成: {daily_forecast.shape}")
    return daily_forecast


def generate_daily_forecast_with_tft(df: pd.DataFrame, 
                                     sku_list: list, 
                                     store_list: list,
                                     tft_checkpoint_path: str = "records/models/tft_model_quantile.ckpt",
                                     forecast_days: int = 84) -> np.ndarray:
    """
    使用TFT分位数模型生成日度需求预测
    
    Args:
        df: 原始DataFrame（历史销售数据）
        sku_list: SKU列表
        store_list: 门店列表
        tft_checkpoint_path: TFT模型检查点路径
        forecast_days: 预测天数
    
    Returns:
        日度预测数组 (forecast_days, len(sku_list))
    """
    logger.info("=" * 70)
    logger.info("  使用TFT分位数模型生成需求预测")
    logger.info("=" * 70)
    
    # 1. 加载TFT模型
    logger.info(f"  加载TFT模型: {tft_checkpoint_path}")
    try:
        integration = create_tft_drl_integration(
            checkpoint_path=tft_checkpoint_path,
            num_skus=len(sku_list)  # 传入正确的 SKU 数量
        )
        logger.info(f"  ✓ TFT模型加载成功")
        logger.info(f"    - Loss type: {getattr(integration.tft.model_config, 'loss_type', 'N/A')}")
        logger.info(f"    - Quantiles: {getattr(integration.tft.model_config, 'quantiles', 'N/A')}")
    except Exception as e:
        logger.error(f"  ✗ TFT模型加载失败: {e}")
        logger.info("  回退到简化预测方法")
        return generate_daily_forecast(df, sku_list, store_list, forecast_days)
    
    # 2. 生成预测
    logger.info(f"  生成预测（{forecast_days}天）...")
    try:
        forecasts_df = integration.prepare_forecasts_for_ppo(
            historical_data=df,
            num_steps=forecast_days
        )
        logger.info(f"  ✓ 预测生成成功: {len(forecasts_df)} 条记录")
    except Exception as e:
        logger.error(f"  ✗ 预测生成失败: {e}")
        logger.error(traceback.format_exc())
        logger.info("  回退到简化预测方法")
        return generate_daily_forecast(df, sku_list, store_list, forecast_days)
    
    # 3. 转换为日度预测数组格式 (forecast_days, num_skus)
    logger.info("  转换预测格式...")
    daily_forecast = np.zeros((forecast_days, len(sku_list)))
    
    sku_to_idx = {str(sku): idx for idx, sku in enumerate(sku_list)}
    
    # 统计匹配的 SKU 数量
    matched_skus = 0
    unmatched_skus = []
    
    for day in range(forecast_days):
        day_data = forecasts_df[forecasts_df['step'] == day]
        for _, row in day_data.iterrows():
            sku = row['sku_id']
            
            # 尝试多种匹配方式
            matched = False
            
            # 方式1: 直接匹配（字符串）
            sku_str = str(sku)
            if sku_str in sku_to_idx:
                idx = sku_to_idx[sku_str]
                daily_forecast[day, idx] = row['demand_median']
                matched_skus = max(matched_skus, idx + 1)
                matched = True
            
            # 方式2: 尝试整数匹配
            if not matched and str(sku).isdigit():
                sku_int = int(sku)
                if sku_int in sku_to_idx:
                    idx = sku_to_idx[sku_int]
                    daily_forecast[day, idx] = row['demand_median']
                    matched_skus = max(matched_skus, idx + 1)
                    matched = True
                elif str(sku_int) in sku_to_idx:
                    idx = sku_to_idx[str(sku_int)]
                    daily_forecast[day, idx] = row['demand_median']
                    matched_skus = max(matched_skus, idx + 1)
                    matched = True
            
            # 方式3: 尝试去除前缀匹配
            if not matched:
                import re
                match = re.search(r'(\d+)', str(sku))
                if match:
                    sku_num = match.group(1)
                    for key_format in ['{:0' + str(len(sku_num)) + 'd}', '{:d}']:
                        key = key_format.format(int(sku_num))
                        if key in sku_to_idx:
                            idx = sku_to_idx[key]
                            daily_forecast[day, idx] = row['demand_median']
                            matched_skus = max(matched_skus, idx + 1)
                            matched = True
                            break
            
            if not matched:
                unmatched_skus.append(str(sku))
    
    # 输出匹配统计
    if unmatched_skus:
        unique_unmatched = list(set(unmatched_skus))
        logger.warning(f"⚠️ 维度警告：{len(unique_unmatched)} 个 SKU 在预测数据中未找到匹配！")
        logger.warning(f"    未匹配的 SKU ID 示例: {unique_unmatched[:5]}")
        logger.warning(f"    sku_list 示例: {list(sku_to_idx.keys())[:5]}")
    else:
        logger.info(f"  ✓ 所有 SKU 匹配成功！")
    
    assert daily_forecast.shape[1] == len(sku_list), \
        f"维度不匹配：daily_forecast.shape[1]={daily_forecast.shape[1]}, len(sku_list)={len(sku_list)}"
    
    logger.info(f"  ✓ 预测格式转换完成: {daily_forecast.shape}")
    logger.info(f"  预测统计: mean={np.mean(daily_forecast):.2f}, "
                 f"std={np.std(daily_forecast):.2f}, "
                 f"min={np.min(daily_forecast):.2f}, "
                 f"max={np.max(daily_forecast):.2f}")
    
    return daily_forecast


def create_env(env_class, daily_forecast: np.ndarray, sku_list: list,
              store_list: list, config: Optional[EnvConfig] = None,
              initial_inventory: Optional[np.ndarray] = None) -> Any:
    """
    创建供应链环境
    
    Args:
        env_class: 环境类（如SupplyChainEnv）
        daily_forecast: 日度预测数组 (days, num_skus)
        sku_list: SKU列表（仅用于推断 num_skus）
        store_list: 门店列表（仅用于推断 num_stores）
        config: 环境配置
        initial_inventory: 初始库存数据 (num_skus,), 如果提供则使用真实数据
    
    Returns:
        包装后的向量化环境
    """
    if config is None:
        from config import env_config
        config = env_config
    
    num_skus = len(sku_list)
    num_stores = len(store_list)
    
    logger.info(f"创建环境: {env_class.__name__} (num_skus={num_skus}, num_stores={num_stores})")
    
    def _init():
        env = env_class(
            num_skus=num_skus,
            num_stores=num_stores,
            demand_forecasts=daily_forecast,
            initial_inventory=initial_inventory,
            sku_list=sku_list,
            store_list=store_list,
        )
        env = Monitor(env, filename=None)
        return env
    
    # 向量化环境
    vec_env = DummyVecEnv([_init])
    
    # Wrap with VecNormalize
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
        epsilon=1e-8,
    )
    
    logger.info(f"环境创建完成: obs_space={vec_env.observation_space.shape}, act_space={vec_env.action_space.shape}")
    logger.info(f"  VecNormalize enabled: norm_obs={vec_env.norm_obs}, norm_reward={vec_env.norm_reward}")
    return vec_env
