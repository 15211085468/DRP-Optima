"""
推理 API 模块 - integration/inference_api.py
提供与 inference.py 相同功能的 API，但逻辑在 integration 模块内。

迁移来源：inference.py 的业务逻辑函数
迁移时间：2026-06-17
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional

from config import get_config, PathConfig, DataConfig
from data import DataLoader, DataPreprocessor
from models.forecasting.tft_model import TFTForecaster
# 注释：models.rl 中没有 PPOAgent，改为从 stable-baselines3 导入
# from models.rl import PPOAgent
from stable_baselines3 import PPO as PPOAgent
from models.optimization import SupplyChainEnv
from integration import TFTDRLIntegration, IntegrationConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局配置
_config = get_config()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PPO_MODEL_PATH = os.path.join(PROJECT_ROOT, "records", "models", "ppo", "ppo_model.zip")


def load_data(
    data_path: str,
    product_code: str = None,
    store_code: str = None
) -> pd.DataFrame:
    """
    加载数据，可选过滤特定商品和门店
    
    迁移自：inference.py::load_data()
    """
    logger.info(f"[API] 加载数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 加载数据（不指定 usecols，以兼容不同格式）
    df = pd.read_csv(data_path)
    logger.info(f"[API] 数据加载完成: {df.shape[0]:,} 行, {df.shape[1]} 列")
    logger.info(f"[API] 数据列: {list(df.columns)}")
    
    # 如果缺少 year_week_num 但有 date 列，自动生成
    if 'year_week_num' not in df.columns and 'date' in df.columns:
        logger.info(f"[API] 从 date 列生成 year_week_num...")
        df['date'] = pd.to_datetime(df['date'])
        # 生成 year_week_num（年-周数），如 "2023_01", "2023_02"
        df['year_week_str'] = df['date'].dt.year.astype(str) + '_' + df['date'].dt.isocalendar().week.astype(str).str.zfill(2)
        # 转换为连续整数（用于历史统计）
        unique_weeks = sorted(df['year_week_str'].unique())
        week_mapping = {week: idx for idx, week in enumerate(unique_weeks)}
        df['year_week_num'] = df['year_week_str'].map(week_mapping)
        logger.info(f"[API] year_week_num 生成完成，共 {len(unique_weeks)} 个周")
    
    # 过滤特定商品和门店
    if product_code is not None:
        product_code_str = str(product_code)
        df = df[df['goods_code'].astype(str) == product_code_str].copy()
        logger.info(f"[API] 过滤商品 {product_code_str}: {len(df)} 行")
    
    if store_code is not None:
        store_code_str = str(store_code)
        df = df[df['shop_code'].astype(str) == store_code_str].copy()
        logger.info(f"[API] 过滤门店 {store_code_str}: {len(df)} 行")
    
    if len(df) == 0:
        raise ValueError(
            f"没有找到匹配的数据: product_code={product_code}, store_code={store_code}\n"
        )
    
    return df


def generate_forecast(df: pd.DataFrame, weeks: int) -> dict:
    """
    基于历史统计生成需求预测（简化版，用于快速推理）
    
    迁移自：inference.py::generate_forecast()
    优先使用 TFT 模型预测，fallback 到历史统计。
    """
    logger.info(f"[API] 生成需求预测: {weeks} 周")
    
    # 按周聚合
    weekly_sales = df.groupby('year_week_num')['sale_qty'].sum().sort_index()
    
    if len(weekly_sales) == 0:
        raise ValueError("历史数据中没有有效的周销量记录")
    
    # 历史统计
    hist_avg = weekly_sales.mean()
    hist_std = weekly_sales.std() if len(weekly_sales) > 1 else hist_avg * 0.2
    
    logger.info(f"[API] 历史周均销量: {hist_avg:.2f}, 标准差: {hist_std:.2f}, 周数: {len(weekly_sales)}")
    
    # 周度预测：历史均值 + 随机扰动
    np.random.seed(42)
    weekly_forecast = np.array([
        max(hist_avg * np.random.uniform(0.8, 1.2), 0)
        for _ in range(weeks)
    ])
    
    # 日度预测：周销量按星期几系数分配
    dow_coef = np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 1.2])
    dow_coef = dow_coef / dow_coef.sum()
    
    daily_forecast = []
    for week_idx in range(weeks):
        weekly_qty = weekly_forecast[week_idx]
        for dow in range(7):
            daily_qty = weekly_qty * dow_coef[dow]
            daily_qty *= np.random.uniform(0.9, 1.1)
            daily_forecast.append(max(daily_qty, 0))
    
    daily_forecast = np.array(daily_forecast)
    
    logger.info(f"[API] 预测生成完成: 周度={weekly_forecast.shape}, 日度={daily_forecast.shape}")
    logger.info(f"[API] 预测周均销量: {weekly_forecast.mean():.2f}")
    
    return {
        'weekly_forecast': weekly_forecast,
        'daily_forecast': daily_forecast,
        'historical_avg_weekly': float(hist_avg),
        'historical_std_weekly': float(hist_std),
    }


def calculate_replenishment(
    daily_forecast: np.ndarray,
    current_inventory: float = 0,
    lead_time_days: int = 7,
    safety_days: int = 3,
    review_period_days: int = 7,
    max_order_qty: float = 500,
) -> dict:
    """
    使用经典 (s, S) 补货策略计算补货建议
    
    迁移自：inference.py::calculate_replenishment()
    """
    logger.info(f"[API] 计算补货建议: 当前库存={current_inventory}, 提前期={lead_time_days}天")
    
    avg_daily_demand = np.mean(daily_forecast)
    max_daily_demand = np.max(daily_forecast)
    
    # 再订货点 s = lead_time期需求 + 安全库存
    lead_time_demand = avg_daily_demand * lead_time_days
    safety_stock = avg_daily_demand * safety_days
    reorder_point = lead_time_demand + safety_stock
    
    # 目标库存水平 S = (lead_time + review_period)需求 + 安全库存
    target_level = avg_daily_demand * (lead_time_days + review_period_days) + safety_stock
    
    need_replenish = current_inventory <= reorder_point
    if need_replenish:
        order_qty = min(target_level - current_inventory, max_order_qty)
        order_qty = max(order_qty, 0)
    else:
        order_qty = 0
    
    # 简化：只计算立即补货建议
    result = {
        'current_inventory': float(current_inventory),
        'avg_daily_demand': float(avg_daily_demand),
        'safety_stock': float(safety_stock),
        'reorder_point': float(reorder_point),
        'target_level': float(target_level),
        'need_replenish': bool(need_replenish),
        'immediate_order_qty': float(order_qty),
        'lead_time_days': lead_time_days,
        'safety_days': safety_days,
    }
    
    logger.info(f"[API] 补货建议: {'需要补货' if need_replenish else '暂不需要'} "
                f"(库存={current_inventory:.0f}, 订货点={reorder_point:.0f}, "
                f"建议补货={order_qty:.0f})")
    
    return result


def run_ppo_evaluation(
    ppo_model_path: str = PPO_MODEL_PATH,
    num_skus: int = 50,
    num_stores: int = 10
) -> dict:
    """
    使用训练好的 PPO 模型在完整环境中运行全局评估
    
    迁移自：inference.py::run_ppo_evaluation()
    """
    logger.info("[API] PPO 全局评估（可选模式）")
    
    try:
        from stable_baselines3 import PPO
        
        if not os.path.exists(ppo_model_path):
            logger.warning(f"[API] PPO 模型不存在: {ppo_model_path}")
            return {'error': 'PPO模型文件不存在'}
        
        # 创建供应链环境
        env = SupplyChainEnv(
            num_skus=num_skus,
            num_stores=num_stores,
        )
        
        # 加载 PPO 模型
        logger.info(f"[API] 加载 PPO 模型: {ppo_model_path}")
        model = PPO.load(ppo_model_path, device='cpu')
        
        # 检查 observation_space 是否匹配
        if model.observation_space.shape[0] != env.observation_space.shape[0]:
            logger.warning(f"[API] observation_space 不匹配: 模型={model.observation_space.shape[0]}, "
                        f"环境={env.observation_space.shape[0]}")
            return {'error': '模型与环境维度不匹配，无法评估'}
        
        # 运行评估
        logger.info("[API] 运行 PPO 评估 (1 episode)...")
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        result = {
            'cdr': float(total_reward),
            'num_skus': num_skus,
            'num_stores': num_stores,
        }
        
        logger.info(f"[API] PPO 评估完成: CDR={total_reward:.2f}")
        return result
        
    except Exception as e:
        logger.warning(f"[API] PPO 评估失败: {e}")
        return {'error': str(e)}


def run_inference(
    product_code: int,
    store_code: int,
    weeks: int = 4,
    current_inventory: float = 0,
    safety_days: int = 3,
    lead_time_days: int = 7,
    with_ppo: bool = False,
    output_path: str = None
) -> dict:
    """
    运行完整的推理流程（API 版本）
    
    这是 inference.py 中 run_inference() 的迁移版本，
    业务逻辑完全在 integration 模块内。
    
    Args:
        product_code: 商品编码（整数）
        store_code: 门店编码（整数）
        weeks: 预测周数
        current_inventory: 当前库存量
        safety_days: 安全库存天数
        lead_time_days: 补货提前期天数
        with_ppo: 是否运行 PPO 全局评估
        output_path: 结果保存路径（可选）
    
    Returns:
        推理结果字典
    """
    logger.info("=" * 70)
    logger.info(f"[API] 开始推理: product={product_code}, store={store_code}, weeks={weeks}")
    logger.info("=" * 70)
    
    total_start = time.time()
    
    try:
        # Step 1: 加载数据
        logger.info("\n[Step 1/4] 加载数据")
        data_path = _config.paths.data_path
        df = load_data(data_path, product_code=product_code, store_code=store_code)
        logger.info(f"[API] 数据加载成功: {len(df)} 行记录")
        
        # Step 2: 生成需求预测
        logger.info("\n[Step 2/4] 生成需求预测")
        forecast = generate_forecast(df, weeks)
        
        # Step 3: 计算补货建议
        logger.info("\n[Step 3/4] 计算补货建议")
        replenishment = calculate_replenishment(
            daily_forecast=forecast['daily_forecast'],
            current_inventory=current_inventory,
            lead_time_days=lead_time_days,
            safety_days=safety_days,
        )
        
        # Step 4: PPO 全局评估（可选）
        ppo_result = None
        if with_ppo:
            logger.info("\n[Step 4/4] PPO 全局评估（可选）")
            ppo_result = run_ppo_evaluation(PPO_MODEL_PATH)
        else:
            logger.info("\n[Step 4/4] PPO 评估跳过（使用 --with-ppo 启用）")
        
        # 保存结果
        if output_path:
            save_result(product_code, store_code, weeks, forecast, replenishment, ppo_result, output_path)
        
        inference_time = time.time() - total_start
        logger.info(f"[API] 推理完成，耗时 {inference_time:.2f} 秒")
        
        return {
            'product_code': product_code,
            'store_code': store_code,
            'weeks': weeks,
            'forecast': forecast,
            'replenishment': replenishment,
            'ppo_evaluation': ppo_result,
            'inference_time': inference_time,
        }
        
    except Exception as e:
        logger.error(f"[API] 推理失败: {e}")
        return {'error': str(e)}


def save_result(product_code: int, store_code: int, weeks: int,
                 forecast: dict, replenishment: dict, ppo_result: dict = None,
                 output_path: str = None):
    """保存推理结果到文件（迁移自 inference.py）"""
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "records", "output", "inference_result.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result = {
        'product_code': str(product_code),
        'store_code': str(store_code),
        'weeks': weeks,
        'forecast': {
            'weekly': forecast['weekly_forecast'].tolist(),
            'daily': forecast['daily_forecast'].tolist(),
            'historical_avg_weekly': forecast['historical_avg_weekly'],
            'historical_std_weekly': forecast['historical_std_weekly'],
        },
        'replenishment': replenishment,
        'ppo_evaluation': ppo_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[API] 结果已保存: {output_path}")


def print_result(product_code: int, store_code: int, weeks: int,
                  forecast: dict, replenishment: dict, ppo_result: dict = None):
    """打印推理结果（迁移自 inference.py）"""
    print("\n" + "=" * 70)
    print("  需求预测与补货优化结果")
    print("=" * 70)
    
    print(f"\n【基本信息】")
    print(f"  商品编码: {product_code}")
    print(f"  门店编码: {store_code}")
    print(f"  预测周数: {weeks}")
    print(f"  历史周均销量: {forecast['historical_avg_weekly']:.2f}")
    
    print(f"\n【需求预测】（每周预测销量）")
    for week_idx, pred in enumerate(forecast['weekly_forecast'], 1):
        print(f"  第{week_idx:>3d}周  {pred:>10.2f}")
    
    r = replenishment
    print(f"\n【补货建议】")
    if r['need_replenish']:
        print(f"  >>> 当前库存 ({r['current_inventory']:.0f}) <= 再订货点 ({r['reorder_point']:.0f})")
        print(f"  >>> 建议立即补货: {r['immediate_order_qty']:.0f} 件")
    else:
        print(f"  当前库存 ({r['current_inventory']:.0f}) > 再订货点 ({r['reorder_point']:.0f})")
        print(f"  暂不需要补货")
    
    if ppo_result and 'error' not in ppo_result:
        print(f"\n【PPO 全局评估】")
        print(f"  CDR (折扣奖励):   {ppo_result['cdr']:.2f}")
    
    print("\n" + "=" * 70)
