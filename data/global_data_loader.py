"""
全局数据加载器（Global Data Loader）
用于支持82门店大一统PPO模型训练

核心特性：
1. 内存哈希字典（极速查询，避免CSV/DataFrame慢查询）
2. 门店画像（Store Profile）支持
3. 需求数据字典（按门店+SKU+周索引）
4. 库存数据字典（按门店+SKU+周索引）
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


class GlobalDataLoader:
    """
    全局数据加载器
    
    功能：
    1. 加载82门店 × 94 SKU的全量数据
    2. 构建内存哈希字典（极速查询）
    3. 提供门店画像（Store Profile）
    4. 支持按门店+SKU+周索引查询需求和库存
    """
    
    def __init__(
        self,
        demand_path: str = "data/processed/estimated_demand_weekly.csv",  # 🔧 改用周粒度数据
        shop_stock_path: str = "data/processed/real_shop_stock_weekly_full.csv",  # 🔧 使用全量库存数据（覆盖率100%）
        dc_stock_path: str = "data/processed/real_dc_stock_intersect.csv",
        store_profiles_path: Optional[str] = None,
    ):
        """
        初始化全局数据加载器
        
        Args:
            demand_path: 需求数据路径（周粒度，estimated_demand_weekly.csv）
            shop_stock_path: 门店库存数据路径（周粒度，real_shop_stock_weekly.csv）
            dc_stock_path: 仓库库存数据路径
            store_profiles_path: 门店画像文件路径（可选）
        """
        logger.info("=" * 80)
        logger.info("GlobalDataLoader 初始化")
        logger.info("=" * 80)
        
        # 1. 加载需求数据（周粒度）
        self.demand_dict = self._load_demand_data(demand_path)
        
        # 2. 加载门店库存数据（周粒度）
        self.shop_stock_dict = self._load_shop_stock_data(shop_stock_path)
        
        # 3. 加载仓库库存数据
        self.dc_stock_dict = self._load_dc_stock_data(dc_stock_path)
        
        # 3.5 加载SKU单价（从仓库库存数据的移动平均价）
        self.sku_prices_dict = self._load_sku_prices(dc_stock_path)
        
        # 3.7 提取有效的门店和SKU列表（确保是字符串类型）
        # 🌟 架构师强制要求：必须在调用 _init_stock_cache() 之前计算 valid_skus
        self.valid_stores = sorted([str(s) for s in self.demand_dict.keys()])
        self.valid_skus = [str(s) for s in self._extract_valid_skus()]
        logger.info(f"  ✓ 提前计算 valid_skus: {len(self.valid_skus)} 个")
        
        # 4. 构建初始库存缓存（内存计算，不依赖文件）
        self._init_stock_cache()
        
        # 5. 加载或生成门店画像
        self.store_profiles = self._load_or_generate_store_profiles(
            store_profiles_path, demand_path
        )
        
        # 6. 提取有效的门店和SKU列表（确保是字符串类型，并去除 .0 后缀）
        self.valid_stores = sorted([str(s).replace('.0', '') if '.0' in str(s) else str(s) for s in self.demand_dict.keys()])
        self.valid_skus = [s.replace('.0', '') if '.0' in s else s for s in self._extract_valid_skus()]
        
        # 🌟 验证缓存与 valid_skus 对齐
        cache_skus = set()
        for (store_id, sku_id) in self._initial_stock_cache:
            cache_skus.add(sku_id)
        
        valid_skus_set = set(self.valid_skus)
        if not valid_skus_set.issubset(cache_skus):
            missing = valid_skus_set - cache_skus
            raise ValueError(
                f"❌ 初始库存缓存缺失部分 valid_skus!\n"
                f"缺失的 SKU: {missing}"
            )
        
        logger.info(f"  ✓ 缓存与 valid_skus 对齐验证通过")
        logger.info(f"\n✅ GlobalDataLoader 初始化完成")
        logger.info(f"  - 门店数: {len(self.valid_stores)}")
        logger.info(f"  - SKU数: {len(self.valid_skus)}")
        logger.info(f"  - 需求数据: {len(self.demand_dict)} 个门店")
        logger.info(f"  - 库存数据: {len(self.shop_stock_dict)} 个门店")
        logger.info(f"  - 初始库存缓存: {len(self._initial_stock_cache)} 个 (门店, SKU) 组合")
        logger.info(f"  - 门店画像: {len(self.store_profiles)} 个门店")
    
    def _sanitize_ids(self, df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
        """
        架构师全局类型净化：将所有 ID 列强制转换为纯净的字符串
        消除 int64, float64 ('.0'), 以及首尾空格的干扰
        
        这是一个"降维打击"式的修复，确保后续所有 merge/query 都不会因格式不匹配而失败。
        """
        for col in id_cols:
            if col in df.columns:
                df[col] = (df[col]
                       .astype(str)
                       .str.replace('.0', '', regex=False)
                       .str.strip())
        return df
    
    def _load_demand_data(self, demand_path: str) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        加载需求数据，构建哈希字典
        
        返回结构: {store_id: {sku_id: {week_index: demand_value}}}
        """
        logger.info(f"\n[1/5] 加载需求数据: {demand_path}")
        
        if not os.path.exists(demand_path):
            raise FileNotFoundError(f"需求数据文件不存在: {demand_path}")
        
        df = pd.read_csv(demand_path, encoding='utf-8')
        df = self._sanitize_ids(df, ['shop_code', 'goods_code'])  # 🌟 架构师全局净化
        logger.info(f"  ✓ 需求数据: {len(df):,} 行")
        logger.info(f"  - 列名: {df.columns.tolist()}")
        
        # 构建哈希字典
        demand_dict = defaultdict(lambda: defaultdict(dict))
        
        for _, row in df.iterrows():
            store_id = str(row['shop_code'])
            sku_id = str(row['goods_code'])
            
            # 获取周索引
            if 'week_index' in df.columns:
                week_idx = int(row['week_index'])
            elif 'ds' in df.columns:
                # 从日期列计算周索引
                date = pd.to_datetime(row['ds'])
                week_idx = date.isocalendar()[1] + (date.isocalendar()[0] - 2024) * 52
            else:
                week_idx = 0
            
            # 获取需求值
            if 'estimated_demand' in df.columns:
                demand_value = float(row['estimated_demand'])
            elif 'sale_qty' in df.columns:
                demand_value = float(row['sale_qty'])
            else:
                demand_value = 0.0
            
            demand_dict[store_id][sku_id][week_idx] = demand_value
        
        logger.info(f"  ✓ 哈希字典构建完成: {len(demand_dict)} 个门店")
        
        return dict(demand_dict)
    
    def _load_shop_stock_data(self, shop_stock_path: str) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        加载门店库存数据，构建哈希字典
        
        返回结构: {store_id: {sku_id: {week_index: stock_value}}}
        """
        logger.info(f"\n[2/5] 加载门店库存数据: {shop_stock_path}")
        
        if not os.path.exists(shop_stock_path):
            logger.warning(f"  门店库存数据文件不存在: {shop_stock_path}")
            return {}
        
        df = pd.read_csv(shop_stock_path, encoding='utf-8')
        df = self._sanitize_ids(df, ['shop_code', 'goods_code'])  # 💥 架构师全局净化
        logger.info(f"  ✓ 门店库存数据: {len(df):,} 行")
        
        # 构建哈希字典
        stock_dict = defaultdict(lambda: defaultdict(dict))
        
        for _, row in df.iterrows():
            store_id = str(row['shop_code'])
            sku_id = str(row['goods_code'])
            
            # 获取周索引
            if 'dt' in df.columns:
                date = pd.to_datetime(row['dt'])
                week_idx = date.isocalendar()[1] + (date.isocalendar()[0] - 2024) * 52
            else:
                week_idx = 0
            
            # 获取库存值
            if 'stock_qty' in df.columns:
                stock_value = float(row['stock_qty'])
            else:
                stock_value = 0.0
            
            stock_dict[store_id][sku_id][week_idx] = stock_value
        
        logger.info(f"  ✓ 哈希字典构建完成: {len(stock_dict)} 个门店")
        
        return dict(stock_dict)
    
    def _load_dc_stock_data(self, dc_stock_path: str) -> Dict[str, Dict[int, float]]:
        """
        加载仓库库存数据，构建哈希字典
        
        返回结构: {sku_id: {week_index: stock_value}}
        """
        logger.info(f"\n[3/5] 加载仓库库存数据: {dc_stock_path}")
        
        if not os.path.exists(dc_stock_path):
            logger.warning(f"  仓库库存数据文件不存在: {dc_stock_path}")
            return {}
        
        df = pd.read_csv(dc_stock_path, encoding='utf-8')
        df = self._sanitize_ids(df, ['goods_code'])  # 💥 架构师全局净化
        logger.info(f"  ✓ 仓库库存数据: {len(df):,} 行")
        
        # 构建哈希字典
        dc_stock_dict = defaultdict(dict)
        
        for _, row in df.iterrows():
            sku_id = str(row['goods_code'])
            
            # 获取周索引
            if 'dt' in df.columns:
                date = pd.to_datetime(row['dt'])
                week_idx = date.isocalendar()[1] + (date.isocalendar()[0] - 2024) * 52
            else:
                week_idx = 0
            
            # 获取库存值
            if 'stock_qty' in df.columns:
                stock_value = float(row['stock_qty'])
            else:
                stock_value = 0.0
            
            dc_stock_dict[sku_id][week_idx] = stock_value
        
        logger.info(f"  ✓ 哈希字典构建完成: {len(dc_stock_dict)} 个SKU")
        
        return dict(dc_stock_dict)
    
    def _init_stock_cache(self):
        """
        从 initial_stock_by_store.csv 构建初始库存缓存
        
        逻辑：
        1. 从 initial_stock_by_store.csv 中读取初始库存（覆盖率95.21%）
        2. 若缺失，用该 SKU 的周平均需求 × 2 估算
        3. 构建缓存字典，key 为 (store_id, sku_id)
        """
        logger.info(f"\n[4/5] 构建初始库存缓存（从 initial_stock_by_store.csv）")
        
        # 1. 读取初始库存文件
        init_stock_path = "data/processed/initial_stock_by_store.csv"
        if not os.path.exists(init_stock_path):
            raise FileNotFoundError(f"初始库存文件不存在: {init_stock_path}")
        
        init_stock_df = pd.read_csv(init_stock_path, encoding='utf-8')
        init_stock_df = self._sanitize_ids(init_stock_df, ['shop_code', 'goods_code'])  # 💥 架构师全局净化
        logger.info(f"  ✓ 初始库存文件: {len(init_stock_df):,} 行")
        
        # 只保留第0周的数据
        init_stock_df = init_stock_df[init_stock_df['week_index'] == 0].copy()
        init_stock_df = init_stock_df[['shop_code', 'goods_code', 'initial_stock']]
        init_stock_df['initial_stock'] = init_stock_df['initial_stock'].fillna(0)
        init_stock_df['initial_stock'] = init_stock_df['initial_stock'].astype(float)
        
        logger.info(f"  ✓ 第0周初始库存: {len(init_stock_df)} 行")
        
        # 2. 获取所有 (门店, SKU) 组合（从需求数据中提取）
        all_combos = []
        for store_id in self.demand_dict:
            for sku_id in self.demand_dict[store_id]:
                all_combos.append((str(store_id), str(sku_id)))
        
        all_combos = pd.DataFrame(all_combos, columns=['shop_code', 'goods_code'])
        logger.info(f"  ✓ 所有 (门店, SKU) 组合: {len(all_combos)} 个")
        
        # 3. 左连接，确保所有组合都有库存值（缺失则用0）
        merged = all_combos.merge(init_stock_df, on=['shop_code', 'goods_code'], how='left')
        merged['initial_stock'] = merged['initial_stock'].fillna(0)
        logger.info(f"  ✓ 合并后数据: {len(merged)} 行")
        
        # 4. 计算每个 SKU 的周平均需求（用于估算）
        self.avg_demand = {}
        for store_id in self.demand_dict:
            for sku_id in self.demand_dict[store_id]:
                demand_values = list(self.demand_dict[store_id][sku_id].values())
                if demand_values:
                    self.avg_demand[sku_id] = np.mean(demand_values)
        
        logger.info(f"  ✓ 平均需求计算完成: {len(self.avg_demand)} 个 SKU")
        
        # 5. 构建缓存
        cache = {}
        real_count = 0
        estimated_count = 0
        zero_count = 0
        
        for _, row in merged.iterrows():
            store_id = str(row['shop_code'])
            sku_id = str(row['goods_code'])
            
            # 🔧 修复：去除 .0 后缀（如果是浮点数格式）
            if '.0' in store_id:
                store_id = store_id.replace('.0', '')
            if '.0' in sku_id:
                sku_id = sku_id.replace('.0', '')
            
            stock = float(row['initial_stock'])
            
            if stock == 0:
                # 若库存为0或缺失，用平均需求×2估算（2周安全库存）
                stock = self.avg_demand.get(sku_id, 5.0) * 2.0
                estimated_count += 1
            else:
                real_count += 1
            
            if stock == 0:
                zero_count += 1
            
            cache[(store_id, sku_id)] = float(stock)
        
        self._initial_stock_cache = cache
        
        logger.info(f"  ✓ 初始库存缓存构建完成:")
        logger.info(f"    - 总组合数: {len(cache)}")
        logger.info(f"    - 使用真实库存: {real_count} 个")
        logger.info(f"    - 使用估算值: {estimated_count} 个")
        logger.info(f"    - 零值数量: {zero_count} 个")
    
    def _load_or_generate_store_profiles(
        self, 
        store_profiles_path: Optional[str],
        demand_path: str
    ) -> Dict[str, Dict[str, float]]:
        """
        加载或生成门店画像
        
        门店画像包含：
        - scale: 规模（历史平均销量归一化到0-1）
        - type: 类型（商圈=1, 社区=0, 医院周边=2）
        - avg_daily_sales: 平均日销量
        """
        logger.info(f"\n[5/5] 加载或生成门店画像")
        
        if store_profiles_path and os.path.exists(store_profiles_path):
            # 从文件加载
            df = pd.read_csv(store_profiles_path, encoding='utf-8')
            df = self._sanitize_ids(df, ['shop_code'])  # 💥 架构师全局净化
            logger.info(f"  ✓ 从文件加载门店画像: {len(df)} 个门店")
            return df.set_index('shop_code').to_dict('index')
        
        else:
            # 从需求数据生成（读取完整文件）
            logger.info(f"  ⚠️ 门店画像文件不存在，从需求数据生成")
            
            # 读取完整文件（不使用nrows限制）
            df = pd.read_csv(demand_path, encoding='utf-8')
            df = self._sanitize_ids(df, ['shop_code', 'goods_code'])  # 💥 架构师全局净化
            
            # 计算门店规模（历史平均销量）
            store_avg_sales = df.groupby('shop_code')['sale_qty'].mean()
            max_avg_sales = store_avg_sales.max()
            store_scale = store_avg_sales / (max_avg_sales + 1e-8)
            
            # 生成门店画像
            store_profiles = {}
            for store_id in store_avg_sales.index:
                store_profiles[str(store_id)] = {
                    'scale': float(store_scale[store_id]),
                    'type': 1,  # 默认商圈类型
                    'avg_daily_sales': float(store_avg_sales[store_id]),
                }
            
            logger.info(f"  ✓ 门店画像生成完成: {len(store_profiles)} 个门店")
            
            return store_profiles
    
    def _extract_valid_skus(self) -> List[str]:
        """提取所有门店共有的SKU列表"""
        if not self.demand_dict:
            return []
        
        # 获取第一个门店的SKU列表
        first_store = list(self.demand_dict.keys())[0]
        valid_skus = set(self.demand_dict[first_store].keys())
        
        # 与其他门店取交集
        for store_id in self.demand_dict:
            valid_skus &= set(self.demand_dict[store_id].keys())
        
        # 💥 架构师全局净化：确保返回的SKU列表是纯净的
        valid_skus = [str(s).replace('.0', '').strip() for s in valid_skus]
        
        return sorted(list(valid_skus))
    
    def get_demand(self, store_id: str, sku_id: str, week_index: int) -> float:
        """
        极速查询：获取指定门店+SKU+周的需求
        
        使用哈希字典，时间复杂度 O(1)
        """
        if store_id not in self.demand_dict:
            return 0.0
        
        if sku_id not in self.demand_dict[store_id]:
            return 0.0
        
        return self.demand_dict[store_id][sku_id].get(week_index, 0.0)
    
    def get_initial_state(self, store_id: str, week_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定门店的初始状态（用于reset()）
        
        返回：
        - initial_stock: 门店库存 (shape: [num_skus])
        - transit_stock: 在途库存 (shape: [num_skus])
        
        🌟 使用内存缓存查询，不依赖文件
        """
        # 🔧 修复：确保 store_id 是字符串类型
        store_id_str = str(int(float(store_id))) if '.' in str(store_id) else str(store_id)
        
        # 从内存缓存中查询
        initial_stock = np.zeros(len(self.valid_skus))
        transit_stock = np.zeros(len(self.valid_skus))  # 初始在途库存为0
        
        cache_miss = 0
        for i, sku_id in enumerate(self.valid_skus):
            key = (store_id_str, str(sku_id))
            
            if key in self._initial_stock_cache:
                initial_stock[i] = self._initial_stock_cache[key]
            else:
                # 理论上不会发生，因为 _init_stock_cache() 已确保所有组合都有值
                cache_miss += 1
                # 使用估算值兜底
                avg_demand = 5.0
                if store_id_str in self.demand_dict:
                    if sku_id in self.demand_dict[store_id_str]:
                        demand_values = list(self.demand_dict[store_id_str][sku_id].values())
                        if demand_values:
                            avg_demand = np.mean(demand_values)
                
                initial_stock[i] = avg_demand * 2.0  # 2周安全库存
                print(f"[Warning] Cache miss: Store {store_id_str} SKU {sku_id}. Using fallback: {initial_stock[i]}")
        
        if cache_miss > 0:
            logger.warning(f"⚠️ 门店 {store_id_str} 有 {cache_miss} 个 SKU 缓存未命中，已使用估算值兜底")
        
        return initial_stock, transit_stock
    
    def get_store_profile(self, store_id: str) -> Dict[str, float]:
        """获取门店画像"""
        return self.store_profiles.get(store_id, {'scale': 0.5, 'type': 0, 'avg_daily_sales': 1.0})
    
    def get_all_store_ids(self) -> List[str]:
        """
        获取所有门店ID列表
        
        从 _initial_stock_cache 字典的keys中提取唯一的门店ID。
        
        返回:
            List[str]: 所有门店ID的列表（已排序）
        """
        if not hasattr(self, '_initial_stock_cache'):
            logger.warning("⚠️ _initial_stock_cache 不存在，无法获取门店ID列表")
            return []
        
        # 从缓存keys中提取唯一门店ID
        store_ids = set()
        for (store_id, sku_id) in self._initial_stock_cache.keys():
            store_ids.add(store_id)
        
        store_ids_sorted = sorted(list(store_ids))
        logger.info(f"  ✓ 提取到 {len(store_ids_sorted)} 个唯一门店ID")
        return store_ids_sorted
    
    # ==========================================
    # 🌟 架构师特供：数据阵列化翻译官
    # ==========================================
    
    def get_historical_demand_array(self, store_id: str, target_skus: list, time_steps: int = 52) -> np.ndarray:
        """
        🌟 架构师特供：将历史需求转换为 (T, num_skus) 的 2D 数组
        
        把嵌套字典/DataFrame 翻译成环境需要的矩阵。
        
        参数:
            store_id: 门店ID
            target_skus: 环境使用的 SKU 列表 (必须是 valid_skus)
            time_steps: 历史时间步长度 (默认 52 周)
        
        返回:
            demand_matrix: shape 为 (time_steps, len(target_skus)) 的 numpy 数组
        """
        num_skus = len(target_skus)
        demand_matrix = np.zeros((time_steps, num_skus), dtype=np.float32)
        
        # 检查 store_id 是否存在
        if store_id not in self.demand_dict:
            logger.warning(f"  ⚠️ 门店 {store_id} 不在 demand_dict 中！使用全局均值填充")
            # 使用全局均值填充
            for sku_idx, sku_id in enumerate(target_skus):
                avg_demand = self.avg_demand.get(sku_id, 5.0)
                demand_matrix[:, sku_idx] = avg_demand
            return demand_matrix
        
        store_data = self.demand_dict[store_id]
        
        # 遍历每个 SKU，填充需求数据
        for sku_idx, sku_id in enumerate(target_skus):
            if sku_id in store_data:
                sku_data = store_data[sku_id]
                
                # 填充数据（只填充存在的 week_idx）
                filled_count = 0
                for week_idx in range(time_steps):
                    if week_idx in sku_data:
                        demand_matrix[week_idx, sku_idx] = sku_data[week_idx]
                        filled_count += 1
                
                # 如果没有任何数据，使用全局均值
                if filled_count == 0:
                    avg_demand = self.avg_demand.get(sku_id, 5.0)
                    demand_matrix[:, sku_idx] = avg_demand
                    logger.debug(f"    SKU {sku_id}: 无历史数据，使用全局均值 {avg_demand:.2f}")
            else:
                # SKU 不在门店数据中，使用全局均值
                avg_demand = self.avg_demand.get(sku_id, 5.0)
                demand_matrix[:, sku_idx] = avg_demand
                logger.debug(f"    SKU {sku_id}: 不在门店数据中，使用全局均值 {avg_demand:.2f}")
        
        # 🚨 架构师防线：绝不允许全零数据流入！
        if np.max(demand_matrix) < 1e-5:
            logger.warning(f"  ⚠️ 门店 {store_id} 的需求矩阵全为 0！使用全局均值填充")
            for sku_idx, sku_id in enumerate(target_skus):
                avg_demand = self.avg_demand.get(sku_id, 5.0)
                demand_matrix[:, sku_idx] = avg_demand
        
        logger.info(f"  ✓ 历史需求阵列化完成: shape={demand_matrix.shape}, 均值={np.mean(demand_matrix):.2f}")
        return demand_matrix

    def _load_sku_prices(self, dc_stock_path: str) -> Dict[str, float]:
        """
        从仓库库存数据加载SKU单价（移动平均价）
        
        数据来源：real_dc_stock_intersect.csv 中的 current_move_avg_price 列
        取每个SKU最新日期的价格
        
        返回：{sku_id: price} 字典
        """
        logger.info(f"\n[3.6] 加载SKU单价（移动平均价）")
        
        if not os.path.exists(dc_stock_path):
            logger.warning(f"  仓库库存数据文件不存在: {dc_stock_path}，使用默认单价1.0")
            return {}
        
        try:
            df = pd.read_csv(dc_stock_path, encoding='utf-8')
            df = self._sanitize_ids(df, ['goods_code'])  # 🌟 架构师全局净化
            
            # 取每个SKU最新日期的价格
            df['dt'] = pd.to_datetime(df['dt'], format='%Y%m%d')
            latest_date = df['dt'].max()
            df_latest = df[df['dt'] == latest_date].copy()
            
            # 构建价格字典
            prices_dict = {}
            for _, row in df_latest.iterrows():
                sku_id = str(row['goods_code'])
                price = float(row['current_move_avg_price'])
                # 🌟 脏数据防御：确保价格合理
                if price <= 0:
                    price = 10.0  # 默认10元
                prices_dict[sku_id] = price
            
            logger.info(f"  ✓ SKU单价加载完成: {len(prices_dict)} 个SKU")
            logger.info(f"  - 最新日期: {latest_date}")
            logger.info(f"  - 价格均值: {np.mean(list(prices_dict.values())):.2f}")
            logger.info(f"  - 价格范围: {min(prices_dict.values()):.2f} ~ {max(prices_dict.values()):.2f}")
            
            return prices_dict
            
        except Exception as e:
            logger.warning(f"  ⚠️ 加载SKU单价失败: {e}，使用默认单价1.0")
            return {}
    
    def get_sku_prices(self, valid_skus: List[str]) -> np.ndarray:
        """
        获取指定SKU的单价数组
        
        Args:
            valid_skus: SKU ID列表
        
        Returns:
            shape为 (len(valid_skus),) 的numpy数组，每个元素是该SKU的单价
        """
        prices = []
        default_price = 10.0  # 默认单价10元
        
        for sku_id in valid_skus:
            if sku_id in self.sku_prices_dict:
                price = self.sku_prices_dict[sku_id]
                # 🌟 脏数据防御
                if price <= 0:
                    price = default_price
            else:
                # 如果SKU不在价格字典中，使用均值或默认值
                if self.sku_prices_dict:
                    price = np.mean(list(self.sku_prices_dict.values()))
                else:
                    price = default_price
                logger.debug(f"  SKU {sku_id} 无单价数据，使用默认值 {price:.2f}")
            
            prices.append(price)
        
        return np.array(prices, dtype=np.float32)
