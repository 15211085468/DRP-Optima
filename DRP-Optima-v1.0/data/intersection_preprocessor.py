"""
库存数据和销售数据交集准备模块 - data/intersection_preprocessor.py
用于将门店库存、仓库库存和销售数据取交集，并生成训练用的NumPy数组

集成到数据通道中，作为数据预处理的步骤之一
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 默认文件路径
DEFAULT_SHOP_STOCK_PATH = "data/raw/shop_stock.csv"
DEFAULT_DC_STOCK_PATH = "data/raw/dc_stock_data.csv"
DEFAULT_SALES_DATA_PATH = "data/processed/real_sales_intersect.csv"

# 输出目录
OUTPUT_DIR = Path("data/intersection")


class IntersectionDataPreprocessor:
    """库存数据和销售数据交集预处理器"""
    
    def __init__(self,
                 shop_stock_path: str = DEFAULT_SHOP_STOCK_PATH,
                 dc_stock_path: str = DEFAULT_DC_STOCK_PATH,
                 sales_data_path: str = DEFAULT_SALES_DATA_PATH,
                 output_dir: str = None):
        """
        初始化交集数据预处理器
        
        Args:
            shop_stock_path: 门店库存数据路径
            dc_stock_path: 仓库库存数据路径
            sales_data_path: 销售数据路径
            output_dir: 输出目录（如果为None，使用默认目录）
        """
        self.shop_stock_path = shop_stock_path
        self.dc_stock_path = dc_stock_path
        self.sales_data_path = sales_data_path
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据
        self.shop_stock_df = None
        self.dc_stock_df = None
        self.sales_df = None
        
        # 交集数据
        self.shop_stock_intersect = None
        self.dc_stock_intersect = None
        self.sales_intersect = None
        
        # 共同元素
        self.common_sku = None
        self.common_shops = None
        self.common_dates = None
        
        # NumPy数组
        self.sales_array = None
        self.shop_stock_array = None
        self.dc_stock_array = None
        self.metadata = None
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载并预处理三个数据文件"""
        logger.info("=" * 80)
        logger.info("步骤1: 加载并预处理数据")
        logger.info("=" * 80)
        
        # 1. 读取门店库存数据
        logger.info(f"读取门店库存数据: {self.shop_stock_path}")
        shop_stock_df = pd.read_csv(self.shop_stock_path)
        logger.info(f"  原始数据形状: {shop_stock_df.shape}")
        
        # 重命名列（保持一致性）
        shop_stock_df = shop_stock_df.rename(columns={
            'dept_id': 'shop_code',
            'dt': 'date',
            'stock_qty': 'shop_stock_qty'
        })
        
        # 转换日期格式（YYYYMMDD → YYYY-MM-DD）
        shop_stock_df['date'] = pd.to_datetime(
            shop_stock_df['date'].astype(str), format='%Y%m%d'
        ).dt.strftime('%Y-%m-%d')
        
        logger.info(f"  唯一门店数: {shop_stock_df['shop_code'].nunique()}")
        logger.info(f"  唯一SKU数: {shop_stock_df['goods_code'].nunique()}")
        logger.info(f"  日期范围: {shop_stock_df['date'].min()} 到 {shop_stock_df['date'].max()}")
        
        # 2. 读取仓库库存数据
        logger.info(f"\n读取仓库库存数据: {self.dc_stock_path}")
        dc_stock_df = pd.read_csv(self.dc_stock_path)
        logger.info(f"  原始数据形状: {dc_stock_df.shape}")
        
        # 重命名列（保持一致性）
        dc_stock_df = dc_stock_df.rename(columns={
            'dt': 'date',
            'stock_qty': 'dc_stock_qty'
        })
        
        # 转换日期格式（YYYYMMDD → YYYY-MM-DD）
        dc_stock_df['date'] = pd.to_datetime(
            dc_stock_df['date'].astype(str), format='%Y%m%d'
        ).dt.strftime('%Y-%m-%d')
        
        logger.info(f"  唯一SKU数: {dc_stock_df['goods_code'].nunique()}")
        logger.info(f"  日期范围: {dc_stock_df['date'].min()} 到 {dc_stock_df['date'].max()}")
        
        # 3. 读取销售数据
        logger.info(f"\n读取销售数据: {self.sales_data_path}")
        sales_df = pd.read_csv(self.sales_data_path)
        logger.info(f"  原始数据形状: {sales_df.shape}")
        
        # 确保日期格式一致
        sales_df['ds'] = pd.to_datetime(sales_df['ds']).dt.strftime('%Y-%m-%d')
        sales_df = sales_df.rename(columns={'ds': 'date'})
        
        logger.info(f"  唯一门店数: {sales_df['shop_code'].nunique()}")
        logger.info(f"  唯一SKU数: {sales_df['goods_code'].nunique()}")
        logger.info(f"  日期范围: {sales_df['date'].min()} 到 {sales_df['date'].max()}")
        
        self.shop_stock_df = shop_stock_df
        self.dc_stock_df = dc_stock_df
        self.sales_df = sales_df
        
        return shop_stock_df, dc_stock_df, sales_df
    
    def find_intersection(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set, set, set]:
        """找到三个数据的交集"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤2: 找到数据的交集")
        logger.info("=" * 80)
        
        if self.shop_stock_df is None:
            self.load_and_preprocess_data()
        
        shop_stock_df = self.shop_stock_df
        dc_stock_df = self.dc_stock_df
        sales_df = self.sales_df
        
        # 1. 找到共同的SKU
        logger.info("\n1. 找到共同的SKU...")
        shop_sku_set = set(shop_stock_df['goods_code'].unique())
        dc_sku_set = set(dc_stock_df['goods_code'].unique())
        sales_sku_set = set(sales_df['goods_code'].unique())
        
        common_sku = shop_sku_set & dc_sku_set & sales_sku_set
        logger.info(f"  门店库存SKU数: {len(shop_sku_set)}")
        logger.info(f"  仓库库存SKU数: {len(dc_sku_set)}")
        logger.info(f"  销售数据SKU数: {len(sales_sku_set)}")
        logger.info(f"  共同SKU数: {len(common_sku)}")
        
        # 2. 找到共同的门店
        logger.info("\n2. 找到共同的门店...")
        shop_code_set = set(shop_stock_df['shop_code'].unique())
        sales_shop_set = set(sales_df['shop_code'].unique())
        
        common_shops = shop_code_set & sales_shop_set
        logger.info(f"  门店库存门店数: {len(shop_code_set)}")
        logger.info(f"  销售数据门店数: {len(sales_shop_set)}")
        logger.info(f"  共同门店数: {len(common_shops)}")
        
        # 3. 找到共同的日期范围
        logger.info("\n3. 找到共同的日期范围...")
        shop_date_set = set(shop_stock_df['date'].unique())
        dc_date_set = set(dc_stock_df['date'].unique())
        sales_date_set = set(sales_df['date'].unique())
        
        common_dates = shop_date_set & dc_date_set & sales_date_set
        logger.info(f"  门店库存日期数: {len(shop_date_set)}")
        logger.info(f"  仓库库存日期数: {len(dc_date_set)}")
        logger.info(f"  销售数据日期数: {len(sales_date_set)}")
        logger.info(f"  共同日期数: {len(common_dates)}")
        
        if len(common_dates) > 0:
            common_dates_sorted = sorted(common_dates)
            logger.info(f"  共同日期范围: {common_dates_sorted[0]} 到 {common_dates_sorted[-1]}")
        
        # 4. 过滤数据，只保留交集部分
        logger.info("\n4. 过滤数据...")
        shop_stock_intersect = shop_stock_df[
            shop_stock_df['goods_code'].isin(common_sku) &
            shop_stock_df['shop_code'].isin(common_shops) &
            shop_stock_df['date'].isin(common_dates)
        ].copy()
        
        dc_stock_intersect = dc_stock_df[
            dc_stock_df['goods_code'].isin(common_sku) &
            dc_stock_df['date'].isin(common_dates)
        ].copy()
        
        sales_intersect = sales_df[
            sales_df['goods_code'].isin(common_sku) &
            sales_df['shop_code'].isin(common_shops) &
            sales_df['date'].isin(common_dates)
        ].copy()
        
        logger.info(f"  门店库存交集形状: {shop_stock_intersect.shape}")
        logger.info(f"  仓库库存交集形状: {dc_stock_intersect.shape}")
        logger.info(f"  销售数据交集形状: {sales_intersect.shape}")
        
        self.shop_stock_intersect = shop_stock_intersect
        self.dc_stock_intersect = dc_stock_intersect
        self.sales_intersect = sales_intersect
        self.common_sku = common_sku
        self.common_shops = common_shops
        self.common_dates = common_dates
        
        return shop_stock_intersect, dc_stock_intersect, sales_intersect, common_sku, common_shops, common_dates
    
    def save_intersection_data(self) -> Tuple[Path, Path, Path]:
        """保存交集数据"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤3: 保存交集数据")
        logger.info("=" * 80)
        
        if self.shop_stock_intersect is None:
            self.find_intersection()
        
        # 保存门店库存交集
        shop_output_path = self.output_dir / "shop_stock_intersection.csv"
        self.shop_stock_intersect.to_csv(shop_output_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ 门店库存交集已保存: {shop_output_path}")
        
        # 保存仓库库存交集
        dc_output_path = self.output_dir / "dc_stock_intersection.csv"
        self.dc_stock_intersect.to_csv(dc_output_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ 仓库库存交集已保存: {dc_output_path}")
        
        # 保存销售数据交集
        sales_output_path = self.output_dir / "sales_intersection.csv"
        self.sales_intersect.to_csv(sales_output_path, index=False, encoding='utf-8-sig')
        logger.info(f"✅ 销售数据交集已保存: {sales_output_path}")
        
        return shop_output_path, dc_output_path, sales_output_path
    
    def create_training_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """创建用于训练的NumPy数组"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤4: 创建训练数据（NumPy数组）")
        logger.info("=" * 80)
        
        if self.sales_intersect is None:
            self.find_intersection()
        
        sales_intersect = self.sales_intersect
        shop_stock_intersect = self.shop_stock_intersect
        dc_stock_intersect = self.dc_stock_intersect
        common_sku = self.common_sku
        common_shops = self.common_shops
        common_dates = self.common_dates
        
        # 1. 创建SKU和门店的映射
        sku_list = sorted(list(common_sku))
        shop_list = sorted(list(common_shops))
        date_list = sorted(list(common_dates))
        
        sku_to_idx = {sku: idx for idx, sku in enumerate(sku_list)}
        shop_to_idx = {shop: idx for idx, shop in enumerate(shop_list)}
        
        num_skus = len(sku_list)
        num_shops = len(shop_list)
        num_dates = len(date_list)
        
        logger.info(f"  SKU数量: {num_skus}")
        logger.info(f"  门店数量: {num_shops}")
        logger.info(f"  日期数量: {num_dates}")
        
        # 2. 创建销售数据数组 (num_dates, num_shops, num_skus)
        logger.info("\n1. 创建销售数据数组...")
        sales_array = np.zeros((num_dates, num_shops, num_skus), dtype=np.float32)
        
        for _, row in sales_intersect.iterrows():
            try:
                date_idx = date_list.index(row['date'])
                shop_idx = shop_to_idx[row['shop_code']]
                sku_idx = sku_to_idx[row['goods_code']]
                sales_array[date_idx, shop_idx, sku_idx] = row['sale_qty']
            except (ValueError, KeyError):
                continue
        
        logger.info(f"  销售数据数组形状: {sales_array.shape}")
        logger.info(f"  非零元素数: {np.count_nonzero(sales_array)}")
        
        # 保存销售数据数组
        sales_array_path = self.output_dir / "sales_array.npy"
        np.save(sales_array_path, sales_array)
        logger.info(f"  ✅ 销售数据数组已保存: {sales_array_path}")
        
        # 3. 创建门店库存数组 (num_dates, num_shops, num_skus)
        logger.info("\n2. 创建门店库存数组...")
        shop_stock_array = np.zeros((num_dates, num_shops, num_skus), dtype=np.float32)
        
        for _, row in shop_stock_intersect.iterrows():
            try:
                date_idx = date_list.index(row['date'])
                shop_idx = shop_to_idx[row['shop_code']]
                sku_idx = sku_to_idx[row['goods_code']]
                shop_stock_array[date_idx, shop_idx, sku_idx] = row['shop_stock_qty']
            except (ValueError, KeyError):
                continue
        
        logger.info(f"  门店库存数组形状: {shop_stock_array.shape}")
        logger.info(f"  非零元素数: {np.count_nonzero(shop_stock_array)}")
        
        # 保存门店库存数组
        shop_stock_array_path = self.output_dir / "shop_stock_array.npy"
        np.save(shop_stock_array_path, shop_stock_array)
        logger.info(f"  ✅ 门店库存数组已保存: {shop_stock_array_path}")
        
        # 4. 创建仓库库存数组 (num_dates, num_skus)
        logger.info("\n3. 创建仓库库存数组...")
        dc_stock_array = np.zeros((num_dates, num_skus), dtype=np.float32)
        
        for _, row in dc_stock_intersect.iterrows():
            try:
                date_idx = date_list.index(row['date'])
                sku_idx = sku_to_idx[row['goods_code']]
                dc_stock_array[date_idx, sku_idx] = row['dc_stock_qty']
            except (ValueError, KeyError):
                continue
        
        logger.info(f"  仓库库存数组形状: {dc_stock_array.shape}")
        logger.info(f"  非零元素数: {np.count_nonzero(dc_stock_array)}")
        
        # 保存仓库库存数组
        dc_stock_array_path = self.output_dir / "dc_stock_array.npy"
        np.save(dc_stock_array_path, dc_stock_array)
        logger.info(f"  ✅ 仓库库存数组已保存: {dc_stock_array_path}")
        
        # 5. 保存元数据
        logger.info("\n4. 保存元数据...")
        metadata = {
            'sku_list': sku_list,
            'shop_list': shop_list,
            'date_list': date_list,
            'num_skus': num_skus,
            'num_shops': num_shops,
            'num_dates': num_dates
        }
        
        metadata_path = self.output_dir / "metadata.npy"
        np.save(metadata_path, metadata)
        logger.info(f"  ✅ 元数据已保存: {metadata_path}")
        
        self.sales_array = sales_array
        self.shop_stock_array = shop_stock_array
        self.dc_stock_array = dc_stock_array
        self.metadata = metadata
        
        return sales_array, shop_stock_array, dc_stock_array, metadata
    
    def prepare_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        执行完整的数据准备流程
        
        Returns:
            Tuple: (sales_array, shop_stock_array, dc_stock_array, metadata)
        """
        logger.info("=" * 80)
        logger.info("开始准备库存数据和销售数据的交集")
        logger.info("=" * 80)
        
        # 步骤1: 加载并预处理数据
        self.load_and_preprocess_data()
        
        # 步骤2: 找到数据的交集
        self.find_intersection()
        
        # 步骤3: 保存交集数据
        self.save_intersection_data()
        
        # 步骤4: 创建训练数据（NumPy数组）
        self.create_training_arrays()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ 数据准备完成！")
        logger.info("=" * 80)
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"共同SKU数: {len(self.common_sku)}")
        logger.info(f"共同门店数: {len(self.common_shops)}")
        logger.info(f"共同日期数: {len(self.common_dates)}")
        logger.info(f"\n生成的文件:")
        logger.info(f"  - shop_stock_intersection.csv")
        logger.info(f"  - dc_stock_intersection.csv")
        logger.info(f"  - sales_intersection.csv")
        logger.info(f"  - sales_array.npy")
        logger.info(f"  - shop_stock_array.npy")
        logger.info(f"  - dc_stock_array.npy")
        logger.info(f"  - metadata.npy")
        
        return self.sales_array, self.shop_stock_array, self.dc_stock_array, self.metadata
    
    def load_prepared_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        加载已准备的数据（如果已存在）
        
        Returns:
            Tuple: (sales_array, shop_stock_array, dc_stock_array, metadata)
        """
        sales_array_path = self.output_dir / "sales_array.npy"
        shop_stock_array_path = self.output_dir / "shop_stock_array.npy"
        dc_stock_array_path = self.output_dir / "dc_stock_array.npy"
        metadata_path = self.output_dir / "metadata.npy"
        
        if all(p.exists() for p in [sales_array_path, shop_stock_array_path, dc_stock_array_path, metadata_path]):
            logger.info("✅ 发现已准备的数据，直接加载...")
            self.sales_array = np.load(sales_array_path)
            self.shop_stock_array = np.load(shop_stock_array_path)
            self.dc_stock_array = np.load(dc_stock_array_path)
            self.metadata = np.load(metadata_path, allow_pickle=True).item()
            
            logger.info(f"  销售数据数组形状: {self.sales_array.shape}")
            logger.info(f"  门店库存数组形状: {self.shop_stock_array.shape}")
            logger.info(f"  仓库库存数组形状: {self.dc_stock_array.shape}")
            logger.info(f"  SKU数量: {self.metadata['num_skus']}")
            logger.info(f"  门店数量: {self.metadata['num_shops']}")
            logger.info(f"  日期数量: {self.metadata['num_dates']}")
            
            return self.sales_array, self.shop_stock_array, self.dc_stock_array, self.metadata
        else:
            logger.info("未找到已准备的数据，需要重新准备...")
            return self.prepare_all()


def prepare_intersection_data(shop_stock_path: str = None,
                              dc_stock_path: str = None,
                              sales_data_path: str = None,
                              output_dir: str = None,
                              force_reprepare: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    准备库存数据和销售数据的交集（便捷函数）
    
    Args:
        shop_stock_path: 门店库存数据路径（如果为None，使用默认路径）
        dc_stock_path: 仓库库存数据路径（如果为None，使用默认路径）
        sales_data_path: 销售数据路径（如果为None，使用默认路径）
        output_dir: 输出目录（如果为None，使用默认目录）
        force_reprepare: 是否强制重新准备（即使已存在）
    
    Returns:
        Tuple: (sales_array, shop_stock_array, dc_stock_array, metadata)
    """
    preprocessor = IntersectionDataPreprocessor(
        shop_stock_path=shop_stock_path or DEFAULT_SHOP_STOCK_PATH,
        dc_stock_path=dc_stock_path or DEFAULT_DC_STOCK_PATH,
        sales_data_path=sales_data_path or DEFAULT_SALES_DATA_PATH,
        output_dir=output_dir
    )
    
    if force_reprepare:
        return preprocessor.prepare_all()
    else:
        return preprocessor.load_prepared_data()


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description='准备库存数据和销售数据的交集')
    parser.add_argument('--shop-stock-path', type=str, default=None, help='门店库存数据路径')
    parser.add_argument('--dc-stock-path', type=str, default=None, help='仓库库存数据路径')
    parser.add_argument('--sales-data-path', type=str, default=None, help='销售数据路径')
    parser.add_argument('--output-dir', type=str, default=None, help='输出目录')
    parser.add_argument('--force-reprepare', action='store_true', help='强制重新准备')
    
    args = parser.parse_args()
    
    prepare_intersection_data(
        shop_stock_path=args.shop_stock_path,
        dc_stock_path=args.dc_stock_path,
        sales_data_path=args.sales_data_path,
        output_dir=args.output_dir,
        force_reprepare=args.force_reprepare
    )
