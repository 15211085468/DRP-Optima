"""
数据加载模块 - data/data_loader.py
负责从原始数据源加载数据
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from config import PathConfig, DataConfig
from config.exceptions import DataError
from utils.logger import get_logger
logger = get_logger(__name__)


class DataLoader:
    """数据加载器"""
    
    def __init__(self, path_config: Optional[PathConfig] = None, 
                 data_config: Optional[DataConfig] = None):
        """
        初始化数据加载器
        
        Args:
            path_config: 路径配置
            data_config: 数据配置
        """
        self.path_config = path_config or PathConfig()
        self.data_config = data_config or DataConfig()
        self._dtype_dict: Optional[Dict] = None
        self._column_info: Optional[Dict] = None
    
    def load_dtype_dict(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        加载数据类型字典和列配置

        Returns:
            Tuple: (main_dtypes_dict, static_cate_col, static_reals_col,
                   time_varying_known_reals_col, time_varying_unknown_reals_col)
        """
        dtype_path = self.path_config.dtypes_dict_file  # 现在使用绝对路径

        # 根据文件扩展名确定加载格式
        ext = Path(dtype_path).suffix.lower()

        if ext == '.json':
            # JSON格式：优先尝试JSON加载
            try:
                with open(dtype_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    main_dtypes_dict = data.get('main_dtypes_dict', {})
                    static_cate_col = data.get('static_cate_col', [])
                    static_reals_col = data.get('static_reals_col', [])
                    time_varying_known_reals_col = data.get('time_varying_known_reals_col', [])
                    time_varying_unknown_reals_col = data.get('time_varying_unknown_reals_col', [])
            except Exception as e:
                # JSON加载失败，尝试pickle（文件扩展名可能不准确）
                logger.warning(f"JSON load failed for {dtype_path}: {e}, trying pickle...")
                try:
                    with open(dtype_path, 'rb') as f:
                        main_dtypes_dict = pickle.load(f)
                        static_cate_col = pickle.load(f)
                        static_reals_col = pickle.load(f)
                        time_varying_known_reals_col = pickle.load(f)
                        time_varying_unknown_reals_col = pickle.load(f)
                except Exception as pe:
                    raise DataError(f"Cannot load dtype dict from {dtype_path}: json err={e}, pickle err={pe}")
        else:
            # 非JSON扩展名：优先尝试pickle加载
            try:
                with open(dtype_path, 'rb') as f:
                    main_dtypes_dict = pickle.load(f)
                    static_cate_col = pickle.load(f)
                    static_reals_col = pickle.load(f)
                    time_varying_known_reals_col = pickle.load(f)
                    time_varying_unknown_reals_col = pickle.load(f)
            except Exception as e:
                # pickle加载失败，尝试JSON
                logger.warning(f"Pickle load failed for {dtype_path}: {e}, trying JSON...")
                try:
                    with open(dtype_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        main_dtypes_dict = data.get('main_dtypes_dict', {})
                        static_cate_col = data.get('static_cate_col', [])
                        static_reals_col = data.get('static_reals_col', [])
                        time_varying_known_reals_col = data.get('time_varying_known_reals_col', [])
                        time_varying_unknown_reals_col = data.get('time_varying_unknown_reals_col', [])
                except Exception as je:
                    raise DataError(f"Cannot load dtype dict from {dtype_path}: pickle err={e}, json err={je}")
        
        self._dtype_dict = main_dtypes_dict
        self._column_info = {
            'static_cate': static_cate_col,
            'static_reals': static_reals_col,
            'time_varying_known_reals': time_varying_known_reals_col,
            'time_varying_unknown_reals': time_varying_unknown_reals_col
        }
        
        return (main_dtypes_dict, static_cate_col, static_reals_col,
                time_varying_known_reals_col, time_varying_unknown_reals_col)
    
    def load_main_sales_data(self) -> pd.DataFrame:
        """
        加载主销售数据集
        
        Returns:
            pd.DataFrame: 销售数据 (约394万行)
        """
        # 直接使用绝对路径（path_config.sales_file 已是绝对路径）
        data_path = self.path_config.sales_file
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise DataError(f"Sales data file not found: {data_path}")
        
        # 使用已保存的dtype加快加载速度
        if self._dtype_dict is None:
            self.load_dtype_dict()
        
        try:
            df = pd.read_csv(data_path, dtype=self._dtype_dict)
        except Exception as e:
            raise DataError(f"Failed to load sales data from {data_path}: {e}")
        
        logger.info(f"加载销售数据: {df.shape[0]:,} 行, {df.shape[1]} 列")
        return df
    
    def load_shop_info(self) -> pd.DataFrame:
        """
        加载店铺信息数据集
        
        Returns:
            pd.DataFrame: 店铺信息 (379行, 59字段)
        """
        shop_path = self.path_config.shop_file  # 使用绝对路径
        
        # 检查文件是否存在
        if not os.path.exists(shop_path):
            raise DataError(f"Shop info file not found: {shop_path}")
        
        try:
            df = pd.read_csv(shop_path)
        except Exception as e:
            raise DataError(f"Failed to load shop info from {shop_path}: {e}")
        
        logger.info(f"加载店铺信息: {df.shape[0]} 行, {df.shape[1]} 字段")
        return df
    
    def load_goods_info(self) -> pd.DataFrame:
        """
        加载商品信息数据集
        
        Returns:
            pd.DataFrame: 商品信息 (500行, 36字段)
        """
        goods_path = self.path_config.goods_file  # 使用绝对路径
        
        # 如果文件不存在，返回空DataFrame
        if not os.path.exists(goods_path):
            logger.warning(f"警告: 商品信息文件不存在: {goods_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(goods_path)
        except Exception as e:
            raise DataError(f"Failed to load goods info from {goods_path}: {e}")
        
        logger.info(f"加载商品信息: {df.shape[0]} 行, {df.shape[1]} 字段")
        return df
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载所有数据集
        
        Returns:
            Tuple: (sales_df, shop_df, goods_df)
        """
        sales_df = self.load_main_sales_data()
        shop_df = self.load_shop_info()
        goods_df = self.load_goods_info()
        
        return sales_df, shop_df, goods_df
    
    def load_column_config(self) -> Dict[str, List[str]]:
        """
        获取TFT模型所需的列配置
        
        Returns:
            Dict: 各类型列的列表
        """
        if self._column_info is None:
            self.load_dtype_dict()
        return self._column_info
    
    def load_main_csv(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        直接加载主数据文件 (main.csv)
        
        此方法用于加载已预处理的数据，跳过原始数据加载和预处理步骤。
        
        Args:
            nrows: 读取的行数（None表示全部）
        
        Returns:
            pd.DataFrame: 主数据（已预处理）
        """
        data_path = self.path_config.data_path
        
        if not os.path.exists(data_path):
            raise DataError(f"主数据文件不存在: {data_path}")
        
        try:
            if nrows is not None and nrows > 0:
                df = pd.read_csv(data_path, nrows=nrows)
            else:
                df = pd.read_csv(data_path)
        except Exception as e:
            raise DataError(f"无法加载主数据文件 {data_path}: {e}")
        
        logger.info(f"加载主数据: {df.shape[0]:,} 行 x {df.shape[1]} 列")
        
        return df
    
    def filter_by_stores(self, df: pd.DataFrame, 
                        store_list: List[str], 
                        col: str = "shop_code") -> pd.DataFrame:
        """
        按门店列表过滤数据
        
        Args:
            df: 输入数据
            store_list: 门店编码列表
            col: 门店编码列名
        
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        return df[df[col].isin(store_list)]
    
    def filter_by_skus(self, df: pd.DataFrame,
                      sku_list: List[str],
                      col: str = "goods_code") -> pd.DataFrame:
        """
        按商品列表过滤数据
        
        Args:
            df: 输入数据
            sku_list: 商品编码列表
            col: 商品编码列名
        
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        return df[df[col].isin(sku_list)]
    
    def split_train_val_test(self, df: pd.DataFrame, 
                             time_idx_col: str = "time_idx",
                             test_weeks: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间索引分割训练集、验证集和测试集
        
        Args:
            df: 输入数据
            time_idx_col: 时间索引列名
            test_weeks: 测试集周数
        
        Returns:
            Tuple: (train_df, val_df, test_df)
        """
        if test_weeks is None:
            test_weeks = self.data_config.test_weeks
        
        max_time_idx = df[time_idx_col].max()
        train_cutoff = max_time_idx - test_weeks
        val_cutoff = train_cutoff - self.data_config.val_months * 4  # 约12个月
        
        train_df = df[df[time_idx_col] <= val_cutoff].copy()
        val_df = df[(df[time_idx_col] > val_cutoff) & 
                    (df[time_idx_col] <= train_cutoff)].copy()
        test_df = df[df[time_idx_col] > train_cutoff].copy()
        
        logger.info(f"数据分割: 训练集 {len(train_df):,} 行, "
                    f"验证集 {len(val_df):,} 行, 测试集 {len(test_df):,} 行")
        
        return train_df, val_df, test_df
    
    def get_unique_stores(self, df: pd.DataFrame, 
                         col: str = "shop_code") -> List[str]:
        """获取数据中唯一的门店列表"""
        return df[col].unique().tolist()
    
    def get_unique_skus(self, df: pd.DataFrame,
                       col: str = "goods_code") -> List[str]:
        """获取数据中唯一的商品列表"""
        return df[col].unique().tolist()
    
    def get_sample_data(self, df: pd.DataFrame,
                       shop_code: str,
                       goods_code: str) -> pd.DataFrame:
        """
        获取单个门店-商品组合的数据
        
        Args:
            df: 输入数据
            shop_code: 门店编码
            goods_code: 商品编码
        
        Returns:
            pd.DataFrame: 单个组合的数据
        """
        return df[(df["shop_code"] == shop_code) & 
                  (df["goods_code"] == goods_code)].copy()
