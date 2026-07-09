"""
数据预处理模块 - data/data_preprocessor.py
负责数据清洗、缺失值处理、数据标准化等
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        """初始化预处理器"""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.column_info: Optional[Dict] = None
    
    def set_column_info(self, static_cate: List[str],
                       static_reals: List[str],
                       time_varying_known: List[str],
                       time_varying_unknown: List[str]):
        """
        设置TFT模型所需的列配置
        
        Args:
            static_cate: 静态分类变量
            static_reals: 静态连续变量
            time_varying_known: 时变已知变量
            time_varying_unknown: 时变未知变量
        """
        self.column_info = {
            'static_cate': static_cate,
            'static_reals': static_reals,
            'time_varying_known': time_varying_known,
            'time_varying_unknown': time_varying_unknown
        }
    
    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'forward_fill',
                             fill_value: Any = 0) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入数据
            strategy: 填充策略 ('forward_fill', 'backward_fill', 'mean', 'zero', 'median')
            fill_value: 固定填充值
        
        Returns:
            pd.DataFrame: 处理后的数据
        """
        df = df.copy()
        
        # 处理数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().any():
                if strategy == 'forward_fill':
                    df[col] = df.groupby(['shop_code', 'goods_code'])[col].ffill().bfill()
                elif strategy == 'backward_fill':
                    df[col] = df.groupby(['shop_code', 'goods_code'])[col].bfill().ffill()
                elif strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'zero':
                    df[col] = df[col].fillna(0)
                elif strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
        
        # 处理非数值列（category + object统一用'unknown'填充）
        # 先将category列转为object，统一处理后再转回，避免两条路径行为不一致
        non_numeric_cols = df.select_dtypes(include=['category', 'object']).columns
        for col in non_numeric_cols:
            if df[col].isna().any():
                if df[col].dtype.name == 'category':
                    # category列：先添加'unknown'类别，再填充
                    if 'unknown' not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories(['unknown'])
                    df[col] = df[col].fillna('unknown')
                else:
                    # object列：直接填充
                    df[col] = df[col].fillna('unknown')
        
        # 剩余数值列缺失值用0填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame,
                           columns: List[str],
                           fit: bool = True) -> pd.DataFrame:
        """
        编码分类变量
        
        Args:
            df: 输入数据
            columns: 需要编码的列
            fit: 是否拟合编码器（False时使用已保存的编码器转换）
        
        Returns:
            pd.DataFrame: 编码后的数据
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # 处理缺失值
            df[col] = df[col].astype(str).fillna('unknown')
            
            if fit:
                encoder = LabelEncoder()
                df[f"{col}_encoded"] = encoder.fit_transform(df[col])
                self.label_encoders[col] = encoder
            else:
                encoder = self.label_encoders.get(col)
                if encoder:
                    # 处理未见过的类别
                    known_classes = set(encoder.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_classes else encoder.classes_[0]
                    )
                    df[f"{col}_encoded"] = encoder.transform(df[col])
        
        return df
    
    def scale_features(self, df: pd.DataFrame,
                      columns: List[str],
                      fit: bool = True) -> pd.DataFrame:
        """
        标准化连续变量
        
        Args:
            df: 输入数据
            columns: 需要标准化的列
            fit: 是否拟合标准化器
        
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if fit:
                scaler = StandardScaler()
                df[f"{col}_scaled"] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                scaler = self.scalers.get(col)
                if scaler:
                    df[f"{col}_scaled"] = scaler.transform(df[[col]])
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame,
                       column: str,
                       n_std: float = 3.0,
                       method: str = 'clip') -> pd.DataFrame:
        """
        移除或限制异常值
        
        Args:
            df: 输入数据
            column: 列名
            n_std: 标准差倍数
            method: 'remove'（删除）或'clip'（限制）
        
        Returns:
            pd.DataFrame: 处理后的数据
        """
        df = df.copy()
        mean = df[column].mean()
        std = df[column].std()
        
        lower = mean - n_std * std
        upper = mean + n_std * std
        
        if method == 'remove':
            df = df[(df[column] >= lower) & (df[column] <= upper)]
        elif method == 'clip':
            df[column] = df[column].clip(lower, upper)
        
        return df
    
    def validate_data(self, df: pd.DataFrame,
                     required_cols: List[str] = None) -> Tuple[bool, List[str]]:
        """
        验证数据完整性（只读操作，不修改传入的DataFrame）

        Args:
            df: 输入数据
            required_cols: 必需列列表

        Returns:
            Tuple: (is_valid, error_messages)
        """
        errors = []

        # 检查必需列
        if required_cols:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                errors.append(f"缺失必需列: {missing_cols}")

        # 检查缺失值
        null_counts = df.isnull().sum()
        high_null_cols = null_counts[null_counts > 0]
        if len(high_null_cols) > 0:
            errors.append(f"高缺失列: {high_null_cols.to_dict()}")

        # 检查重复
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            errors.append(f"存在 {dup_count} 条重复记录")

        # 检查可选列 is_active_sale 的类型问题
        if 'is_active_sale' in df.columns:
            col = df['is_active_sale']
            if col.dtype != bool:
                errors.append("is_active_sale 列类型非布尔，需调用 normalize_active_sale_column() 转换")
        else:
            errors.append("is_active_sale 列不存在，将跳过活跃销售过滤")

        is_valid = len(errors) == 0
        return is_valid, errors

    def normalize_active_sale_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将 is_active_sale 列标准化为布尔类型

        Args:
            df: 输入数据

        Returns:
            pd.DataFrame: 转换后的数据（副本）
        """
        df = df.copy()
        if 'is_active_sale' in df.columns:
            df['is_active_sale'] = df['is_active_sale'].astype(str).map({
                '1': True, 'true': True, 'True': True,
                '0': False, 'false': False, 'False': False
            }).fillna(False)
        return df
    
    def prepare_tft_data(self, df: pd.DataFrame,
                        object_cols: List[str] = None) -> pd.DataFrame:
        """
        准备TFT模型所需的数据格式
        
        Args:
            df: 输入数据
            object_cols: 需要编码的对象类型列
        
        Returns:
            pd.DataFrame: 准备好的数据
        """
        df = df.copy()
        
        # 0.5 创建 year_week_num 列（如果 date 列存在且 year_week_num 不存在）
        if 'date' in df.columns and 'year_week_num' not in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year_week_num'] = df['date'].dt.strftime('%Y%U').astype(int)
            print("  [prepare_tft_data] year_week_num 列已创建", flush=True)
        
        # 0. 标准化 is_active_sale 列
        df = self.normalize_active_sale_column(df)

        # 1. 处理缺失值
        df = self.handle_missing_values(df, strategy='forward_fill')
        
        # 2. 确保数值类型
        numeric_cols = ['sale_qty', 'sale_amt'] + \
                      [col for col in df.columns if col.startswith(('lag_', 'rolling_'))]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 3. 创建唯一索引
        if 'idx' not in df.columns:
            df['idx'] = df['shop_code'].astype(str) + '_' + df['goods_code'].astype(str)
        
        # 4. 编码分类变量
        if object_cols:
            df = self.encode_categorical(df, object_cols, fit=True)
        
        # 5. 确保time_idx是整数
        if 'time_idx' in df.columns:
            df['time_idx'] = df['time_idx'].astype(int)
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        获取特征分组信息
        
        Returns:
            Dict: 各类型特征列表
        """
        if self.column_info is None:
            return {}
        
        return self.column_info.copy()
    
    def inverse_transform_target(self, scaled_values: np.ndarray,
                                scaler: StandardScaler = None) -> np.ndarray:
        """
        反标准化目标变量
        
        Args:
            scaled_values: 标准化后的值
            scaler: 标准化器
        
        Returns:
            np.ndarray: 原始尺度的值
        """
        if scaler is None:
            scaler = self.scalers.get('sale_qty')
        
        if scaler:
            return scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
        
        return scaled_values
    
    def generate_main_csv(self,
                          sale_data_path: str,
                          shop_info_path: str,
                          output_path: str,
                          encoding: str = 'utf-8',
                          nrows: int = None) -> pd.DataFrame:
        """
        从原始数据生成 main.csv（完整预处理流程）
        
        流程：
        1. 加载 sale_data.csv
        2. 重命名 ds -> date
        3. 加载并合并 shop_info.csv
        4. 创建 time_idx / group_id
        5. 调用 FeatureEngineer 生成特征
        6. 调用 prepare_tft_data() 预处理
        7. 保存为 main.csv
        
        Args:
            sale_data_path: sale_data.csv 路径
            shop_info_path: shop_info.csv 路径
            output_path: 输出 main.csv 路径
            encoding: 文件编码（默认 utf-8）
            nrows: 可选，限制读取行数（用于测试）
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        import os
        from .feature_engineering import FeatureEngineer
        
        print("=" * 70, flush=True)
        print("[generate_main_csv] 开始生成 main.csv", flush=True)
        print("=" * 70, flush=True)
        
        # ============================================================
        # 1. 加载原始销售数据
        # ============================================================
        print("\n[1/7] 加载原始销售数据...", flush=True)
        df = pd.read_csv(sale_data_path, nrows=nrows, encoding=encoding)
        print(f"  [OK] sale_data.csv: {df.shape[0]:,} 行 x {df.shape[1]} 列", flush=True)
        print(f"  列名: {list(df.columns)}", flush=True)
        
        # ============================================================
        # 2. 重命名 ds -> date
        # ============================================================
        print("\n[2/7] 重命名日期列...", flush=True)
        if 'ds' in df.columns and 'date' not in df.columns:
            df = df.rename(columns={'ds': 'date'})
            print("  [OK] 重命名: ds -> date", flush=True)
        elif 'date' in df.columns:
            print("  [OK] date 列已存在，跳过重命名", flush=True)
        else:
            print("  [WARN] 未找到 ds 或 date 列，请检查数据！", flush=True)
        
        # 转换日期类型
        df['date'] = pd.to_datetime(df['date'])
        print(f"  [OK] date 列已转换为 datetime", flush=True)
        print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}", flush=True)
        
        # ============================================================
        # 3. 排序（特征工程需要按时间顺序）
        # ============================================================
        print("\n[3/7] 数据排序...", flush=True)
        df = df.sort_values(['shop_code', 'goods_code', 'date']).reset_index(drop=True)
        print(f"  [OK] 已按 [shop_code, goods_code, date] 排序", flush=True)
        
        # ============================================================
        # 4. 创建 time_idx 和 group_id
        # ============================================================
        print("\n[4/7] 创建 time_idx 和 group_id...", flush=True)
        
        # time_idx: 每个分组内从0开始递增
        df['time_idx'] = df.groupby(['shop_code', 'goods_code']).cumcount()
        print(f"  [OK] time_idx 已创建，范围: {df['time_idx'].min()} - {df['time_idx'].max()}", flush=True)
        
        # group_id: 唯一标识每个 (shop_code, goods_code) 分组
        df['group_id'] = df['shop_code'].astype(str) + '_' + df['goods_code'].astype(str)
        n_groups = df['group_id'].nunique()
        print(f"  [OK] group_id 已创建，唯一分组数: {n_groups:,}", flush=True)
        
        # ============================================================
        # 4.5 创建 year_week_num 列（供 inference.py 使用）
        # ============================================================
        print("\n[4.5/7] 创建 year_week_num 列...", flush=True)
        # year_week_num 格式：YYYYWW（如 202301 表示 2023 年第 1 周）
        df['year_week_num'] = df['date'].dt.strftime('%Y%U').astype(int)
        print(f"  [OK] year_week_num 已创建，范围: {df['year_week_num'].min()} - {df['year_week_num'].max()}", flush=True)
        
        # ============================================================
        # 5. 合并门店信息 shop_info.csv
        # ============================================================
        print("\n[5/7] 合并门店信息...", flush=True)
        shop_df = pd.read_csv(shop_info_path, encoding=encoding)
        print(f"  [OK] shop_info.csv: {shop_df.shape[0]:,} 行 x {shop_df.shape[1]} 列", flush=True)
        
        # 检查 shop_code 匹配情况
        shop_codes_sale = set(df['shop_code'].unique())
        shop_codes_shop = set(shop_df['shop_code'].unique()) if 'shop_code' in shop_df.columns else set()
        missing = shop_codes_sale - shop_codes_shop
        if missing:
            print(f"  [WARN] {len(missing)} 个 shop_code 在 shop_info.csv 中未找到（将用 NaN 填充）", flush=True)
        
        # 合并
        df = df.merge(shop_df, on='shop_code', how='left')
        print(f"  [OK] 合并完成，合并后维度: {df.shape}", flush=True)
        
        # ============================================================
        # 6. 特征工程（调用现有的 FeatureEngineer）
        # ============================================================
        print("\n[6/7] 特征工程（调用 FeatureEngineer.create_all_features()）...", flush=True)
        engineer = FeatureEngineer()
        df = engineer.create_all_features(df)
        print(f"  [OK] 特征工程完成，维度: {df.shape}", flush=True)
        
        # ============================================================
        # 7. 数据预处理（调用现有的 prepare_tft_data）
        # ============================================================
        print("\n[7/7] 数据预处理（调用 prepare_tft_data()）...", flush=True)
        
        # 自动识别 object 类型列（需要编码的分类列）
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        # 排除 date 列（已经是 datetime 类型）
        if 'date' in object_cols:
            object_cols.remove('date')
        print(f"  需要编码的分类列（前10）: {object_cols[:10]}", flush=True)
        
        df = self.prepare_tft_data(df, object_cols)
        print(f"  [OK] 数据预处理完成，最终维度: {df.shape}", flush=True)
        
        # ============================================================
        # 8. 保存为 main.csv
        # ============================================================
        print("\n[保存] 写入 main.csv...", flush=True)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False, encoding=encoding)
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  [OK] 已保存到: {output_path}", flush=True)
        print(f"  文件大小: {file_size_mb:.2f} MB", flush=True)
        print(f"  行数: {len(df):,}", flush=True)
        print(f"  列数: {df.shape[1]}", flush=True)
        
        print("\n" + "=" * 70, flush=True)
        print("[generate_main_csv] 完成！", flush=True)
        print("=" * 70, flush=True)
        
        return df

# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description='数据预处理：从原始数据生成 main.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python data/data_preprocessor.py --sale-data data/sale_data.csv --shop-info data/shop_info.csv --output data/main.csv
  python data/data_preprocessor.py --sale-data data/sale_data.csv --output data/main.csv --nrows 10000
        """
    )
    parser.add_argument('--sale-data', type=str, default='data/sale_data.csv',
                        help='sale_data.csv 路径（默认：data/sale_data.csv）')
    parser.add_argument('--shop-info', type=str, default='data/shop_info.csv',
                        help='shop_info.csv 路径（默认：data/shop_info.csv）')
    parser.add_argument('--output', type=str, default='data/main.csv',
                        help='输出 main.csv 路径（默认：data/main.csv）')
    parser.add_argument('--encoding', type=str, default='utf-8',
                        help='文件编码（默认：utf-8）')
    parser.add_argument('--nrows', type=int, default=None,
                        help='限制读取行数（用于测试）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.sale_data):
        print(f"[ERROR] 文件不存在: {args.sale_data}")
        exit(1)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.generate_main_csv(
        sale_data_path=args.sale_data,
        shop_info_path=args.shop_info,
        output_path=args.output,
        encoding=args.encoding,
        nrows=args.nrows,
    )
    print(f"\n[完成] main.csv 已生成: {args.output}")

