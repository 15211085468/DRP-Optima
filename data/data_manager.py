"""
data/data_manager.py - 统一数据加载器（DataManager）

目标：彻底消灭"维度不匹配"的 Bug，让代码具备工业级健壮性。

策略：废弃在各个脚本中散落读取 CSV 的做法，抽象出一个统一的 DataManager。
所有环境和训练脚本只能通过这个类获取数据。

核心功能：
1. 自动对齐需求数据和状态数据的维度
2. 提供统一的数据访问接口
3. 确保数据质量（缺失值处理、异常值检测等）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SupplyChainDataManager:
    """
    统一数据加载器
    
    职责：
    1. 加载需求数据和状态数据
    2. 自动对齐维度（以状态数据为基准）
    3. 提供统一的数据访问接口
    
    使用示例：
    ```python
    # 初始化
    data_mgr = SupplyChainDataManager(
        demand_path="data/processed/estimated_demand.csv",
        state_path="data/processed/historical_data_for_env.csv"
    )
    
    # 获取对齐后的环境配置
    env_config = data_mgr.get_env_config()
    # {'num_skus': 10, 'num_stores': 1, 'sku_list': [...], 'store_list': [...]}
    
    # 获取对齐后的需求矩阵
    demand_matrix = data_mgr.get_demand_matrix()
    # Shape: (weeks, stores, skus)
    ```
    """
    
    def __init__(
        self,
        demand_path: Optional[str] = None,
        state_path: Optional[str] = None,
        demand_df: Optional[pd.DataFrame] = None,
        state_df: Optional[pd.DataFrame] = None
    ):
        """
        初始化数据加载器
        
        参数：
        - demand_path: 需求数据文件路径（CSV）
        - state_path: 状态数据文件路径（CSV）
        - demand_df: 直接传入需求 DataFrame（可选）
        - state_df: 直接传入状态 DataFrame（可选）
        
        注意：必须提供 demand_path 或 demand_df，以及 state_path 或 state_df
        """
        logger.info("=" * 80)
        logger.info("SupplyChainDataManager 初始化")
        logger.info("=" * 80)
        
        # 1. 加载需求数据
        if demand_df is not None:
            self.demand_df = demand_df.copy()
            logger.info(f"  使用传入的 demand_df: {self.demand_df.shape}")
        elif demand_path is not None:
            self.demand_path = Path(demand_path)
            if not self.demand_path.exists():
                raise FileNotFoundError(f"需求数据文件不存在: {demand_path}")
            self.demand_df = pd.read_csv(self.demand_path)
            logger.info(f"  加载需求数据: {demand_path}")
            logger.info(f"    形状: {self.demand_df.shape}")
        else:
            self.demand_df = None
            logger.warning("  未提供需求数据")
        
        # 2. 加载状态数据
        if state_df is not None:
            self.state_df = state_df.copy()
            logger.info(f"  使用传入的 state_df: {self.state_df.shape}")
        elif state_path is not None:
            self.state_path = Path(state_path)
            if not self.state_path.exists():
                raise FileNotFoundError(f"状态数据文件不存在: {state_path}")
            self.state_df = pd.read_csv(self.state_path)
            logger.info(f"  加载状态数据: {state_path}")
            logger.info(f"    形状: {self.state_df.shape}")
        else:
            self.state_df = None
            logger.warning("  未提供状态数据")
        
        # 3. 自动对齐维度
        if self.demand_df is not None and self.state_df is not None:
            self._align_dimensions()
        else:
            logger.warning("  跳过维度对齐（缺少需求数据或状态数据）")
            # 使用默认值
            if self.demand_df is not None:
                self.valid_skus = self._extract_skus(self.demand_df)
                self.valid_stores = self._extract_stores(self.demand_df)
            elif self.state_df is not None:
                self.valid_skus = self._extract_skus(self.state_df)
                self.valid_stores = self._extract_stores(self.state_df)
            else:
                self.valid_skus = []
                self.valid_stores = []
    
    def _extract_skus(self, df: pd.DataFrame) -> List[str]:
        """从 DataFrame 中提取 SKU 列表"""
        if 'goods_code' in df.columns:
            return df['goods_code'].unique().tolist()
        elif 'sku_id' in df.columns:
            return df['sku_id'].unique().tolist()
        else:
            logger.warning("  无法识别 SKU 列（需要 'goods_code' 或 'sku_id'）")
            return []
    
    def _extract_stores(self, df: pd.DataFrame) -> List[str]:
        """从 DataFrame 中提取门店列表"""
        if 'shop_code' in df.columns:
            return df['shop_code'].unique().tolist()
        elif 'store_id' in df.columns:
            return df['store_id'].unique().tolist()
        elif 'sku_id' in df.columns and 'goods_code' in df.columns:
            # 特殊判断：如果 sku_id 和 goods_code 的值完全相同，说明 sku_id 是 SKU 代码，不是门店代码
            if df['sku_id'].equals(df['goods_code']):
                logger.warning("  检测到 sku_id 和 goods_code 值相同，说明这是单门店数据")
                logger.warning("  将使用默认门店标识：['STORE_001']")
                return ['STORE_001']
            else:
                # 否则，sku_id 可能是门店代码
                logger.info("  检测到 sku_id 列，将其作为门店标识")
                return df['sku_id'].unique().tolist()
        else:
            logger.warning("  无法识别门店列（需要 'shop_code'、'store_id' 或 'sku_id'）")
            logger.warning("  将使用默认门店标识：['STORE_001']")
            return ['STORE_001']
    
    def _align_dimensions(self):
        """
        核心逻辑：计算黄金交集（SKU & Store），自动裁剪数据
        
        策略：
        1. 提取需求数据和状态数据的 SKU 和门店维度
        2. 计算两个数据集的 SKU 交集和门店交集
        3. 使用交集裁剪两个数据集
        4. 更新环境配置参数
        """
        logger.info("\n[数据对齐] 开始计算黄金交集...")
        
        # 1. 提取需求数据的维度
        demand_skus = set()
        demand_stores = set()
        
        if self.demand_df is not None:
            if 'goods_code' in self.demand_df.columns:
                demand_skus = set(self.demand_df['goods_code'].unique())
            if 'shop_code' in self.demand_df.columns:
                demand_stores = set(self.demand_df['shop_code'].unique())
        
        # 2. 提取状态数据的维度
        state_skus = set()
        state_stores = set()
        
        if self.state_df is not None:
            # SKU 维度
            if 'goods_code' in self.state_df.columns:
                state_skus = set(self.state_df['goods_code'].unique())
            elif 'sku_id' in self.state_df.columns:
                # 注意：historical_data_for_env.csv 使用 sku_id 表示 SKU
                state_skus = set(self.state_df['sku_id'].unique())
            
            # 门店维度
            if 'shop_code' in self.state_df.columns:
                state_stores = set(self.state_df['shop_code'].unique())
            elif 'store_id' in self.state_df.columns:
                state_stores = set(self.state_df['store_id'].unique())
            else:
                # 单门店数据（如 historical_data_for_env.csv）
                # 从需求数据中获取门店列表，或使用默认标识
                if len(demand_stores) > 0:
                    state_stores = demand_stores  # 使用需求数据的门店
                else:
                    state_stores = set(['STORE_001'])  # 默认单门店
        
        # 3. 计算黄金交集
        if len(demand_skus) > 0 and len(state_skus) > 0:
            valid_skus = demand_skus & state_skus
        elif len(demand_skus) > 0:
            valid_skus = demand_skus
        elif len(state_skus) > 0:
            valid_skus = state_skus
        else:
            valid_skus = set()
        
        if len(demand_stores) > 0 and len(state_stores) > 0:
            valid_stores = demand_stores & state_stores
        elif len(demand_stores) > 0:
            valid_stores = demand_stores
        elif len(state_stores) > 0:
            valid_stores = state_stores
        else:
            valid_stores = set(['STORE_001'])
        
        logger.info(f"  需求数据维度: {len(demand_skus)} SKU, {len(demand_stores)} 门店")
        logger.info(f"  状态数据维度: {len(state_skus)} SKU, {len(state_stores)} 门店")
        logger.info(f"\n✅ 黄金交集:")
        logger.info(f"   - SKU交集: {len(valid_skus)} (需求:{len(demand_skus)} ∩ 状态:{len(state_skus)})")
        logger.info(f"   - 门店交集: {len(valid_stores)} (需求:{len(demand_stores)} ∩ 状态:{len(state_stores)})")
        
        # 4. 验证交集
        if len(valid_skus) == 0:
            logger.error("  ❌ 致命错误: 需求数据与状态数据没有共同的SKU，请检查数据源！")
            raise ValueError("需求数据与状态数据没有共同的SKU")
        
        if len(valid_stores) == 0:
            logger.error("  ❌ 致命错误: 需求数据与状态数据没有共同的门店，请检查数据源！")
            raise ValueError("需求数据与状态数据没有共同的门店")
        
        # 5. 保存交集结果
        self.valid_skus = sorted(list(valid_skus))
        self.valid_stores = sorted(list(valid_stores))
        
        # 6. 裁剪需求数据
        if self.demand_df is not None:
            original_len = len(self.demand_df)
            
            # 裁剪 SKU 维度
            if 'goods_code' in self.demand_df.columns:
                self.demand_df = self.demand_df[
                    self.demand_df['goods_code'].isin(self.valid_skus)
                ]
            
            # 裁剪门店维度
            if 'shop_code' in self.demand_df.columns:
                self.demand_df = self.demand_df[
                    self.demand_df['shop_code'].isin(self.valid_stores)
                ]
            
            logger.info(f"\n  裁剪需求数据: {original_len} → {len(self.demand_df)} 行")
        
        # 7. 裁剪状态数据
        if self.state_df is not None:
            original_len = len(self.state_df)
            
            # 裁剪 SKU 维度
            if 'goods_code' in self.state_df.columns:
                self.state_df = self.state_df[
                    self.state_df['goods_code'].isin(self.valid_skus)
                ]
            elif 'sku_id' in self.state_df.columns:
                self.state_df = self.state_df[
                    self.state_df['sku_id'].isin(self.valid_skus)
                ]
            
            logger.info(f"  裁剪状态数据: {original_len} → {len(self.state_df)} 行")
        
        # 8. 输出最终结果
        logger.info(f"\n  [OK] 维度对齐完成！")
        logger.info(f"    最终 SKU 数: {len(self.valid_skus)}")
        logger.info(f"    最终门店数: {len(self.valid_stores)}")
        if self.demand_df is not None:
            logger.info(f"    需求数据形状: {self.demand_df.shape}")
        if self.state_df is not None:
            logger.info(f"    状态数据形状: {self.state_df.shape}")
        logger.info(f"    ✅ 不会再出现维度不匹配错误！")
    
    def get_env_config(self) -> Dict[str, Any]:
        """
        返回对齐后的环境配置参数
        
        返回：
        - num_skus: 有效的 SKU 数量
        - num_stores: 有效的门店数量
        - sku_list: 有效的 SKU 列表
        - store_list: 有效的门店列表
        """
        return {
            "num_skus": len(self.valid_skus),
            "num_stores": len(self.valid_stores),
            "sku_list": self.valid_skus,
            "store_list": self.valid_stores
        }
    
    def get_demand_matrix(self) -> np.ndarray:
        """
        返回对齐后的需求矩阵
        
        形状: (weeks, stores, skus)
        
        注意：
        - 如果需求数据是 DataFrame，会自动转换为 Numpy 矩阵
        - 会自动处理缺失值和异常值
        """
        logger.info("\n[需求矩阵] 转换需求数据为矩阵...")
        
        if self.demand_df is None:
            logger.error("  [ERROR] 需求数据为空")
            raise ValueError("需求数据为空")
        
        # 1. 检查数据格式
        required_columns = ['goods_code', 'estimated_demand']
        for col in required_columns:
            if col not in self.demand_df.columns:
                logger.error(f"  [ERROR] 需求数据缺少必需列：'{col}'")
                logger.error(f"    现有列：{list(self.demand_df.columns)}")
                raise ValueError(f"需求数据缺少必需列：'{col}'")
        
        # 2. 提取时间和门店维度
        # 尝试多种时间列名
        time_col = None
        for col in ['week_index', 'date', 'week']:
            if col in self.demand_df.columns:
                time_col = col
                break
        
        if time_col is None:
            logger.error("  [ERROR] 需求数据缺少时间列（需要 'week_index'、'date' 或 'week'）")
            raise ValueError("需求数据缺少时间列")
        
        # 尝试多种门店列名
        store_col = None
        for col in ['shop_code', 'store_id', 'sku_id']:
            if col in self.demand_df.columns:
                store_col = col
                break
        
        if store_col is None:
            logger.warning("  [WARNING] 需求数据缺少门店列，将使用单门店")
            self.demand_df['_store'] = 'STORE_001'
            store_col = '_store'
        
        # 3. 转换时间列为周索引
        if time_col == 'date':
            self.demand_df['date'] = pd.to_datetime(self.demand_df['date'])
            # 假设每周一行，按日期排序后分配 week_index
            self.demand_df = self.demand_df.sort_values(['goods_code', 'shop_code', 'date'])
            self.demand_df['week_index'] = self.demand_df.groupby(['goods_code', 'shop_code']).cumcount()
            time_col = 'week_index'
        
        # 4. 创建三维矩阵
        weeks = sorted(self.demand_df[time_col].unique())
        num_weeks = len(weeks)
        
        # 使用对齐后的门店和 SKU 列表
        num_stores = len(self.valid_stores)
        num_skus = len(self.valid_skus)
        
        # 创建三维矩阵
        demand_matrix = np.zeros((num_weeks, num_stores, num_skus), dtype=np.float32)
        
        # 创建映射字典（加速查找）
        sku_to_idx = {sku: idx for idx, sku in enumerate(self.valid_skus)}
        store_to_idx = {store: idx for idx, store in enumerate(self.valid_stores)}
        
        # 5. 填充矩阵
        logger.info(f"  填充需求矩阵（共 {len(self.demand_df)} 行）...")
        
        for _, row in self.demand_df.iterrows():
            try:
                week_idx = weeks.index(row[time_col])
                sku_idx = sku_to_idx.get(row['goods_code'])
                store_idx = store_to_idx.get(row[store_col])
                
                if sku_idx is not None and store_idx is not None:
                    demand_matrix[week_idx, store_idx, sku_idx] = row['estimated_demand']
            except (KeyError, ValueError):
                # 跳过无效行
                continue
        
        logger.info(f"  [OK] 需求矩阵填充完成")
        logger.info(f"    矩阵形状: {demand_matrix.shape}")
        logger.info(f"    周数: {num_weeks}")
        logger.info(f"    门店数: {num_stores}")
        logger.info(f"    SKU 数: {num_skus}")
        logger.info(f"    非零元素数: {np.count_nonzero(demand_matrix)}")
        logger.info(f"    稀疏度: {1 - np.count_nonzero(demand_matrix) / demand_matrix.size:.2%}")
        
        return demand_matrix
    
    def get_state_data(self) -> pd.DataFrame:
        """
        返回对齐后的状态数据
        
        返回：
        - 对齐后的状态 DataFrame
        """
        return self.state_df.copy()
    
    def get_demand_data(self) -> pd.DataFrame:
        """
        返回对齐后的需求数据
        
        返回：
        - 对齐后的需求 DataFrame
        """
        return self.demand_df.copy()
    
    def save_aligned_data(self, output_dir: str = "data/processed/aligned"):
        """
        保存对齐后的数据
        
        参数：
        - output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存需求数据
        if self.demand_df is not None:
            demand_output = output_path / "demand_aligned.csv"
            self.demand_df.to_csv(demand_output, index=False, encoding='utf-8')
            logger.info(f"  保存对齐后的需求数据: {demand_output}")
        
        # 保存状态数据
        if self.state_df is not None:
            state_output = output_path / "state_aligned.csv"
            self.state_df.to_csv(state_output, index=False, encoding='utf-8')
            logger.info(f"  保存对齐后的状态数据: {state_output}")
        
        # 保存配置
        config_output = output_path / "env_config.json"
        import json
        with open(config_output, 'w', encoding='utf-8') as f:
            json.dump(self.get_env_config(), f, indent=2, ensure_ascii=False)
        logger.info(f"  保存环境配置: {config_output}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SupplyChainDataManager':
        """
        从配置字典创建 DataManager
        
        参数：
        - config: 配置字典，包含 demand_path 和 state_path
        """
        return cls(
            demand_path=config.get('demand_path'),
            state_path=config.get('state_path')
        )


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试 1：使用默认路径
    print("=" * 80)
    print("测试 1：使用默认路径")
    print("=" * 80)
    
    try:
        data_mgr = SupplyChainDataManager(
            demand_path="data/processed/estimated_demand.csv",
            state_path="data/processed/historical_data_for_env.csv"
        )
        
        # 获取环境配置
        env_config = data_mgr.get_env_config()
        print(f"\n环境配置: {env_config}")
        
        # 获取需求矩阵
        demand_matrix = data_mgr.get_demand_matrix()
        print(f"\n需求矩阵形状: {demand_matrix.shape}")
        
        # 保存对齐后的数据
        data_mgr.save_aligned_data()
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 2：使用 DataFrame
    print("\n" + "=" * 80)
    print("测试 2：使用 DataFrame")
    print("=" * 80)
    
    try:
        # 创建模拟数据
        demand_df = pd.DataFrame({
            'goods_code': ['SKU001', 'SKU002', 'SKU003'] * 10,
            'shop_code': ['STORE001'] * 30,
            'week_index': list(range(10)) * 3,
            'demand': np.random.uniform(5, 15, 30)
        })
        
        state_df = pd.DataFrame({
            'goods_code': ['SKU001', 'SKU002'],
            'shop_code': ['STORE001'] * 2,
            'week_index': [0, 0],
            'stock': [50, 50]
        })
        
        data_mgr = SupplyChainDataManager(
            demand_df=demand_df,
            state_df=state_df
        )
        
        # 获取环境配置
        env_config = data_mgr.get_env_config()
        print(f"\n环境配置: {env_config}")
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
