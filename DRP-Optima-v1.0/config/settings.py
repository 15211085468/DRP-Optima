"""
统一配置管理 - DRP-Optima 项目（最小化可用版本）
使用 Pydantic BaseSettings，支持 .env 文件和环境变量覆盖
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from dataclasses import dataclass
import os

@dataclass
class MetricsConfig:
    """评估指标配置"""
    smape_weight: float = 1.0
    mae_weight: float = 1.0
    rmse_weight: float = 1.0
    mape_weight: float = 1.0
    bias_threshold: float = 0.1


class AppSettings(BaseSettings):
    """应用总配置（扁平化设计 - 最小化可用版本）"""
    
    # ========= 路径配置 =========
    DRP_PROJECT_ROOT: str = Field(
        default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        description="项目根目录"
    )
    DRP_DATA_PATH: str = Field(
        default="data/processed/real_sales_intersect.csv",
        description="销售数据路径（相对于项目根目录）"
    )
    DRP_INVENTORY_DATA_PATH: str = Field(
        default="data/processed/real_shop_stock_intersect.csv",
        description="库存数据路径（相对于项目根目录）"
    )
    DRP_OUTPUT_DIR: str = Field(
        default="output",
        description="输出目录（相对于项目根目录）"
    )
    DRP_MODELS_DIR: str = Field(
        default="models",
        description="模型保存目录（相对于项目根目录）"
    )
    DRP_LOG_DIR: str = Field(
        default="logs",
        description="日志目录（相对于输出目录）"
    )
    
    # ========= 数据配置 =========
    DRP_DATA_SKU_LIMIT: int = Field(
        default=500,
        ge=1,
        description="SKU数量限制"
    )
    DRP_DATA_STORE_LIMIT: int = Field(
        default=100,
        ge=1,
        description="门店数量限制"
    )
    
    # ========= PPO 配置 =========
    DRP_PPO_TOTAL_TIMESTEPS: int = Field(
        default=1_000_000,
        ge=10_000,
        description="总训练步数"
    )
    DRP_PPO_N_STEPS: int = Field(
        default=2048,
        ge=128,
        description="每次更新收集的样本数"
    )
    DRP_PPO_BATCH_SIZE: int = Field(
        default=512,
        ge=32,
        description="批次大小"
    )
    DRP_PPO_POLICY_LR: float = Field(
        default=5e-5,
        gt=0,
        description="策略学习率"
    )
    DRP_PPO_VF_COEF: float = Field(
        default=1.0,
        ge=0,
        description="价值函数系数"
    )
    
    # ========= 环境配置 =========
    DRP_ENV_LEAD_TIME: int = Field(
        default=7,
        ge=1,
        description="提前期（天）"
    )
    DRP_ENV_WAREHOUSE_CAPACITY: float = Field(
        default=100.0,
        gt=0,
        description="仓库容量（每SKU）"
    )
    
    # ========= 日志配置 =========
    DRP_LOG_LEVEL: str = Field(
        default="INFO",
        description="日志级别（DEBUG/INFO/WARNING/ERROR）"
    )
    
    # ========= 随机种子 =========
    DRP_SEED: int = Field(
        default=42,
        description="随机种子"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "DRP_",
    }
    
    def get_absolute_path(self, relative_path: str) -> str:
        """将相对路径转换为绝对路径"""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.DRP_PROJECT_ROOT, relative_path)
    
    def create_dirs(self):
        """创建必要的目录"""
        output_dir = self.get_absolute_path(self.DRP_OUTPUT_DIR)
        models_dir = self.get_absolute_path(self.DRP_MODELS_DIR)
        log_dir = os.path.join(output_dir, self.DRP_LOG_DIR)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        return {
            "output_dir": output_dir,
            "models_dir": models_dir,
            "log_dir": log_dir,
        }


# 全局单例
settings = AppSettings()


if __name__ == "__main__":
    # 测试配置加载
    print("=" * 70)
    print("  DRP-Optima 统一配置管理 - 测试")
    print("=" * 70)
    print(f"\n路径配置:")
    print(f"  DRP_PROJECT_ROOT: {settings.DRP_PROJECT_ROOT}")
    print(f"  DRP_DATA_PATH: {settings.DRP_DATA_PATH}")
    print(f"  DRP_OUTPUT_DIR: {settings.DRP_OUTPUT_DIR}")
    print(f"  DRP_MODELS_DIR: {settings.DRP_MODELS_DIR}")
    print(f"\n数据配置:")
    print(f"  DRP_DATA_SKU_LIMIT: {settings.DRP_DATA_SKU_LIMIT}")
    print(f"\nPPO配置:")
    print(f"  DRP_PPO_TOTAL_TIMESTEPS: {settings.DRP_PPO_TOTAL_TIMESTEPS}")
    print(f"  DRP_PPO_POLICY_LR: {settings.DRP_PPO_POLICY_LR}")
    print(f"\n环境配置:")
    print(f"  DRP_ENV_LEAD_TIME: {settings.DRP_ENV_LEAD_TIME}")
    print(f"\n" + "=" * 70)
    
    # 测试路径计算
    print("\n测试路径计算:")
    dirs = settings.create_dirs()
    print(f"  输出目录: {dirs['output_dir']}")
    print(f"  模型目录: {dirs['models_dir']}")
    print(f"  日志目录: {dirs['log_dir']}")
    
    print("\n" + "=" * 70)
    print("  ✅ 配置加载成功！")
    print("=" * 70)
