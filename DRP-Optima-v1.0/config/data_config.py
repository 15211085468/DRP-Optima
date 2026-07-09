"""
配置管理模块 - config/data_config.py
包含PPO训练、环境、路径、数据等所有配置（V8 终极无坑版）
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .validator import validate_optuna_config
from utils.logger import get_logger
from .exceptions import ConfigError

logger = get_logger(__name__)


# ============= 项目根目录（模块级常量） =============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============= 路径配置 =============
@dataclass
class PathConfig:
    """路径配置类"""
    # 🔧 修复：使用模块级常量 PROJECT_ROOT（避免硬编码路径）
    project_root: str = PROJECT_ROOT
    
    # 主数据文件（预处理后的数据）
    data_path: str = os.path.join(PROJECT_ROOT, "data", "main_medium_test.csv")
    
    # 原始数据文件
    sales_file: str = os.path.join(PROJECT_ROOT, "data", "main_medium_test.csv")
    shop_file: str = os.path.join(PROJECT_ROOT, "data", "shop_info.csv")
    goods_file: str = os.path.join(PROJECT_ROOT, "data", "goods_info.csv")
    dc_stock_file: str = os.path.join(PROJECT_ROOT, "data", "shop_info.csv")  # 临时使用 shop_info.csv
    
    # 数据类型字典文件
    # 🔧 修复：改为相对路径（避免硬编码本地绝对路径）
    dtypes_dict_file: str = os.path.join(PROJECT_ROOT, "data", "raw", "main_dtypes_dict.json")
    
    # 输出目录
    models_dir: str = os.path.join(PROJECT_ROOT, "models")
    output_dir: str = os.path.join(PROJECT_ROOT, "output")
    log_dir: str = os.path.join(output_dir, "logs")
    
    def __post_init__(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# ============= 数据配置 =============
@dataclass
class DataConfig:
    """数据配置类"""
    # 🔧 修复：改为相对路径（避免硬编码本地绝对路径）
    data_path: str = os.path.join(PROJECT_ROOT, "data", "raw", "main.csv")
    nrows: Optional[int] = None
    sku_limit: int = 500
    num_skus: int = 500  # Trainer 别名，与 sku_limit 同义
    store_limit: int = 100
    num_train_stores: int = 100  # Trainer 别名，与 store_limit 同义
    test_sku_limit: int = 100
    test_store_limit: int = 10
    test_weeks: int = 7  # 测试集周数（用于分割训练/验证集）
    
    # 特征工程配置（供 FeatureEngineer 使用）
    lag_windows: List[int] = None  # 滞后特征窗口，默认 [1, 2, 4]
    rolling_windows: List[int] = None  # 滚动统计窗口，默认 [4, 12, 26, 52]
    rolling_stats: List[str] = None  # 滚动统计量，默认 ['mean', 'std', 'max']
    
    # 时间对齐配置（用于周度→日度转换）
    demo_days_per_week: int = 7  # 每周天数（用于周度→日度转换）
    demo_daily_demand_fluctuation: tuple = (0.8, 1.2)  # 日度需求波动范围
    demo_random_seed: int = 42  # 随机种子（确保可重现）
    
    def __post_init__(self):
        """设置默认值（使用列表字面量会在 dataclass 中共享状态，需在此处初始化）"""
        if self.lag_windows is None:
            self.lag_windows = [1, 2, 4]
        if self.rolling_windows is None:
            self.rolling_windows = [4, 12, 26, 52]
        if self.rolling_stats is None:
            self.rolling_stats = ['mean', 'std', 'max']


# ============= 环境配置 =============
@dataclass
class EnvConfig:
    """
    环境配置类（V8 终极无坑版）
    
    核心改进：
    1. 恢复 60 件物理上限（废除 30 件硬截断）
    2. 下调 Sigmoid 阈值至 0.85（确保训练初期有密集梯度信号）
    3. 纠正熵系数衰减方向（初期高熵 0.05，后期低熵 0.001）
    4. 保留小额补货惩罚（系数 1.0，防止"因噎废食"）
    """
    
    # ============ 1. 动作与物理约束 ============
    base_reorder_unit: float = 100.0
    max_order_quantity_ratio: float = 0.6  # 🟢 恢复 60.0 上限 (100 * 0.6)，废除 30 的硬限制
    min_reorder_qty: float = 2.0       # 业务最小起订量 (仅作为软惩罚阈值，不做物理截断)

    # ============ 2. 全局业务硬约束 ============
    max_store_capacity_units: float = 5000.0  # 门店后仓最大物理容量（件）
    max_store_budget: float = 500000.0     # 门店最大库存资金占用（元）

    # ============ 3. Sigmoid+Log 混合满足率奖励（V8.2） ============
    fill_reward_scale: float = 3.0         # 满足率基础奖励权重
    sigmoid_threshold: float = 0.85        # 同 V8.1，保持阈值陡峭度
    sigmoid_steepness: float = 15.0      # 同 V8.1，Sigmoid 陡峭度
    log_k: float = 5.0                   # V8.2 新增：对数项形状参数，控制曲率
    w_sig: float = 0.85                   # V8.2 新增：Sigmoid 项权重占比（对数项占 15%）

    # ============ 4. 成本与溢出惩罚 ============
    stockout_penalty_per_unit: float = 2.0  # 缺货惩罚 (每件)
    
    holding_cost_per_unit_weekly: float = 0.02  # 基础持有成本 (消除魔法数字)
    holding_cost_penalty_scale: float = 0.1    # 持有成本整体缩放系数
    holding_excess_penalty_multiplier: float = 20.0  # 超过 70% 容量时的二次惩罚乘数
    
    unit_overflow_penalty_rate: float = 0.5   # 容量溢出惩罚 (每超出 1 件扣 0.5)
    budget_overflow_penalty_rate: float = 0.01  # V8.1 方案：恢复至 0.01

    # ============ 5. 行为引导 (微操抑制与不补货奖励) ============
    small_reorder_penalty_rate: float = 1.5  # V8.1 方案：从 1.0 上调至 1.5，抑制 1 件补货
    no_reorder_bonus_per_sku: float = 0.0001  # V8.1 方案：从 0.001 下调至 0.0001，避免奖励过大
    no_reorder_inventory_threshold: float = 0.7  # 触发不补货奖励的全局库存阈值 (70%)

    # ============ 6. 奖励裁剪 ============
    reward_clip_min: float = -10.0
    reward_clip_max: float = 10.0
    
    # ============ 7. 简化奖励模式（用于跑通测试） ============
    simple_reward: bool = False  # 如果为True，只奖励Fill Rate，去掉所有惩罚
    
    # ============ 8. 向后兼容参数 ============
    max_sku_inventory: float = 200.0  # 单品最大库存（用于计算库存比例）


# ============= PPO 配置 =============
@dataclass
class PPOConfig:
    """
    PPO 配置类（V8 终极无坑版）
    
    核心改进：
    1. 纠正熵系数衰减方向（初期高熵 0.05，后期低熵 0.001）
    2. 下调价值函数系数（从 1.5 降至 0.5，防止 Critic 更新过猛）
    3. 增加训练步数（200 万步，确保充分训练）
    """
    # 训练参数（针对高维动作空间优化）
    total_timesteps: int = 2_000_000  # 🟢 建议 200 万步 (100 万是底线，94 个 SKU 空间较大)
    n_steps: int = 4096            # 每次更新的步数 (较长轨迹适合高维空间)
    batch_size: int = 256           # 批次大小
    n_epochs: int = 10              # 每次更新的 epoch 数
    
    # 超参数（针对高维动作空间调整）
    policy_lr: float = 1e-4         # 初始学习率 (配合 linear 衰减)
    gamma: float = 0.99             # 折扣因子（标准值）
    gae_lambda: float = 0.95        # GAE Lambda（平衡偏差和方差）
    clip_range: float = 0.2          # PPO 标准裁剪范围
    clip_range_vf: float = 0.2       # Value function 裁剪范围
    ent_coef: float = 0.05          # 🟢 初始高熵 (配合热启动，强行冲刷旧策略偏见)
    vf_coef: float = 0.5            # 🟢 从 1.5 下调至 0.5，防止 Critic 更新过猛掩盖 Actor 梯度
    max_grad_norm: float = 0.5       # 梯度裁剪阈值
    
    # 网络架构（大容量网络，拟合复杂策略）
    net_arch: List[int] = field(default_factory=lambda: [512, 256, 128])  # 三层 MLP
    
    # 学习率和熵衰减（线性衰减，确保收敛）
    lr_schedule: str = "linear"       # 学习率衰减策略
    ent_schedule: str = "linear"      # 熵系数衰减策略
    lr_end: float = 1e-6             # 最终学习率
    ent_end: float = 0.001           # 🟢 最终熵系数 (后期收敛)
    
    # 评估频率
    eval_freq: int = 50_000
    n_eval_episodes: int = 5
    
    # 探索参数（PPO 通常不需要探索噪声）
    exploration_initial_eps: float = 0.0
    exploration_fraction: float = 0.0
    exploration_final_eps: float = 0.0
    
    # 奖励权重
    weight_ofr_reward: float = 2.0
    weight_itr_reward: float = 1.0
    weight_sor_penalty: float = 3.0
    
    # 奖励裁剪
    reward_clip_min: float = -10.0
    reward_clip_max: float = 10.0
    
    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [512, 256, 128]


# ============= 兼容层：get_config() 函数 =============
def get_config():
    """
    获取完整配置对象（兼容旧代码）
    """
    from types import SimpleNamespace
    
    config = SimpleNamespace()
    config.paths = PathConfig()
    config.data = DataConfig()
    config.model = TFTModelConfig()
    config.hardware = HardwareConfig()
    config.env = EnvConfig()
    config.ppo = PPOConfig()
    
    # 同步别名属性（确保 Trainer 能正常工作）
    config.data.num_skus = config.data.sku_limit
    
    return config


# ============= TFT 模型配置 =============
@dataclass
class TFTModelConfig:
    """TFT 模型配置类（支持决策感知联合训练）"""
    # 模型结构参数
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    output_size: int = 1
    max_prediction_length: int = 1
    max_encoder_length: int = 30
    min_encoder_length_ratio: float = 0.5  # min_encoder_length = max_encoder_length * ratio
    
    # 数据预处理参数
    target_normalizer: str = "standard"  # "standard" 或 TorchNormalizer 实例
    add_relative_time_idx: bool = True
    add_target_scales: bool = True
    add_encoder_length: bool = True
    allow_missing_timesteps: bool = False
    
    # 损失函数类型
    loss_type: str = "quantile"  # 修改为 quantile，启用分位数预测
    quantiles: List[float] = None  # 分位数：将在 __post_init__ 中设置为 [0.1, 0.5, 0.9]
    
    # 训练参数
    learning_rate: float = 0.001
    optimizer: str = "adam"  # "adam" 或 "sgd"
    batch_size: int = 32  # 批次大小（用于 create_dataloaders）
    log_interval: int = 10
    max_epochs: int = 50
    reduce_on_plateau_patience: int = 5
    gradient_clip_val: float = 0.5
    limit_train_batches: float = 1.0  # 可用于快速调试（如 0.1 = 10% 数据）
    
    # EarlyStopping 参数
    early_stop_min_delta: float = 0.001
    early_stop_patience: int = 10
    
    # ============= 决策感知联合训练参数 =============
    use_decision_aware: bool = False  # 是否启用决策感知训练
    decision_weight: float = 0.1  # 决策成本在总损失中的权重 (lambda)
    holding_cost: float = 1.0  # 单位库存持有成本 (Ch)
    shortage_penalty: float = 5.0  # 单位缺货惩罚成本 (Cp)
    decision_smooth: bool = True  # 是否使用 Softplus 平滑（保证严格可导）
    decision_warmup_epochs: int = 0  # 预热轮数（在此期间 decision_weight 从 0 线性增加）
    
    # ============= 数据集参数（训练时设置，保存至 checkpoint，加载时恢复）=============
    # 这些参数确保训练和预测使用完全相同的数据格式
    dataset_time_idx: str = "time_idx"
    dataset_target: str = "sale_qty"
    dataset_group_ids: List[str] = None  # 将在 __post_init__ 中设置默认值
    dataset_static_categoricals: List[str] = None  # 将在 __post_init__ 中设置默认值
    dataset_static_reals: List[str] = None  # 将在 __post_init__ 中设置默认值
    dataset_time_varying_known_reals: List[str] = None  # 将在 __post_init__ 中设置默认值
    dataset_time_varying_unknown_reals: List[str] = None  # 将在 __post_init__ 中设置默认值
    dataset_max_encoder_length: int = 30
    dataset_max_prediction_length: int = 7
    dataset_min_encoder_length: Optional[int] = None
    dataset_target_normalizer: str = "standard"
    dataset_add_relative_time_idx: bool = True
    dataset_add_target_scales: bool = True
    dataset_add_encoder_length: bool = True
    
    def __post_init__(self):
        """设置列表字段的默认值（避免使用可变默认参数）"""
        if self.dataset_group_ids is None:
            self.dataset_group_ids = ['shop_code', 'goods_code']
        if self.dataset_static_categoricals is None:
            self.dataset_static_categoricals = ['shop_code', 'goods_code']
        if self.dataset_static_reals is None:
            self.dataset_static_reals = []
        if self.dataset_time_varying_known_reals is None:
            self.dataset_time_varying_known_reals = []
        if self.dataset_time_varying_unknown_reals is None:
            self.dataset_time_varying_unknown_reals = ['sale_qty']
        if self.dataset_min_encoder_length is None:
            self.dataset_min_encoder_length = int(self.dataset_max_encoder_length * 0.5)
        
        # 设置分位数默认值（如果未提供）
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]  # 默认使用3个分位数


# ============= 硬件配置 =============
@dataclass
class HardwareConfig:
    """硬件配置（简化版，仅保留必要字段）"""
    device: str = "auto"
    num_workers: int = 0
    float32_matmul_precision: str = "high"


# 全局配置实例（兼容旧代码）
path_config = PathConfig()
data_config = DataConfig()
env_config = EnvConfig()
ppo_config = PPOConfig()
tft_config = TFTModelConfig()


def update_ppo_config(config: PPOConfig, **kwargs) -> PPOConfig:
    """更新 PPO 配置"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Invalid config parameter: {key}")
    return config


# ============= V10.3 环境配置 =============
@dataclass
class EnvConfigV9:
    """
    环境配置类（V10.3 生产方案）
    
    核心改进：
    1. MultiDiscrete 动作空间（7 bins × 88 SKUs）
    2. Action Masking（掩码非法动作）
    3. Curriculum Learning（软惩罚 → 硬截断）
    4. 评估隔离（eval_mode + eval_stores）
    5. 向量化需求查询（demand_matrix）
    6. 数据循环保护（allow_demand_loop）
    """
    
    # ============ 1. 环境与数据配置 ============
    num_stores: int = 82              # 门店数量
    num_skus: int = 88               # SKU 数量（82店 × 88 SKU）
    max_weeks: int = 100             # 每回合最大周数
    data_weeks: int = 94             # 数据中的周数（estimated_demand_weekly.csv）
    lead_time_weeks: int = 1         # 补货提前期（周，周期性Review）
    
    # ============ 2. 动作空间配置 ============
    # MultiDiscrete: 每个 SKU 有 7 个离散动作 (0-6 coverage weeks)
    action_n_bins: int = 7           # 每个 SKU 的动作桶数
    coverage_bins: List[float] = None  # 覆盖周数分桶 [0, 1, 2, 3, 4, 5, 6]
    
    # ============ 3. 观察空间配置 ============
    # 观察空间维度: 2*num_skus + 4
    # - current_stock (num_skus)
    # - pending_order (num_skus)
    # - store_features (2): [store_avg_weekly_demand, store_std_weekly_demand]
    # - ratios (2): [fill_rate, stockout_rate]
    obs_include_store_features: bool = True
    obs_include_ratios: bool = True
    
    # ============ 4. 覆盖目标（按 SKU 类别） ============
    cov_target_high: float = 0.95     # A 类 SKU 覆盖目标（高周转）
    cov_target_med: float = 0.90      # B 类 SKU 覆盖目标（中周转）
    cov_target_low: float = 0.85      # C 类 SKU 覆盖目标（低周转）
    
    # ============ 5. 奖励权重配置 ============
    # 🔧 改造3：重新平衡Reward权重（让惩罚足够痛）
    # 5.1 满足率奖励（正向）
    reward_fill_rate_weight: float = 5.0  # 3.0 → 5.0 (增加满足率奖励)
    fill_rate_smooth: bool = True      # 使用 Sigmoid 平滑满足率奖励
    
    # 5.2 缺货惩罚（必须非常痛！）
    penalty_stockout_per_unit: float = 10.0  # 2.0 → 10.0 (显著增加缺货惩罚)
    penalty_stockout_scale: float = 2.0   # 1.0 → 2.0 (增加缩放)
    
    # 5.3 溢出惩罚（库存过多）
    penalty_overstock_per_unit: float = 2.0  # 0.5 → 2.0 (增加溢出惩罚)
    penalty_overstock_scale: float = 1.5   # 1.0 → 1.5 (增加缩放)
    
    # 5.4 持有成本（细水长流的惩罚）
    holding_cost_per_unit: float = 0.1   # 0.02 → 0.1 (增加持有成本)
    holding_cost_scale: float = 0.5       # 0.1 → 0.5 (增加缩放)
    
    # 5.5 覆盖奖励（鼓励达到覆盖目标）
    coverage_reward_weight: float = 1.0
    coverage_cap: float = 2.0         # 覆盖奖励上限
    
    # 5.6 动作截断惩罚（Curriculum Learning）
    truncation_penalty: float = 5.0    # 硬截断惩罚（after curriculum）
    soft_penalty_scale: float = 0.1    # 软惩罚缩放（during curriculum）
    
    # ============ 6. Curriculum Learning 配置 ============
    enable_hard_truncation: bool = True   # 是否启用硬截断
    curriculum_soft_epochs: int = 50_000   # 软惩罚阶段步数（前 50k 步）
    curriculum_hard_epochs: int = 0        # 硬截断阶段起始步数（after 50k）
    
    # ============ 7. Action Masking 配置 ============
    # 掩码条件：当前库存 + 在途订单 > 容量上限时，掩码补货动作
    enable_action_mask: bool = True
    mask_capacity_threshold: float = 0.95  # 容量阈值（95% 时开始掩码）
    
    # ============ 8. 评估隔离配置 ============
    eval_mode: bool = False            # 是否评估模式
    eval_stores: List[str] = None      # 评估门店列表（None = 所有门店）
    
    # ============ 9. 数据循环与质量配置 ============
    allow_demand_loop: bool = True     # 需求数据循环（当 max_weeks > data_weeks 时）
    dropped_row_warn_ratio: float = 0.05  # 丢弃行警告比例（>5% 时警告）
    
    # ============ 10. 物理约束配置 ============
    # 🔧 修复：添加缺失的物理约束参数（避免getattr fallback）
    max_store_capacity_units: float = 5000.0  # 门店后仓最大物理容量
    
    # ============ 10. 数值稳定性 ============
    reward_eps: float = 1e-8         # 奖励计算 epsilon（防止除零）
    min_demand: float = 0.01          # 最小需求（防止全零）
    
    # ============ 11. 日志与调试 ============
    verbose: int = 1                  # 日志级别（0=静默, 1=信息, 2=调试）
    log_interval: int = 1000          # 日志间隔（步数）
    
    def __post_init__(self):
        """初始化列表字段和验证参数"""
        if self.coverage_bins is None:
            self.coverage_bins = [0, 1, 2, 3, 4, 5, 6]
        
        if self.eval_stores is None:
            self.eval_stores = []
        
        # 验证参数
        if self.num_skus <= 0:
            raise ValueError(f"num_skus must be positive, got {self.num_skus}")
        
        if self.num_stores <= 0:
            raise ValueError(f"num_stores must be positive, got {self.num_stores}")
        
        if self.max_weeks <= 0:
            raise ValueError(f"max_weeks must be positive, got {self.max_weeks}")
        
        if self.action_n_bins != len(self.coverage_bins):
            raise ValueError(
                f"action_n_bins ({self.action_n_bins}) != len(coverage_bins) ({len(self.coverage_bins)})"
            )
        
        if self.verbose >= 1:
            logger.info(f"EnvConfigV9 初始化完成:")
            logger.info(f"  - num_stores: {self.num_stores}")
            logger.info(f"  - num_skus: {self.num_skus}")
            logger.info(f"  - action_space: MultiDiscrete({self.action_n_bins} × {self.num_skus})")
            logger.info(f"  - max_weeks: {self.max_weeks}")
            logger.info(f"  - curriculum_soft_epochs: {self.curriculum_soft_epochs}")
            logger.info(f"  - enable_hard_truncation: {self.enable_hard_truncation}")


# ============= 线性衰减函数（用于学习率和熵系数）=============
def linear_schedule(initial_value, final_value):
    """
    线性衰减函数
    
    参数：
        initial_value: 初始值
        final_value: 最终值
    
    返回：
        函数 f(progress_remaining) -> current_value
    """
    def func(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return func
