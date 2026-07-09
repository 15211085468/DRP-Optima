"""
Configuration Validator
配置验证模块 - 集中管理所有配置验证
"""

from .exceptions import ConfigError


def validate_gradient_clip_range(range_):
    """
    验证梯度裁剪范围
    
    Args:
        range_: 元组 (min_value, max_value)
    
    Raises:
        ConfigError: 如果范围无效
    """
    if range_[0] <= 0:
        raise ConfigError(
            f"gradient_clip_range[0] must be > 0, got {range_[0]}. "
            f"Suggestion: set to a positive value like 0.1"
        )
    if range_[0] > range_[1]:
        raise ConfigError(
            f"gradient_clip_range invalid: {range_[0]} > {range_[1]}. "
            f"Suggestion: ensure lower bound <= upper bound"
        )


def validate_hidden_size_range(range_):
    """
    验证隐藏层大小范围
    
    Args:
        range_: 元组 (min_value, max_value)
    
    Raises:
        ConfigError: 如果范围无效
    """
    if range_[0] < 1:
        raise ConfigError(
            f"hidden_size must be >= 1, got {range_[0]}. "
            f"Suggestion: use a positive integer like 64 or 128"
        )
    if range_[0] > range_[1]:
        raise ConfigError(
            f"hidden_size range invalid: {range_[0]} > {range_[1]}. "
            f"Suggestion: lower bound should be <= upper bound"
        )


def validate_dropout(value):
    """
    验证dropout值
    
    Args:
        value: dropout概率值
    
    Raises:
        ConfigError: 如果值不在(0,1)区间内
    """
    if not (0 < value < 1):
        raise ConfigError(
            f"dropout must be in (0,1), got {value}. "
            f"Suggestion: use a value between 0.1 and 0.5"
        )


def validate_dropout_range(range_):
    """
    验证dropout范围
    
    Args:
        range_: 元组 (min_value, max_value)
    
    Raises:
        ConfigError: 如果范围无效
    """
    if not (0 < range_[0] < 1):
        raise ConfigError(
            f"dropout_range[0] must be in (0,1), got {range_[0]}. "
            f"Suggestion: use a value between 0.1 and 0.5"
        )
    if not (0 < range_[1] < 1):
        raise ConfigError(
            f"dropout_range[1] must be in (0,1), got {range_[1]}. "
            f"Suggestion: use a value between 0.1 and 0.5"
        )
    if range_[0] > range_[1]:
        raise ConfigError(
            f"dropout_range invalid: {range_[0]} > {range_[1]}. "
            f"Suggestion: ensure lower bound <= upper bound"
        )


def validate_attention_head_range(range_):
    """
    验证注意力头数量范围
    
    Args:
        range_: 元组 (min_value, max_value)
    
    Raises:
        ConfigError: 如果范围无效
    """
    if range_[0] < 1:
        raise ConfigError(
            f"attention_head_range[0] must be >= 1, got {range_[0]}. "
            f"Suggestion: use a positive integer like 1, 2, 4"
        )
    if range_[0] > range_[1]:
        raise ConfigError(
            f"attention_head_range invalid: {range_[0]} > {range_[1]}. "
            f"Suggestion: ensure lower bound <= upper bound"
        )


def validate_learning_rate_range(range_):
    """
    验证学习率范围
    
    Args:
        range_: 元组 (min_value, max_value)
    
    Raises:
        ConfigError: 如果范围无效
    """
    if range_[0] <= 0:
        raise ConfigError(
            f"learning_rate must be > 0, got {range_[0]}. "
            f"Suggestion: try values like 1e-4 or 1e-3"
        )
    if range_[0] > range_[1]:
        raise ConfigError(
            f"learning_rate range invalid: {range_[0]} > {range_[1]}. "
            f"Suggestion: ensure lower bound <= upper bound"
        )


def validate_all(config):
    """
    验证所有配置参数
    
    Args:
        config: 配置对象，包含 gradient_clip_range, hidden_size_range, 
                dropout, learning_rate_range 等属性
    
    Raises:
        ConfigError: 如果任何配置参数无效
    """
    validate_gradient_clip_range(config.gradient_clip_range)
    validate_hidden_size_range(config.hidden_size_range)
    validate_dropout(config.dropout)
    validate_learning_rate_range(config.learning_rate_range)


def validate_optuna_config(config):
    """
    验证Optuna配置参数
    
    Args:
        config: OptunaConfig对象，包含 gradient_clip_range, hidden_size_range,
                dropout_range, learning_rate_range, attention_head_range 等属性
    
    Raises:
        ConfigError: 如果任何配置参数无效
    """
    validate_gradient_clip_range(config.gradient_clip_range)
    validate_hidden_size_range(config.hidden_size_range)
    validate_dropout_range(config.dropout_range)
    validate_learning_rate_range(config.learning_rate_range)
    validate_attention_head_range(config.attention_head_range)
