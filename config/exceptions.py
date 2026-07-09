"""
DRP-Optima Custom Exceptions
自定义异常类模块
"""

class DRPError(Exception):
    """Base exception for DRP-Optima."""
    pass

class ConfigError(DRPError):
    """Configuration error."""
    pass

class DataError(DRPError):
    """Data loading or validation error."""
    pass

class ModelError(DRPError):
    """Model training or prediction error."""
    pass

class ModelNotTrainedError(DRPError):
    """Model used before training."""
    pass

class UnsupportedLossError(DRPError):
    """Unsupported loss function."""
    pass
