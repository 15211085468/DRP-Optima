"""
日志工具模块 - utils/logger.py
提供统一的日志记录接口，支持通过环境变量 DRP_LOG_LEVEL 控制日志级别
支持可选的文件日志输出
"""

import logging
import sys
import os
from pathlib import Path


def setup_basic_logger(log_file: str = None) -> logging.Logger:
    """
    极简日志配置（写死配置，避免权限问题）
    
    Args:
        log_file: 日志文件路径（如果为None，则仅输出到控制台）
    
    Returns:
        logging.Logger: root logger实例
    """
    # 强制移除所有已有handler（避免重复输出）
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # 设置日志级别
    log_level = os.getenv('DRP_LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)
    
    # 创建handler列表
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # 可选文件输出
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            handlers.append(file_handler)
        except:
            pass  # 文件创建失败则仅使用控制台
    
    # 配置root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        handlers=handlers,
        force=True  # Python 3.8+ 强制覆盖已有配置
    )
    
    return logging.getLogger()


def get_logger(name: str = __name__, level: int = None,
           log_file: str = None) -> logging.Logger:
    """
    获取logger实例
    
    Args:
        name: logger名称（通常使用__name__）
        level: 日志级别（如果为None，则从环境变量 DRP_LOG_LEVEL 读取，默认为 INFO）
        log_file: 日志文件路径（如果为None，则从环境变量 DRP_LOG_FILE 读取，默认不输出到文件）
    
    Returns:
        logging.Logger: logger实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 创建控制台handler
        handler = logging.StreamHandler(sys.stdout)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # 添加handler
        logger.addHandler(handler)
        
        # 支持通过环境变量 DRP_LOG_LEVEL 设置日志级别
        if level is None:
            log_level = os.getenv('DRP_LOG_LEVEL', 'INFO').upper()
            level = getattr(logging, log_level, logging.INFO)
        
        logger.setLevel(level)
        
        # 防止日志传播到根logger（避免重复输出）
        logger.propagate = False
    
    # 添加可选的文件handler
    if log_file is None:
        log_file = os.getenv('DRP_LOG_FILE', None)
    
    if log_file and not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(Path(log_file).resolve()) for h in logger.handlers):
        # 确保目录存在
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logger.level)
        logger.addHandler(file_handler)
    
    return logger


def configure_logging(verbose: int = 1, log_file: str = None):
    """
    配置日志（兼容旧代码）
    
    这是 setup_basic_logger 的包装函数，用于兼容训练脚本中的导入。
    
    Args:
        verbose: 日志级别（0=WARNING, 1=INFO, 2=DEBUG）
        log_file: 日志文件路径（可选）
    """
    # 将 verbose 转换为日志级别
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    
    # 设置环境变量（供 get_logger 使用）
    if level == logging.DEBUG:
        os.environ['DRP_LOG_LEVEL'] = 'DEBUG'
    elif level == logging.INFO:
        os.environ['DRP_LOG_LEVEL'] = 'INFO'
    else:
        os.environ['DRP_LOG_LEVEL'] = 'WARNING'
    
    if log_file:
        os.environ['DRP_LOG_FILE'] = log_file
    
    # 调用 setup_basic_logger 进行实际配置
    setup_basic_logger(log_file=log_file)
