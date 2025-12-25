import logging
import random
import numpy as np
import torch


def get_root_logger(log_file=None, log_level=logging.INFO):
    """
    Get root logger.
    
    Args:
        log_file (str, optional): File path for logging. Defaults to None.
        log_level (int, optional): Logging level. Defaults to logging.INFO.
    
    Returns:
        logging.Logger: Root logger.
    """
    logger = logging.getLogger('netmamba_fscil')
    
    # If the logger has been initialized, return it directly
    if logger.hasHandlers():
        return logger
    
    # Set logging format
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)
    
    # Add stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Add file handler if log_file is provided
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set logging level
    logger.setLevel(log_level)
    
    return logger


def set_random_seed(seed, deterministic=False):
    """
    Set random seed.
    
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set deterministic options for CUDNN backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
