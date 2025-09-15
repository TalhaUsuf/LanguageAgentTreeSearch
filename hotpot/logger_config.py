"""
Loguru-based logging configuration for LATS experiments.
Provides colored console output and detailed file logging.
"""

import os
from loguru import logger
import sys


def setup_logger(log_file_path: str = "logs/lats_experiment.log", debug_mode: bool = True):
    """
    Setup loguru logger with console and file handlers.
    
    Args:
        log_file_path: Path to the log file
        debug_mode: If True, file logging at DEBUG level, else INFO level
    """
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Console handler - INFO level with colors
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=console_format,
        level="INFO",
        colorize=True,
        enqueue=True
    )
    
    # File handler - DEBUG/INFO level with detailed format
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{extra[iteration]:<3} | "
        "{extra[node_depth]:<2} | "
        "{message}"
    )
    
    file_level = "DEBUG" if debug_mode else "INFO"
    logger.add(
        log_file_path,
        format=file_format,
        level=file_level,
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        # Add default context
        filter=lambda record: record["extra"].setdefault("iteration", "N/A") or 
                             record["extra"].setdefault("node_depth", "N/A") or True
    )
    
    return logger


def get_context_logger(iteration: int = None, node_depth: int = None):
    """
    Get a logger with context information for iteration and node depth.
    
    Args:
        iteration: Current LATS iteration
        node_depth: Current node depth in the tree
        
    Returns:
        Contextual logger instance
    """
    return logger.bind(
        iteration=iteration if iteration is not None else "N/A",
        node_depth=node_depth if node_depth is not None else "N/A"
    )


# Initialize logger with default settings
setup_logger()