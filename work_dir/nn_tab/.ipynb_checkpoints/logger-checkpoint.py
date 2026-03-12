import logging
import os
from datetime import datetime
import sys


def setup_logger(name="nntab", level=logging.INFO, log_file=None, exp_name=None):
    """
    Set up a comprehensive logger for the neural network tabular package.
    
    Args:
        name (str): Logger name (default: "nntab")
        level: Logging level (default: INFO)
        log_file (str): Custom log file path (optional)
        exp_name (str): Experiment name for log file naming (optional)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - always add
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler - create log file
    if log_file is None:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate log filename with timestamp and experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if exp_name:
            log_filename = f"{exp_name}_{timestamp}.log"
        else:
            log_filename = f"nntab_training_{timestamp}.log"
        
        log_file = os.path.join(logs_dir, log_filename)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log the setup
    logger.info(f"Logger initialized - Log file: {log_file}")
    logger.info(f"Logging level: {logging.getLevelName(level)}")
    
    return logger


def get_logger(name="nntab"):
    """
    Get the existing logger instance.
    
    Args:
        name (str): Logger name (default: "nntab")
    
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    Usage: class MyClass(LoggerMixin): ...
    Then use: self.logger.info("message")
    """
    
    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            self._logger = get_logger()
        return self._logger