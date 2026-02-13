"""
Utility functions for EDA pipeline
Includes logging setup, validation, and common operations
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from Config import EDAConfig


def setup_logging(config: EDAConfig) -> logging.Logger:
    """
    Configure logging with both file and console handlers
    
    Args:
        config: EDAConfig instance with logging settings
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger('FraudEDA')
    logger.setLevel(config.LOG_LEVEL)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Console handler
    if config.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.LOG_TO_FILE:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setLevel(config.LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_dataframe(df: pd.DataFrame, logger: logging.Logger) -> bool:
    """
    Validate input dataframe meets basic requirements
    
    Args:
        df: Input dataframe
        logger: Logger instance
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if df is None:
        raise ValueError("Dataframe is None")
    
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    if len(df) < 10:
        logger.warning(f"Very small dataset: only {len(df)} rows")
    
    if len(df.columns) < 3:
        raise ValueError(f"Too few columns: {len(df.columns)}")
    
    logger.info(f"Dataframe validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def validate_file_exists(filepath: str, logger: logging.Logger) -> bool:
    """
    Check if file exists before attempting to read
    
    Args:
        filepath: Path to file
        logger: Logger instance
        
    Returns:
        True if exists, raises FileNotFoundError otherwise
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.info(f"File found: {filepath}")
    return True


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Result of division or default value
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        return default


def get_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get list of numeric columns, excluding specified columns
    
    Args:
        df: Input dataframe
        exclude: List of column names to exclude
        
    Returns:
        List of numeric column names
    """
    exclude = exclude or []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def get_categorical_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Get list of categorical columns, excluding specified columns
    
    Args:
        df: Input dataframe
        exclude: List of column names to exclude
        
    Returns:
        List of categorical column names
    """
    exclude = exclude or []
    categorical_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    return [col for col in categorical_cols if col not in exclude]


def calculate_data_quality_score(df: pd.DataFrame, config: EDAConfig, logger: logging.Logger) -> Dict[str, float]:
    """
    Calculate comprehensive data quality metrics
    
    Args:
        df: Input dataframe
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Dictionary with quality metrics
    """
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    
    # Completeness: percentage of non-null values
    completeness = safe_divide((total_cells - missing_cells), total_cells, 0) * 100
    
    # Uniqueness: percentage of non-duplicate rows
    duplicates = df.duplicated().sum()
    uniqueness = safe_divide((len(df) - duplicates), len(df), 0) * 100
    
    # Consistency: check data type consistency within columns
    consistency_score = 100.0  # Start with perfect score
    inconsistent_cols = 0
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column should be numeric
            try:
                pd.to_numeric(df[col].dropna(), errors='raise')
                # If successful, column is consistently numeric but typed as object
                inconsistent_cols += 1
            except (ValueError, TypeError):
                pass  # Column is properly categorical
    
    if len(df.columns) > 0:
        consistency_score = (1 - inconsistent_cols / len(df.columns)) * 100
    
    # Validity: check for valid values (basic checks)
    validity_score = 100.0
    invalid_count = 0
    
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        # Check for infinite values
        if np.isinf(df[col]).any():
            invalid_count += 1
    
    if len(numeric_cols) > 0:
        validity_score = (1 - invalid_count / len(numeric_cols)) * 100
    
    quality_metrics = {
        'completeness': completeness,
        'uniqueness': uniqueness,
        'consistency': consistency_score,
        'validity': validity_score,
    }
    
    # Overall quality score (weighted average)
    overall_quality = np.mean([
        quality_metrics['completeness'] * 0.4,  # Completeness is most important
        quality_metrics['uniqueness'] * 0.2,
        quality_metrics['consistency'] * 0.2,
        quality_metrics['validity'] * 0.2,
    ])
    
    quality_metrics['overall'] = overall_quality
    
    logger.info(f"Data quality calculated - Overall: {overall_quality:.2f}%")
    
    return quality_metrics


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> Dict[str, Any]:
    """
    Detect outliers using IQR method
    
    Args:
        series: Pandas series
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Dictionary with outlier information
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers_mask = (series < lower_bound) | (series > upper_bound)
    
    return {
        'count': int(outliers_mask.sum()),
        'percentage': float(safe_divide(outliers_mask.sum(), len(series), 0) * 100),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'iqr': float(IQR),
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detect outliers using Z-score method
    
    Args:
        series: Pandas series
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Dictionary with outlier information
    """
    from scipy import stats
    
    z_scores = np.abs(stats.zscore(series.dropna()))
    outliers_mask = z_scores > threshold
    
    return {
        'count': int(outliers_mask.sum()),
        'percentage': float(safe_divide(outliers_mask.sum(), len(series), 0) * 100),
        'threshold': float(threshold),
    }


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def print_section_header(title: str, width: int = 120, logger: Optional[logging.Logger] = None):
    """
    Print formatted section header
    
    Args:
        title: Section title
        width: Width of separator line
        logger: Optional logger instance
    """
    separator = "=" * width
    message = f"\n{separator}\n{title}\n{separator}"
    
    if logger:
        logger.info(title)
    print(message)


def create_output_directory(output_dir: str = "eda_outputs", logger: Optional[logging.Logger] = None) -> Path:
    """
    Create output directory if it doesn't exist
    
    Args:
        output_dir: Directory name
        logger: Optional logger instance
        
    Returns:
        Path object for the directory
    """
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    
    if logger:
        logger.info(f"Output directory: {path.absolute()}")
    
    return path