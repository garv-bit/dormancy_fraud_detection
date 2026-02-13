"""
Configuration file for Financial Fraud Detection EDA
All magic numbers and settings are defined here for easy modification
"""

from dataclasses import dataclass
from typing import Optional
import logging


@dataclass
class EDAConfig:
    """Configuration class for EDA parameters"""
    
    # ============================================================================
    # FILE PATHS
    # ============================================================================
    INPUT_CSV: str = "Dataset/financial_fraud_detection_dataset.csv"
    OUTPUT_PICKLE: str = "eda_results.joblib"
    OUTPUT_PDF: str = "eda_report.pdf"
    LOG_FILE: str = "eda_analysis.log"
    
    # ============================================================================
    # STATISTICAL THRESHOLDS
    # ============================================================================
    CORRELATION_THRESHOLD: float = 0.5  # Threshold for strong correlation
    ALPHA: float = 0.05  # Significance level for hypothesis tests
    IQR_MULTIPLIER: float = 1.5  # Standard IQR multiplier for outliers
    Z_SCORE_THRESHOLD: float = 3.0  # Z-score threshold for outliers
    NORMALITY_ALPHA: float = 0.05  # Alpha for normality tests
    
    # ============================================================================
    # MACHINE LEARNING PARAMETERS
    # ============================================================================
    N_ESTIMATORS: int = 100  # Number of trees in Random Forest
    RANDOM_STATE: int = 42  # Random state for reproducibility
    N_JOBS: int = -1  # Use all CPU cores
    PCA_VARIANCE_THRESHOLD: float = 0.95  # Variance to retain in PCA
    MIN_CLUSTERS: int = 2  # Minimum number of clusters for K-means
    MAX_CLUSTERS: int = 11  # Maximum number of clusters to test
    TEST_SIZE: float = 0.2  # Train/test split ratio
    CV_FOLDS: int = 5  # Cross-validation folds
    
    # ============================================================================
    # DATA QUALITY THRESHOLDS
    # ============================================================================
    COMPLETENESS_THRESHOLD: float = 90.0  # Minimum completeness percentage
    MIN_SAMPLE_SIZE: int = 1000  # Minimum number of samples
    MIN_FRAUD_RATE: float = 0.001  # Minimum fraud rate (0.1%)
    MAX_FRAUD_RATE: float = 0.99  # Maximum fraud rate
    MIN_FEATURES: int = 5  # Minimum number of features
    
    # ============================================================================
    # VISUALIZATION PARAMETERS
    # ============================================================================
    PLOT_DPI: int = 300  # DPI for saved plots
    PLOT_STYLE: str = "seaborn-v0_8-darkgrid"
    COLOR_PALETTE: str = "husl"
    FIGURE_WIDTH: int = 16  # Base figure width
    SUBPLOT_HEIGHT: int = 4  # Height per subplot row
    COLS_PER_ROW: int = 4  # Columns per row in subplot grids
    HISTOGRAM_BINS: int = 30  # Number of bins for histograms
    
    # ============================================================================
    # DISPLAY PARAMETERS
    # ============================================================================
    MAX_CATEGORICAL_UNIQUE: int = 20  # Max unique values to show for categorical
    MAX_CATEGORICAL_DISPLAY: int = 10  # Max categories to display in plots
    TOP_N_VALUES: int = 5  # Top N values to show in reports
    TOP_N_FEATURES: int = 15  # Top N features for importance plots
    MAX_CHI_SQUARE_TESTS: int = 10  # Maximum chi-square tests to perform
    
    # ============================================================================
    # MEMORY OPTIMIZATION
    # ============================================================================
    STORE_RAW_DATA: bool = False  # Don't store raw data in results
    SAMPLE_SIZE_FOR_STORAGE: int = 100  # Sample size if storing data
    MAX_PCA_POINTS: int = 1000  # Max points to store for PCA visualization
    MAX_CLUSTER_POINTS: int = 1000  # Max points to store for clustering viz
    
    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True
    
    # ============================================================================
    # FEATURE ENGINEERING
    # ============================================================================
    TEMPORAL_FEATURES: bool = True  # Extract temporal features
    WEEKEND_DAYS: list = None  # Will be set to [5, 6] in __post_init__
    
    # ============================================================================
    # EXCLUSION LISTS
    # ============================================================================
    ID_COLUMNS: list = None  # Will be set in __post_init__
    SKIP_CATEGORICAL_ANALYSIS: list = None  # Will be set in __post_init__
    
    def __post_init__(self):
        """Initialize lists after dataclass creation"""
        if self.WEEKEND_DAYS is None:
            self.WEEKEND_DAYS = [5, 6]  # Saturday, Sunday
        
        if self.ID_COLUMNS is None:
            self.ID_COLUMNS = ["transaction_id", "ip_address"]
        
        if self.SKIP_CATEGORICAL_ANALYSIS is None:
            self.SKIP_CATEGORICAL_ANALYSIS = ["transaction_id", "timestamp", "ip_address"]
    
    def validate(self):
        """Validate configuration parameters"""
        assert 0 < self.CORRELATION_THRESHOLD <= 1, "Correlation threshold must be between 0 and 1"
        assert 0 < self.ALPHA < 1, "Alpha must be between 0 and 1"
        assert self.IQR_MULTIPLIER > 0, "IQR multiplier must be positive"
        assert self.Z_SCORE_THRESHOLD > 0, "Z-score threshold must be positive"
        assert self.N_ESTIMATORS > 0, "Number of estimators must be positive"
        assert 0 < self.PCA_VARIANCE_THRESHOLD <= 1, "PCA variance threshold must be between 0 and 1"
        assert self.MIN_CLUSTERS >= 2, "Minimum clusters must be at least 2"
        assert self.MIN_CLUSTERS < self.MAX_CLUSTERS, "Min clusters must be less than max clusters"
        assert 0 < self.TEST_SIZE < 1, "Test size must be between 0 and 1"
        assert self.CV_FOLDS > 1, "CV folds must be greater than 1"
        assert 0 <= self.COMPLETENESS_THRESHOLD <= 100, "Completeness threshold must be between 0 and 100"
        assert self.MIN_SAMPLE_SIZE > 0, "Minimum sample size must be positive"
        assert self.PLOT_DPI > 0, "DPI must be positive"
        
        return True


# Create default configuration instance
config = EDAConfig()