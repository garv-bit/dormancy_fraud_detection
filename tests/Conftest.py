"""
pytest configuration and shared fixtures
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_dataframe():
    """
    Create a sample dataframe for testing
    Simulates fraud detection dataset structure
    """
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(n_samples)],
        'amount': np.random.uniform(10, 1000, n_samples),
        'time_since_last_transaction': np.random.uniform(0, 30, n_samples),
        'spending_deviation_score': np.random.randn(n_samples),
        'velocity_score': np.random.randint(0, 20, n_samples),
        'geo_anomaly_score': np.random.uniform(0, 1, n_samples),
        'transaction_type': np.random.choice(['deposit', 'payment', 'transfer', 'withdrawal'], n_samples),
        'device_used': np.random.choice(['mobile', 'web', 'atm', 'pos'], n_samples),
        'location': np.random.choice(['New York', 'Tokyo', 'London', 'Singapore'], n_samples),
        'merchant_category': np.random.choice(['retail', 'travel', 'restaurant', 'entertainment'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.96, 0.04]),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.choice([True, False], n_samples)
    })
    
    return df


@pytest.fixture
def small_dataframe():
    """
    Create a very small dataframe for quick tests
    """
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['x', 'y', 'z', 'x', 'y']
    })


@pytest.fixture
def dataframe_with_missing():
    """
    Create dataframe with missing values
    """
    return pd.DataFrame({
        'complete': [1, 2, 3, 4, 5],
        'partial': [1, np.nan, 3, np.nan, 5],
        'mostly_missing': [np.nan, np.nan, np.nan, np.nan, 1]
    })


@pytest.fixture
def dataframe_with_outliers():
    """
    Create dataframe with obvious outliers
    """
    return pd.DataFrame({
        'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'with_outliers': [1, 2, 3, 4, 5, 6, 7, 8, 100, 200]
    })


@pytest.fixture
def test_config():
    """
    Create a test configuration object
    """
    from src import EDAConfig
    
    config = EDAConfig()
    config.INPUT_CSV = "test_data.csv"
    config.OUTPUT_FILE = "test_results.joblib"
    config.LOG_FILE = "test.log"
    config.N_ESTIMATORS = 50  # Smaller for faster tests
    
    return config


@pytest.fixture
def temp_csv_file(tmp_path, sample_dataframe):
    """
    Create a temporary CSV file for testing
    """
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    
    return str(csv_path)


@pytest.fixture
def mock_logger():
    """
    Create a mock logger for testing
    """
    import logging
    
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    # Add null handler to prevent output during tests
    logger.addHandler(logging.NullHandler())
    
    return logger


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """
    Create a temporary directory for test data
    Available for entire test session
    """
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def numeric_series():
    """
    Create a numeric series for testing
    """
    np.random.seed(42)
    return pd.Series(np.random.randn(100))


@pytest.fixture
def categorical_series():
    """
    Create a categorical series for testing
    """
    return pd.Series(['A', 'B', 'C', 'A', 'B', 'C'] * 10)


@pytest.fixture
def fraud_detection_features():
    """
    Create feature array for ML testing
    """
    np.random.seed(42)
    n_samples = 500
    
    features = np.column_stack([
        np.random.uniform(10, 1000, n_samples),  # amount
        np.random.uniform(0, 30, n_samples),     # time_since_last
        np.random.randn(n_samples),              # deviation_score
        np.random.randint(0, 20, n_samples),     # velocity
        np.random.uniform(0, 1, n_samples)       # geo_anomaly
    ])
    
    return features


@pytest.fixture
def fraud_labels():
    """
    Create fraud labels for ML testing
    """
    np.random.seed(42)
    return np.random.choice([0, 1], 500, p=[0.96, 0.04])


# Test markers
def pytest_configure(config):
    """
    Configure custom pytest markers
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Pytest options
def pytest_addoption(parser):
    """
    Add custom command line options
    """
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip slow tests unless --run-slow is given
    """
    if config.getoption("--run-slow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)