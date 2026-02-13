"""
Production-grade Financial Fraud Detection EDA Pipeline
Modular, testable, and professionally structured
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, normaltest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import joblib

from Config import EDAConfig
from Utils import (
    setup_logging,
    validate_dataframe,
    validate_file_exists,
    safe_divide,
    get_numeric_columns,
    get_categorical_columns,
    calculate_data_quality_score,
    detect_outliers_iqr,
    detect_outliers_zscore,
    print_section_header,
    create_output_directory,
    format_bytes,
)

warnings.filterwarnings("ignore")


class FraudEDA:
    """
    Comprehensive EDA pipeline for financial fraud detection datasets
    
    This class provides modular, reusable analysis components with proper
    error handling, logging, and configuration management.
    """
    
    def __init__(self, config: Optional[EDAConfig] = None):
        """
        Initialize EDA pipeline
        
        Args:
            config: Optional configuration object. Uses defaults if not provided.
        """
        self.config = config or EDAConfig()
        self.config.validate()
        
        self.logger = setup_logging(self.config)
        self.df = None
        self.results = {
            'metadata': {},
            'data_quality': {},
            'numerical_analysis': {},
            'categorical_analysis': {},
            'temporal_analysis': {},
            'correlation_analysis': {},
            'feature_importance': {},
            'pca_analysis': {},
            'clustering_analysis': {},
            'statistical_tests': {},
            'model_readiness': {},
            'recommendations': [],
            'visualizations': {},
        }
        
        # Setup plotting style
        plt.style.use(self.config.PLOT_STYLE)
        sns.set_palette(self.config.COLOR_PALETTE)
        
        self.logger.info("FraudEDA pipeline initialized")
    
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from CSV file with validation
        
        Args:
            filepath: Path to CSV file. Uses config default if not provided.
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If DataFrame is invalid
        """
        filepath = filepath or self.config.INPUT_CSV
        
        try:
            validate_file_exists(filepath, self.logger)
            self.logger.info(f"Loading data from: {filepath}")
            
            self.df = pd.read_csv(filepath)
            validate_dataframe(self.df, self.logger)
            
            self.logger.info(f"Data loaded successfully: {len(self.df)} rows, {len(self.df.columns)} columns")
            return self.df
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {filepath}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"CSV file is empty: {filepath}")
            raise ValueError(f"CSV file is empty: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_metadata(self) -> Dict[str, Any]:
        """
        Analyze dataset metadata and structure
        
        Returns:
            Dictionary containing metadata information
        """
        print_section_header("1. DATASET METADATA & PROFILING", logger=self.logger)
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        metadata = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_bytes': int(self.df.memory_usage(deep=True).sum()),
            'memory_mb': float(self.df.memory_usage(deep=True).sum() / (1024**2)),
            'bytes_per_row': float(self.df.memory_usage(deep=True).sum() / len(self.df)),
            'column_info': {},
            'dtypes_summary': self.df.dtypes.value_counts().to_dict(),
        }
        
        # Detailed column information
        for col in self.df.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'non_null': int(self.df[col].count()),
                'null_count': int(self.df[col].isnull().sum()),
                'null_pct': float(safe_divide(self.df[col].isnull().sum(), len(self.df), 0) * 100),
                'unique': int(self.df[col].nunique()),
                'unique_pct': float(safe_divide(self.df[col].nunique(), len(self.df), 0) * 100),
            }
            metadata['column_info'][col] = col_info
        
        # Print summary
        print(f"\n📊 Dataset Overview:")
        print(f"   Rows: {metadata['total_rows']:,}")
        print(f"   Columns: {metadata['total_columns']}")
        print(f"   Memory: {format_bytes(metadata['memory_bytes'])}")
        print(f"   Bytes/Row: {metadata['bytes_per_row']:.0f}")
        
        self.results['metadata'] = metadata
        self.logger.info(f"Metadata analysis complete")
        
        return metadata
    
    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        
        Returns:
            Dictionary containing quality metrics
        """
        print_section_header("2. DATA QUALITY ASSESSMENT", logger=self.logger)
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Calculate quality metrics using utility function
        quality_metrics = calculate_data_quality_score(self.df, self.config, self.logger)
        
        # Missing values analysis
        total_missing = self.df.isnull().sum().sum()
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        
        data_quality = {
            'total_missing': int(total_missing),
            'completeness_pct': float(quality_metrics['completeness']),
            'duplicate_rows': int(duplicates),
            'duplicate_pct': float(safe_divide(duplicates, len(self.df), 0) * 100),
            'quality_metrics': quality_metrics,
            'overall_quality': float(quality_metrics['overall']),
            'missing_by_column': self.df.isnull().sum().to_dict(),
        }
        
        # Print summary
        print(f"\n✅ Quality Metrics:")
        print(f"   Completeness: {quality_metrics['completeness']:.2f}%")
        print(f"   Uniqueness: {quality_metrics['uniqueness']:.2f}%")
        print(f"   Consistency: {quality_metrics['consistency']:.2f}%")
        print(f"   Validity: {quality_metrics['validity']:.2f}%")
        print(f"   Overall Quality: {quality_metrics['overall']:.2f}%")
        print(f"\n   Duplicates: {duplicates} ({data_quality['duplicate_pct']:.2f}%)")
        
        # Warn about quality issues
        if quality_metrics['overall'] < self.config.COMPLETENESS_THRESHOLD:
            self.logger.warning(f"Low overall quality: {quality_metrics['overall']:.2f}%")
        
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate rows")
        
        self.results['data_quality'] = data_quality
        self.logger.info("Data quality assessment complete")
        
        return data_quality
    
    def analyze_numerical_features(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of numerical features
        
        Returns:
            Dictionary containing numerical feature statistics
        """
        print_section_header("3. NUMERICAL FEATURES ANALYSIS", logger=self.logger)
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        numerical_cols = get_numeric_columns(self.df, exclude=['is_fraud'])
        
        print(f"\n📊 Analyzing {len(numerical_cols)} numerical features...")
        
        numerical_analysis = {
            'feature_count': len(numerical_cols),
            'features': {},
        }
        
        for col in numerical_cols:
            if self.df[col].notna().sum() == 0:
                self.logger.warning(f"Column {col} has no non-null values")
                continue
            
            try:
                data = self.df[col].dropna()
                
                if len(data) < 3:
                    self.logger.warning(f"Column {col} has less than 3 non-null values")
                    continue
                
                # Basic statistics
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                # Outlier detection
                outliers_iqr = detect_outliers_iqr(data, self.config.IQR_MULTIPLIER)
                outliers_zscore = detect_outliers_zscore(data, self.config.Z_SCORE_THRESHOLD)
                
                # Normality test
                try:
                    _, normality_p = normaltest(data)
                    is_normal = bool(normality_p > self.config.NORMALITY_ALPHA)
                except Exception as e:
                    self.logger.warning(f"Normality test failed for {col}: {str(e)}")
                    normality_p = None
                    is_normal = None
                
                # Coefficient of variation
                cv = safe_divide(data.std(), data.mean(), 0) * 100
                
                feature_stats = {
                    'count': int(data.count()),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q1': float(Q1),
                    'q3': float(Q3),
                    'iqr': float(IQR),
                    'range': float(data.max() - data.min()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'cv': float(cv),
                    'outliers_iqr': outliers_iqr,
                    'outliers_zscore': outliers_zscore,
                    'normality_p_value': float(normality_p) if normality_p is not None else None,
                    'is_normal': is_normal,
                }
                
                numerical_analysis['features'][col] = feature_stats
                
                print(f"\n{col}:")
                print(f"   Mean: {feature_stats['mean']:.4f}, Median: {feature_stats['median']:.4f}")
                print(f"   Skewness: {feature_stats['skewness']:.4f}, Kurtosis: {feature_stats['kurtosis']:.4f}")
                print(f"   Outliers (IQR): {outliers_iqr['count']} ({outliers_iqr['percentage']:.2f}%)")
                print(f"   Normal distribution: {is_normal}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing numerical column {col}: {str(e)}")
                continue
        
        self.results['numerical_analysis'] = numerical_analysis
        self.logger.info(f"Numerical analysis complete for {len(numerical_analysis['features'])} features")
        
        return numerical_analysis
    
    def analyze_categorical_features(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of categorical features
        
        Returns:
            Dictionary containing categorical feature statistics
        """
        print_section_header("4. CATEGORICAL FEATURES ANALYSIS", logger=self.logger)
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        categorical_cols = get_categorical_columns(self.df, exclude=self.config.SKIP_CATEGORICAL_ANALYSIS)
        
        print(f"\n📊 Analyzing {len(categorical_cols)} categorical features...")
        
        categorical_analysis = {
            'feature_count': len(categorical_cols),
            'features': {},
        }
        
        for col in categorical_cols:
            try:
                value_counts = self.df[col].value_counts()
                
                if len(value_counts) == 0:
                    self.logger.warning(f"Column {col} has no values")
                    continue
                
                # Calculate entropy
                try:
                    entropy_value = float(stats.entropy(value_counts))
                except Exception as e:
                    self.logger.warning(f"Entropy calculation failed for {col}: {str(e)}")
                    entropy_value = None
                
                # Determine cardinality
                unique_ratio = safe_divide(self.df[col].nunique(), len(self.df), 0)
                if unique_ratio > 0.5:
                    cardinality = "high"
                elif unique_ratio > 0.1:
                    cardinality = "medium"
                else:
                    cardinality = "low"
                
                feature_info = {
                    'unique_count': int(self.df[col].nunique()),
                    'top_value': str(value_counts.index[0]),
                    'top_count': int(value_counts.iloc[0]),
                    'top_pct': float(safe_divide(value_counts.iloc[0], len(self.df), 0) * 100),
                    'entropy': entropy_value,
                    'cardinality': cardinality,
                    'value_counts': value_counts.head(self.config.TOP_N_VALUES).to_dict(),
                }
                
                categorical_analysis['features'][col] = feature_info
                
                print(f"\n{col}: {feature_info['unique_count']} unique values ({cardinality} cardinality)")
                if feature_info['unique_count'] <= self.config.MAX_CATEGORICAL_DISPLAY:
                    for val, cnt in list(value_counts.items())[:self.config.TOP_N_VALUES]:
                        pct = safe_divide(cnt, len(self.df), 0) * 100
                        print(f"   • {val}: {cnt} ({pct:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Error analyzing categorical column {col}: {str(e)}")
                continue
        
        self.results['categorical_analysis'] = categorical_analysis
        self.logger.info(f"Categorical analysis complete for {len(categorical_analysis['features'])} features")
        
        return categorical_analysis
    
    def analyze_temporal_features(self) -> Dict[str, Any]:
        """
        Analyze temporal patterns if timestamp column exists
        
        Returns:
            Dictionary containing temporal analysis results
        """
        if 'timestamp' not in self.df.columns:
            self.logger.info("No timestamp column found, skipping temporal analysis")
            return {}
        
        print_section_header("5. TEMPORAL ANALYSIS", logger=self.logger)
        
        try:
            # Parse timestamp
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='ISO8601')
            
            # Extract temporal features if configured
            if self.config.TEMPORAL_FEATURES:
                self.df['hour'] = self.df['timestamp'].dt.hour
                self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
                self.df['day_name'] = self.df['timestamp'].dt.day_name()
                self.df['month'] = self.df['timestamp'].dt.month
                self.df['month_name'] = self.df['timestamp'].dt.month_name()
                self.df['is_weekend'] = self.df['day_of_week'].isin(self.config.WEEKEND_DAYS)
                self.df['quarter'] = self.df['timestamp'].dt.quarter
            
            time_span = (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            
            temporal_analysis = {
                'min_date': str(self.df['timestamp'].min()),
                'max_date': str(self.df['timestamp'].max()),
                'time_span_days': int(time_span),
                'peak_hour': int(self.df['hour'].mode().values[0]) if len(self.df['hour'].mode()) > 0 else None,
                'peak_day': str(self.df['day_name'].mode().values[0]) if len(self.df['day_name'].mode()) > 0 else None,
                'peak_month': str(self.df['month_name'].mode().values[0]) if len(self.df['month_name'].mode()) > 0 else None,
                'weekend_pct': float(safe_divide(self.df['is_weekend'].sum(), len(self.df), 0) * 100),
                'hourly_distribution': self.df['hour'].value_counts().sort_index().to_dict(),
                'daily_distribution': self.df['day_name'].value_counts().to_dict(),
                'monthly_distribution': self.df['month_name'].value_counts().to_dict(),
            }
            
            print(f"\n📅 Temporal Coverage: {time_span} days")
            print(f"   From: {temporal_analysis['min_date']}")
            print(f"   To: {temporal_analysis['max_date']}")
            print(f"\n⏰ Activity Patterns:")
            print(f"   • Peak Hour: {temporal_analysis['peak_hour']}:00")
            print(f"   • Peak Day: {temporal_analysis['peak_day']}")
            print(f"   • Weekend Activity: {temporal_analysis['weekend_pct']:.1f}%")
            
            self.results['temporal_analysis'] = temporal_analysis
            self.logger.info("Temporal analysis complete")
            
            return temporal_analysis
            
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {str(e)}")
            return {}


# Import additional methods from other modules
from Fraud_eda_part2 import (
    analyze_correlation,
    analyze_feature_importance,
    analyze_pca,
    analyze_clustering,
)

from Fraud_eda_part3 import (
    analyze_statistical_tests,
    assess_model_readiness,
    generate_recommendations,
    run_full_analysis,
    save_results,
)

# Attach methods to FraudEDA class
FraudEDA.analyze_correlation = analyze_correlation
FraudEDA.analyze_feature_importance = analyze_feature_importance
FraudEDA.analyze_pca = analyze_pca
FraudEDA.analyze_clustering = analyze_clustering
FraudEDA.analyze_statistical_tests = analyze_statistical_tests
FraudEDA.assess_model_readiness = assess_model_readiness
FraudEDA.generate_recommendations = generate_recommendations
FraudEDA.run_full_analysis = run_full_analysis
FraudEDA.save_results = save_results