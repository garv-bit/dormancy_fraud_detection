"""
FraudEDA Part 3: Statistical Tests, Model Readiness, and Main Runner
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import joblib

from Utils import (
    safe_divide,
    get_numeric_columns,
    get_categorical_columns,
    print_section_header,
    format_bytes,
)
from Config import EDAConfig


def analyze_statistical_tests(self) -> Dict[str, Any]:
    """
    Perform statistical hypothesis tests
    
    Returns:
        Dictionary containing test results
    """
    print_section_header("10. STATISTICAL HYPOTHESIS TESTS", logger=self.logger)
    
    if self.df is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    statistical_tests = {
        'chi_square_tests': [],
        't_tests': [],
    }
    
    # Chi-square tests for categorical independence
    categorical_cols = get_categorical_columns(self.df, exclude=self.config.SKIP_CATEGORICAL_ANALYSIS)
    
    # Filter columns suitable for chi-square
    cat_cols_test = [
        c for c in categorical_cols
        if self.df[c].nunique() > 1 and self.df[c].nunique() < self.config.MAX_CATEGORICAL_UNIQUE
    ]
    
    if len(cat_cols_test) >= 2:
        print("\n📊 Chi-Square Independence Tests:")
        
        test_count = 0
        for i, col1 in enumerate(cat_cols_test):
            if test_count >= self.config.MAX_CHI_SQUARE_TESTS:
                break
                
            for col2 in cat_cols_test[i + 1:]:
                if test_count >= self.config.MAX_CHI_SQUARE_TESTS:
                    break
                
                try:
                    contingency = pd.crosstab(self.df[col1], self.df[col2])
                    
                    # Check if contingency table is valid
                    if contingency.size == 0:
                        continue
                    
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    test_result = {
                        'variable1': col1,
                        'variable2': col2,
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'dof': int(dof),
                        'significant': bool(p_value < self.config.ALPHA),
                    }
                    
                    statistical_tests['chi_square_tests'].append(test_result)
                    
                    significance = 'Significant' if p_value < self.config.ALPHA else 'Not significant'
                    print(f"\n{col1} vs {col2}:")
                    print(f"   χ² = {chi2:.4f}, p = {p_value:.4f}, {significance}")
                    
                    test_count += 1
                    
                except ValueError as e:
                    self.logger.warning(f"Chi-square test failed for {col1} vs {col2}: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Unexpected error in chi-square test: {str(e)}")
        
        self.logger.info(f"Completed {test_count} chi-square tests")
    else:
        self.logger.info("Not enough suitable categorical columns for chi-square tests")
    
    self.results['statistical_tests'] = statistical_tests
    return statistical_tests


def assess_model_readiness(self) -> Dict[str, Any]:
    """
    Assess if dataset is ready for model training
    
    Returns:
        Dictionary containing readiness assessment
    """
    print_section_header("11. MODEL READINESS ASSESSMENT", logger=self.logger)
    
    if self.df is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    # Get quality metrics
    quality_data = self.results.get('data_quality', {})
    completeness = quality_data.get('completeness_pct', 0)
    
    numerical_cols = get_numeric_columns(self.df, exclude=['is_fraud'])
    categorical_cols = get_categorical_columns(self.df, exclude=self.config.SKIP_CATEGORICAL_ANALYSIS)
    total_features = len(numerical_cols) + len(categorical_cols)
    
    model_readiness = {
        'completeness_score': float(completeness),
        'completeness_pass': bool(completeness >= self.config.COMPLETENESS_THRESHOLD),
        'sample_size': len(self.df),
        'sample_size_pass': bool(len(self.df) >= self.config.MIN_SAMPLE_SIZE),
        'feature_count': total_features,
        'feature_pass': bool(total_features >= self.config.MIN_FEATURES),
        'class_balance_score': 0.0,
        'class_balance_pass': False,
        'overall_readiness': 'NOT READY',
    }
    
    # Check if target variable exists and has fraud cases
    if 'is_fraud' in self.df.columns:
        fraud_count = self.df['is_fraud'].sum()
        fraud_rate = safe_divide(fraud_count, len(self.df), 0)
        
        model_readiness['class_balance_score'] = float(fraud_rate * 100)
        model_readiness['class_balance_pass'] = bool(
            fraud_rate >= self.config.MIN_FRAUD_RATE and 
            fraud_rate <= self.config.MAX_FRAUD_RATE
        )
        
        print(f"\n🎯 Model Readiness Checklist:")
        print(f"   1. Data Completeness: {completeness:.1f}% - {'✅ PASS' if model_readiness['completeness_pass'] else '❌ FAIL'}")
        print(f"   2. Sample Size: {len(self.df):,} - {'✅ PASS' if model_readiness['sample_size_pass'] else '❌ FAIL'}")
        print(f"   3. Class Balance: {fraud_rate * 100:.2f}% fraud - {'✅ PASS' if model_readiness['class_balance_pass'] else '❌ FAIL'}")
        print(f"   4. Feature Count: {total_features} features - {'✅ PASS' if model_readiness['feature_pass'] else '❌ FAIL'}")
        
        # Overall assessment
        checks_passed = sum([
            model_readiness['completeness_pass'],
            model_readiness['sample_size_pass'],
            model_readiness['class_balance_pass'],
            model_readiness['feature_pass'],
        ])
        
        if checks_passed >= 3:
            model_readiness['overall_readiness'] = 'READY'
        elif checks_passed >= 2:
            model_readiness['overall_readiness'] = 'NEEDS WORK'
        else:
            model_readiness['overall_readiness'] = 'NOT READY'
        
        print(f"\n   ⚖️  Overall Assessment: {model_readiness['overall_readiness']}")
        
    else:
        print("\n⚠️  No target variable (is_fraud) found")
        self.logger.warning("No target variable found in dataset")
    
    self.results['model_readiness'] = model_readiness
    self.logger.info(f"Model readiness: {model_readiness['overall_readiness']}")
    
    return model_readiness


def generate_recommendations(self) -> List[Dict[str, str]]:
    """
    Generate actionable recommendations based on analysis
    
    Returns:
        List of recommendation dictionaries
    """
    print_section_header("12. RECOMMENDATIONS & ACTION ITEMS", logger=self.logger)
    
    recommendations = []
    
    # Get analysis results
    quality_data = self.results.get('data_quality', {})
    completeness = quality_data.get('completeness_pct', 100)
    duplicates = quality_data.get('duplicate_rows', 0)
    
    model_data = self.results.get('model_readiness', {})
    fraud_rate = model_data.get('class_balance_score', 0) / 100
    
    # Data Quality Recommendations
    if completeness < self.config.COMPLETENESS_THRESHOLD:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Data Quality',
            'issue': f'Data is only {completeness:.1f}% complete',
            'action': 'Implement imputation strategy (mean/median/mode) or collect more complete data',
        })
    
    if duplicates > len(self.df) * 0.01:  # More than 1% duplicates
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Data Quality',
            'issue': f'{duplicates} duplicate rows found',
            'action': 'Remove duplicates or investigate if they are valid repeated transactions',
        })
    
    # Target Variable Recommendations
    if 'is_fraud' in self.df.columns:
        if self.df['is_fraud'].sum() == 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Target Variable',
                'issue': 'No fraud cases in dataset',
                'action': 'Collect diverse data including fraud examples, or use synthetic data generation (SMOTE)',
            })
        elif fraud_rate < self.config.MIN_FRAUD_RATE:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Class Imbalance',
                'issue': f'Severe class imbalance ({fraud_rate*100:.2f}% fraud)',
                'action': 'Use SMOTE, class weights, or ensemble methods (Random Forest, XGBoost with scale_pos_weight)',
            })
        elif fraud_rate > 0.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Class Balance',
                'issue': f'Unusually high fraud rate ({fraud_rate*100:.2f}%)',
                'action': 'Verify data collection process - this may indicate biased sampling',
            })
    
    # Sample Size Recommendations
    if len(self.df) < self.config.MIN_SAMPLE_SIZE:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Sample Size',
            'issue': f'Only {len(self.df)} transactions',
            'action': f'Collect more data for robust model training (target: {self.config.MIN_SAMPLE_SIZE}+)',
        })
    
    # Feature Engineering Recommendations
    recommendations.append({
        'priority': 'MEDIUM',
        'category': 'Feature Engineering',
        'issue': 'Additional features could improve model performance',
        'action': 'Create interaction features, ratios, temporal aggregations, and behavioral features',
    })
    
    # Temporal Recommendations
    if 'timestamp' in self.df.columns:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Model Selection',
            'issue': 'Dataset has temporal dependencies',
            'action': 'Use time-aware cross-validation (TimeSeriesSplit) and consider sequential models',
        })
    
    # Correlation Recommendations
    corr_data = self.results.get('correlation_analysis', {})
    strong_corr_count = len(corr_data.get('strong_correlations', []))
    
    if strong_corr_count > 10:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Multicollinearity',
            'issue': f'{strong_corr_count} pairs of strongly correlated features',
            'action': 'Consider feature selection, PCA, or regularization (Ridge/Lasso)',
        })
    
    # Sort by priority
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    # Print recommendations
    print("\n🎯 Actionable Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        icon = {
            'CRITICAL': '🔴',
            'HIGH': '🟠',
            'MEDIUM': '🟡',
            'LOW': '🟢',
        }.get(rec['priority'], '⚪')
        
        print(f"{i}. {icon} [{rec['priority']}] {rec['category']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Action: {rec['action']}\n")
    
    self.results['recommendations'] = recommendations
    self.logger.info(f"Generated {len(recommendations)} recommendations")
    
    return recommendations


def run_full_analysis(self, filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Run complete EDA pipeline
    
    Args:
        filepath: Optional path to CSV file
        
    Returns:
        Dictionary containing all analysis results
    """
    self.logger.info("=" * 120)
    self.logger.info("STARTING COMPREHENSIVE EDA PIPELINE")
    self.logger.info("=" * 120)
    
    try:
        # Load data
        self.load_data(filepath)
        
        # Run all analyses
        self.analyze_metadata()
        self.analyze_data_quality()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        self.analyze_temporal_features()
        self.analyze_correlation()
        self.analyze_feature_importance()
        self.analyze_pca()
        self.analyze_clustering()
        self.analyze_statistical_tests()
        self.assess_model_readiness()
        self.generate_recommendations()
        
        # Add timestamp
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['dataset_name'] = filepath or self.config.INPUT_CSV
        
        # Note: Don't store raw data (memory efficient)
        if self.config.STORE_RAW_DATA:
            self.results['data_sample'] = self.df.head(self.config.SAMPLE_SIZE_FOR_STORAGE).to_dict()
            self.logger.info(f"Stored {self.config.SAMPLE_SIZE_FOR_STORAGE} sample rows")
        
        self.logger.info("=" * 120)
        self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 120)
        
        return self.results
        
    except Exception as e:
        self.logger.error(f"Fatal error in EDA pipeline: {str(e)}", exc_info=True)
        raise


def save_results(self, output_path: Optional[str] = None):
    """
    Save analysis results to file using joblib (safer than pickle)
    
    Args:
        output_path: Optional output file path
    """
    output_path = output_path or self.config.OUTPUT_PICKLE
    
    try:
        joblib.dump(self.results, output_path)
        file_size = Path(output_path).stat().st_size
        self.logger.info(f"Results saved to: {output_path} ({format_bytes(file_size)})")
        print(f"\n💾 Results saved to: {output_path}")
        
    except Exception as e:
        self.logger.error(f"Error saving results: {str(e)}")
        raise