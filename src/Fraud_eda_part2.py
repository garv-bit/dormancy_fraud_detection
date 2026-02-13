"""
FraudEDA Part 2: Advanced Analysis Methods
Correlation, Feature Importance, PCA, Clustering
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from Utils import (
    print_section_header,
    get_numeric_columns,
    get_categorical_columns,
    safe_divide,
)
from Config import EDAConfig


def analyze_correlation(self, numerical_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze correlations between numerical features
    
    Args:
        numerical_cols: Optional list of numerical columns. Auto-detected if None.
        
    Returns:
        Dictionary containing correlation matrices and strong correlations
    """
    print_section_header("6. CORRELATION & MULTICOLLINEARITY ANALYSIS", logger=self.logger)
    
    if self.df is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    if numerical_cols is None:
        numerical_cols = get_numeric_columns(self.df, exclude=['is_fraud'])
    
    correlation_analysis = {
        'pearson_matrix': {},
        'spearman_matrix': {},
        'strong_correlations': [],
    }
    
    if len(numerical_cols) <= 1:
        self.logger.warning("Need at least 2 numerical columns for correlation analysis")
        return correlation_analysis
    
    try:
        # Pearson correlation
        pearson_corr = self.df[numerical_cols].corr(method='pearson')
        correlation_analysis['pearson_matrix'] = pearson_corr.to_dict()
        
        # Spearman correlation
        spearman_corr = self.df[numerical_cols].corr(method='spearman')
        correlation_analysis['spearman_matrix'] = spearman_corr.to_dict()
        
        print("\n📊 Correlation Analysis:")
        print("Pearson Correlation Matrix:")
        print(pearson_corr.round(3))
        
        # Find strong correlations using vectorized approach
        corr_matrix = pearson_corr.values
        spearman_matrix = spearman_corr.values
        cols = pearson_corr.columns
        
        # Create upper triangle mask
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Find strong correlations
        strong_indices = np.where((np.abs(corr_matrix) > self.config.CORRELATION_THRESHOLD) & mask & ~np.isnan(corr_matrix))
        
        for i, j in zip(strong_indices[0], strong_indices[1]):
            correlation_analysis['strong_correlations'].append({
                'feature1': cols[i],
                'feature2': cols[j],
                'pearson': float(corr_matrix[i, j]),
                'spearman': float(spearman_matrix[i, j]),
            })
        
        print(f"\n🔍 Strong Correlations (|r| > {self.config.CORRELATION_THRESHOLD}): {len(correlation_analysis['strong_correlations'])}")
        
        for corr in correlation_analysis['strong_correlations'][:10]:  # Show top 10
            print(f"   • {corr['feature1']} ↔ {corr['feature2']}: r={corr['pearson']:.3f}")
        
        self.results['correlation_analysis'] = correlation_analysis
        self.logger.info(f"Correlation analysis complete, found {len(correlation_analysis['strong_correlations'])} strong correlations")
        
    except Exception as e:
        self.logger.error(f"Error in correlation analysis: {str(e)}")
    
    return correlation_analysis


def analyze_feature_importance(self) -> Dict[str, Any]:
    """
    Analyze feature importance using multiple methods
    Handles both supervised (with target) and unsupervised scenarios
    
    Returns:
        Dictionary containing feature importance scores
    """
    print_section_header("7. FEATURE IMPORTANCE ANALYSIS", logger=self.logger)
    
    if self.df is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    # Prepare ML dataset
    df_ml = self.df.copy()
    label_encoders = {}
    
    # Encode categorical features
    categorical_cols = get_categorical_columns(self.df, exclude=self.config.SKIP_CATEGORICAL_ANALYSIS)
    
    for col in categorical_cols:
        try:
            le = LabelEncoder()
            df_ml[col + '_enc'] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le
        except Exception as e:
            self.logger.warning(f"Failed to encode column {col}: {str(e)}")
    
    # Get numeric features
    numeric_features = [
        c for c in df_ml.select_dtypes(include=[np.number]).columns
        if c not in ['is_fraud'] and df_ml[c].notna().sum() > 0
    ]
    
    feature_importance = {
        'has_target': False,
        'variance_analysis': {},
        'rf_importance': {},
        'gb_importance': {},
        'mutual_info': {},
    }
    
    has_target = 'is_fraud' in self.df.columns and self.df['is_fraud'].sum() > 0
    
    if has_target and len(numeric_features) > 0:
        print("\n🎯 SUPERVISED FEATURE IMPORTANCE")
        feature_importance['has_target'] = True
        
        try:
            X = df_ml[numeric_features].fillna(df_ml[numeric_features].mean())
            y = df_ml['is_fraud'].astype(int)
            
            # Train/test split for proper evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.TEST_SIZE,
                random_state=self.config.RANDOM_STATE,
                stratify=y if y.sum() > 1 else None
            )
            
            # Random Forest
            self.logger.info("Training Random Forest for feature importance...")
            rf = RandomForestClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS
            )
            rf.fit(X_train, y_train)
            feature_importance['rf_importance'] = dict(zip(numeric_features, rf.feature_importances_))
            
            # Gradient Boosting
            self.logger.info("Training Gradient Boosting for feature importance...")
            gb = GradientBoostingClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                random_state=self.config.RANDOM_STATE
            )
            gb.fit(X_train, y_train)
            feature_importance['gb_importance'] = dict(zip(numeric_features, gb.feature_importances_))
            
            # Mutual Information
            self.logger.info("Calculating mutual information scores...")
            mi_scores = mutual_info_classif(X_train, y_train, random_state=self.config.RANDOM_STATE)
            feature_importance['mutual_info'] = dict(zip(numeric_features, mi_scores))
            
            # Print top features
            rf_sorted = sorted(
                feature_importance['rf_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"\n📊 Top {self.config.TOP_N_FEATURES} Features (Random Forest):")
            for feat, imp in rf_sorted[:self.config.TOP_N_FEATURES]:
                print(f"   • {feat}: {imp:.4f}")
            
            self.logger.info("Supervised feature importance analysis complete")
            
        except Exception as e:
            self.logger.error(f"Error in supervised feature importance: {str(e)}")
    
    else:
        print("\n📊 UNSUPERVISED ANALYSIS (No fraud target or no fraud cases)")
        
        if len(numeric_features) > 0:
            try:
                X = df_ml[numeric_features].fillna(df_ml[numeric_features].mean())
                
                # Variance analysis
                variances = X.var()
                feature_importance['variance_analysis'] = variances.to_dict()
                
                print(f"\n📊 Top {self.config.TOP_N_FEATURES} Features by Variance:")
                for feat, var in variances.sort_values(ascending=False).head(self.config.TOP_N_FEATURES).items():
                    print(f"   • {feat}: {var:.4f}")
                
                self.logger.info("Unsupervised variance analysis complete")
                
            except Exception as e:
                self.logger.error(f"Error in unsupervised analysis: {str(e)}")
    
    self.results['feature_importance'] = feature_importance
    return feature_importance


def analyze_pca(self, numerical_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform PCA dimensionality reduction analysis
    
    Args:
        numerical_cols: Optional list of numerical columns
        
    Returns:
        Dictionary containing PCA analysis results
    """
    print_section_header("8. DIMENSIONALITY REDUCTION (PCA)", logger=self.logger)
    
    if self.df is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    if numerical_cols is None:
        numerical_cols = get_numeric_columns(self.df, exclude=['is_fraud'])
    
    pca_analysis = {}
    
    if len(numerical_cols) < 2:
        self.logger.warning("Need at least 2 numerical columns for PCA")
        return pca_analysis
    
    try:
        X = self.df[numerical_cols].fillna(self.df[numerical_cols].mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = int(np.argmax(cumulative_variance >= self.config.PCA_VARIANCE_THRESHOLD) + 1)
        
        pca_analysis = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'explained_variance': pca.explained_variance_.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'n_components_95': n_components_95,
            'total_components': len(pca.explained_variance_ratio_),
            'reduction_potential_pct': float(
                (1 - n_components_95 / len(numerical_cols)) * 100
            ),
        }
        
        print(f"\n📊 PCA Results:")
        print(f"   • Components for {self.config.PCA_VARIANCE_THRESHOLD*100:.0f}% variance: {n_components_95}/{len(numerical_cols)}")
        print(f"   • Dimensionality reduction potential: {pca_analysis['reduction_potential_pct']:.1f}%")
        
        # Transform to 2D for visualization (limited points)
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        # Store limited points for memory efficiency
        max_points = min(self.config.MAX_PCA_POINTS, len(X_pca_2d))
        pca_analysis['pca_2d_data'] = X_pca_2d[:max_points].tolist()
        
        if len(X_pca_2d) > max_points:
            self.logger.info(f"Stored {max_points} of {len(X_pca_2d)} PCA points for visualization")
        
        self.results['pca_analysis'] = pca_analysis
        self.logger.info("PCA analysis complete")
        
    except Exception as e:
        self.logger.error(f"Error in PCA analysis: {str(e)}")
    
    return pca_analysis


def analyze_clustering(self, numerical_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform K-means clustering analysis
    
    Args:
        numerical_cols: Optional list of numerical columns
        
    Returns:
        Dictionary containing clustering results
    """
    print_section_header("9. CLUSTERING ANALYSIS", logger=self.logger)
    
    if self.df is None:
        raise ValueError("No data loaded. Call load_data() first.")
    
    if numerical_cols is None:
        numerical_cols = get_numeric_columns(self.df, exclude=['is_fraud'])
    
    clustering_analysis = {}
    
    if len(numerical_cols) < 2:
        self.logger.warning("Need at least 2 numerical columns for clustering")
        return clustering_analysis
    
    try:
        X = self.df[numerical_cols].fillna(self.df[numerical_cols].mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test different numbers of clusters
        max_k = min(self.config.MAX_CLUSTERS, len(self.df))
        inertias = []
        silhouette_scores = []
        
        self.logger.info(f"Testing K-means with k from {self.config.MIN_CLUSTERS} to {max_k}")
        
        for k in range(self.config.MIN_CLUSTERS, max_k):
            kmeans = KMeans(n_clusters=k, random_state=self.config.RANDOM_STATE, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            
            try:
                sil_score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(sil_score)
            except Exception as e:
                self.logger.warning(f"Silhouette score failed for k={k}: {str(e)}")
                silhouette_scores.append(0)
        
        # Find optimal k
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + self.config.MIN_CLUSTERS
        
        # Fit final model with optimal k
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=self.config.RANDOM_STATE, n_init=10)
        cluster_labels = kmeans_final.fit_predict(X_scaled)
        
        # Store limited points for memory efficiency
        max_points = min(self.config.MAX_CLUSTER_POINTS, len(cluster_labels))
        
        clustering_analysis = {
            'optimal_k': int(optimal_k),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'cluster_labels': cluster_labels[:max_points].tolist(),
            'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict(),
        }
        
        print(f"\n📊 K-Means Clustering:")
        print(f"   • Optimal clusters: {optimal_k}")
        print(f"   • Silhouette score: {silhouette_scores[optimal_k - self.config.MIN_CLUSTERS]:.4f}")
        print(f"\n   Cluster distribution:")
        for cluster, size in sorted(clustering_analysis['cluster_sizes'].items()):
            pct = safe_divide(size, len(cluster_labels), 0) * 100
            print(f"      Cluster {cluster}: {size} ({pct:.1f}%)")
        
        if len(cluster_labels) > max_points:
            self.logger.info(f"Stored {max_points} of {len(cluster_labels)} cluster labels")
        
        self.results['clustering_analysis'] = clustering_analysis
        self.logger.info("Clustering analysis complete")
        
    except Exception as e:
        self.logger.error(f"Error in clustering analysis: {str(e)}")
    
    return clustering_analysis