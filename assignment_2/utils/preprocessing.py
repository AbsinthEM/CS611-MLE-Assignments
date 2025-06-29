"""
ML Pipeline Preprocessing Module
Provides model-specific preprocessing for credit risk modeling

This module handles the preprocessing requirements for different ML models:
- Logistic Regression: Feature scaling, one-hot encoding, feature selection
- Random Forest: Label encoding, missing value imputation, feature engineering
- XGBoost: Missing value handling, label encoding, feature optimization
"""

import os
import pandas as pd
import numpy as np
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class BasePreprocessor(ABC):
    """
    Abstract base class for model-specific preprocessors
    
    Defines the common interface for all preprocessing implementations
    ensuring consistency across different model types
    """
    
    def __init__(self, random_state: int = 88):
        self.random_state = random_state
        self.transformers = {}
        self.feature_columns = []
        self.is_fitted = False
        self.preprocessing_stats = {}
        
    @abstractmethod
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Fit preprocessor and transform training data"""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        pass
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names after preprocessing"""
        return self.feature_columns
    
    def save_transformers(self, filepath: str) -> None:
        """Save fitted transformers to disk"""
        transformer_data = {
            'transformers': self.transformers,
            'feature_columns': self.feature_columns,
            'preprocessing_stats': self.preprocessing_stats,
            'model_type': self.__class__.__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(transformer_data, f)
            
    def load_transformers(self, filepath: str) -> None:
        """Load fitted transformers from disk"""
        with open(filepath, 'rb') as f:
            transformer_data = pickle.load(f)
            
        self.transformers = transformer_data['transformers']
        self.feature_columns = transformer_data['feature_columns']
        self.preprocessing_stats = transformer_data['preprocessing_stats']
        self.is_fitted = True


class LogisticRegressionPreprocessor(BasePreprocessor):
    """
    Preprocessing pipeline optimized for Logistic Regression
    
    Key features:
    - Standardization of numerical features (critical for LR)
    - One-hot encoding for categorical features
    - Feature selection to avoid multicollinearity
    - Conservative missing value handling
    """
    
    def __init__(self, random_state: int = 88, max_features: int = 50):
        super().__init__(random_state)
        self.max_features = max_features
        
    def _select_features_for_lr(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Select features most suitable for logistic regression"""
        
        # Core financial features (most predictive for credit risk)
        financial_features = [
            'annual_income', 'monthly_salary', 'debt_to_income_ratio', 
            'emi_to_income_ratio', 'credit_utilization_ratio',
            'financial_health_score', 'outstanding_debt'
        ]
        
        # Key demographic features
        demographic_features = [
            'age', 'credit_history_months'
        ]
        
        # Important categorical features
        categorical_features = [
            'age_group', 'occupation_category', 'credit_mix', 
            'payment_of_min_amount'
        ]
        
        # Selected clickstream statistical features (avoid too many to prevent overfitting)
        clickstream_features = [
            col for col in df.columns 
            if col.startswith('fe_') and col.endswith(('_mean', '_std')) 
            and any(str(i) in col for i in [1, 2, 3, 5, 7, 10])  # Select key fe_ features
        ]
        
        # Data quality indicators
        quality_features = [
            'data_completeness_score', 'has_financial_data', 'has_clickstream_data'
        ]
        
        # Filter to only include columns that exist in the dataframe
        feature_groups = {
            'financial': [f for f in financial_features if f in df.columns],
            'demographic': [f for f in demographic_features if f in df.columns],
            'categorical': [f for f in categorical_features if f in df.columns],
            'clickstream': [f for f in clickstream_features if f in df.columns][:10],  # Limit clickstream features
            'quality': [f for f in quality_features if f in df.columns]
        }
        
        return feature_groups
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Fit preprocessor and transform training data for Logistic Regression"""
        
        print(f"[LR Preprocessor] Starting preprocessing for {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Select appropriate features for LR
        feature_groups = self._select_features_for_lr(X_train)
        
        # Combine all selected features
        selected_features = []
        for group_name, features in feature_groups.items():
            selected_features.extend(features)
            print(f"[LR Preprocessor] Selected {len(features)} {group_name} features")
        
        X_selected = X_train[selected_features].copy()
        
        # Separate numerical and categorical features
        numerical_features = feature_groups['financial'] + feature_groups['demographic'] + \
                           feature_groups['clickstream'] + feature_groups['quality']
        categorical_features = feature_groups['categorical']
        
        # Handle missing values for numerical features (conservative approach)
        num_imputer = SimpleImputer(strategy='median')
        X_selected[numerical_features] = num_imputer.fit_transform(X_selected[numerical_features])
        self.transformers['numerical_imputer'] = num_imputer
        
        # Handle missing values for categorical features
        cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        X_selected[categorical_features] = cat_imputer.fit_transform(X_selected[categorical_features])
        self.transformers['categorical_imputer'] = cat_imputer
        
        # One-hot encode categorical features
        onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        categorical_encoded = onehot_encoder.fit_transform(X_selected[categorical_features])
        
        # Get feature names for one-hot encoded features
        categorical_feature_names = []
        for i, cat_feature in enumerate(categorical_features):
            categories = onehot_encoder.categories_[i][1:]  # Skip first category (dropped)
            for category in categories:
                categorical_feature_names.append(f"{cat_feature}_{category}")
        
        self.transformers['onehot_encoder'] = onehot_encoder
        
        # Combine numerical and categorical features
        X_numerical = X_selected[numerical_features].values
        X_combined = np.hstack([X_numerical, categorical_encoded])
        
        # Feature names for combined dataset
        combined_feature_names = numerical_features + categorical_feature_names
        
        # Standardize all features (critical for Logistic Regression)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        self.transformers['scaler'] = scaler
        
        # Feature selection to reduce multicollinearity
        feature_selector = SelectKBest(f_classif, k=min(self.max_features, X_scaled.shape[1]))
        X_final = feature_selector.fit_transform(X_scaled, y_train)
        self.transformers['feature_selector'] = feature_selector
        
        # Get final feature names
        selected_indices = feature_selector.get_support(indices=True)
        self.feature_columns = [combined_feature_names[i] for i in selected_indices]
        
        # Store preprocessing statistics
        self.preprocessing_stats = {
            'original_features': X_train.shape[1],
            'selected_features': len(selected_features),
            'final_features': X_final.shape[1],
            'missing_value_percentage': X_train.isnull().sum().mean(),
            'categorical_features_count': len(categorical_features),
            'numerical_features_count': len(numerical_features)
        }
        
        self.is_fitted = True
        
        print(f"[LR Preprocessor] Preprocessing complete: {X_train.shape[1]} -> {X_final.shape[1]} features")
        
        return X_final, self.feature_columns
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Select same features as training
        feature_groups = self._select_features_for_lr(X)
        selected_features = []
        for features in feature_groups.values():
            selected_features.extend(features)
        
        X_selected = X[selected_features].copy()
        
        # Separate features same way as training
        numerical_features = feature_groups['financial'] + feature_groups['demographic'] + \
                           feature_groups['clickstream'] + feature_groups['quality']
        categorical_features = feature_groups['categorical']
        
        # Apply same transformations
        X_selected[numerical_features] = self.transformers['numerical_imputer'].transform(X_selected[numerical_features])
        X_selected[categorical_features] = self.transformers['categorical_imputer'].transform(X_selected[categorical_features])
        
        # One-hot encode categorical features
        categorical_encoded = self.transformers['onehot_encoder'].transform(X_selected[categorical_features])
        
        # Combine features
        X_numerical = X_selected[numerical_features].values
        X_combined = np.hstack([X_numerical, categorical_encoded])
        
        # Scale and select features
        X_scaled = self.transformers['scaler'].transform(X_combined)
        X_final = self.transformers['feature_selector'].transform(X_scaled)
        
        return X_final


class RandomForestPreprocessor(BasePreprocessor):
    """
    Preprocessing pipeline optimized for Random Forest
    
    Key features:
    - Label encoding for categorical features (preserves ordinal relationships)
    - Minimal feature scaling (RF is scale-invariant)
    - More comprehensive feature usage
    - Robust missing value handling
    """
    
    def __init__(self, random_state: int = 88):
        super().__init__(random_state)
        self.fitted_categorical_columns = []
        self.fitted_numerical_columns = []
        
    def _select_features_for_rf(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Select features suitable for Random Forest (can handle more features)"""
        
        # Use more comprehensive feature set for RF
        financial_features = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in [
                'income', 'salary', 'debt', 'emi', 'credit', 'balance', 
                'outstanding', 'financial', 'payment'
            ])
        ]
        
        # All clickstream statistical features
        clickstream_features = [
            col for col in df.columns 
            if col.startswith('fe_') and any(suffix in col for suffix in ['_mean', '_std', '_min', '_max', '_cv'])
        ]
        
        # Include interaction features (RF can benefit from these)
        interaction_features = [
            col for col in df.columns 
            if 'interaction' in col or 'ratio' in col or 'relative' in col
        ]
        
        # Demographic and categorical features
        demographic_features = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in [
                'age', 'occupation', 'history', 'group', 'category'
            ])
        ]
        
        # Data quality features
        quality_features = [
            col for col in df.columns 
            if col.startswith('has_') or 'completeness' in col or 'quality' in col
        ]
        
        feature_groups = {
            'financial': [f for f in financial_features if f in df.columns],
            'clickstream': [f for f in clickstream_features if f in df.columns],
            'interaction': [f for f in interaction_features if f in df.columns],
            'demographic': [f for f in demographic_features if f in df.columns],
            'quality': [f for f in quality_features if f in df.columns]
        }
        
        return feature_groups
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Fit preprocessor and transform training data for Random Forest"""
        
        print(f"[RF Preprocessor] Starting preprocessing for {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Select features for RF
        feature_groups = self._select_features_for_rf(X_train)
        
        # Combine all selected features
        selected_features = []
        for group_name, features in feature_groups.items():
            selected_features.extend(features)
            print(f"[RF Preprocessor] Selected {len(features)} {group_name} features")
        
        X_selected = X_train[selected_features].copy()
        
        # Identify categorical and numerical columns
        categorical_columns = X_selected.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = X_selected.select_dtypes(include=[np.number]).columns.tolist()

        # Save feature categories
        self.fitted_categorical_columns = categorical_columns
        self.fitted_numerical_columns = numerical_columns
        
        # Handle missing values
        # For numerical: use median (robust to outliers)
        if numerical_columns:
            num_imputer = SimpleImputer(strategy='median')
            X_selected[numerical_columns] = num_imputer.fit_transform(X_selected[numerical_columns])
            self.transformers['numerical_imputer'] = num_imputer
        
        # For categorical: use most frequent value
        if categorical_columns:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_selected[categorical_columns] = cat_imputer.fit_transform(X_selected[categorical_columns])
            self.transformers['categorical_imputer'] = cat_imputer
        
        # Label encode categorical features (RF can handle this better than one-hot)
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X_selected[col] = le.fit_transform(X_selected[col].astype(str))
            label_encoders[col] = le
        
        self.transformers['label_encoders'] = label_encoders
        
        # Convert to numpy array
        X_final = X_selected.values
        self.feature_columns = selected_features
        
        # Store preprocessing statistics
        self.preprocessing_stats = {
            'original_features': X_train.shape[1],
            'selected_features': len(selected_features),
            'final_features': X_final.shape[1],
            'missing_value_percentage': X_train.isnull().sum().mean(),
            'categorical_features_count': len(categorical_columns),
            'numerical_features_count': len(numerical_columns)
        }
        
        self.is_fitted = True
        
        print(f"[RF Preprocessor] Preprocessing complete: {X_train.shape[1]} -> {X_final.shape[1]} features")
        
        return X_final, self.feature_columns
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Select same features as training
        self.feature_columns = [col for col in self.feature_columns if col in X.columns]
        X_selected = X[self.feature_columns].copy()
        
        # Identify categorical and numerical columns
        categorical_columns = [col for col in self.fitted_categorical_columns if col in self.feature_columns]
        numerical_columns = [col for col in self.fitted_numerical_columns if col in self.feature_columns]
        
        # Apply same transformations
        if numerical_columns:
            X_selected[numerical_columns] = self.transformers['numerical_imputer'].transform(X_selected[numerical_columns])
        
        if categorical_columns:
            X_selected[categorical_columns] = self.transformers['categorical_imputer'].transform(X_selected[categorical_columns])
        
        # Label encode categorical features
        for col in categorical_columns:
            le = self.transformers['label_encoders'][col]
            X_selected[col] = X_selected[col].astype(str)
            
            # Handle unknown categories
            unknown_mask = ~X_selected[col].isin(le.classes_)
            X_selected[col] = X_selected[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
            X_selected[col] = le.transform(X_selected[col])
        
        return X_selected.values


class XGBoostPreprocessor(BasePreprocessor):
    """
    Preprocessing pipeline optimized for XGBoost
    
    Key features:
    - Minimal preprocessing (XGB handles missing values internally)
    - Label encoding for categorical features
    - Uses most comprehensive feature set
    - Special missing value indicators
    """
    
    def __init__(self, random_state: int = 88, missing_value_indicator: float = -999):
        super().__init__(random_state)
        self.missing_value_indicator = missing_value_indicator
        
    def _select_features_for_xgb(self, df: pd.DataFrame) -> List[str]:
        """Select features for XGBoost (can use almost all features)"""
        
        # XGBoost can handle most features, so be comprehensive
        exclude_columns = [
            'Customer_ID', 'snapshot_date', 'loan_id', 'label', 'label_def',
            'ingestion_timestamp', 'data_source', 'processing_date'
        ]
        
        selected_features = [
            col for col in df.columns 
            if col not in exclude_columns and not col.startswith('Unnamed')
        ]
        
        return selected_features
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Fit preprocessor and transform training data for XGBoost"""
        
        print(f"[XGB Preprocessor] Starting preprocessing for {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Select features for XGBoost
        selected_features = self._select_features_for_xgb(X_train)
        X_selected = X_train[selected_features].copy()
        
        print(f"[XGB Preprocessor] Selected {len(selected_features)} features for training")
        
        # Identify categorical columns
        categorical_columns = X_selected.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Label encode categorical features
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            # Handle missing values in categorical columns
            X_selected[col] = X_selected[col].fillna('Missing').astype(str)
            X_selected[col] = le.fit_transform(X_selected[col])
            label_encoders[col] = le
        
        self.transformers['label_encoders'] = label_encoders
        
        # Handle missing values in numerical columns with special indicator
        numerical_columns = X_selected.select_dtypes(include=[np.number]).columns.tolist()
        X_selected[numerical_columns] = X_selected[numerical_columns].fillna(self.missing_value_indicator)
        
        # Convert to numpy array
        X_final = X_selected.values.astype(np.float32)
        self.feature_columns = selected_features
        
        # Store preprocessing statistics
        self.preprocessing_stats = {
            'original_features': X_train.shape[1],
            'selected_features': len(selected_features),
            'final_features': X_final.shape[1],
            'missing_value_percentage': X_train.isnull().sum().mean(),
            'categorical_features_count': len(categorical_columns),
            'numerical_features_count': len(numerical_columns),
            'missing_value_indicator': self.missing_value_indicator
        }
        
        self.is_fitted = True
        
        print(f"[XGB Preprocessor] Preprocessing complete: {X_train.shape[1]} -> {X_final.shape[1]} features")
        
        return X_final, self.feature_columns
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Select same features as training
        X_selected = X[self.feature_columns].copy()
        
        # Identify categorical columns (same as training)
        categorical_columns = list(self.transformers.get('label_encoders', {}).keys())
        
        # Label encode categorical features
        for col in categorical_columns:
            le = self.transformers['label_encoders'][col]
            X_selected[col] = X_selected[col].fillna('Missing').astype(str)
            
            # Handle unknown categories by mapping to most frequent class
            unknown_mask = ~X_selected[col].isin(le.classes_)
            if unknown_mask.any():
                X_selected.loc[unknown_mask, col] = le.classes_[0]
            
            X_selected[col] = le.transform(X_selected[col])
        
        # Handle missing values in numerical columns
        numerical_columns = X_selected.select_dtypes(include=[np.number]).columns.tolist()
        X_selected[numerical_columns] = X_selected[numerical_columns].fillna(self.missing_value_indicator)
        
        return X_selected.values.astype(np.float32)


class PreprocessorFactory:
    """
    Factory class to create appropriate preprocessor based on model type
    
    Centralizes preprocessor creation and ensures consistent interface
    across different model types
    """
    
    @staticmethod
    def create_preprocessor(model_type: str, **kwargs) -> BasePreprocessor:
        """Create preprocessor instance based on model type"""
        
        model_type = model_type.lower().replace('_', '').replace('-', '')
        
        if model_type in ['logisticregression', 'logistic', 'lr']:
            return LogisticRegressionPreprocessor(**kwargs)
        elif model_type in ['randomforest', 'rf', 'forest']:
            return RandomForestPreprocessor(**kwargs)
        elif model_type in ['xgboost', 'xgb', 'boost']:
            return XGBoostPreprocessor(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_supported_models() -> List[str]:
        """Return list of supported model types"""
        return ['logistic_regression', 'random_forest', 'xgboost']


class DataQualityChecker:
    """
    Data quality validation for ML pipeline
    
    Performs comprehensive data quality checks before and after preprocessing
    to ensure data integrity and flag potential issues
    """
    
    @staticmethod
    def validate_training_data(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data quality validation for training data"""
        
        quality_report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': {
                'rows': X.shape[0],
                'columns': X.shape[1]
            },
            'missing_values': {
                'total_missing': X.isnull().sum().sum(),
                'missing_percentage': (X.isnull().sum().sum() / X.size) * 100,
                'columns_with_missing': X.columns[X.isnull().any()].tolist(),
                'high_missing_columns': X.columns[X.isnull().mean() > 0.5].tolist()
            },
            'label_distribution': {
                'class_counts': y.value_counts().to_dict(),
                'class_proportions': y.value_counts(normalize=True).to_dict(),
                'is_balanced': abs(y.value_counts(normalize=True).iloc[0] - 0.5) < 0.3
            },
            'feature_types': {
                'numerical': X.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': X.select_dtypes(include=['object', 'category']).columns.tolist()
            },
            'data_quality_flags': []
        }
        
        # Add quality flags based on validation rules
        if quality_report['missing_values']['missing_percentage'] > 20:
            quality_report['data_quality_flags'].append('HIGH_MISSING_VALUES')
        
        if not quality_report['label_distribution']['is_balanced']:
            quality_report['data_quality_flags'].append('IMBALANCED_LABELS')
        
        if X.shape[0] < 1000:
            quality_report['data_quality_flags'].append('SMALL_SAMPLE_SIZE')
        
        if len(quality_report['missing_values']['high_missing_columns']) > 0:
            quality_report['data_quality_flags'].append('HIGH_MISSING_FEATURES')
        
        return quality_report
    
    @staticmethod
    def validate_feature_store_format(df: pd.DataFrame) -> bool:
        """Validate that dataframe matches expected feature store format"""
        
        required_columns = ['Customer_ID', 'feature_snapshot_date']
        
        # Check if required columns exist
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            print(f"Missing required columns: {missing_required}")
            return False
        
        # Check if there are any feature columns
        feature_columns = [col for col in df.columns if not col.startswith(('Customer_ID', 'snapshot', 'ingestion', 'data_source'))]
        if len(feature_columns) == 0:
            print("No feature columns found in dataframe")
            return False
        
        print(f"Feature store validation passed: {len(feature_columns)} features found")
        return True


# Utility functions for integration with existing pipeline
def load_and_prepare_data(feature_store_path: str, label_store_path: str, 
                         config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load feature and label data from gold layer stores
    
    Integrates with existing Assignment 1 data pipeline outputs
    """
    import glob
    import pyspark
    from pyspark.sql.functions import col
    
    # Initialize Spark session
    spark = pyspark.sql.SparkSession.builder \
        .appName("ML_Pipeline_Preprocessing") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # Load label store
        label_files = [label_store_path + os.path.basename(f) 
                      for f in glob.glob(os.path.join(label_store_path, '*'))]
        label_store_sdf = spark.read.option("header", "true").parquet(*label_files)
        
        # Load feature store  
        feature_files = [feature_store_path + os.path.basename(f) 
                        for f in glob.glob(os.path.join(feature_store_path, '*'))]
        feature_store_sdf = spark.read.option("header", "true").parquet(*feature_files)
        
        # Filter by time window
        labels_sdf = label_store_sdf.filter(
            (col("snapshot_date") >= config["train_test_start_date"]) & 
            (col("snapshot_date") <= config["oot_end_date"])
        )
        
        features_sdf = feature_store_sdf.filter(
            (col("feature_snapshot_date") >= config["train_test_start_date"]) & 
            (col("feature_snapshot_date") <= config["oot_end_date"])
        )
        
        # Convert to Pandas for easier preprocessing
        labels_pdf = labels_sdf.toPandas()
        features_pdf = features_sdf.toPandas()
        
        print(f"Loaded {labels_pdf.shape[0]} label records and {features_pdf.shape[0]} feature records")
        
        return features_pdf, labels_pdf
        
    finally:
        spark.stop()


if __name__ == "__main__":
    # Example usage and testing
    print("ML Pipeline Preprocessing Module")
    print("Supported models:", PreprocessorFactory.get_supported_models())
    
    # Create sample preprocessors
    for model_type in PreprocessorFactory.get_supported_models():
        preprocessor = PreprocessorFactory.create_preprocessor(model_type)
        print(f"Created {preprocessor.__class__.__name__} for {model_type}")
