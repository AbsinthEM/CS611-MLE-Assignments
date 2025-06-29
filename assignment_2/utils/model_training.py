"""
Model Training Utilities Module
Provides production-ready training functionality for ML models

This module handles:
- Training configuration management
- Data loading and preprocessing for training
- Multi-model training with hyperparameter optimization
- Model evaluation and selection
- Model artifact management and storage
- Batch training across multiple time periods
- Airflow-compatible training tasks
"""

import os
import glob
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import pyspark
from pyspark.sql.functions import col

# Import preprocessing utilities
from utils.preprocessing import PreprocessorFactory, DataQualityChecker


def build_training_config(training_date_str: str, models_to_train: List[str] = None,
                         train_test_period_months: int = 12, oot_period_months: int = 2,
                         train_test_ratio: float = 0.8, cv_folds: int = 3,
                         hyperparameter_iterations: int = 10, random_state: int = 88,
                         feature_store_path: str = "datamart/gold/feature_store/",
                         label_store_path: str = "datamart/gold/label_store/",
                         model_bank_directory: str = "models/") -> Dict[str, Any]:
    """
    Build configuration dictionary for model training
    
    Args:
        training_date_str: Training cutoff date in YYYY-MM-DD format
        models_to_train: List of model types to train
        train_test_period_months: Number of months for training/testing period
        oot_period_months: Number of months for out-of-time validation
        train_test_ratio: Ratio for train/test split
        cv_folds: Number of cross-validation folds
        hyperparameter_iterations: Number of hyperparameter search iterations
        random_state: Random state for reproducibility
        feature_store_path: Path to feature store data
        label_store_path: Path to label store data
        model_bank_directory: Directory to save trained models
        
    Returns:
        Configuration dictionary containing all training parameters
    """
    
    if models_to_train is None:
        models_to_train = ["logistic_regression", "random_forest", "xgboost"]
    
    # Calculate date ranges
    model_train_date = datetime.strptime(training_date_str, "%Y-%m-%d")
    oot_end_date = model_train_date - timedelta(days=1)
    oot_start_date = model_train_date - relativedelta(months=oot_period_months)
    train_test_end_date = oot_start_date - timedelta(days=1)
    train_test_start_date = oot_start_date - relativedelta(months=train_test_period_months)
    
    config = {
        # Training dates
        "model_train_date_str": training_date_str,
        "model_train_date": model_train_date,
        "train_test_start_date": train_test_start_date,
        "train_test_end_date": train_test_end_date,
        "oot_start_date": oot_start_date,
        "oot_end_date": oot_end_date,
        
        # Training parameters
        "train_test_period_months": train_test_period_months,
        "oot_period_months": oot_period_months,
        "train_test_ratio": train_test_ratio,
        "models_to_train": models_to_train,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "hyperparameter_iterations": hyperparameter_iterations,
        
        # Data paths
        "feature_store_path": feature_store_path,
        "label_store_path": label_store_path,
        "model_bank_directory": model_bank_directory,
        
        # Output configuration
        "model_artifacts_output": {}
    }
    
    return config


def load_training_data(config: Dict[str, Any], spark) -> Optional[pd.DataFrame]:
    """
    Load and merge feature store and label store data for training
    
    Args:
        config: Configuration dictionary
        spark: SparkSession object
        
    Returns:
        DataFrame containing merged training data or None if loading fails
    """
    
    try:
        print(f"Loading training data...")
        print(f"Training period: {config['train_test_start_date'].date()} to {config['train_test_end_date'].date()}")
        print(f"OOT period: {config['oot_start_date'].date()} to {config['oot_end_date'].date()}")
        
        # Load label store
        label_store_path = config["label_store_path"]
        label_files = glob.glob(os.path.join(label_store_path, "*.parquet"))
        
        if not label_files:
            print(f"No label files found in: {label_store_path}")
            return None
        
        # Load all label files
        label_files_full_path = [label_store_path + os.path.basename(f) for f in label_files]
        label_store_sdf = spark.read.option("header", "true").parquet(*label_files_full_path)
        
        # Filter labels by time window
        labels_sdf = label_store_sdf.filter(
            (col("snapshot_date") >= config["train_test_start_date"]) & 
            (col("snapshot_date") <= config["oot_end_date"])
        )
        
        print(f"Loaded {labels_sdf.count()} label records")
        
        # Load feature store
        feature_store_path = config["feature_store_path"]
        feature_files = glob.glob(os.path.join(feature_store_path, "*.parquet"))
        
        if not feature_files:
            print(f"No feature files found in: {feature_store_path}")
            return None
        
        # Load all feature files
        feature_files_full_path = [feature_store_path + os.path.basename(f) for f in feature_files]
        feature_store_sdf = spark.read.option("header", "true").parquet(*feature_files_full_path)
        
        # Filter features by time window
        features_sdf = feature_store_sdf.filter(
            (col("feature_snapshot_date") >= config["train_test_start_date"]) & 
            (col("feature_snapshot_date") <= config["oot_end_date"])
        )
        
        print(f"Loaded {features_sdf.count()} feature records")
        
        # Join features and labels
        merged_sdf = labels_sdf.join(
            features_sdf,
            (labels_sdf.Customer_ID == features_sdf.Customer_ID) & 
            (labels_sdf.snapshot_date == features_sdf.feature_snapshot_date),
            how="left"
        ).select(
            labels_sdf["*"],  # All label columns
            *[features_sdf[col] for col in features_sdf.columns 
              if col not in ["Customer_ID", "feature_snapshot_date"]]  # Feature columns except join keys
        )
        
        # Convert to Pandas
        data_pdf = merged_sdf.toPandas()
        print(f"Merged dataset shape: {data_pdf.shape}")
        
        return data_pdf
        
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return None


def validate_training_data(data_pdf: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate training data quality before model training
    
    Args:
        data_pdf: Training data DataFrame
        config: Configuration dictionary
        
    Returns:
        Dictionary containing validation results
    """
    
    try:
        print(f"Validating training data quality...")
        
        # Get feature columns (exclude identifiers and labels)
        feature_cols = [col for col in data_pdf.columns 
                       if col not in ['Customer_ID', 'snapshot_date', 'label', 'label_def']]
        
        # Extract features and labels
        X = data_pdf[feature_cols]
        y = data_pdf['label']
        
        # Run data quality validation
        quality_report = DataQualityChecker.validate_training_data(X, y, config)
        
        print(f"Data quality validation completed")
        if quality_report['data_quality_flags']:
            print(f"WARNING: Data quality issues detected: {quality_report['data_quality_flags']}")
        else:
            print("Data quality validation passed")
        
        return quality_report
        
    except Exception as e:
        print(f"Error validating training data: {str(e)}")
        return {'error': str(e), 'data_quality_flags': ['validation_failed']}


def prepare_training_datasets(data_pdf: pd.DataFrame, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Prepare training datasets by splitting into train/test/oot periods
    
    Args:
        data_pdf: Merged training data
        config: Configuration dictionary
        
    Returns:
        Dictionary containing split datasets or None if preparation fails
    """
    
    try:
        print(f"Preparing training datasets...")
        
        # Get feature columns
        feature_cols = [col for col in data_pdf.columns 
                       if col not in ['Customer_ID', 'snapshot_date', 'label', 'label_def']]
        
        # Split data by time periods
        oot_pdf = data_pdf[
            (data_pdf['snapshot_date'] >= config["oot_start_date"].date()) & 
            (data_pdf['snapshot_date'] <= config["oot_end_date"].date())
        ]
        
        train_test_pdf = data_pdf[
            (data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) & 
            (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())
        ]
        
        # Extract features and labels for OOT period
        X_oot = oot_pdf[feature_cols]
        y_oot = oot_pdf["label"]
        
        # Split train/test period into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            train_test_pdf[feature_cols], 
            train_test_pdf["label"], 
            test_size=1 - config["train_test_ratio"],
            random_state=config["random_state"],
            shuffle=False  # Preserve temporal order to avoid data leakage
        )
        
        datasets = {
            'X_train': X_train,
            'X_test': X_test,
            'X_oot': X_oot,
            'y_train': y_train,
            'y_test': y_test,
            'y_oot': y_oot,
            'feature_columns': feature_cols
        }
        
        print(f"Dataset preparation completed:")
        print(f"  Train: {X_train.shape[0]} samples, label rate: {y_train.mean():.3f}")
        print(f"  Test: {X_test.shape[0]} samples, label rate: {y_test.mean():.3f}")
        print(f"  OOT: {X_oot.shape[0]} samples, label rate: {y_oot.mean():.3f}")
        print(f"  Features: {len(feature_cols)}")
        
        return datasets
        
    except Exception as e:
        print(f"Error preparing training datasets: {str(e)}")
        return None


def prepare_model_preprocessing(datasets: Dict[str, Any], model_type: str, 
                              random_state: int = 88) -> Optional[Dict[str, Any]]:
    """
    Prepare model-specific preprocessing for training data
    
    Args:
        datasets: Dictionary containing training datasets
        model_type: Type of model for preprocessing
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing preprocessed data or None if preprocessing fails
    """
    
    try:
        print(f"Preparing preprocessing for {model_type}...")
        
        # Create preprocessor
        preprocessor = PreprocessorFactory.create_preprocessor(model_type, random_state=random_state)
        
        # Fit on training data only
        X_processed_train, feature_names = preprocessor.fit_transform(
            datasets['X_train'], datasets['y_train']
        )
        
        # Transform test and OOT data using fitted preprocessor
        X_processed_test = preprocessor.transform(datasets['X_test'])
        X_processed_oot = preprocessor.transform(datasets['X_oot'])
        
        preprocessing_result = {
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'preprocessing_stats': preprocessor.preprocessing_stats,
            'X_train': X_processed_train,
            'X_test': X_processed_test,
            'X_oot': X_processed_oot
        }
        
        stats = preprocessor.preprocessing_stats
        print(f"{model_type}: {stats['original_features']} -> {stats['final_features']} features")
        
        return preprocessing_result
        
    except Exception as e:
        print(f"Error in preprocessing for {model_type}: {str(e)}")
        return None


def get_hyperparameter_space(model_type: str) -> Dict[str, Any]:
    """
    Define hyperparameter search spaces for different model types
    """
    
    if model_type.lower() in ['logistic_regression', 'logistic', 'lr']:
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
            'class_weight': [None, 'balanced']
        }
    
    elif model_type.lower() in ['random_forest', 'randomforest', 'rf']:
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
    
    elif model_type.lower() in ['xgboost', 'xgb', 'boost']:
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.5, 1],
            'min_child_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [1, 1.5, 2, 3],
            'scale_pos_weight': [1, 2, 3]
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_base_model(model_type: str, random_state: int = 88) -> Any:
    """
    Create base model instance with default parameters
    """
    
    if model_type.lower() in ['logistic_regression', 'logistic', 'lr']:
        return LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear'
        )
    
    elif model_type.lower() in ['random_forest', 'randomforest', 'rf']:
        return RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            n_estimators=100
        )
    
    elif model_type.lower() in ['xgboost', 'xgb', 'boost']:
        return xgb.XGBClassifier(
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_single_model(model_type: str, preprocessing_result: Dict[str, Any], 
                      datasets: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Train a single model with hyperparameter optimization
    
    Args:
        model_type: Type of model to train
        preprocessing_result: Dictionary containing preprocessed data
        datasets: Original datasets with labels
        config: Training configuration
        
    Returns:
        Model artifact dictionary or None if training fails
    """
    
    try:
        start_time = time.time()
        
        print(f"\n[{model_type.upper()}] Starting model training...")
        print(f"[{model_type.upper()}] Train samples: {preprocessing_result['X_train'].shape[0]}")
        print(f"[{model_type.upper()}] Test samples: {preprocessing_result['X_test'].shape[0]}")
        print(f"[{model_type.upper()}] OOT samples: {preprocessing_result['X_oot'].shape[0]}")
        print(f"[{model_type.upper()}] Features: {preprocessing_result['X_train'].shape[1]}")
        
        # Get preprocessed data
        X_train_processed = preprocessing_result['X_train']
        X_test_processed = preprocessing_result['X_test']
        X_oot_processed = preprocessing_result['X_oot']
        feature_names = preprocessing_result['feature_names']
        
        # Get labels
        y_train = datasets['y_train']
        y_test = datasets['y_test']
        y_oot = datasets['y_oot']
        
        # Create base model
        base_model = create_base_model(model_type, config["random_state"])
        
        # Get hyperparameter search space
        param_space = get_hyperparameter_space(model_type)
        
        # Setup cross-validation
        cv_splitter = StratifiedKFold(
            n_splits=config["cv_folds"],
            shuffle=True,
            random_state=config["random_state"]
        )
        
        # Perform hyperparameter search
        print(f"[{model_type.upper()}] Starting hyperparameter search...")
        print(f"[{model_type.upper()}] Search iterations: {config['hyperparameter_iterations']}")
        print(f"[{model_type.upper()}] CV folds: {config['cv_folds']}")
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=config["hyperparameter_iterations"],
            scoring='roc_auc',
            cv=cv_splitter,
            random_state=config["random_state"],
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the model
        random_search.fit(X_train_processed, y_train)
        best_model = random_search.best_estimator_
        
        print(f"[{model_type.upper()}] Best CV score: {random_search.best_score_:.4f}")
        
        # Evaluate model performance
        performance_results = evaluate_model_performance(
            best_model, X_train_processed, X_test_processed, X_oot_processed,
            y_train, y_test, y_oot, model_type
        )
        
        # Get feature importance if available
        feature_importance = extract_feature_importance(best_model, feature_names)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Create model artifact
        model_artifact = create_model_artifact(
            best_model, model_type, preprocessing_result, config,
            performance_results, random_search, feature_importance,
            training_time, datasets
        )
        
        print(f"[{model_type.upper()}] Training completed in {training_time:.2f} seconds")
        
        return model_artifact
        
    except Exception as e:
        print(f"[{model_type.upper()}] Training failed: {str(e)}")
        return None


def evaluate_model_performance(model: Any, X_train: np.ndarray, X_test: np.ndarray, X_oot: np.ndarray,
                             y_train: pd.Series, y_test: pd.Series, y_oot: pd.Series, 
                             model_type: str) -> Dict[str, float]:
    """
    Evaluate model performance on train, test, and OOT datasets
    
    Args:
        model: Trained model
        X_train, X_test, X_oot: Feature matrices
        y_train, y_test, y_oot: Label series
        model_type: Type of model for logging
        
    Returns:
        Dictionary containing performance metrics
    """
    
    try:
        # Train set evaluation
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        
        # Test set evaluation
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        # OOT evaluation
        y_oot_pred_proba = model.predict_proba(X_oot)[:, 1]
        oot_auc = roc_auc_score(y_oot, y_oot_pred_proba)
        
        # Calculate Gini coefficients
        train_gini = 2 * train_auc - 1
        test_gini = 2 * test_auc - 1
        oot_gini = 2 * oot_auc - 1
        
        print(f"[{model_type.upper()}] Performance Results:")
        print(f"[{model_type.upper()}]   Train AUC: {train_auc:.4f} (Gini: {train_gini:.3f})")
        print(f"[{model_type.upper()}]   Test AUC:  {test_auc:.4f} (Gini: {test_gini:.3f})")
        print(f"[{model_type.upper()}]   OOT AUC:   {oot_auc:.4f} (Gini: {oot_gini:.3f})")
        
        return {
            'auc_train': float(train_auc),
            'auc_test': float(test_auc),
            'auc_oot': float(oot_auc),
            'gini_train': float(train_gini),
            'gini_test': float(test_gini),
            'gini_oot': float(oot_gini)
        }
        
    except Exception as e:
        print(f"Error evaluating model performance: {str(e)}")
        return {}


def extract_feature_importance(model: Any, feature_names: List[str]) -> Optional[Dict[str, float]]:
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance scores or None
    """
    
    try:
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(zip(feature_names, np.abs(model.coef_[0])))
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting feature importance: {str(e)}")
        return None


def create_model_artifact(model: Any, model_type: str, preprocessing_result: Dict[str, Any],
                         config: Dict[str, Any], performance_results: Dict[str, float],
                         random_search: Any, feature_importance: Optional[Dict[str, float]],
                         training_time: float, datasets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create model artifact dictionary for storage
    
    Args:
        model: Trained model
        model_type: Type of model
        preprocessing_result: Preprocessing results
        config: Training configuration
        performance_results: Performance metrics
        random_search: RandomizedSearchCV object
        feature_importance: Feature importance scores
        training_time: Training duration in seconds
        datasets: Original datasets for statistics
        
    Returns:
        Model artifact dictionary
    """
    
    model_artifact = {
        'model': model,
        'model_type': model_type,
        'model_version': f"credit_model_{model_type}_{config['model_train_date_str'].replace('-', '_')}",
        'preprocessing_transformers': {
            'preprocessor': preprocessing_result['preprocessor'],
            'feature_names': preprocessing_result['feature_names'],
            'preprocessing_stats': preprocessing_result['preprocessing_stats']
        },
        'data_dates': config,
        'data_stats': {
            'X_train': datasets['X_train'].shape[0],
            'X_test': datasets['X_test'].shape[0],
            'X_oot': datasets['X_oot'].shape[0],
            'y_train': float(datasets['y_train'].mean()),
            'y_test': float(datasets['y_test'].mean()),
            'y_oot': float(datasets['y_oot'].mean()),
            'feature_count': preprocessing_result['X_train'].shape[1],
            'original_feature_count': len(preprocessing_result['feature_names'])
        },
        'results': {
            **performance_results,
            'cv_score': float(random_search.best_score_)
        },
        'hp_params': random_search.best_params_,
        'feature_importance': feature_importance,
        'training_metadata': {
            'training_time_seconds': training_time,
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cv_results': {
                'mean_cv_score': float(random_search.best_score_),
                'std_cv_score': float(random_search.cv_results_['std_test_score'][random_search.best_index_])
            }
        }
    }
    
    return model_artifact


def save_model_artifacts(model_artifacts: Dict[str, Dict[str, Any]], 
                        config: Dict[str, Any]) -> Dict[str, str]:
    """
    Save all model artifacts to model bank
    
    Args:
        model_artifacts: Dictionary of model artifacts
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping model types to saved file paths
    """
    
    try:
        model_bank_directory = config["model_bank_directory"]
        if not os.path.exists(model_bank_directory):
            os.makedirs(model_bank_directory)
        
        saved_paths = {}
        
        for model_type, artifact in model_artifacts.items():
            if artifact and 'error' not in str(artifact):
                # Save main model artifact
                model_filepath = os.path.join(
                    model_bank_directory,
                    f"{artifact['model_version']}.pkl"
                )
                
                with open(model_filepath, 'wb') as f:
                    pickle.dump(artifact, f)
                
                saved_paths[model_type] = model_filepath
                print(f"✓ {model_type} saved to: {model_filepath}")
                
                # Save preprocessor separately for easy access
                preprocessor_filepath = os.path.join(
                    model_bank_directory,
                    f"{artifact['model_version']}_preprocessor.pkl"
                )
                
                artifact['preprocessing_transformers']['preprocessor'].save_transformers(preprocessor_filepath)
                print(f"✓ {model_type} preprocessor saved to: {preprocessor_filepath}")
        
        return saved_paths
        
    except Exception as e:
        print(f"Error saving model artifacts: {str(e)}")
        return {}


def validate_model_artifacts(model_artifacts: Dict[str, Dict[str, Any]], 
                           datasets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate saved model artifacts by testing inference
    
    Args:
        model_artifacts: Dictionary of model artifacts
        datasets: Test datasets for validation
        
    Returns:
        Dictionary containing validation results
    """
    
    try:
        validation_results = {}
        
        for model_type, artifact in model_artifacts.items():
            if artifact and 'error' not in str(artifact):
                print(f"\n[VALIDATION] Testing {model_type} inference...")
                
                try:
                    # Get preprocessor and model
                    preprocessor = artifact['preprocessing_transformers']['preprocessor']
                    model = artifact['model']
                    
                    # Preprocess test data
                    X_test_processed = preprocessor.transform(datasets['X_test'])
                    
                    # Make predictions
                    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                    y_pred = model.predict(X_test_processed)
                    
                    # Calculate metrics
                    auc_score = roc_auc_score(datasets['y_test'], y_pred_proba)
                    
                    validation_results[model_type] = {
                        'validation_successful': True,
                        'auc_score': float(auc_score),
                        'prediction_stats': {
                            'mean_probability': float(y_pred_proba.mean()),
                            'std_probability': float(y_pred_proba.std())
                        }
                    }
                    
                    print(f"[VALIDATION] ✓ {model_type} validation successful - AUC: {auc_score:.4f}")
                    
                except Exception as e:
                    validation_results[model_type] = {
                        'validation_successful': False,
                        'error': str(e)
                    }
                    print(f"[VALIDATION] ✗ {model_type} validation failed: {str(e)}")
        
        return validation_results
        
    except Exception as e:
        print(f"Error validating model artifacts: {str(e)}")
        return {}


def select_best_model(model_artifacts: Dict[str, Dict[str, Any]], 
                     selection_metric: str = 'auc_test') -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Select best model based on specified metric
    
    Args:
        model_artifacts: Dictionary of model artifacts
        selection_metric: Metric to use for model selection
        
    Returns:
        Tuple of (best_model_type, best_model_artifact) or None
    """
    
    try:
        valid_models = {k: v for k, v in model_artifacts.items() 
                       if v and 'error' not in str(v) and selection_metric in v.get('results', {})}
        
        if not valid_models:
            print("No valid models found for selection")
            return None
        
        # Find best model by metric
        best_model_type = max(valid_models.keys(), 
                             key=lambda x: valid_models[x]['results'][selection_metric])
        best_model_artifact = valid_models[best_model_type]
        
        print(f"\nBest model selected: {best_model_type}")
        print(f"Selection metric ({selection_metric}): {best_model_artifact['results'][selection_metric]:.4f}")
        
        return best_model_type, best_model_artifact
        
    except Exception as e:
        print(f"Error selecting best model: {str(e)}")
        return None


def run_full_training_pipeline(config: Dict[str, Any], spark) -> Dict[str, Any]:
    """
    Run the complete training pipeline
    
    Args:
        config: Training configuration dictionary
        spark: SparkSession object
        
    Returns:
        Dictionary containing training results
    """
    
    try:
        print(f"\n{'='*80}")
        print(f"STARTING FULL TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Training date: {config['model_train_date_str']}")
        print(f"Models to train: {config['models_to_train']}")
        
        # Load training data
        data_pdf = load_training_data(config, spark)
        if data_pdf is None:
            return {'success': False, 'error': 'Failed to load training data'}
        
        # Validate data quality
        quality_report = validate_training_data(data_pdf, config)
        if 'error' in quality_report:
            return {'success': False, 'error': 'Data quality validation failed'}
        
        # Prepare datasets
        datasets = prepare_training_datasets(data_pdf, config)
        if datasets is None:
            return {'success': False, 'error': 'Failed to prepare datasets'}
        
        # Train models
        model_artifacts = {}
        preprocessing_results = {}
        
        for model_type in config["models_to_train"]:
            print(f"\n{'='*60}")
            print(f"TRAINING {model_type.upper()}")
            print(f"{'='*60}")
            
            # Prepare preprocessing for this model
            preprocessing_result = prepare_model_preprocessing(
                datasets, model_type, config["random_state"]
            )
            
            if preprocessing_result is None:
                print(f"✗ {model_type} preprocessing failed")
                continue
            
            preprocessing_results[model_type] = preprocessing_result
            
            # Train model
            model_artifact = train_single_model(
                model_type, preprocessing_result, datasets, config
            )
            
            if model_artifact is None:
                print(f"✗ {model_type} training failed")
                continue
            
            model_artifacts[model_type] = model_artifact
            print(f"✓ {model_type} training completed successfully")
        
        # Save model artifacts
        saved_paths = save_model_artifacts(model_artifacts, config)
        
        # Validate artifacts
        validation_results = validate_model_artifacts(model_artifacts, datasets)
        
        # Select best model
        best_model_info = select_best_model(model_artifacts)
        
        # Prepare results summary
        results = {
            'success': True,
            'training_date': config['model_train_date_str'],
            'models_trained': list(model_artifacts.keys()),
            'models_failed': [m for m in config['models_to_train'] if m not in model_artifacts],
            'saved_paths': saved_paths,
            'validation_results': validation_results,
            'best_model': best_model_info[0] if best_model_info else None,
            'model_artifacts': model_artifacts,
            'data_stats': {
                'total_samples': len(data_pdf),
                'train_samples': datasets['X_train'].shape[0],
                'test_samples': datasets['X_test'].shape[0],
                'oot_samples': datasets['X_oot'].shape[0]
            }
        }
        
        print(f"\n{'='*80}")
        print(f"TRAINING PIPELINE COMPLETED")
        print(f"{'='*80}")
        print(f"Models trained successfully: {len(model_artifacts)}/{len(config['models_to_train'])}")
        if best_model_info:
            print(f"Best model: {best_model_info[0]}")
        
        return results
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        return {'success': False, 'error': str(e)}


def run_batch_training(training_dates: List[str], base_config: Dict[str, Any], 
                      spark) -> List[Dict[str, Any]]:
    """
    Run training for multiple dates in batch
    
    Args:
        training_dates: List of training dates in YYYY-MM-DD format
        base_config: Base configuration to use for all training runs
        spark: SparkSession object
        
    Returns:
        List of training results for each date
    """
    
    results = []
    
    print(f"Starting batch training for {len(training_dates)} dates")
    print(f"Training dates: {training_dates}")
    
    for i, training_date in enumerate(training_dates, 1):
        print(f"\nProcessing training date {i}/{len(training_dates)}: {training_date}")
        
        try:
            # Update config for this training date
            training_config = build_training_config(
                training_date_str=training_date,
                models_to_train=base_config.get('models_to_train'),
                train_test_period_months=base_config.get('train_test_period_months', 12),
                oot_period_months=base_config.get('oot_period_months', 2),
                train_test_ratio=base_config.get('train_test_ratio', 0.8),
                cv_folds=base_config.get('cv_folds', 3),
                hyperparameter_iterations=base_config.get('hyperparameter_iterations', 10),
                random_state=base_config.get('random_state', 88),
                feature_store_path=base_config.get('feature_store_path', "datamart/gold/feature_store/"),
                label_store_path=base_config.get('label_store_path', "datamart/gold/label_store/"),
                model_bank_directory=base_config.get('model_bank_directory', "models/")
            )
            
            # Run training pipeline
            result = run_full_training_pipeline(training_config, spark)
            result['training_date'] = training_date
            results.append(result)
            
            if result['success']:
                print(f"✓ Successfully completed training for {training_date}")
            else:
                print(f"✗ Failed training for {training_date}: {result['error']}")
                
        except Exception as e:
            print(f"✗ Error in batch training for {training_date}: {str(e)}")
            results.append({
                'success': False,
                'training_date': training_date,
                'error': str(e)
            })
    
    return results


# Airflow-compatible task functions
def training_airflow_task(training_date: str, models_to_train: List[str] = None, **context) -> str:
    """
    Airflow-compatible training task
    
    Args:
        training_date: Training date in YYYY-MM-DD format
        models_to_train: List of model types to train
        context: Airflow context
        
    Returns:
        Success message with training summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"Training_{training_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting training task for {training_date}")
        
        # Build configuration
        config = build_training_config(
            training_date_str=training_date,
            models_to_train=models_to_train or ["logistic_regression", "random_forest", "xgboost"],
            feature_store_path="/opt/airflow/datamart/gold/feature_store/",
            label_store_path="/opt/airflow/datamart/gold/label_store/",
            model_bank_directory="/opt/airflow/models/"
        )
        
        # Run training pipeline
        result = run_full_training_pipeline(config, spark)
        
        if result['success']:
            message = f"Training completed: {len(result['models_trained'])} models trained"
            if result['best_model']:
                message += f", best model: {result['best_model']}"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Training failed: {result['error']}")
            
    finally:
        try:
            if 'spark' in locals() and spark is not None:
                print("[SPARK CLEANUP] Stopping Spark session...")
                spark.stop()
                print("[SPARK CLEANUP] ✓ Spark session stopped successfully")
        except Exception as e:
            print(f"[SPARK CLEANUP] Warning: Error stopping Spark session: {e}")
            print("[SPARK CLEANUP] Training completed successfully despite cleanup warning")
            try:
                from pyspark.sql import SparkSession
                SparkSession._instantiatedSession = None
                SparkSession._activeSession = None
                print("[SPARK CLEANUP] ✓ Alternative cleanup completed")
            except:
                print("[SPARK CLEANUP] Alternative cleanup failed, proceeding anyway")



def batch_training_airflow_task(start_date: str, end_date: str, 
                               models_to_train: List[str] = None, **context) -> str:
    """
    Airflow-compatible batch training task
    
    Args:
        start_date: Start date for batch training in YYYY-MM-DD format
        end_date: End date for batch training in YYYY-MM-DD format
        models_to_train: List of model types to train
        context: Airflow context
        
    Returns:
        Success message with batch training summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"BatchTraining") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting batch training from {start_date} to {end_date}")
        
        # Generate training date list (monthly)
        def generate_monthly_dates(start_str, end_str):
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d")
            dates = []
            current = datetime(start.year, start.month, 1)
            
            while current <= end:
                dates.append(current.strftime("%Y-%m-%d"))
                if current.month == 12:
                    current = datetime(current.year + 1, 1, 1)
                else:
                    current = datetime(current.year, current.month + 1, 1)
            return dates
        
        training_dates = generate_monthly_dates(start_date, end_date)
        
        # Base configuration
        base_config = {
            'models_to_train': models_to_train or ["logistic_regression", "random_forest", "xgboost"],
            'feature_store_path': "/opt/airflow/datamart/gold/feature_store/",
            'label_store_path': "/opt/airflow/datamart/gold/label_store/",
            'model_bank_directory': "/opt/airflow/models/"
        }
        
        # Run batch training
        results = run_batch_training(training_dates, base_config, spark)
        
        successful = len([r for r in results if r['success']])
        total_models = sum([len(r.get('models_trained', [])) for r in results if r['success']])
        
        message = f"Batch training completed: {successful}/{len(results)} dates processed, {total_models} total models trained"
        print(f"[AIRFLOW] {message}")
        return message
        
    finally:
        try:
            if 'spark' in locals() and spark is not None:
                print("[SPARK CLEANUP] Stopping Spark session...")
                spark.stop()
                print("[SPARK CLEANUP] ✓ Spark session stopped successfully")
        except Exception as e:
            print(f"[SPARK CLEANUP] Warning: Error stopping Spark session: {e}")
            print("[SPARK CLEANUP] Training completed successfully despite cleanup warning")
            try:
                from pyspark.sql import SparkSession
                SparkSession._instantiatedSession = None
                SparkSession._activeSession = None
                print("[SPARK CLEANUP] ✓ Alternative cleanup completed")
            except:
                print("[SPARK CLEANUP] Alternative cleanup failed, proceeding anyway")



if __name__ == "__main__":
    # Example usage and testing
    print("Model Training Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")
