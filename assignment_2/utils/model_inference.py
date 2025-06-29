"""
Model Inference Utilities Module
Provides production-ready inference functionality for ML models

This module handles:
- Model artifact loading and validation
- Feature data extraction and preprocessing  
- Batch inference processing
- Prediction storage and validation
- Airflow-compatible task functions
"""

import os
import glob
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Any, Optional
import pyspark
from pyspark.sql.functions import col


def build_inference_config(snapshot_date_str: str, model_version: str, 
                          model_bank_directory: str = "models/",
                          feature_store_path: str = "datamart/gold/feature_store/",
                          predictions_output_path: str = "datamart/gold/model_predictions/") -> Dict[str, Any]:
    """
    Build configuration dictionary for model inference
    
    Args:
        snapshot_date_str: Date for inference in YYYY-MM-DD format
        model_version: Name of model version to use (without .pkl extension)
        model_bank_directory: Directory containing saved models
        feature_store_path: Path to feature store data
        predictions_output_path: Path to save predictions
    
    Returns:
        Configuration dictionary containing all inference parameters
    """
    
    config = {
        # Date configuration
        "snapshot_date_str": snapshot_date_str,
        "snapshot_date": datetime.strptime(snapshot_date_str, "%Y-%m-%d"),
        
        # Model configuration
        "model_version": model_version,
        "model_bank_directory": model_bank_directory,
        "model_artifact_filepath": os.path.join(model_bank_directory, f"{model_version}.pkl"),
        
        # Data paths
        "feature_store_path": feature_store_path,
        "predictions_output_path": predictions_output_path,
        
        # Output configuration
        "predictions_table_name": f"{model_version}_predictions",
        "prediction_columns": ["Customer_ID", "snapshot_date", "model_version", 
                              "model_prediction_proba", "model_prediction_binary", 
                              "prediction_timestamp"]
    }
    
    return config


def load_model_artifact(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load model artifact from model bank
    
    Args:
        config: Configuration dictionary containing model file path
        
    Returns:
        Model artifact dictionary or None if loading fails
    """
    
    try:
        model_filepath = config["model_artifact_filepath"]
        
        if not os.path.exists(model_filepath):
            print(f"Model file not found: {model_filepath}")
            return None
            
        with open(model_filepath, 'rb') as f:
            model_artifact = pickle.load(f)
        
        # Validate model artifact structure
        required_keys = ['model', 'model_type', 'preprocessing_transformers']
        missing_keys = [key for key in required_keys if key not in model_artifact]
        
        if missing_keys:
            print(f"Invalid model artifact - missing keys: {missing_keys}")
            return None
            
        print(f"Model artifact loaded successfully from: {model_filepath}")
        return model_artifact
        
    except Exception as e:
        print(f"Error loading model artifact: {str(e)}")
        return None


def load_inference_features(config: Dict[str, Any], spark) -> Optional[pd.DataFrame]:
    """
    Load feature data for inference from feature store
    
    Args:
        config: Configuration dictionary
        spark: SparkSession object
        
    Returns:
        DataFrame containing features for inference or None if loading fails
    """
    
    try:
        feature_store_path = config["feature_store_path"]
        snapshot_date = config["snapshot_date"]
        
        print(f"Loading features from: {feature_store_path}")
        print(f"Target snapshot date: {snapshot_date}")
        
        # Get all feature store files
        feature_files = glob.glob(os.path.join(feature_store_path, "*.parquet"))
        
        if not feature_files:
            print(f"No feature files found in: {feature_store_path}")
            return None
        
        # Load all feature store data
        feature_files_full_path = [feature_store_path + os.path.basename(f) for f in feature_files]
        features_sdf = spark.read.option("header", "true").parquet(*feature_files_full_path)
        
        # Filter by snapshot date
        features_filtered_sdf = features_sdf.filter(col("feature_snapshot_date") == snapshot_date)
        
        # Convert to Pandas
        features_pdf = features_filtered_sdf.toPandas()
        
        if len(features_pdf) == 0:
            print(f"No feature data found for snapshot date: {snapshot_date}")
            return None
            
        print(f"Feature data loaded: {len(features_pdf)} records")
        return features_pdf
        
    except Exception as e:
        print(f"Error loading inference features: {str(e)}")
        return None


def prepare_inference_features(feature_data: pd.DataFrame, model_artifact: Dict[str, Any], 
                             config: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Prepare features for model inference using saved preprocessor
    
    Args:
        feature_data: Raw feature data from feature store
        model_artifact: Loaded model artifact containing preprocessor
        config: Configuration dictionary
        
    Returns:
        Tuple of (processed_features_array, customer_info_dataframe)
    """
    
    try:
        print(f"Preparing features for inference...")
        
        # Extract preprocessor from model artifact
        preprocessor = model_artifact['preprocessing_transformers']['preprocessor']
        feature_names = model_artifact['preprocessing_transformers']['feature_names']
        
        print(f"Using preprocessor: {preprocessor.__class__.__name__}")
        print(f"Expected features: {len(feature_names)}")
        
        # Prepare customer info for later use
        customer_info = feature_data[['Customer_ID', 'feature_snapshot_date']].copy()
        customer_info['snapshot_date'] = config['snapshot_date_str']
        
        # Select feature columns (exclude identifiers)
        feature_cols = [col for col in feature_data.columns 
                       if col not in ['Customer_ID', 'feature_snapshot_date']]
        
        X_raw = feature_data[feature_cols]
        print(f"Raw features shape: {X_raw.shape}")
        
        # Apply preprocessing using saved preprocessor
        X_processed = preprocessor.transform(X_raw)
        
        print(f"Processed features shape: {X_processed.shape}")
        print(f"Preprocessing completed successfully")
        
        return X_processed, customer_info
        
    except Exception as e:
        print(f"Error preparing inference features: {str(e)}")
        return None, None


def generate_model_predictions(X_inference: np.ndarray, customer_info: pd.DataFrame,
                             model_artifact: Dict[str, Any], config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Generate predictions using the loaded model
    
    Args:
        X_inference: Preprocessed feature matrix
        customer_info: Customer identification information
        model_artifact: Loaded model artifact
        config: Configuration dictionary
        
    Returns:
        DataFrame containing predictions or None if prediction fails
    """
    
    try:
        print(f"Generating model predictions...")
        
        # Load model from artifact
        model = model_artifact['model']
        model_type = model_artifact['model_type']
        
        print(f"Using model type: {model_type}")
        print(f"Inference samples: {X_inference.shape[0]}")
        
        # Generate probability predictions
        y_pred_proba = model.predict_proba(X_inference)[:, 1]
        
        # Generate binary predictions (using 0.5 threshold)
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        
        # Create predictions DataFrame
        predictions_df = customer_info.copy()
        predictions_df['model_version'] = config['model_version']
        predictions_df['model_prediction_proba'] = y_pred_proba
        predictions_df['model_prediction_binary'] = y_pred_binary
        predictions_df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"Predictions generated successfully")
        print(f"Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
        print(f"Binary predictions distribution: {np.bincount(y_pred_binary)}")
        
        return predictions_df
        
    except Exception as e:
        print(f"Error generating model predictions: {str(e)}")
        return None


def save_predictions_to_datamart(predictions_df: pd.DataFrame, config: Dict[str, Any], 
                                spark) -> Optional[str]:
    """
    Save predictions to datamart following medallion architecture
    
    Args:
        predictions_df: DataFrame containing predictions
        config: Configuration dictionary
        spark: SparkSession object
        
    Returns:
        Output file path or None if saving fails
    """
    
    try:
        # Create output directory structure
        model_version = config['model_version']
        snapshot_date_str = config['snapshot_date_str']
        predictions_output_path = config['predictions_output_path']
        
        output_directory = os.path.join(predictions_output_path, model_version)
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created output directory: {output_directory}")
        
        # Create partition filename
        partition_name = f"{model_version}_predictions_{snapshot_date_str.replace('-', '_')}.parquet"
        output_filepath = os.path.join(output_directory, partition_name)
        
        # Convert to Spark DataFrame and save
        predictions_sdf = spark.createDataFrame(predictions_df)
        predictions_sdf.write.mode("overwrite").parquet(output_filepath)
        
        print(f"Predictions saved to: {output_filepath}")
        print(f"Records saved: {len(predictions_df)}")
        
        return output_filepath
        
    except Exception as e:
        print(f"Error saving predictions to datamart: {str(e)}")
        return None


def validate_saved_predictions(config: Dict[str, Any], spark) -> Dict[str, Any]:
    """
    Validate that predictions were saved correctly
    
    Args:
        config: Configuration dictionary
        spark: SparkSession object
        
    Returns:
        Dictionary containing validation results
    """
    
    try:
        model_version = config['model_version']
        snapshot_date_str = config['snapshot_date_str']
        predictions_output_path = config['predictions_output_path']
        
        # Check if prediction files exist
        output_directory = os.path.join(predictions_output_path, model_version)
        partition_name = f"{model_version}_predictions_{snapshot_date_str.replace('-', '_')}.parquet"
        expected_filepath = os.path.join(output_directory, partition_name)
        
        if not os.path.exists(expected_filepath):
            return {
                'success': False,
                'error': f"Prediction file not found: {expected_filepath}"
            }
        
        # Load and validate saved data
        saved_df = spark.read.parquet(expected_filepath)
        record_count = saved_df.count()
        
        # Count prediction files for this model
        prediction_files = glob.glob(os.path.join(output_directory, "*.parquet"))
        file_count = len(prediction_files)
        
        return {
            'success': True,
            'record_count': record_count,
            'file_count': file_count,
            'output_path': expected_filepath
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def run_single_inference(snapshot_date_str: str, model_version: str, spark,
                        model_bank_directory: str = "models/",
                        feature_store_path: str = "datamart/gold/feature_store/",
                        predictions_output_path: str = "datamart/gold/model_predictions/") -> Dict[str, Any]:
    """
    Run inference for a single date (Airflow-compatible function)
    
    Args:
        snapshot_date_str: Date for inference in YYYY-MM-DD format
        model_version: Model version to use
        spark: SparkSession object
        model_bank_directory: Path to model bank
        feature_store_path: Path to feature store
        predictions_output_path: Path to save predictions
        
    Returns:
        Dictionary containing inference results
    """
    
    try:
        print(f"Running inference for {snapshot_date_str} using {model_version}")
        
        # Build configuration
        config = build_inference_config(
            snapshot_date_str=snapshot_date_str,
            model_version=model_version,
            model_bank_directory=model_bank_directory,
            feature_store_path=feature_store_path,
            predictions_output_path=predictions_output_path
        )
        
        # Load model artifact
        model_artifact = load_model_artifact(config)
        if not model_artifact:
            return {'success': False, 'error': 'Failed to load model artifact'}
        
        # Load features
        feature_data = load_inference_features(config, spark)
        if feature_data is None:
            return {'success': False, 'error': 'Failed to load feature data'}
        
        # Prepare features
        X_inference, customer_info = prepare_inference_features(feature_data, model_artifact, config)
        if X_inference is None:
            return {'success': False, 'error': 'Failed to prepare features'}
        
        # Generate predictions
        predictions_df = generate_model_predictions(X_inference, customer_info, model_artifact, config)
        if predictions_df is None:
            return {'success': False, 'error': 'Failed to generate predictions'}
        
        # Save predictions
        output_path = save_predictions_to_datamart(predictions_df, config, spark)
        if not output_path:
            return {'success': False, 'error': 'Failed to save predictions'}
        
        return {
            'success': True,
            'date': snapshot_date_str,
            'model_version': model_version,
            'records_processed': len(predictions_df),
            'output_path': output_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'date': snapshot_date_str,
            'error': str(e)
        }


def run_batch_inference(date_list: List[str], model_version: str, spark,
                       model_bank_directory: str = "models/",
                       feature_store_path: str = "datamart/gold/feature_store/",
                       predictions_output_path: str = "datamart/gold/model_predictions/") -> List[Dict[str, Any]]:
    """
    Run inference for multiple dates in batch
    
    Args:
        date_list: List of dates for inference in YYYY-MM-DD format
        model_version: Model version to use
        spark: SparkSession object
        model_bank_directory: Path to model bank
        feature_store_path: Path to feature store
        predictions_output_path: Path to save predictions
        
    Returns:
        List of dictionaries containing results for each date
    """
    
    results = []
    
    print(f"Starting batch inference for {len(date_list)} dates")
    print(f"Model: {model_version}")
    
    for i, snapshot_date_str in enumerate(date_list, 1):
        print(f"\nProcessing date {i}/{len(date_list)}: {snapshot_date_str}")
        
        result = run_single_inference(
            snapshot_date_str=snapshot_date_str,
            model_version=model_version,
            spark=spark,
            model_bank_directory=model_bank_directory,
            feature_store_path=feature_store_path,
            predictions_output_path=predictions_output_path
        )
        
        results.append(result)
        
        if result['success']:
            print(f"✓ Successfully processed {snapshot_date_str}")
        else:
            print(f"✗ Failed to process {snapshot_date_str}: {result['error']}")
    
    return results


def verify_predictions_datamart(model_version: str, predictions_output_path: str, 
                               spark) -> Dict[str, Any]:
    """
    Verify all predictions in datamart for a specific model
    
    Args:
        model_version: Model version to verify
        predictions_output_path: Path to predictions datamart
        spark: SparkSession object
        
    Returns:
        Dictionary containing verification results
    """
    
    try:
        model_directory = os.path.join(predictions_output_path, model_version)
        
        if not os.path.exists(model_directory):
            return {
                'success': False,
                'error': f"Model directory not found: {model_directory}"
            }
        
        # Get all prediction files
        prediction_files = glob.glob(os.path.join(model_directory, "*.parquet"))
        
        if not prediction_files:
            return {
                'success': False,
                'error': f"No prediction files found in: {model_directory}"
            }
        
        # Load all predictions
        prediction_files_full_path = [model_directory + "/" + os.path.basename(f) for f in prediction_files]
        all_predictions_sdf = spark.read.option("header", "true").parquet(*prediction_files_full_path)
        
        # Collect summary statistics
        total_records = all_predictions_sdf.count()
        
        # Get date range
        date_stats = all_predictions_sdf.select("snapshot_date").distinct().toPandas()
        min_date = date_stats['snapshot_date'].min()
        max_date = date_stats['snapshot_date'].max()
        
        # Sample data
        sample_data = all_predictions_sdf.limit(5).toPandas()
        
        return {
            'success': True,
            'total_files': len(prediction_files),
            'total_records': total_records,
            'date_range': {'min': str(min_date), 'max': str(max_date)},
            'avg_records_per_date': total_records / len(prediction_files),
            'sample_data': sample_data
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Airflow-compatible task functions
def inference_airflow_task(snapshot_date: str, model_version: str, **context) -> str:
    """
    Airflow-compatible inference task
    
    Args:
        snapshot_date: Date for inference in YYYY-MM-DD format
        model_version: Model version to use
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"Inference_{model_version}_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting inference task for {snapshot_date}")
        
        result = run_single_inference(
            snapshot_date_str=snapshot_date,
            model_version=model_version,
            spark=spark,
            model_bank_directory="/opt/airflow/models/",
            feature_store_path="/opt/airflow/datamart/gold/feature_store/",
            predictions_output_path="/opt/airflow/datamart/gold/model_predictions/"
        )
        
        if result['success']:
            message = f"Inference completed: {result['records_processed']} predictions generated"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Inference failed: {result['error']}")
            
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



def batch_inference_airflow_task(start_date: str, end_date: str, model_version: str, **context) -> str:
    """
    Airflow-compatible batch inference task
    
    Args:
        start_date: Start date for batch inference in YYYY-MM-DD format
        end_date: End date for batch inference in YYYY-MM-DD format  
        model_version: Model version to use
        context: Airflow context
        
    Returns:
        Success message with batch processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"BatchInference_{model_version}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting batch inference from {start_date} to {end_date}")
        
        # Generate date list
        from datetime import datetime, timedelta
        
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
        
        date_list = generate_monthly_dates(start_date, end_date)
        
        results = run_batch_inference(
            date_list=date_list,
            model_version=model_version,
            spark=spark,
            model_bank_directory="/opt/airflow/models/",
            feature_store_path="/opt/airflow/datamart/gold/feature_store/",
            predictions_output_path="/opt/airflow/datamart/gold/model_predictions/"
        )
        
        successful = len([r for r in results if r['success']])
        total_records = sum([r.get('records_processed', 0) for r in results if r['success']])
        
        message = f"Batch inference completed: {successful}/{len(results)} dates processed, {total_records} total predictions"
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
    print("Model Inference Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")
