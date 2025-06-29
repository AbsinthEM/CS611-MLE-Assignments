"""
Data Validation Utilities Module
Simple data availability validation for ML pipelines

This module provides:
- Basic feature store and label store availability checks
- Simple data quality validation
- Airflow-compatible validation tasks
"""

import os
import glob
from datetime import datetime
from typing import Dict, Any
import pyspark
from pyspark.sql import SparkSession


def validate_data_stores(snapshot_date_str: str, spark,
                        feature_store_path: str = "datamart/gold/feature_store/",
                        label_store_path: str = "datamart/gold/label_store/") -> Dict[str, Any]:
    """
    Simple validation of feature store and label store availability
    
    Args:
        snapshot_date_str: Date for validation in YYYY-MM-DD format
        spark: SparkSession object
        feature_store_path: Path to feature store data
        label_store_path: Path to label store data
        
    Returns:
        Dictionary containing validation results
    """
    
    print(f"Validating data stores for {snapshot_date_str}")
    
    validation_results = {
        'validation_date': snapshot_date_str,
        'feature_store_available': False,
        'label_store_available': False,
        'feature_record_count': 0,
        'label_record_count': 0,
        'validation_passed': False,
        'validation_message': ''
    }
    
    try:
        # Check feature store
        feature_files = glob.glob(os.path.join(feature_store_path, "*.parquet"))
        if feature_files:
            feature_files_full_path = [feature_store_path + os.path.basename(f) for f in feature_files]
            features_sdf = spark.read.option("header", "true").parquet(*feature_files_full_path)
            feature_count = features_sdf.count()
            
            validation_results['feature_store_available'] = True
            validation_results['feature_record_count'] = feature_count
            print(f"Feature store: {feature_count} records found")
        else:
            print(f"Feature store: No files found in {feature_store_path}")
        
        # Check label store
        label_files = glob.glob(os.path.join(label_store_path, "*.parquet"))
        if label_files:
            label_files_full_path = [label_store_path + os.path.basename(f) for f in label_files]
            labels_sdf = spark.read.option("header", "true").parquet(*label_files_full_path)
            label_count = labels_sdf.count()
            
            validation_results['label_store_available'] = True
            validation_results['label_record_count'] = label_count
            print(f"Label store: {label_count} records found")
        else:
            print(f"Label store: No files found in {label_store_path}")
        
        # Determine overall validation status
        if validation_results['feature_store_available'] and validation_results['label_store_available']:
            validation_results['validation_passed'] = True
            validation_results['validation_message'] = 'Both feature and label stores are available'
        elif validation_results['feature_store_available']:
            validation_results['validation_message'] = 'Only feature store is available'
        elif validation_results['label_store_available']:
            validation_results['validation_message'] = 'Only label store is available'
        else:
            validation_results['validation_message'] = 'Neither feature nor label store is available'
        
        print(f"Data validation completed: {validation_results['validation_message']}")
        return validation_results
        
    except Exception as e:
        print(f"Error during data validation: {str(e)}")
        validation_results['validation_message'] = f'Validation error: {str(e)}'
        return validation_results


def validate_feature_and_label_stores_airflow_task(**context) -> Dict[str, Any]:
    """
    Airflow-compatible data validation task
    
    Args:
        context: Airflow context
        
    Returns:
        Simple validation results
    """
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"DataValidation_{context['ds']}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW DATA VALIDATION] Starting validation for {context['ds']}")
        
        # Run simple validation
        validation_results = validate_data_stores(
            snapshot_date_str=context['ds'],
            spark=spark,
            feature_store_path="/opt/airflow/datamart/gold/feature_store/",
            label_store_path="/opt/airflow/datamart/gold/label_store/"
        )
        
        print(f"[AIRFLOW DATA VALIDATION] Validation completed: {validation_results['validation_passed']}")
        
        return validation_results
        
    finally:
        spark.stop()


if __name__ == "__main__":
    print("Data Validation Utilities Module")
    print("Simple data store availability validation")