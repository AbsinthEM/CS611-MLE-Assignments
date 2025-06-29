import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col, lit, current_timestamp
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, TimestampType


def process_clickstream_bronze_table(snapshot_date_str, bronze_clickstream_directory, spark):
    """
    Process clickstream data for bronze layer - preserve raw data with metadata
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format for filtering data
        bronze_clickstream_directory: Output directory for bronze clickstream data
        spark: SparkSession object
    
    Returns:
        Spark DataFrame containing processed bronze clickstream data
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to source backend - load raw clickstream data
    csv_file_path = "data/feature_clickstream.csv"
    
    # Load data and filter by snapshot date
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(f"Clickstream {snapshot_date_str} row count: {df.count()}")
    
    # Add metadata columns for data lineage
    df = df.withColumn("ingestion_timestamp", current_timestamp())
    df = df.withColumn("data_source", lit("feature_clickstream"))
    df = df.withColumn("processing_date", lit(snapshot_date_str))
    
    # Save bronze table to datamart
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print(f'Clickstream bronze saved to: {filepath}')
    
    return df


def process_attributes_bronze_table(snapshot_date_str, bronze_attributes_directory, spark):
    """
    Process customer attributes data for bronze layer - preserve raw data with metadata
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format for filtering data
        bronze_attributes_directory: Output directory for bronze attributes data
        spark: SparkSession object
    
    Returns:
        Spark DataFrame containing processed bronze attributes data
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to source backend - load raw attributes data
    csv_file_path = "data/features_attributes.csv"
    
    # Load data and filter by snapshot date
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(f"Attributes {snapshot_date_str} row count: {df.count()}")
    
    # Add metadata columns for data lineage
    df = df.withColumn("ingestion_timestamp", current_timestamp())
    df = df.withColumn("data_source", lit("features_attributes"))
    df = df.withColumn("processing_date", lit(snapshot_date_str))
    
    # Save bronze table to datamart
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print(f'Attributes bronze saved to: {filepath}')
    
    return df


def process_financials_bronze_table(snapshot_date_str, bronze_financials_directory, spark):
    """
    Process financial features data for bronze layer - preserve raw data with metadata
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format for filtering data
        bronze_financials_directory: Output directory for bronze financial data
        spark: SparkSession object
    
    Returns:
        Spark DataFrame containing processed bronze financial data
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to source backend - load raw financial data
    csv_file_path = "data/features_financials.csv"
    
    # Load data and filter by snapshot date
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(f"Financials {snapshot_date_str} row count: {df.count()}")
    
    # Add metadata columns for data lineage
    df = df.withColumn("ingestion_timestamp", current_timestamp())
    df = df.withColumn("data_source", lit("features_financials"))
    df = df.withColumn("processing_date", lit(snapshot_date_str))
    
    # Save bronze table to datamart
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print(f'Financials bronze saved to: {filepath}')
    
    return df


def build_feature_bronze_config(execution_date_str: str,
                               bronze_clickstream_path: str = "datamart/bronze/clickstream/",
                               bronze_attributes_path: str = "datamart/bronze/attributes/",
                               bronze_financials_path: str = "datamart/bronze/financials/",
                               data_source_base_path: str = "data/") -> dict:
    """
    Build configuration dictionary for feature bronze processing
    
    Args:
        execution_date_str: Date for processing in YYYY-MM-DD format
        bronze_clickstream_path: Path to save bronze clickstream data
        bronze_attributes_path: Path to save bronze attributes data
        bronze_financials_path: Path to save bronze financials data
        data_source_base_path: Base path to source data files
        
    Returns:
        Configuration dictionary containing all processing parameters
    """
    
    config = {
        "execution_date_str": execution_date_str,
        "execution_date": datetime.strptime(execution_date_str, "%Y-%m-%d"),
        "bronze_paths": {
            "clickstream": bronze_clickstream_path,
            "attributes": bronze_attributes_path,
            "financials": bronze_financials_path
        },
        "data_sources": {
            "clickstream": os.path.join(data_source_base_path, "feature_clickstream.csv"),
            "attributes": os.path.join(data_source_base_path, "features_attributes.csv"),
            "financials": os.path.join(data_source_base_path, "features_financials.csv")
        },
        "processing_timestamp": datetime.now()
    }
    
    return config


def run_single_feature_bronze_processing(snapshot_date_str: str, feature_type: str, spark,
                                        bronze_output_path: str = None,
                                        data_source_path: str = None) -> dict:
    """
    Run bronze processing for single feature type and date (Airflow-compatible function)
    
    Args:
        snapshot_date_str: Date for processing in YYYY-MM-DD format
        feature_type: Type of feature ('clickstream', 'attributes', 'financials')
        spark: SparkSession object
        bronze_output_path: Path to save bronze data
        data_source_path: Path to source data
        
    Returns:
        Dictionary containing processing results
    """
    
    try:
        print(f"Processing {feature_type} bronze data for {snapshot_date_str}")
        
        # Set default paths if not provided
        if bronze_output_path is None:
            bronze_output_path = f"datamart/bronze/{feature_type}/"
        if data_source_path is None:
            data_source_path = f"data/feature_{feature_type}.csv"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(bronze_output_path):
            os.makedirs(bronze_output_path)
            print(f"Created directory: {bronze_output_path}")
        
        # Process bronze table using existing functions
        if feature_type == 'clickstream':
            processed_df = process_clickstream_bronze_table(snapshot_date_str, bronze_output_path, spark)
        elif feature_type == 'attributes':
            processed_df = process_attributes_bronze_table(snapshot_date_str, bronze_output_path, spark)
        elif feature_type == 'financials':
            processed_df = process_financials_bronze_table(snapshot_date_str, bronze_output_path, spark)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Validate processing results
        record_count = processed_df.count()
        
        return {
            'success': True,
            'date': snapshot_date_str,
            'feature_type': feature_type,
            'records_processed': record_count,
            'output_path': bronze_output_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'date': snapshot_date_str,
            'feature_type': feature_type,
            'error': str(e)
        }


def run_batch_feature_bronze_processing(date_list: list, feature_type: str, spark,
                                      bronze_output_path: str = None,
                                      data_source_path: str = None) -> list:
    """
    Run bronze processing for single feature type across multiple dates
    
    Args:
        date_list: List of dates for processing in YYYY-MM-DD format
        feature_type: Type of feature ('clickstream', 'attributes', 'financials')
        spark: SparkSession object
        bronze_output_path: Path to save bronze data
        data_source_path: Path to source data
        
    Returns:
        List of dictionaries containing results for each date
    """
    
    results = []
    
    print(f"Starting batch {feature_type} bronze processing for {len(date_list)} dates")
    
    for i, snapshot_date_str in enumerate(date_list, 1):
        print(f"\nProcessing {feature_type} date {i}/{len(date_list)}: {snapshot_date_str}")
        
        result = run_single_feature_bronze_processing(
            snapshot_date_str=snapshot_date_str,
            feature_type=feature_type,
            spark=spark,
            bronze_output_path=bronze_output_path,
            data_source_path=data_source_path
        )
        
        results.append(result)
        
        if result['success']:
            print(f"✓ Successfully processed {feature_type} {snapshot_date_str}")
        else:
            print(f"✗ Failed to process {feature_type} {snapshot_date_str}: {result['error']}")
    
    return results


# Airflow-compatible task functions for each feature type
def clickstream_bronze_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible clickstream bronze processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"ClickstreamBronze_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting clickstream bronze processing for {snapshot_date}")
        
        result = run_single_feature_bronze_processing(
            snapshot_date_str=snapshot_date,
            feature_type='clickstream',
            spark=spark,
            bronze_output_path="/opt/airflow/datamart/bronze/clickstream/",
            data_source_path="/opt/airflow/data/feature_clickstream.csv"
        )
        
        if result['success']:
            message = f"Clickstream bronze processing completed: {result['records_processed']} records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Clickstream bronze processing failed: {result['error']}")
            
    finally:
        spark.stop()


def attributes_bronze_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible attributes bronze processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"AttributesBronze_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting attributes bronze processing for {snapshot_date}")
        
        result = run_single_feature_bronze_processing(
            snapshot_date_str=snapshot_date,
            feature_type='attributes',
            spark=spark,
            bronze_output_path="/opt/airflow/datamart/bronze/attributes/",
            data_source_path="/opt/airflow/data/features_attributes.csv"
        )
        
        if result['success']:
            message = f"Attributes bronze processing completed: {result['records_processed']} records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Attributes bronze processing failed: {result['error']}")
            
    finally:
        spark.stop()


def financials_bronze_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible financials bronze processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"FinancialsBronze_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting financials bronze processing for {snapshot_date}")
        
        result = run_single_feature_bronze_processing(
            snapshot_date_str=snapshot_date,
            feature_type='financials',
            spark=spark,
            bronze_output_path="/opt/airflow/datamart/bronze/financials/",
            data_source_path="/opt/airflow/data/features_financials.csv"
        )
        
        if result['success']:
            message = f"Financials bronze processing completed: {result['records_processed']} records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Financials bronze processing failed: {result['error']}")
            
    finally:
        spark.stop()


def all_features_bronze_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible task to process all feature types in bronze layer
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary for all feature types
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"AllFeaturesBronze_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting all features bronze processing for {snapshot_date}")
        
        feature_types = ['clickstream', 'attributes', 'financials']
        results = []
        
        for feature_type in feature_types:
            result = run_single_feature_bronze_processing(
                snapshot_date_str=snapshot_date,
                feature_type=feature_type,
                spark=spark,
                bronze_output_path=f"/opt/airflow/datamart/bronze/{feature_type}/",
                data_source_path=f"/opt/airflow/data/feature_{feature_type}.csv"
            )
            results.append(result)
        
        successful = len([r for r in results if r['success']])
        total_records = sum([r.get('records_processed', 0) for r in results if r['success']])
        
        if successful == len(feature_types):
            message = f"All features bronze processing completed: {total_records} total records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            failed_types = [r['feature_type'] for r in results if not r['success']]
            raise ValueError(f"Bronze processing failed for: {failed_types}")
            
    finally:
        spark.stop()


if __name__ == "__main__":
    # Example usage and testing
    print("Feature Bronze Processing Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")