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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    """
    Process loan daily data for bronze layer - preserve raw data with metadata
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format for filtering data
        bronze_lms_directory: Output directory for bronze loan data
        spark: SparkSession object
    
    Returns:
        Spark DataFrame containing processed bronze loan data
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


def build_label_bronze_config(execution_date_str: str,
                             bronze_output_path: str = "datamart/bronze/lms/",
                             data_source_path: str = "data/lms_loan_daily.csv") -> dict:
    """
    Build configuration dictionary for label bronze processing
    
    Args:
        execution_date_str: Date for processing in YYYY-MM-DD format
        bronze_output_path: Path to save bronze label data
        data_source_path: Path to source loan data
        
    Returns:
        Configuration dictionary containing all processing parameters
    """
    
    config = {
        "execution_date_str": execution_date_str,
        "execution_date": datetime.strptime(execution_date_str, "%Y-%m-%d"),
        "bronze_output_path": bronze_output_path,
        "data_source_path": data_source_path,
        "processing_timestamp": datetime.now()
    }
    
    return config


def run_single_label_bronze_processing(snapshot_date_str: str, spark,
                                     bronze_output_path: str = "datamart/bronze/lms/",
                                     data_source_path: str = "data/lms_loan_daily.csv") -> dict:
    """
    Run bronze processing for single date (Airflow-compatible function)
    
    Args:
        snapshot_date_str: Date for processing in YYYY-MM-DD format
        spark: SparkSession object
        bronze_output_path: Path to save bronze data
        data_source_path: Path to source data
        
    Returns:
        Dictionary containing processing results
    """
    
    try:
        print(f"Processing label bronze data for {snapshot_date_str}")
        
        # Build configuration
        config = build_label_bronze_config(
            execution_date_str=snapshot_date_str,
            bronze_output_path=bronze_output_path,
            data_source_path=data_source_path
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(bronze_output_path):
            os.makedirs(bronze_output_path)
            print(f"Created directory: {bronze_output_path}")
        
        # Process bronze table using existing function
        processed_df = process_bronze_table(snapshot_date_str, bronze_output_path, spark)
        
        # Validate processing results
        record_count = processed_df.count()
        
        return {
            'success': True,
            'date': snapshot_date_str,
            'records_processed': record_count,
            'output_path': bronze_output_path,
            'processing_timestamp': config['processing_timestamp'].isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'date': snapshot_date_str,
            'error': str(e)
        }


def run_batch_label_bronze_processing(date_list: list, spark,
                                    bronze_output_path: str = "datamart/bronze/lms/",
                                    data_source_path: str = "data/lms_loan_daily.csv") -> list:
    """
    Run bronze processing for multiple dates in batch
    
    Args:
        date_list: List of dates for processing in YYYY-MM-DD format
        spark: SparkSession object
        bronze_output_path: Path to save bronze data
        data_source_path: Path to source data
        
    Returns:
        List of dictionaries containing results for each date
    """
    
    results = []
    
    print(f"Starting batch label bronze processing for {len(date_list)} dates")
    
    for i, snapshot_date_str in enumerate(date_list, 1):
        print(f"\nProcessing date {i}/{len(date_list)}: {snapshot_date_str}")
        
        result = run_single_label_bronze_processing(
            snapshot_date_str=snapshot_date_str,
            spark=spark,
            bronze_output_path=bronze_output_path,
            data_source_path=data_source_path
        )
        
        results.append(result)
        
        if result['success']:
            print(f"✓ Successfully processed {snapshot_date_str}")
        else:
            print(f"✗ Failed to process {snapshot_date_str}: {result['error']}")
    
    return results


# Airflow-compatible task functions
def label_bronze_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible label bronze processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"LabelBronze_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting label bronze processing for {snapshot_date}")
        
        result = run_single_label_bronze_processing(
            snapshot_date_str=snapshot_date,
            spark=spark,
            bronze_output_path="/opt/airflow/datamart/bronze/lms/",
            data_source_path="/opt/airflow/data/lms_loan_daily.csv"
        )
        
        if result['success']:
            message = f"Label bronze processing completed: {result['records_processed']} records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Label bronze processing failed: {result['error']}")
            
    finally:
        spark.stop()


def batch_label_bronze_airflow_task(start_date: str, end_date: str, **context) -> str:
    """
    Airflow-compatible batch label bronze processing task
    
    Args:
        start_date: Start date for batch processing in YYYY-MM-DD format
        end_date: End date for batch processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with batch processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"BatchLabelBronze") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting batch label bronze processing from {start_date} to {end_date}")
        
        # Generate date list (monthly)
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
        
        results = run_batch_label_bronze_processing(
            date_list=date_list,
            spark=spark,
            bronze_output_path="/opt/airflow/datamart/bronze/lms/",
            data_source_path="/opt/airflow/data/lms_loan_daily.csv"
        )
        
        successful = len([r for r in results if r['success']])
        total_records = sum([r.get('records_processed', 0) for r in results if r['success']])
        
        message = f"Batch label bronze processing completed: {successful}/{len(results)} dates processed, {total_records} total records"
        print(f"[AIRFLOW] {message}")
        return message
        
    finally:
        spark.stop()


if __name__ == "__main__":
    # Example usage and testing
    print("Label Bronze Processing Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")