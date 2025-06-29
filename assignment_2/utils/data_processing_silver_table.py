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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    """
    Process loan daily data for silver layer - clean and standardize loan metrics
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format
        bronze_lms_directory: Input directory for bronze loan data
        silver_loan_daily_directory: Output directory for silver loan data
        spark: SparkSession object
    
    Returns:
        Spark DataFrame containing processed silver loan data
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def build_label_silver_config(execution_date_str: str,
                             bronze_input_path: str = "datamart/bronze/lms/",
                             silver_output_path: str = "datamart/silver/loan_daily/") -> dict:
    """
    Build configuration dictionary for label silver processing
    
    Args:
        execution_date_str: Date for processing in YYYY-MM-DD format
        bronze_input_path: Path to bronze loan data
        silver_output_path: Path to save silver loan data
        
    Returns:
        Configuration dictionary containing all processing parameters
    """
    
    config = {
        "execution_date_str": execution_date_str,
        "execution_date": datetime.strptime(execution_date_str, "%Y-%m-%d"),
        "bronze_input_path": bronze_input_path,
        "silver_output_path": silver_output_path,
        "processing_timestamp": datetime.now()
    }
    
    return config


def run_single_label_silver_processing(snapshot_date_str: str, spark,
                                      bronze_input_path: str = "datamart/bronze/lms/",
                                      silver_output_path: str = "datamart/silver/loan_daily/") -> dict:
    """
    Run silver processing for single date (Airflow-compatible function)
    
    Args:
        snapshot_date_str: Date for processing in YYYY-MM-DD format
        spark: SparkSession object
        bronze_input_path: Path to bronze loan data
        silver_output_path: Path to save silver loan data
        
    Returns:
        Dictionary containing processing results
    """
    
    try:
        print(f"Processing label silver data for {snapshot_date_str}")
        
        # Build configuration
        config = build_label_silver_config(
            execution_date_str=snapshot_date_str,
            bronze_input_path=bronze_input_path,
            silver_output_path=silver_output_path
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(silver_output_path):
            os.makedirs(silver_output_path)
            print(f"Created directory: {silver_output_path}")
        
        # Process silver table using existing function
        processed_df = process_silver_table(snapshot_date_str, bronze_input_path, silver_output_path, spark)
        
        # Validate processing results
        record_count = processed_df.count()
        
        return {
            'success': True,
            'date': snapshot_date_str,
            'records_processed': record_count,
            'output_path': silver_output_path,
            'processing_timestamp': config['processing_timestamp'].isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'date': snapshot_date_str,
            'error': str(e)
        }


# Airflow-compatible task functions
def label_silver_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible label silver processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"LabelSilver_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting label silver processing for {snapshot_date}")
        
        result = run_single_label_silver_processing(
            snapshot_date_str=snapshot_date,
            spark=spark,
            bronze_input_path="/opt/airflow/datamart/bronze/lms/",
            silver_output_path="/opt/airflow/datamart/silver/loan_daily/"
        )
        
        if result['success']:
            message = f"Label silver processing completed: {result['records_processed']} records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Label silver processing failed: {result['error']}")
            
    finally:
        spark.stop()


if __name__ == "__main__":
    # Example usage and testing
    print("Label Silver Processing Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")