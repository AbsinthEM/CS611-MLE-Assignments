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


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    """
    Process loan daily data for gold layer - create ML-ready label store
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format
        silver_loan_daily_directory: Input directory for silver loan data
        gold_label_store_directory: Output directory for gold label store
        spark: SparkSession object
        dpd: Days past due threshold for label definition
        mob: Month on book for label extraction
    
    Returns:
        Spark DataFrame containing processed gold label store
    """
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def build_label_gold_config(execution_date_str: str,
                           silver_input_path: str = "datamart/silver/loan_daily/",
                           gold_output_path: str = "datamart/gold/label_store/",
                           dpd_threshold: int = 30,
                           mob_threshold: int = 6) -> dict:
    """
    Build configuration dictionary for label gold processing
    
    Args:
        execution_date_str: Date for processing in YYYY-MM-DD format
        silver_input_path: Path to silver loan data
        gold_output_path: Path to save gold label store
        dpd_threshold: Days past due threshold for label definition
        mob_threshold: Month on book for label extraction
        
    Returns:
        Configuration dictionary containing all processing parameters
    """
    
    config = {
        "execution_date_str": execution_date_str,
        "execution_date": datetime.strptime(execution_date_str, "%Y-%m-%d"),
        "silver_input_path": silver_input_path,
        "gold_output_path": gold_output_path,
        "dpd_threshold": dpd_threshold,
        "mob_threshold": mob_threshold,
        "processing_timestamp": datetime.now()
    }
    
    return config


def run_single_label_gold_processing(snapshot_date_str: str, spark,
                                    silver_input_path: str = "datamart/silver/loan_daily/",
                                    gold_output_path: str = "datamart/gold/label_store/",
                                    dpd_threshold: int = 30,
                                    mob_threshold: int = 6) -> dict:
    """
    Run gold processing for single date (Airflow-compatible function)
    
    Args:
        snapshot_date_str: Date for processing in YYYY-MM-DD format
        spark: SparkSession object
        silver_input_path: Path to silver loan data
        gold_output_path: Path to save gold label store
        dpd_threshold: Days past due threshold for label definition
        mob_threshold: Month on book for label extraction
        
    Returns:
        Dictionary containing processing results
    """
    
    try:
        print(f"Processing label gold data for {snapshot_date_str}")
        
        # Build configuration
        config = build_label_gold_config(
            execution_date_str=snapshot_date_str,
            silver_input_path=silver_input_path,
            gold_output_path=gold_output_path,
            dpd_threshold=dpd_threshold,
            mob_threshold=mob_threshold
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(gold_output_path):
            os.makedirs(gold_output_path)
            print(f"Created directory: {gold_output_path}")
        
        # Process gold table using existing function
        processed_df = process_labels_gold_table(
            snapshot_date_str, 
            silver_input_path, 
            gold_output_path, 
            spark, 
            dpd_threshold, 
            mob_threshold
        )
        
        # Validate processing results
        record_count = processed_df.count()
        
        return {
            'success': True,
            'date': snapshot_date_str,
            'records_processed': record_count,
            'output_path': gold_output_path,
            'label_definition': f"{dpd_threshold}dpd_{mob_threshold}mob",
            'processing_timestamp': config['processing_timestamp'].isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'date': snapshot_date_str,
            'error': str(e)
        }


# Airflow-compatible task functions
def label_gold_airflow_task(snapshot_date: str, dpd_threshold: int = 30, mob_threshold: int = 6, **context) -> str:
    """
    Airflow-compatible label gold processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        dpd_threshold: Days past due threshold for label definition
        mob_threshold: Month on book for label extraction
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"LabelGold_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting label gold processing for {snapshot_date}")
        
        result = run_single_label_gold_processing(
            snapshot_date_str=snapshot_date,
            spark=spark,
            silver_input_path="/opt/airflow/datamart/silver/loan_daily/",
            gold_output_path="/opt/airflow/datamart/gold/label_store/",
            dpd_threshold=dpd_threshold,
            mob_threshold=mob_threshold
        )
        
        if result['success']:
            message = f"Label gold processing completed: {result['records_processed']} records processed ({result['label_definition']})"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Label gold processing failed: {result['error']}")
            
    finally:
        spark.stop()


if __name__ == "__main__":
    # Example usage and testing
    print("Label Gold Processing Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")