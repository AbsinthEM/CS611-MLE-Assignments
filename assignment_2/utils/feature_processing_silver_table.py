import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import (
    col, lit, when, regexp_replace, split, regexp_extract,
    sum as spark_sum
)
from functools import reduce

from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_clickstream_silver_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Process clickstream data for silver layer - clean and standardize features
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format
        bronze_clickstream_directory: Input directory for bronze clickstream data
        silver_clickstream_directory: Output directory for silver clickstream data
        spark: SparkSession object
    
    Returns:
        Spark DataFrame containing processed silver clickstream data
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Clickstream silver loaded from: {filepath}, row count: {df.count()}')
    
    # Clean and standardize clickstream features
    # Enforce data types for all clickstream features
    feature_cols = [col_name for col_name in df.columns if col_name.startswith('fe_')]
    
    for feature_col in feature_cols:
        # Cast to double for consistency and handle any string values
        df = df.withColumn(feature_col, col(feature_col).cast(FloatType()))
    
    # Cast Customer_ID to string and ensure proper format
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    
    # Cast snapshot_date to proper date format
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    
    # Add data quality flags
    # Check for null values in key features
    null_checks = [when(col(c).isNull(), 1).otherwise(0) for c in feature_cols]
    if null_checks:
        sum_expr = reduce(lambda x, y: x + y, null_checks)
        df = df.withColumn("has_null_features", sum_expr > 0)
    else:
        df = df.withColumn("has_null_features", lit(False))
    
    # Add feature count for validation
    df = df.withColumn("feature_count", lit(len(feature_cols)))
    
    # Save silver table to datamart
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f'Clickstream silver saved to: {filepath}')
    
    return df


def process_attributes_silver_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Process customer attributes data for silver layer - clean and validate demographic data
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format
        bronze_attributes_directory: Input directory for bronze attributes data
        silver_attributes_directory: Output directory for silver attributes data
        spark: SparkSession object
    
    Returns:
        Spark DataFrame containing processed silver attributes data
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Attributes silver loaded from: {filepath}, row count: {df.count()}')
    
    # Clean Age field - remove underscores and validate range
    df = df.withColumn("Age_clean", regexp_replace(col("Age").cast(StringType()), "_", ""))
    df = df.withColumn("Age_clean", col("Age_clean").cast(IntegerType()))
    df = df.withColumn("Age_valid", 
                       when((col("Age_clean") >= 18) & (col("Age_clean") <= 90), True).otherwise(False))
    
    # Clean and standardize Occupation field
    df = df.withColumn("Occupation_clean", 
                       when(col("Occupation").rlike("^_+$"), "Unknown")
                       .otherwise(col("Occupation")))
    
    # Validate SSN format - flag potentially corrupted entries
    df = df.withColumn("SSN_valid", 
                       when(col("SSN").rlike("^[0-9\\-]+$"), True).otherwise(False))
    
    # Standardize Customer_ID format
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    
    # Cast snapshot_date to proper date format
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    
    # Add overall data quality score
    df = df.withColumn("data_quality_score", 
                       (col("Age_valid").cast(IntegerType()) + 
                        col("SSN_valid").cast(IntegerType()) + 
                        when(col("Occupation_clean") != "Unknown", 1).otherwise(0)) / 3.0)
    
    # Save silver table to datamart
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f'Attributes silver saved to: {filepath}')
    
    return df


def process_financials_silver_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Process financial features data for silver layer - clean and standardize financial metrics
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format
        bronze_financials_directory: Input directory for bronze financial data
        silver_financials_directory: Output directory for silver financial data
        spark: SparkSession object
    
    Returns:
        Spark DataFrame containing processed silver financial data
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Financials silver loaded from: {filepath}, row count: {df.count()}')
    
    # Clean numeric fields with underscores
    numeric_fields_to_clean = ['Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Outstanding_Debt']
    
    for field in numeric_fields_to_clean:
        if field in df.columns:
            # Remove underscores and convert to numeric
            df = df.withColumn(f"{field}_clean", 
                               regexp_replace(col(field).cast(StringType()), "_", "").cast(FloatType()))
    
    # Clean and standardize Credit_Mix
    df = df.withColumn("Credit_Mix_clean", 
                       when(col("Credit_Mix") == "_", "Unknown")
                       .otherwise(col("Credit_Mix")))
    
    # Clean Payment_of_Min_Amount - standardize values
    df = df.withColumn("Payment_of_Min_Amount_clean",
                       when(col("Payment_of_Min_Amount") == "NM", "No")
                       .otherwise(col("Payment_of_Min_Amount")))
    
    # Parse Credit_History_Age and convert to months
    # Extract years and months from "X Years and Y Months" format
    df = df.withColumn("credit_history_years", 
                       regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Years?", 1).cast(IntegerType()))
    df = df.withColumn("credit_history_months_part", 
                       regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Months?", 1).cast(IntegerType()))
    
    # Calculate total months
    df = df.withColumn("Credit_History_Months", 
                       col("credit_history_years") * 12 + 
                       when(col("credit_history_months_part").isNull(), 0).otherwise(col("credit_history_months_part")))
    
    # Drop intermediate columns
    df = df.drop("credit_history_years", "credit_history_months_part")
    
    # Validate and clean other numeric fields
    df = df.withColumn("Credit_Utilization_Ratio", col("Credit_Utilization_Ratio").cast(FloatType()))
    df = df.withColumn("Delay_from_due_date", col("Delay_from_due_date").cast(IntegerType()))
    df = df.withColumn("Total_EMI_per_month", col("Total_EMI_per_month").cast(FloatType()))
    df = df.withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType()))
    df = df.withColumn("Amount_invested_monthly", col("Amount_invested_monthly").cast(FloatType()))
    df = df.withColumn("Monthly_Balance", col("Monthly_Balance").cast(FloatType()))
    
    # Standardize Customer_ID format
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    
    # Cast snapshot_date to proper date format
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    
    # Add financial health indicators
    # Debt-to-Income ratio
    df = df.withColumn("debt_to_income_ratio", 
                       when(col("Annual_Income_clean") > 0, 
                            col("Outstanding_Debt_clean") / col("Annual_Income_clean"))
                       .otherwise(None))
    
    # EMI-to-Income ratio
    df = df.withColumn("emi_to_income_ratio", 
                       when(col("Monthly_Inhand_Salary") > 0, 
                            col("Total_EMI_per_month") / col("Monthly_Inhand_Salary"))
                       .otherwise(None))
    
    # Save silver table to datamart
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print(f'Financials silver saved to: {filepath}')
    
    return df


def run_single_feature_silver_processing(snapshot_date_str: str, feature_type: str, spark,
                                        bronze_input_path: str = None,
                                        silver_output_path: str = None) -> dict:
    """
    Run silver processing for single feature type and date (Airflow-compatible function)
    
    Args:
        snapshot_date_str: Date for processing in YYYY-MM-DD format
        feature_type: Type of feature ('clickstream', 'attributes', 'financials')
        spark: SparkSession object
        bronze_input_path: Path to bronze feature data
        silver_output_path: Path to save silver feature data
        
    Returns:
        Dictionary containing processing results
    """
    
    try:
        print(f"Processing {feature_type} silver data for {snapshot_date_str}")
        
        # Set default paths if not provided
        if bronze_input_path is None:
            bronze_input_path = f"datamart/bronze/{feature_type}/"
        if silver_output_path is None:
            silver_output_path = f"datamart/silver/{feature_type}/"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(silver_output_path):
            os.makedirs(silver_output_path)
            print(f"Created directory: {silver_output_path}")
        
        # Process silver table using existing functions
        if feature_type == 'clickstream':
            processed_df = process_clickstream_silver_table(snapshot_date_str, bronze_input_path, silver_output_path, spark)
        elif feature_type == 'attributes':
            processed_df = process_attributes_silver_table(snapshot_date_str, bronze_input_path, silver_output_path, spark)
        elif feature_type == 'financials':
            processed_df = process_financials_silver_table(snapshot_date_str, bronze_input_path, silver_output_path, spark)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Validate processing results
        record_count = processed_df.count()
        
        return {
            'success': True,
            'date': snapshot_date_str,
            'feature_type': feature_type,
            'records_processed': record_count,
            'output_path': silver_output_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'date': snapshot_date_str,
            'feature_type': feature_type,
            'error': str(e)
        }


# Airflow-compatible task functions for each feature type
def clickstream_silver_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible clickstream silver processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"ClickstreamSilver_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting clickstream silver processing for {snapshot_date}")
        
        result = run_single_feature_silver_processing(
            snapshot_date_str=snapshot_date,
            feature_type='clickstream',
            spark=spark,
            bronze_input_path="/opt/airflow/datamart/bronze/clickstream/",
            silver_output_path="/opt/airflow/datamart/silver/clickstream/"
        )
        
        if result['success']:
            message = f"Clickstream silver processing completed: {result['records_processed']} records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Clickstream silver processing failed: {result['error']}")
            
    finally:
        spark.stop()


def attributes_silver_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible attributes silver processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"AttributesSilver_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting attributes silver processing for {snapshot_date}")
        
        result = run_single_feature_silver_processing(
            snapshot_date_str=snapshot_date,
            feature_type='attributes',
            spark=spark,
            bronze_input_path="/opt/airflow/datamart/bronze/attributes/",
            silver_output_path="/opt/airflow/datamart/silver/attributes/"
        )
        
        if result['success']:
            message = f"Attributes silver processing completed: {result['records_processed']} records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Attributes silver processing failed: {result['error']}")
            
    finally:
        spark.stop()


def financials_silver_airflow_task(snapshot_date: str, **context) -> str:
    """
    Airflow-compatible financials silver processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"FinancialsSilver_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting financials silver processing for {snapshot_date}")
        
        result = run_single_feature_silver_processing(
            snapshot_date_str=snapshot_date,
            feature_type='financials',
            spark=spark,
            bronze_input_path="/opt/airflow/datamart/bronze/financials/",
            silver_output_path="/opt/airflow/datamart/silver/financials/"
        )
        
        if result['success']:
            message = f"Financials silver processing completed: {result['records_processed']} records processed"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Financials silver processing failed: {result['error']}")
            
    finally:
        spark.stop()


if __name__ == "__main__":
    # Example usage and testing
    print("Feature Silver Processing Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")