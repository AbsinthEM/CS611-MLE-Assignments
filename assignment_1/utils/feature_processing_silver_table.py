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
    df = df.withColumn("has_null_features", 
                       sum([when(col(c).isNull(), 1).otherwise(0) for c in feature_cols]) > 0)
    
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