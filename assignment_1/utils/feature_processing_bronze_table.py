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