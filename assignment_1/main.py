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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# Import label processing modules
import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# Import feature processing modules
import utils.feature_processing_bronze_table
import utils.feature_processing_silver_table
import utils.feature_processing_gold_table


def initialize_spark():
    """Initialize Spark session with appropriate configuration"""
    spark = pyspark.sql.SparkSession.builder \
        .appName("ML_Engineering_Pipeline") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def generate_first_of_month_dates(start_date_str, end_date_str):
    """Generate list of first-of-month dates between start and end dates"""
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def create_directory_structure():
    """Create all required directories for the medallion architecture"""
    directories = [
        # Label store directories
        "datamart/bronze/lms/",
        "datamart/silver/loan_daily/", 
        "datamart/gold/label_store/",
        
        # Feature store directories  
        "datamart/bronze/clickstream/",
        "datamart/bronze/attributes/",
        "datamart/bronze/financials/",
        "datamart/silver/clickstream/",
        "datamart/silver/attributes/",
        "datamart/silver/financials/",
        "datamart/gold/feature_store/"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def run_label_pipeline(dates_str_lst, spark, dpd=30, mob=6):
    """Execute the complete label engineering pipeline"""
    print("\n" + "="*60)
    print("LABEL ENGINEERING PIPELINE")
    print("="*60)
    print(f"Label definition: {dpd}dpd_{mob}mob")
    
    # Directory paths
    bronze_lms_directory = "datamart/bronze/lms/"
    silver_loan_daily_directory = "datamart/silver/loan_daily/"
    gold_label_store_directory = "datamart/gold/label_store/"
    
    # Bronze layer processing for labels
    print(f"\n[LABELS] Processing Bronze Layer...")
    for date_str in dates_str_lst:
        utils.data_processing_bronze_table.process_bronze_table(
            date_str, bronze_lms_directory, spark)
    
    # Silver layer processing for labels  
    print(f"\n[LABELS] Processing Silver Layer...")
    for date_str in dates_str_lst:
        utils.data_processing_silver_table.process_silver_table(
            date_str, bronze_lms_directory, silver_loan_daily_directory, spark)
    
    # Gold layer processing for labels
    print(f"\n[LABELS] Processing Gold Layer...")
    for date_str in dates_str_lst:
        utils.data_processing_gold_table.process_labels_gold_table(
            date_str, silver_loan_daily_directory, gold_label_store_directory, 
            spark, dpd=dpd, mob=mob)
    
    print(f"[LABELS] Pipeline completed successfully!")
    return gold_label_store_directory


def run_feature_pipeline(dates_str_lst, spark, lookback_months=6):
    """Execute the complete feature engineering pipeline"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    print(f"Lookback period: {lookback_months} months")
    
    # Directory paths
    bronze_clickstream_directory = "datamart/bronze/clickstream/"
    bronze_attributes_directory = "datamart/bronze/attributes/"
    bronze_financials_directory = "datamart/bronze/financials/"
    silver_clickstream_directory = "datamart/silver/clickstream/"
    silver_attributes_directory = "datamart/silver/attributes/"
    silver_financials_directory = "datamart/silver/financials/"
    gold_feature_store_directory = "datamart/gold/feature_store/"
    
    # Bronze layer processing for features
    print(f"\n[FEATURES] Processing Bronze Layer...")
    
    print("Processing Bronze Clickstream Features...")
    for date_str in dates_str_lst:
        utils.feature_processing_bronze_table.process_clickstream_bronze_table(
            date_str, bronze_clickstream_directory, spark)
    
    print("Processing Bronze Attributes Features...")
    for date_str in dates_str_lst:
        utils.feature_processing_bronze_table.process_attributes_bronze_table(
            date_str, bronze_attributes_directory, spark)
    
    print("Processing Bronze Financial Features...")
    for date_str in dates_str_lst:
        utils.feature_processing_bronze_table.process_financials_bronze_table(
            date_str, bronze_financials_directory, spark)
    
    # Silver layer processing for features
    print(f"\n[FEATURES] Processing Silver Layer...")
    
    print("Processing Silver Clickstream Features...")
    for date_str in dates_str_lst:
        utils.feature_processing_silver_table.process_clickstream_silver_table(
            date_str, bronze_clickstream_directory, silver_clickstream_directory, spark)
    
    print("Processing Silver Attributes Features...")
    for date_str in dates_str_lst:
        utils.feature_processing_silver_table.process_attributes_silver_table(
            date_str, bronze_attributes_directory, silver_attributes_directory, spark)
    
    print("Processing Silver Financial Features...")
    for date_str in dates_str_lst:
        utils.feature_processing_silver_table.process_financials_silver_table(
            date_str, bronze_financials_directory, silver_financials_directory, spark)
    
    # Gold layer processing for features
    print(f"\n[FEATURES] Processing Gold Layer...")
    start_date_str = dates_str_lst[0]
    
    for date_str in dates_str_lst:
        # Only process dates that have sufficient lookback history
        current_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        
        # Check if we have enough history for this date
        months_since_start = (current_date.year - start_date.year) * 12 + current_date.month - start_date.month
        
        if months_since_start >= lookback_months:
            utils.feature_processing_gold_table.process_features_gold_table(
                date_str, 
                silver_clickstream_directory, 
                silver_attributes_directory, 
                silver_financials_directory, 
                gold_feature_store_directory, 
                spark, 
                lookback_months=lookback_months
            )
        else:
            print(f"Skipping {date_str} - insufficient lookback history (need {lookback_months} months)")
    
    print(f"[FEATURES] Pipeline completed successfully!")
    return gold_feature_store_directory


def inspect_data_stores(gold_label_store_directory, gold_feature_store_directory, spark):
    """Inspect and summarize the final gold layer data stores"""
    print("\n" + "="*60)
    print("DATA STORES INSPECTION")
    print("="*60)
    
    # Inspect Label Store
    print(f"\n[LABEL STORE INSPECTION]")
    label_files = [gold_label_store_directory + os.path.basename(f) 
                  for f in glob.glob(os.path.join(gold_label_store_directory, '*'))]
    
    if label_files:
        labels_df = spark.read.option("header", "true").parquet(*label_files)
        label_count = labels_df.count()
        print(f"Label Store Records: {label_count}")
        
        # Show label distribution
        label_dist = labels_df.groupBy("label").count().orderBy("label")
        print("Label Distribution:")
        label_dist.show()
        
        # Show time coverage
        time_coverage = labels_df.groupBy("snapshot_date").count().orderBy("snapshot_date")
        print("Label Time Coverage:")
        time_coverage.show()
        
        print("Label Store Sample:")
        labels_df.show(5)
    else:
        print("No label store files found!")
    
    # Inspect Feature Store
    print(f"\n[FEATURE STORE INSPECTION]")
    feature_files = [gold_feature_store_directory + os.path.basename(f) 
                    for f in glob.glob(os.path.join(gold_feature_store_directory, '*'))]
    
    if feature_files:
        features_df = spark.read.option("header", "true").parquet(*feature_files)
        feature_count = features_df.count()
        feature_columns = len(features_df.columns)
        
        print(f"Feature Store Records: {feature_count}")
        print(f"Feature Store Columns: {feature_columns}")
        
        # Show time distribution
        if "feature_snapshot_date" in features_df.columns:
            time_dist = features_df.groupBy("feature_snapshot_date").count().orderBy("feature_snapshot_date")
            print("Feature Time Distribution:")
            time_dist.show()
        
        # Check data coverage if columns exist
        coverage_columns = ["has_clickstream_data", "has_attributes_data", "has_financial_data", "data_completeness_score"]
        existing_coverage_cols = [col for col in coverage_columns if col in features_df.columns]
        
        if existing_coverage_cols:
            print("Data Coverage Analysis:")
            coverage_stats = features_df.select(*[F.avg(col).alias(f"avg_{col}") for col in existing_coverage_cols]).collect()[0]
            for col in existing_coverage_cols:
                print(f"{col}: {coverage_stats[f'avg_{col}']:.3f}")
        
        print("Feature Store Sample:")
        features_df.show(5, truncate=False)
        
        return label_count, feature_count
    else:
        print("No feature store files found!")
        return label_count if 'label_count' in locals() else 0, 0


def main():
    """Main execution function for the complete ML engineering pipeline"""
    print("="*80)
    print("MACHINE LEARNING ENGINEERING PIPELINE")
    print("CS611 - Assignment 1: Data Processing Pipelines")
    print("="*80)
    
    # Initialize Spark session
    spark = initialize_spark()
    
    # Pipeline configuration
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"
    lookback_months = 6  # For feature engineering
    dpd = 30  # Days past due for label definition
    mob = 6   # Month on book for label definition
    
    print(f"\nPipeline Configuration:")
    print(f"Date range: {start_date_str} to {end_date_str}")
    print(f"Feature lookback period: {lookback_months} months")
    print(f"Label definition: {dpd} days past due at {mob} months on book")
    
    # Generate processing dates
    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
    print(f"Processing {len(dates_str_lst)} monthly snapshots")
    
    # Create directory structure
    create_directory_structure()
    
    # Execute label engineering pipeline
    gold_label_store_directory = run_label_pipeline(dates_str_lst, spark, dpd=dpd, mob=mob)
    
    # Execute feature engineering pipeline  
    gold_feature_store_directory = run_feature_pipeline(dates_str_lst, spark, lookback_months=lookback_months)
    
    # Stop Spark session
    spark.stop()
    print(f"\nSpark session terminated successfully.")

if __name__ == "__main__":
    main()