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

from pyspark.sql.functions import col, lit, when, avg, stddev, min as spark_min, max as spark_max, count, sum as spark_sum
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pyspark.sql.window import Window


def process_features_gold_table(snapshot_date_str, silver_clickstream_directory, silver_attributes_directory, 
                               silver_financials_directory, gold_feature_store_directory, spark, lookback_months=6):
    """
    Process and combine all feature sources for gold layer - create ML-ready feature store
    
    Args:
        snapshot_date_str: Date string in YYYY-MM-DD format for point-in-time features
        silver_clickstream_directory: Input directory for silver clickstream data
        silver_attributes_directory: Input directory for silver attributes data
        silver_financials_directory: Input directory for silver financials data
        gold_feature_store_directory: Output directory for gold feature store
        spark: SparkSession object
        lookback_months: Number of months to look back for historical features
    
    Returns:
        Spark DataFrame containing processed gold feature store
    """
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    lookback_date = snapshot_date - relativedelta(months=lookback_months)
    
    print(f"Processing features for {snapshot_date_str} with lookback to {lookback_date.strftime('%Y-%m-%d')}")
    
    # Get all unique customers as the base for our feature store
    all_customers = get_all_unique_customers(snapshot_date_str, silver_clickstream_directory, 
                                           silver_attributes_directory, silver_financials_directory, 
                                           spark, lookback_date)
    
    # Load and process clickstream features
    clickstream_features = load_and_aggregate_clickstream_features(
        snapshot_date_str, silver_clickstream_directory, spark, lookback_date)
    
    # Load and process attributes features using "most recent available" strategy
    attributes_features = load_and_process_attributes_features_latest(
        snapshot_date_str, silver_attributes_directory, spark)
    
    # Load and process financial features using "most recent available" strategy
    financial_features = load_and_aggregate_financial_features_latest(
        snapshot_date_str, silver_financials_directory, spark, lookback_date)
    
    # Combine all features using customer-centric approach
    combined_features = combine_all_features_customer_centric(all_customers, clickstream_features, 
                                                            attributes_features, financial_features, snapshot_date_str)
    
    # Create cross-source interaction features
    enhanced_features = create_interaction_features(combined_features)
    
    # Add data availability flags
    final_features = add_data_availability_flags(enhanced_features)
    
    # Save gold feature store
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    final_features.write.mode("overwrite").parquet(filepath)
    print(f'Gold feature store saved to: {filepath}')
    
    return final_features


def get_all_unique_customers(snapshot_date_str, silver_clickstream_directory, silver_attributes_directory, 
                           silver_financials_directory, spark, lookback_date):
    """
    Get all unique customers who appear in any of the data sources up to the snapshot date
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Collect all customer IDs from all sources
    all_customer_ids = set()
    
    # Get customers from clickstream data (within lookback period)
    try:
        current_date = lookback_date
        while current_date <= snapshot_date:
            file_date_str = current_date.strftime("%Y-%m-%d")
            partition_name = "silver_clickstream_" + file_date_str.replace('-','_') + '.parquet'
            filepath = silver_clickstream_directory + partition_name
            
            if os.path.exists(filepath):
                df = spark.read.parquet(filepath)
                customer_ids = [row['Customer_ID'] for row in df.select('Customer_ID').distinct().collect()]
                all_customer_ids.update(customer_ids)
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
    except Exception as e:
        print(f"Error collecting clickstream customers: {e}")
    
    # Get customers from attributes data (all available up to snapshot date)
    try:
        attr_files = glob.glob(silver_attributes_directory + "*.parquet")
        for filepath in attr_files:
            # Extract date from filename to check if it's before snapshot_date
            filename = os.path.basename(filepath)
            date_part = filename.replace('silver_attributes_', '').replace('.parquet', '').replace('_', '-')
            try:
                file_date = datetime.strptime(date_part, "%Y-%m-%d")
                if file_date <= snapshot_date:
                    df = spark.read.parquet(filepath)
                    customer_ids = [row['Customer_ID'] for row in df.select('Customer_ID').distinct().collect()]
                    all_customer_ids.update(customer_ids)
            except ValueError:
                continue
    except Exception as e:
        print(f"Error collecting attributes customers: {e}")
    
    # Get customers from financial data (all available up to snapshot date)
    try:
        fin_files = glob.glob(silver_financials_directory + "*.parquet")
        for filepath in fin_files:
            # Extract date from filename to check if it's before snapshot_date
            filename = os.path.basename(filepath)
            date_part = filename.replace('silver_financials_', '').replace('.parquet', '').replace('_', '-')
            try:
                file_date = datetime.strptime(date_part, "%Y-%m-%d")
                if file_date <= snapshot_date:
                    df = spark.read.parquet(filepath)
                    customer_ids = [row['Customer_ID'] for row in df.select('Customer_ID').distinct().collect()]
                    all_customer_ids.update(customer_ids)
            except ValueError:
                continue
    except Exception as e:
        print(f"Error collecting financial customers: {e}")
    
    # Convert to DataFrame
    if all_customer_ids:
        customer_list = [(cust_id,) for cust_id in all_customer_ids]
        customers_df = spark.createDataFrame(customer_list, ["Customer_ID"])
        print(f"Found {len(all_customer_ids)} unique customers across all data sources")
        return customers_df
    else:
        # Create empty customer DataFrame
        return spark.createDataFrame([], ["Customer_ID"])


def load_and_aggregate_clickstream_features(snapshot_date_str, silver_clickstream_directory, spark, lookback_date):
    """
    Load clickstream data and create time-aware aggregated features (unchanged from original)
    """
    try:
        # Get all clickstream files from lookback period to current date
        clickstream_files = []
        current_date = lookback_date
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        while current_date <= snapshot_date:
            file_date_str = current_date.strftime("%Y-%m-%d")
            partition_name = "silver_clickstream_" + file_date_str.replace('-','_') + '.parquet'
            filepath = silver_clickstream_directory + partition_name
            
            if os.path.exists(filepath):
                clickstream_files.append(filepath)
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        if not clickstream_files:
            print(f"No clickstream files found for date range, creating empty DataFrame")
            return create_empty_clickstream_features(spark)
        
        # Read all relevant clickstream files
        df = spark.read.parquet(*clickstream_files)
        
        # Get feature columns
        feature_cols = [col_name for col_name in df.columns if col_name.startswith('fe_')]
        
        # Create aggregated features per customer
        # Statistical aggregations
        agg_exprs = []
        for feature_col in feature_cols:
            agg_exprs.extend([
                avg(col(feature_col)).alias(f"{feature_col}_mean"),
                stddev(col(feature_col)).alias(f"{feature_col}_std"),
                spark_min(col(feature_col)).alias(f"{feature_col}_min"),
                spark_max(col(feature_col)).alias(f"{feature_col}_max")
            ])
        
        # Add record count and time span
        agg_exprs.extend([
            count("*").alias("clickstream_record_count"),
            F.datediff(spark_max("snapshot_date"), spark_min("snapshot_date")).alias("clickstream_time_span_days")
        ])
        
        clickstream_features = df.groupBy("Customer_ID").agg(*agg_exprs)
        
        # Add derived features
        # Calculate feature stability (coefficient of variation)
        for feature_col in feature_cols:
            clickstream_features = clickstream_features.withColumn(
                f"{feature_col}_cv",
                when(col(f"{feature_col}_mean") != 0, 
                     col(f"{feature_col}_std") / F.abs(col(f"{feature_col}_mean")))
                .otherwise(0)
            )
        
        print(f"Clickstream features processed for {clickstream_features.count()} customers")
        return clickstream_features
        
    except Exception as e:
        print(f"Error processing clickstream features: {e}")
        return create_empty_clickstream_features(spark)


def load_and_process_attributes_features_latest(snapshot_date_str, silver_attributes_directory, spark):
    """
    Load and process customer attributes using most recent available data up to snapshot date
    """
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # Get all attributes files up to snapshot date
        attr_files = []
        all_files = glob.glob(silver_attributes_directory + "*.parquet")
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            date_part = filename.replace('silver_attributes_', '').replace('.parquet', '').replace('_', '-')
            try:
                file_date = datetime.strptime(date_part, "%Y-%m-%d")
                if file_date <= snapshot_date:
                    attr_files.append((filepath, file_date))
            except ValueError:
                continue
        
        if not attr_files:
            print(f"No attributes files found up to {snapshot_date_str}, creating empty DataFrame")
            return create_empty_attributes_features(spark)
        
        # Read all relevant attributes files
        all_attr_data = []
        for filepath, file_date in attr_files:
            df = spark.read.parquet(filepath)
            df = df.withColumn("file_date", lit(file_date))
            all_attr_data.append(df)
        
        # Combine all attributes data
        combined_attr_df = all_attr_data[0]
        for df in all_attr_data[1:]:
            combined_attr_df = combined_attr_df.union(df)
        
        # Get the most recent record for each customer
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("file_date").desc())
        latest_attributes = combined_attr_df.withColumn("row_number", F.row_number().over(window_spec)) \
                                          .filter(col("row_number") == 1) \
                                          .drop("row_number", "file_date")
        
        # Select and process key attributes features
        attributes_features = latest_attributes.select(
            col("Customer_ID"),
            col("Age_clean").alias("age"),
            col("Age_valid").alias("age_is_valid"),
            col("Occupation_clean").alias("occupation"),
            col("SSN_valid").alias("ssn_is_valid"),
            col("data_quality_score").alias("attributes_quality_score")
        )
        
        # Create age group features
        attributes_features = attributes_features.withColumn(
            "age_group",
            when((col("age") >= 18) & (col("age") < 30), "18-29")
            .when((col("age") >= 30) & (col("age") < 40), "30-39")
            .when((col("age") >= 40) & (col("age") < 50), "40-49")
            .when((col("age") >= 50) & (col("age") < 60), "50-59")
            .when(col("age") >= 60, "60+")
            .otherwise("Unknown")
        )
        
        # Create occupation category groups (based on EDA findings)
        attributes_features = attributes_features.withColumn(
            "occupation_category",
            when(col("occupation").isin(["Lawyer", "Doctor", "Architect", "Engineer"]), "Professional")
            .when(col("occupation").isin(["Teacher", "Scientist"]), "Education_Research")
            .when(col("occupation").isin(["Developer", "Media_Manager"]), "Technology_Media")
            .when(col("occupation").isin(["Mechanic", "Accountant"]), "Technical_Services")
            .when(col("occupation") == "Unknown", "Unknown")
            .otherwise("Other")
        )
        
        print(f"Attributes features processed for {attributes_features.count()} customers")
        return attributes_features
        
    except Exception as e:
        print(f"Error processing attributes features: {e}")
        return create_empty_attributes_features(spark)


def load_and_aggregate_financial_features_latest(snapshot_date_str, silver_financials_directory, spark, lookback_date):
    """
    Load financial data using most recent available data up to snapshot date
    """
    try:
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # Get all financial files up to snapshot date
        fin_files = []
        all_files = glob.glob(silver_financials_directory + "*.parquet")
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            date_part = filename.replace('silver_financials_', '').replace('.parquet', '').replace('_', '-')
            try:
                file_date = datetime.strptime(date_part, "%Y-%m-%d")
                if file_date <= snapshot_date:
                    fin_files.append((filepath, file_date))
            except ValueError:
                continue
        
        if not fin_files:
            print(f"No financial files found up to {snapshot_date_str}, creating empty DataFrame")
            return create_empty_financial_features(spark)
        
        # Read all relevant financial files
        all_fin_data = []
        for filepath, file_date in fin_files:
            df = spark.read.parquet(filepath)
            df = df.withColumn("file_date", lit(file_date))
            all_fin_data.append(df)
        
        # Combine all financial data
        combined_fin_df = all_fin_data[0]
        for df in all_fin_data[1:]:
            combined_fin_df = combined_fin_df.union(df)
        
        # Get the most recent record for each customer
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("file_date").desc())
        latest_financial = combined_fin_df.withColumn("row_number", F.row_number().over(window_spec)) \
                                         .filter(col("row_number") == 1) \
                                         .drop("row_number", "file_date")
        
        # Create financial features
        financial_features = latest_financial.select(
            col("Customer_ID"),
            col("Annual_Income_clean").alias("annual_income"),
            col("Monthly_Inhand_Salary").alias("monthly_salary"),
            col("Credit_Mix_clean").alias("credit_mix"),
            col("Payment_of_Min_Amount_clean").alias("payment_of_min_amount"),
            col("Credit_History_Months").alias("credit_history_months"),
            col("Credit_Utilization_Ratio").alias("credit_utilization_ratio"),
            col("Delay_from_due_date").alias("delay_from_due_date"),
            col("Num_of_Loan_clean").alias("num_loans"),
            col("Num_of_Delayed_Payment_clean").alias("num_delayed_payments"),
            col("Outstanding_Debt_clean").alias("outstanding_debt"),
            col("Total_EMI_per_month").alias("total_emi_monthly"),
            col("Amount_invested_monthly").alias("amount_invested_monthly"),
            col("Monthly_Balance").alias("monthly_balance"),
            col("debt_to_income_ratio"),
            col("emi_to_income_ratio")
        )
        
        # Create additional financial risk indicators
        financial_features = financial_features.withColumn(
            "high_credit_utilization",
            when(col("credit_utilization_ratio") > 30, 1).otherwise(0)
        )
        
        financial_features = financial_features.withColumn(
            "high_debt_burden",
            when(col("debt_to_income_ratio") > 0.4, 1).otherwise(0)
        )
        
        financial_features = financial_features.withColumn(
            "payment_issues",
            when((col("num_delayed_payments") > 5) | (col("delay_from_due_date") > 30), 1).otherwise(0)
        )
        
        # Create financial health score
        financial_features = financial_features.withColumn(
            "financial_health_score",
            (when(col("credit_mix") == "Good", 1).otherwise(0) +
             when(col("payment_of_min_amount") == "Yes", 1).otherwise(0) +
             when(col("high_credit_utilization") == 0, 1).otherwise(0) +
             when(col("high_debt_burden") == 0, 1).otherwise(0) +
             when(col("payment_issues") == 0, 1).otherwise(0)) / 5.0
        )
        
        print(f"Financial features processed for {financial_features.count()} customers")
        return financial_features
        
    except Exception as e:
        print(f"Error processing financial features: {e}")
        return create_empty_financial_features(spark)


def combine_all_features_customer_centric(all_customers, clickstream_features, attributes_features, 
                                         financial_features, snapshot_date_str):
    """
    Combine features from all sources using customer-centric approach with left joins
    """
    # Start with all customers as the base
    combined = all_customers
    
    # Left join with financial features
    if financial_features.count() > 0:
        combined = combined.join(financial_features, on="Customer_ID", how="left")
    
    # Left join with attributes features
    if attributes_features.count() > 0:
        combined = combined.join(attributes_features, on="Customer_ID", how="left")
    
    # Left join with clickstream features
    if clickstream_features.count() > 0:
        combined = combined.join(clickstream_features, on="Customer_ID", how="left")
    
    # Add snapshot date for reference
    combined = combined.withColumn("feature_snapshot_date", lit(snapshot_date_str))
    
    print(f"Combined features for {combined.count()} customers")
    return combined


def create_interaction_features(df):
    """
    Create interaction features between different data sources
    """
    # Age-Income interactions (handle nulls)
    df = df.withColumn(
        "age_income_interaction",
        when((col("age").isNotNull()) & (col("annual_income").isNotNull()),
             col("age") * col("annual_income") / 1000000)  # Normalized
        .otherwise(None)
    )
    
    # Age-Credit History interaction (handle nulls)
    df = df.withColumn(
        "relative_credit_history",
        when((col("age").isNotNull()) & (col("credit_history_months").isNotNull()) & (col("age") > 0), 
             col("credit_history_months") / col("age"))
        .otherwise(None)
    )
    
    # Income-Debt ratio with age adjustment (handle nulls)
    df = df.withColumn(
        "age_adjusted_debt_ratio",
        when((col("age").isNotNull()) & (col("debt_to_income_ratio").isNotNull()) & (col("age") > 0), 
             col("debt_to_income_ratio") * (col("age") / 40))
        .otherwise(col("debt_to_income_ratio"))
    )
    
    return df


def add_data_availability_flags(df):
    """
    Add flags indicating which data sources are available for each customer
    """
    # Check clickstream availability
    df = df.withColumn(
        "has_clickstream_data",
        when(col("clickstream_record_count").isNull(), 0).otherwise(1)
    )
    
    # Check attributes availability  
    df = df.withColumn(
        "has_attributes_data",
        when(col("age").isNull(), 0).otherwise(1)
    )
    
    # Check financial availability
    df = df.withColumn(
        "has_financial_data",
        when(col("annual_income").isNull(), 0).otherwise(1)
    )
    
    # Create overall data completeness score
    df = df.withColumn(
        "data_completeness_score",
        (col("has_clickstream_data") + col("has_attributes_data") + col("has_financial_data")) / 3.0
    )
    
    return df


def create_empty_clickstream_features(spark):
    """Create empty clickstream features DataFrame for cases with no data"""
    schema = "Customer_ID string, clickstream_record_count int"
    return spark.createDataFrame([], schema)


def create_empty_attributes_features(spark):
    """Create empty attributes features DataFrame for cases with no data"""
    schema = "Customer_ID string, age int"
    return spark.createDataFrame([], schema)


def create_empty_financial_features(spark):
    """Create empty financial features DataFrame for cases with no data"""
    schema = "Customer_ID string, annual_income double"
    return spark.createDataFrame([], schema)


def build_feature_gold_config(execution_date_str: str,
                             silver_clickstream_path: str = "datamart/silver/clickstream/",
                             silver_attributes_path: str = "datamart/silver/attributes/",
                             silver_financials_path: str = "datamart/silver/financials/",
                             gold_output_path: str = "datamart/gold/feature_store/",
                             lookback_months: int = 6) -> dict:
    """
    Build configuration dictionary for feature gold processing
    
    Args:
        execution_date_str: Date for processing in YYYY-MM-DD format
        silver_clickstream_path: Path to silver clickstream data
        silver_attributes_path: Path to silver attributes data
        silver_financials_path: Path to silver financials data
        gold_output_path: Path to save gold feature store
        lookback_months: Number of months to look back for historical features
        
    Returns:
        Configuration dictionary containing all processing parameters
    """
    
    config = {
        "execution_date_str": execution_date_str,
        "execution_date": datetime.strptime(execution_date_str, "%Y-%m-%d"),
        "silver_paths": {
            "clickstream": silver_clickstream_path,
            "attributes": silver_attributes_path,
            "financials": silver_financials_path
        },
        "gold_output_path": gold_output_path,
        "lookback_months": lookback_months,
        "processing_timestamp": datetime.now()
    }
    
    return config


def run_single_feature_gold_processing(snapshot_date_str: str, spark,
                                      silver_clickstream_path: str = "datamart/silver/clickstream/",
                                      silver_attributes_path: str = "datamart/silver/attributes/",
                                      silver_financials_path: str = "datamart/silver/financials/",
                                      gold_output_path: str = "datamart/gold/feature_store/",
                                      lookback_months: int = 6) -> dict:
    """
    Run feature gold processing for single date (Airflow-compatible function)
    
    Args:
        snapshot_date_str: Date for processing in YYYY-MM-DD format
        spark: SparkSession object
        silver_clickstream_path: Path to silver clickstream data
        silver_attributes_path: Path to silver attributes data
        silver_financials_path: Path to silver financials data
        gold_output_path: Path to save gold feature store
        lookback_months: Number of months to look back for historical features
        
    Returns:
        Dictionary containing processing results
    """
    
    try:
        print(f"Processing feature gold data for {snapshot_date_str}")
        
        # Build configuration
        config = build_feature_gold_config(
            execution_date_str=snapshot_date_str,
            silver_clickstream_path=silver_clickstream_path,
            silver_attributes_path=silver_attributes_path,
            silver_financials_path=silver_financials_path,
            gold_output_path=gold_output_path,
            lookback_months=lookback_months
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(gold_output_path):
            os.makedirs(gold_output_path)
            print(f"Created directory: {gold_output_path}")
        
        # Process gold table using existing function
        processed_df = process_features_gold_table(
            snapshot_date_str, 
            silver_clickstream_path, 
            silver_attributes_path, 
            silver_financials_path, 
            gold_output_path, 
            spark, 
            lookback_months
        )
        
        # Validate processing results
        record_count = processed_df.count()
        feature_count = len(processed_df.columns)
        
        return {
            'success': True,
            'date': snapshot_date_str,
            'records_processed': record_count,
            'feature_count': feature_count,
            'lookback_months': lookback_months,
            'output_path': gold_output_path,
            'processing_timestamp': config['processing_timestamp'].isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'date': snapshot_date_str,
            'error': str(e)
        }


# Airflow-compatible task function
def feature_gold_airflow_task(snapshot_date: str, lookback_months: int = 6, **context) -> str:
    """
    Airflow-compatible feature gold processing task
    
    Args:
        snapshot_date: Date for processing in YYYY-MM-DD format
        lookback_months: Number of months to look back for historical features
        context: Airflow context
        
    Returns:
        Success message with processing summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"FeatureGold_{snapshot_date}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting feature gold processing for {snapshot_date}")
        
        result = run_single_feature_gold_processing(
            snapshot_date_str=snapshot_date,
            spark=spark,
            silver_clickstream_path="/opt/airflow/datamart/silver/clickstream/",
            silver_attributes_path="/opt/airflow/datamart/silver/attributes/",
            silver_financials_path="/opt/airflow/datamart/silver/financials/",
            gold_output_path="/opt/airflow/datamart/gold/feature_store/",
            lookback_months=lookback_months
        )
        
        if result['success']:
            message = f"Feature gold processing completed: {result['records_processed']} records, {result['feature_count']} features"
            print(f"[AIRFLOW] {message}")
            return message
        else:
            raise ValueError(f"Feature gold processing failed: {result['error']}")
            
    finally:
        spark.stop()


if __name__ == "__main__":
    # Example usage and testing
    print("Feature Gold Processing Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")