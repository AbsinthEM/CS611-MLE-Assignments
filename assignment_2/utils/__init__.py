"""
Utils package for data processing pipeline

This package contains modules for processing both feature and label data through the Medallion Architecture:
- Bronze layer: Raw data ingestion with metadata
- Silver layer: Data cleaning and standardization  
- Gold layer: ML-ready feature engineering and label creation

Modules:
    Feature Processing:
        feature_processing_bronze: Bronze layer feature processing functions
        feature_processing_silver: Silver layer feature cleaning functions
        feature_processing_gold: Gold layer feature engineering functions with customer-centric approach
    
    Label Processing:
        data_processing_bronze_table: Bronze layer loan data processing
        data_processing_silver_table: Silver layer loan data cleaning
        data_processing_gold_table: Gold layer label creation
"""

# Import feature processing bronze layer functions
from .feature_processing_bronze_table import (
    process_clickstream_bronze_table,
    process_attributes_bronze_table,
    process_financials_bronze_table
)

# Import feature processing silver layer functions
from .feature_processing_silver_table import (
    process_clickstream_silver_table,
    process_attributes_silver_table,
    process_financials_silver_table
)

# Import feature processing gold layer functions (updated with latest data strategy)
from .feature_processing_gold_table import (
    process_features_gold_table,
    get_all_unique_customers,
    load_and_aggregate_clickstream_features,
    load_and_process_attributes_features_latest,
    load_and_aggregate_financial_features_latest,
    combine_all_features_customer_centric,
    create_interaction_features,
    add_data_availability_flags
)

# Import label processing functions (existing pipeline)
from .data_processing_bronze_table import (
    process_bronze_table
)

from .data_processing_silver_table import (
    process_silver_table
)

from .data_processing_gold_table import (
    process_labels_gold_table
)

# Define what gets imported when using "from utils import *"
__all__ = [
    # Feature processing bronze layer functions
    'process_clickstream_bronze_table',
    'process_attributes_bronze_table', 
    'process_financials_bronze_table',
    
    # Feature processing silver layer functions
    'process_clickstream_silver_table',
    'process_attributes_silver_table',
    'process_financials_silver_table',
    
    # Feature processing gold layer functions (updated)
    'process_features_gold_table',
    'get_all_unique_customers',
    'load_and_aggregate_clickstream_features',
    'load_and_process_attributes_features_latest', 
    'load_and_aggregate_financial_features_latest',
    'combine_all_features_customer_centric',
    'create_interaction_features',
    'add_data_availability_flags',
    
    # Label processing functions (existing pipeline)
    'process_bronze_table',
    'process_silver_table',
    'process_labels_gold_table'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "Data Science Team"
__description__ = "Data processing utilities for ML feature and label pipeline"