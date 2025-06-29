"""
End-to-End ML Pipeline DAG
Complete orchestration for data processing and machine learning lifecycle

This DAG orchestrates:
- Data processing pipelines (Bronze → Silver → Gold)
- Data validation across feature and label stores
- Conditional model training based on schedule
- Model inference for all available models
- Comprehensive model monitoring and stability analysis
- Automated reporting and notification system
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from datetime import datetime, timedelta
import sys
import os
import glob

# Add project root to Python path for module imports
sys.path.append('/opt/airflow')

# Import ML pipeline modules
import utils.data_validation as dv
import utils.model_training as mt
import utils.model_inference as mi  
import utils.model_monitoring as mm
import utils.reporting as rp
import utils.notification as nt
import utils.model_management as mm_utils

# Import data processing modules
import utils.data_processing_bronze_table as label_bronze
import utils.data_processing_silver_table as label_silver
import utils.data_processing_gold_table as label_gold
import utils.feature_processing_bronze_table as feature_bronze
import utils.feature_processing_silver_table as feature_silver
import utils.feature_processing_gold_table as feature_gold

# DAG configuration
default_args = {
    'owner': 'ml_engineering_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'execution_timeout': timedelta(hours=4),
}

# DAG definition with configurable parameters
dag = DAG(
    'ml_end_to_end_pipeline',
    default_args=default_args,
    description='Complete ML pipeline: Data Processing → Training → Inference → Monitoring',
    schedule_interval='0 2 1 * *',  # Every 1st of month at 2 AM
    start_date=datetime(2023, 1, 1),  # Data processing start date
    end_date=datetime(2024, 12, 1),   # Data coverage end date
    catchup=True,
    max_active_runs=1,  # Prevent overlapping runs
    doc_md=__doc__,
    tags=['ml', 'production', 'end-to-end', 'data-processing'],
    params={
        "training_interval_months": 3,  # Training frequency in months
        "models_to_process": ["logistic_regression", "random_forest", "xgboost"],
        "enable_monitoring": True,
        "send_notifications": True,
        "force_training": False,
        "feature_lookback_months": 6,  # Feature engineering lookback period
        "dpd_threshold": 30,  # Days past due for label definition
        "mob_threshold": 6    # Month on book for label definition
    }
)


def should_run_training(**context) -> bool:
    """
    Determine if training should run based on configurable interval
    Training runs every N months from September 2024 (first training date)
    """
    
    execution_date = context['execution_date']
    dag_instance = context['dag']
    dag_run = context['dag_run']
    
    # Check for force training override
    if dag_run.conf and dag_run.conf.get('force_training', False):
        print(f"[TRAINING CONTROL] Force training enabled for {execution_date}")
        return True

    # Training starts from September 2024
    training_start_date = datetime(2024, 9, 1)
    if execution_date.date() == training_start_date.date():
        print(f"[TRAINING CONTROL] First training date detected - forcing training execution") 
        return True

    # Only run training from September 2024 onwards
    if execution_date.date() < training_start_date.date():
        print(f"[TRAINING CONTROL] Skipping training - before training start date")
        return False
    
    # Get training interval from DAG params
    training_interval_months = dag_instance.params.get('training_interval_months', 3)
    
    # Calculate months since training start
    months_since_start = (execution_date.year - training_start_date.year) * 12 + (execution_date.month - training_start_date.month)
    
    # Training should run if months_since_start is divisible by training_interval_months
    should_train = months_since_start % training_interval_months == 0
    
    print(f"[TRAINING CONTROL] Execution date: {execution_date}")
    print(f"[TRAINING CONTROL] Months since start: {months_since_start}")
    print(f"[TRAINING CONTROL] Training interval: {training_interval_months} months")
    print(f"[TRAINING CONTROL] Should train: {should_train}")
    
    return should_train


def should_run_inference(**context) -> bool:
    """
    Determine if inference should run based on model availability
    Inference runs from September 2024 onwards (when models are available)
    """
    
    execution_date = context['execution_date']
    
    # Inference starts from September 2024 (when first models are available)
    inference_start_date = datetime(2024, 9, 1)
    
    should_infer = execution_date.date() >= inference_start_date.date()
    
    print(f"[INFERENCE CONTROL] Execution date: {execution_date}")
    print(f"[INFERENCE CONTROL] Should run inference: {should_infer}")
    
    return should_infer


def should_run_monitoring(**context) -> bool:
    """
    Determine if monitoring should run based on configuration and model availability
    """
    
    dag_run = context['dag_run']
    execution_date = context['execution_date']
    
    # Check if monitoring is disabled in run configuration
    if dag_run.conf and not dag_run.conf.get('enable_monitoring', True):
        print(f"[MONITORING CONTROL] Monitoring disabled for {execution_date}")
        return False
    
    # Monitoring runs only when inference runs (needs prediction results)
    return should_run_inference(**context)


def decide_execution_branch(**context) -> str:
    """
    Branch decision function to determine execution flow
    
    Returns:
        'training_start' if it's a training date (training → inference → monitoring)
        'inference_only_start' if it's a non-training date but models exist (inference → monitoring, skip training)
        'skip_to_reporting' if before first training date or no models available
    """
    
    execution_date = context['execution_date']
    
    # Define first training date (when ML pipeline starts)
    first_training_date = datetime(2024, 9, 1)
    
    # Check if current date is before first training date
    if execution_date.date() < first_training_date.date():
        print(f"[EXECUTION BRANCH] Before first training date ({first_training_date.date()}) - skipping all ML tasks")
        return 'skip_to_reporting'
    
    # Check if this is a training date
    is_training_date = should_run_training(**context)
    
    if is_training_date:
        print(f"[EXECUTION BRANCH] Training date detected - using execution with training")
        return 'training_start'  
    
    # For non-training dates, check if any models exist for inference
    try:
        models_to_check = ['logistic_regression', 'random_forest', 'xgboost']
        existing_models = []
        
        for model_type in models_to_check:
            latest_model_version = mm.find_latest_trained_model(model_type)  # 使用你的model_management
            if latest_model_version:
                existing_models.append(model_type)
        
        if existing_models:
            print(f"[EXECUTION BRANCH] Non-training date with existing models: {existing_models}")
            print(f"[EXECUTION BRANCH] Using inference-only execution (skip training)")
            return 'inference_only_start'
        else:
            print(f"[EXECUTION BRANCH] No models available - skipping to reporting")
            return 'skip_to_reporting'
            
    except Exception as e:
        print(f"[EXECUTION BRANCH] Error checking model availability: {e}")
        return 'skip_to_reporting'


# Data Processing Task Wrappers
def label_bronze_wrapper(**context):
    """Wrapper for label bronze processing task"""
    return label_bronze.label_bronze_airflow_task(snapshot_date=context['ds'], **context)


def label_silver_wrapper(**context):
    """Wrapper for label silver processing task"""
    return label_silver.label_silver_airflow_task(snapshot_date=context['ds'], **context)


def label_gold_wrapper(**context):
    """Wrapper for label gold processing task"""
    dag_instance = context['dag']
    dpd_threshold = dag_instance.params.get('dpd_threshold', 30)
    mob_threshold = dag_instance.params.get('mob_threshold', 6)
    return label_gold.label_gold_airflow_task(
        snapshot_date=context['ds'],
        dpd_threshold=dpd_threshold,
        mob_threshold=mob_threshold,
        **context
    )


def feature_bronze_wrapper(feature_type: str):
    """Create feature-specific bronze processing wrapper"""
    def _process_feature_bronze(**context):
        if feature_type == 'clickstream':
            return feature_bronze.clickstream_bronze_airflow_task(snapshot_date=context['ds'], **context)
        elif feature_type == 'attributes':
            return feature_bronze.attributes_bronze_airflow_task(snapshot_date=context['ds'], **context)
        elif feature_type == 'financials':
            return feature_bronze.financials_bronze_airflow_task(snapshot_date=context['ds'], **context)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    return _process_feature_bronze


def feature_silver_wrapper(feature_type: str):
    """Create feature-specific silver processing wrapper"""
    def _process_feature_silver(**context):
        if feature_type == 'clickstream':
            return feature_silver.clickstream_silver_airflow_task(snapshot_date=context['ds'], **context)
        elif feature_type == 'attributes':
            return feature_silver.attributes_silver_airflow_task(snapshot_date=context['ds'], **context)
        elif feature_type == 'financials':
            return feature_silver.financials_silver_airflow_task(snapshot_date=context['ds'], **context)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    return _process_feature_silver


def feature_gold_wrapper(**context):
    """Wrapper for feature gold processing task"""
    dag_instance = context['dag']
    lookback_months = dag_instance.params.get('feature_lookback_months', 6)
    return feature_gold.feature_gold_airflow_task(
        snapshot_date=context['ds'],
        lookback_months=lookback_months,
        **context
    )


# ML Pipeline Task Wrappers
def validate_data_stores_wrapper(**context):
    """Wrapper for data validation task"""
    return dv.validate_feature_and_label_stores_airflow_task(**context)


def train_model_wrapper(model_type: str):
    """Create model-specific training wrapper"""
    def _train_model(**context):
        # Check if training should actually run for this execution
        if not should_run_training(**context):
            print(f"[TRAINING] Skipping {model_type} training - not scheduled for this date")
            return {'training_skipped': True, 'model_type': model_type, 'reason': 'not_training_date'}
        
        return mt.training_airflow_task(
            training_date=context['ds'],
            models_to_train=[model_type],
            **context
        )
    return _train_model


def inference_model_wrapper(model_type: str):
    """Create model-specific inference wrapper with improved model detection"""
    def _inference_model(**context):
        execution_date = context['execution_date']
        first_training_date = datetime(2024, 9, 1)
        
        # Skip if before first training date
        if execution_date.date() < first_training_date.date():
            print(f"[INFERENCE] Skipping {model_type} inference - before first training date")
            return {'inference_skipped': True, 'model_type': model_type, 'reason': 'before_first_training'}
        
       # Find the latest available model using model_management
        try:
            print(f"[INFERENCE] Looking for {model_type} models...")
            
            model_version = mm_utils.find_latest_trained_model(model_type)
            
            if not model_version:
                print(f"[INFERENCE] No trained model found for {model_type} - skipping inference")
                return {'inference_skipped': True, 'model_type': model_type, 'reason': 'no_model_available'}
            
            print(f"[INFERENCE] Using model: {model_version}")
            
        except Exception as e:
            print(f"[INFERENCE] Error checking model availability for {model_type}: {e}")
            return {'inference_skipped': True, 'model_type': model_type, 'reason': 'model_check_failed', 'error': str(e)}
        
        # Proceed with inference if model exists
        try:
            return mi.inference_airflow_task(
                snapshot_date=context['ds'],
                model_version=model_version,
                **context
            )
        except Exception as e:
            print(f"[INFERENCE] Inference failed for {model_type}: {e}")
            return {'inference_failed': True, 'model_type': model_type, 'error': str(e)}
    return _inference_model


def monitor_model_wrapper(model_type: str):
    """Create model-specific monitoring wrapper with improved model finding"""
    def _monitor_model(**context):
        if not should_run_monitoring(**context):
            print(f"[MONITORING] Skipping {model_type} monitoring - disabled")
            return {'monitoring_skipped': True, 'model_type': model_type}
        
        try:
            # Find the latest available model using model_management
            print(f"[MONITORING] Looking for {model_type} models...")
            
            model_version = mm_utils.find_latest_trained_model(model_type)
            
            if not model_version:
                print(f"[MONITORING] No trained model found for {model_type} - skipping monitoring")
                return {'monitoring_skipped': True, 'model_type': model_type, 'reason': 'no_model_available'}
            
            print(f"[MONITORING] Using model version: {model_version}")
            
            # Calculate monitoring period (last 30 days)
            from datetime import datetime, timedelta
            end_date = datetime.strptime(context['ds'], '%Y-%m-%d')
            start_date = end_date - timedelta(days=30)
            
            return mm.monitoring_airflow_task(
                model_version=model_version,
                monitoring_start_date=start_date.strftime('%Y-%m-%d'),
                monitoring_end_date=end_date.strftime('%Y-%m-%d'),
                **context
            )
            
        except Exception as e:
            print(f"[MONITORING] Error in monitoring for {model_type}: {e}")
            return {'monitoring_failed': True, 'model_type': model_type, 'error': str(e)}
    return _monitor_model


def generate_pipeline_report_wrapper(**context):
    """Wrapper for pipeline report generation"""
    return rp.generate_comprehensive_pipeline_report_airflow_task(**context)


def send_notifications_wrapper(**context):
    """Wrapper for notification sending"""
    return nt.send_pipeline_notifications_airflow_task(**context)


# Define TaskGroups and tasks
with dag:
    
    # Pipeline start marker
    pipeline_start = DummyOperator(
        task_id='pipeline_start',
        doc_md="Pipeline execution start marker - validates configuration"
    )
    
    # Data Processing Pipeline - Feature and Label branches
    with TaskGroup(group_id='data_processing_pipeline', 
                   tooltip="Data processing pipeline with medallion architecture") as data_processing_pipeline:
        
        # Feature Processing Branch
        with TaskGroup(group_id='feature_processing', 
                       tooltip="Feature data processing: Bronze → Silver → Gold") as feature_processing:
            
            # Feature Bronze Layer
            with TaskGroup(group_id='feature_bronze_layer', 
                           tooltip="Feature bronze layer - raw data ingestion") as feature_bronze_layer:
                
                process_clickstream_bronze = PythonOperator(
                    task_id='process_clickstream_bronze',
                    python_callable=feature_bronze_wrapper('clickstream'),
                    doc_md="Process clickstream data for bronze layer"
                )
                
                process_attributes_bronze = PythonOperator(
                    task_id='process_attributes_bronze',
                    python_callable=feature_bronze_wrapper('attributes'),
                    doc_md="Process customer attributes data for bronze layer"
                )
                
                process_financials_bronze = PythonOperator(
                    task_id='process_financials_bronze',
                    python_callable=feature_bronze_wrapper('financials'),
                    doc_md="Process financial features data for bronze layer"
                )
                
                feature_bronze_completed = DummyOperator(
                    task_id='feature_bronze_completed',
                    doc_md="Feature bronze layer completion marker"
                )
                
                # Parallel bronze processing
                [process_clickstream_bronze, process_attributes_bronze, process_financials_bronze] >> feature_bronze_completed
            
            # Feature Silver Layer
            with TaskGroup(group_id='feature_silver_layer', 
                           tooltip="Feature silver layer - data cleaning and standardization") as feature_silver_layer:
                
                process_clickstream_silver = PythonOperator(
                    task_id='process_clickstream_silver',
                    python_callable=feature_silver_wrapper('clickstream'),
                    doc_md="Clean and standardize clickstream features"
                )
                
                process_attributes_silver = PythonOperator(
                    task_id='process_attributes_silver',
                    python_callable=feature_silver_wrapper('attributes'),
                    doc_md="Clean and validate customer attributes"
                )
                
                process_financials_silver = PythonOperator(
                    task_id='process_financials_silver',
                    python_callable=feature_silver_wrapper('financials'),
                    doc_md="Clean and standardize financial metrics"
                )
                
                feature_silver_completed = DummyOperator(
                    task_id='feature_silver_completed',
                    doc_md="Feature silver layer completion marker"
                )
                
                # Parallel silver processing
                [process_clickstream_silver, process_attributes_silver, process_financials_silver] >> feature_silver_completed
            
            # Feature Gold Layer
            with TaskGroup(group_id='feature_gold_layer', 
                           tooltip="Feature gold layer - ML-ready feature store") as feature_gold_layer:
                
                process_feature_store = PythonOperator(
                    task_id='process_feature_store',
                    python_callable=feature_gold_wrapper,
                    doc_md="Create ML-ready feature store by combining all feature sources"
                )
            
            # Feature processing dependencies
            feature_bronze_layer >> feature_silver_layer >> feature_gold_layer
        
        # Label Processing Branch
        with TaskGroup(group_id='label_processing', 
                       tooltip="Label data processing: Bronze → Silver → Gold") as label_processing:
            
            # Label Bronze Layer
            process_label_bronze = PythonOperator(
                task_id='process_label_bronze',
                python_callable=label_bronze_wrapper,
                doc_md="Process loan daily data for bronze layer"
            )
            
            # Label Silver Layer
            process_label_silver = PythonOperator(
                task_id='process_label_silver',
                python_callable=label_silver_wrapper,
                doc_md="Clean and standardize loan metrics"
            )
            
            # Label Gold Layer
            process_label_gold = PythonOperator(
                task_id='process_label_gold',
                python_callable=label_gold_wrapper,
                doc_md="Create ML-ready label store with default definitions"
            )
            
            # Label processing dependencies
            process_label_bronze >> process_label_silver >> process_label_gold
    
    # Data Validation TaskGroup
    with TaskGroup(group_id='data_validation_group', 
                   tooltip="Data quality and availability validation") as data_validation_group:
        
        validate_data_stores = PythonOperator(
            task_id='validate_data_stores',
            python_callable=validate_data_stores_wrapper,
            doc_md="Validate feature store and label store data quality and availability"
        )
    
    # Execution Branch Decision
    execution_branch_decision = BranchPythonOperator(
        task_id='execution_branch_decision',
        python_callable=decide_execution_branch,
        doc_md="Decide whether to run training and inference sequentially or inference-only"
    )
    
    # Sequential Execution Start Point (Entry task for sequential execution with training)
    training_start = DummyOperator(
        task_id='training_start',
        doc_md="Entry point for training execution path (training dates)"
    )

    
    # Inference-Only Execution Start Point (Entry task for inference-only execution)
    inference_only_start = DummyOperator(
        task_id='inference_only_start',
        doc_md="Entry point for inference-only execution path (non-training dates)"
    )
    
    # Sequential Execution Path (Training Date - Training → Inference → Monitoring)
    with TaskGroup(group_id='training_execution', 
               tooltip="Training execution: Training → Inference → Monitoring") as training_execution:
        
        # Model Training TaskGroup
        with TaskGroup(group_id='seq_model_training_group', 
                       tooltip="Sequential model training") as seq_model_training_group:
            
            seq_train_logistic_regression = PythonOperator(
                task_id='seq_train_logistic_regression',
                python_callable=train_model_wrapper('logistic_regression'),
                doc_md="Train logistic regression model with hyperparameter optimization"
            )
            
            seq_train_random_forest = PythonOperator(
                task_id='seq_train_random_forest',
                python_callable=train_model_wrapper('random_forest'),
                doc_md="Train random forest model with feature importance analysis"
            )
            
            seq_train_xgboost = PythonOperator(
                task_id='seq_train_xgboost',
                python_callable=train_model_wrapper('xgboost'),
                doc_md="Train XGBoost model with gradient boosting optimization"
            )
            
            seq_training_completed = DummyOperator(
                task_id='seq_training_completed',
                doc_md="Sequential training phase completion marker"
            )
            
            [seq_train_logistic_regression, seq_train_random_forest, seq_train_xgboost] >> seq_training_completed
        
        # Model Inference TaskGroup (waits for training)
        with TaskGroup(group_id='seq_model_inference_group', 
                       tooltip="Sequential model inference after training") as seq_model_inference_group:
            
            seq_inference_logistic_regression = PythonOperator(
                task_id='seq_inference_logistic_regression',
                python_callable=inference_model_wrapper('logistic_regression'),
                doc_md="Generate predictions using logistic regression model after training"
            )
            
            seq_inference_random_forest = PythonOperator(
                task_id='seq_inference_random_forest',
                python_callable=inference_model_wrapper('random_forest'),
                doc_md="Generate predictions using random forest model after training"
            )
            
            seq_inference_xgboost = PythonOperator(
                task_id='seq_inference_xgboost',
                python_callable=inference_model_wrapper('xgboost'),
                doc_md="Generate predictions using XGBoost model after training"
            )
            
            seq_inference_completed = DummyOperator(
                task_id='seq_inference_completed',
                doc_md="Sequential inference phase completion marker"
            )
            
            [seq_inference_logistic_regression, seq_inference_random_forest, seq_inference_xgboost] >> seq_inference_completed
        
        # Model Monitoring TaskGroup (after inference)
        with TaskGroup(group_id='seq_model_monitoring_group', 
                       tooltip="Sequential model monitoring after inference") as seq_model_monitoring_group:
            
            seq_monitor_logistic_regression = PythonOperator(
                task_id='seq_monitor_logistic_regression',
                python_callable=monitor_model_wrapper('logistic_regression'),
                doc_md="Monitor logistic regression model performance after inference"
            )
            
            seq_monitor_random_forest = PythonOperator(
                task_id='seq_monitor_random_forest',
                python_callable=monitor_model_wrapper('random_forest'),
                doc_md="Monitor random forest model performance after inference"
            )
            
            seq_monitor_xgboost = PythonOperator(
                task_id='seq_monitor_xgboost',
                python_callable=monitor_model_wrapper('xgboost'),
                doc_md="Monitor XGBoost model performance after inference"
            )
            
            seq_monitoring_completed = DummyOperator(
                task_id='seq_monitoring_completed',
                doc_md="Sequential monitoring phase completion marker"
            )
            
            [seq_monitor_logistic_regression, seq_monitor_random_forest, seq_monitor_xgboost] >> seq_monitoring_completed
        
        # Sequential execution dependencies
        seq_model_training_group >> seq_model_inference_group >> seq_model_monitoring_group
    
    # Inference-Only Execution Path (Non-Training Date - Skip Training, Run Inference → Monitoring)
    with TaskGroup(group_id='inference_only_execution', 
                   tooltip="Inference-only execution: Inference → Monitoring (skip training)") as inference_only_execution:
        
        # Model Inference TaskGroup
        with TaskGroup(group_id='inf_only_model_inference_group', 
                       tooltip="Inference-only model inference") as inf_only_model_inference_group:
            
            inf_only_inference_logistic_regression = PythonOperator(
                task_id='inf_only_inference_logistic_regression',
                python_callable=inference_model_wrapper('logistic_regression'),
                doc_md="Generate predictions using logistic regression model (inference-only execution)"
            )
            
            inf_only_inference_random_forest = PythonOperator(
                task_id='inf_only_inference_random_forest',
                python_callable=inference_model_wrapper('random_forest'),
                doc_md="Generate predictions using random forest model (inference-only execution)"
            )
            
            inf_only_inference_xgboost = PythonOperator(
                task_id='inf_only_inference_xgboost',
                python_callable=inference_model_wrapper('xgboost'),
                doc_md="Generate predictions using XGBoost model (inference-only execution)"
            )
            
            inf_only_inference_completed = DummyOperator(
                task_id='inf_only_inference_completed',
                doc_md="Inference-only inference phase completion marker"
            )
            
            [inf_only_inference_logistic_regression, inf_only_inference_random_forest, inf_only_inference_xgboost] >> inf_only_inference_completed
        
        # Model Monitoring TaskGroup (after inference)
        with TaskGroup(group_id='inf_only_model_monitoring_group', 
                       tooltip="Inference-only model monitoring after inference") as inf_only_model_monitoring_group:
            
            inf_only_monitor_logistic_regression = PythonOperator(
                task_id='inf_only_monitor_logistic_regression',
                python_callable=monitor_model_wrapper('logistic_regression'),
                doc_md="Monitor logistic regression model performance (inference-only)"
            )
            
            inf_only_monitor_random_forest = PythonOperator(
                task_id='inf_only_monitor_random_forest',
                python_callable=monitor_model_wrapper('random_forest'),
                doc_md="Monitor random forest model performance (inference-only)"
            )
            
            inf_only_monitor_xgboost = PythonOperator(
                task_id='inf_only_monitor_xgboost',
                python_callable=monitor_model_wrapper('xgboost'),
                doc_md="Monitor XGBoost model performance (inference-only)"
            )
            
            inf_only_monitoring_completed = DummyOperator(
                task_id='inf_only_monitoring_completed',
                doc_md="Inference-only monitoring phase completion marker"
            )
            
            [inf_only_monitor_logistic_regression, inf_only_monitor_random_forest, inf_only_monitor_xgboost] >> inf_only_monitoring_completed
        
        # Inference-only execution dependencies: inference → monitoring
        inf_only_model_inference_group >> inf_only_model_monitoring_group
    
    # Skip to Reporting (for dates before ML start)
    skip_to_reporting = DummyOperator(
        task_id='skip_to_reporting',
        doc_md="Skip ML tasks for dates before September 2024"
    )
    
    # Convergence point for all execution paths
    execution_convergence = DummyOperator(
        task_id='execution_convergence',
        trigger_rule='none_failed_min_one_success',
        doc_md="Convergence point for all execution paths before reporting"
    )
    
    # Reporting TaskGroup
    with TaskGroup(group_id='reporting_group', 
                   tooltip="Pipeline reporting and notification management") as reporting_group:
        
        generate_pipeline_report = PythonOperator(
            task_id='generate_pipeline_report',
            python_callable=generate_pipeline_report_wrapper,
            doc_md="Generate comprehensive pipeline execution report with insights"
        )
        
        send_notifications = PythonOperator(
            task_id='send_notifications',
            python_callable=send_notifications_wrapper,
            doc_md="Send notifications based on pipeline execution results"
        )
        
        # Reporting task dependencies
        generate_pipeline_report >> send_notifications
    
    # Pipeline end marker
    pipeline_end = DummyOperator(
        task_id='pipeline_end',
        doc_md="Pipeline execution completion marker"
    )
    
    # Define main pipeline dependencies with conditional execution
    pipeline_start >> data_processing_pipeline >> data_validation_group >> execution_branch_decision

    # Branch connections - now pointing to entry point tasks
    execution_branch_decision >> [training_start, inference_only_start, skip_to_reporting]

    # Connect entry points to their respective execution paths
    training_start >> training_execution
    inference_only_start >> inference_only_execution

    # All execution paths converge to the convergence point
    training_execution >> execution_convergence
    inference_only_execution >> execution_convergence
    skip_to_reporting >> execution_convergence

    # Continue to reporting and end
    execution_convergence >> reporting_group >> pipeline_end