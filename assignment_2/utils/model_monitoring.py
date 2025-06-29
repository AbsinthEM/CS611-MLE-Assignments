"""
Model Monitoring Utilities Module
Provides production-ready monitoring functionality for ML models

This module handles:
- Model performance tracking over time
- Data drift and stability analysis
- Alert detection and notification
- Monitoring report generation
- Batch monitoring for multiple models
- Airflow-compatible monitoring tasks
"""

import os
import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from scipy import stats
import pyspark
from pyspark.sql.functions import col


def build_monitoring_config(model_version: str, monitoring_start_date: str, monitoring_end_date: str,
                           predictions_path: str = "datamart/gold/model_predictions/",
                           labels_path: str = "datamart/gold/label_store/", 
                           monitoring_output_path: str = "datamart/gold/model_monitoring/",
                           model_bank_directory: str = "models/") -> Dict[str, Any]:
    """
    Build configuration dictionary for model monitoring
    
    Args:
        model_version: Model version to monitor
        monitoring_start_date: Start date for monitoring period (YYYY-MM-DD)
        monitoring_end_date: End date for monitoring period (YYYY-MM-DD)
        predictions_path: Path to model predictions datamart
        labels_path: Path to labels datamart
        monitoring_output_path: Path to save monitoring results
        model_bank_directory: Directory containing model artifacts
        
    Returns:
        Configuration dictionary containing all monitoring parameters
    """
    
    config = {
        # Model configuration
        "model_version": model_version,
        "model_bank_directory": model_bank_directory,
        "model_artifact_filepath": os.path.join(model_bank_directory, f"{model_version}.pkl"),
        
        # Monitoring period
        "monitoring_start_date": monitoring_start_date,
        "monitoring_end_date": monitoring_end_date,
        "monitoring_start_dt": datetime.strptime(monitoring_start_date, "%Y-%m-%d"),
        "monitoring_end_dt": datetime.strptime(monitoring_end_date, "%Y-%m-%d"),
        
        # Data paths
        "predictions_path": predictions_path,
        "labels_path": labels_path,
        "monitoring_output_path": monitoring_output_path,
        
        # Output paths
        "performance_output_path": os.path.join(monitoring_output_path, model_version, "performance_metrics"),
        "stability_output_path": os.path.join(monitoring_output_path, model_version, "stability_analysis"),
        "alerts_output_path": os.path.join(monitoring_output_path, model_version, "alerts"),
        "visualizations_output_path": os.path.join(monitoring_output_path, model_version, "visualizations"),
        "reports_output_path": os.path.join(monitoring_output_path, model_version, "reports"),
        
        # Monitoring thresholds
        "performance_degradation_threshold": 0.05,  # 5% AUC degradation
        "stability_threshold": 0.2,  # PSI threshold for feature drift
        "alert_thresholds": {
            "auc_critical": 0.10,      # 10% AUC drop = critical
            "auc_warning": 0.05,       # 5% AUC drop = warning
            "psi_critical": 0.25,      # PSI > 0.25 = critical drift
            "psi_warning": 0.10        # PSI > 0.10 = warning drift
        }
    }
    
    return config


def load_training_baseline(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load training baseline performance from model artifact
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing baseline performance metrics or None if loading fails
    """
    
    try:
        model_filepath = config["model_artifact_filepath"]
        
        if not os.path.exists(model_filepath):
            print(f"Model artifact not found: {model_filepath}")
            return None
            
        with open(model_filepath, 'rb') as f:
            model_artifact = pickle.load(f)
        
        # Extract baseline performance
        baseline = {
            "auc_train": model_artifact['results']['auc_train'],
            "auc_test": model_artifact['results']['auc_test'], 
            "auc_oot": model_artifact['results']['auc_oot'],
            "gini_train": model_artifact['results']['gini_train'],
            "gini_test": model_artifact['results']['gini_test'],
            "gini_oot": model_artifact['results']['gini_oot'],
            "training_date": model_artifact.get('data_dates', {}).get('model_train_date_str', 'Unknown'),
            "feature_count": model_artifact['data_stats']['feature_count']
        }
        
        print(f"Training baseline loaded from: {model_filepath}")
        return baseline
        
    except Exception as e:
        print(f"Error loading training baseline: {str(e)}")
        return None


def load_monitoring_data(config: Dict[str, Any], spark) -> Optional[pd.DataFrame]:
    """
    Load and merge model predictions with actual labels for monitoring
    
    Args:
        config: Configuration dictionary
        spark: SparkSession object
        
    Returns:
        DataFrame containing merged predictions and labels or None if loading fails
    """
    
    try:
        model_version = config["model_version"]
        start_date = config["monitoring_start_dt"]
        end_date = config["monitoring_end_dt"]
        
        print(f"Loading monitoring data for {model_version}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Load predictions
        predictions_directory = os.path.join(config["predictions_path"], model_version)
        
        if not os.path.exists(predictions_directory):
            print(f"Predictions directory not found: {predictions_directory}")
            return None
        
        prediction_files = glob.glob(os.path.join(predictions_directory, "*.parquet"))
        
        if not prediction_files:
            print(f"No prediction files found in: {predictions_directory}")
            return None
        
        # Load all prediction files
        prediction_files_full_path = [predictions_directory + "/" + os.path.basename(f) for f in prediction_files]
        predictions_sdf = spark.read.option("header", "true").parquet(*prediction_files_full_path)
        
        # Filter predictions by date range
        predictions_filtered_sdf = predictions_sdf.filter(
            (col("snapshot_date") >= start_date) & 
            (col("snapshot_date") <= end_date)
        )
        
        predictions_pdf = predictions_filtered_sdf.toPandas()
        print(f"Loaded {len(predictions_pdf)} prediction records")
        
        # Load labels
        labels_path = config["labels_path"]
        label_files = glob.glob(os.path.join(labels_path, "*.parquet"))
        
        if not label_files:
            print(f"No label files found in: {labels_path}")
            return None
        
        # Load all label files
        label_files_full_path = [labels_path + "/" + os.path.basename(f) for f in label_files]
        labels_sdf = spark.read.option("header", "true").parquet(*label_files_full_path)
        
        labels_pdf = labels_sdf.toPandas()
        print(f"Loaded {len(labels_pdf)} label records (all periods)")


        # Convert ISO format (2024-12-01T00:00:00.000Z) to simple date format (2024-12-01)
        labels_pdf['snapshot_date'] = labels_pdf['snapshot_date'].astype(str).str[:10]
        predictions_pdf['snapshot_date'] = predictions_pdf['snapshot_date'].astype(str)
        
        # Merge predictions with labels
        monitoring_data = predictions_pdf.merge(
            labels_pdf[['Customer_ID', 'label']].drop_duplicates('Customer_ID'),
            on=['Customer_ID'],
            how='inner',
            suffixes=('_pred', '_label')
        )
        
        # Rename columns for clarity
        monitoring_data = monitoring_data.rename(columns={'label': 'actual_label'})
        
        print(f"Merged monitoring data: {len(monitoring_data)} records")
        print(f"Date coverage: {monitoring_data['snapshot_date'].nunique()} unique dates")
                
        return monitoring_data
        
    except Exception as e:
        print(f"Error loading monitoring data: {str(e)}")
        return None


def calculate_performance_metrics_over_time(monitoring_data: pd.DataFrame, 
                                          config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Calculate model performance metrics over time periods
    
    Args:
        monitoring_data: DataFrame containing predictions and actual labels
        config: Configuration dictionary
        
    Returns:
        DataFrame containing performance metrics by time period or None if calculation fails
    """
    
    try:
        print(f"Calculating performance metrics over time...")
        
        # Group by snapshot date and calculate metrics
        performance_list = []
        
        for snapshot_date in sorted(monitoring_data['snapshot_date'].unique()):
            date_data = monitoring_data[monitoring_data['snapshot_date'] == snapshot_date].copy()
            
            if len(date_data) < 10:  # Skip if too few samples
                continue
                
            # Extract predictions and actual labels
            y_true = date_data['actual_label']
            y_pred_proba = date_data['model_prediction_proba']
            y_pred_binary = date_data['model_prediction_binary']
            
            # Calculate metrics
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
                gini = 2 * auc - 1
                
                # Additional metrics
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Prediction distribution metrics
                mean_pred = y_pred_proba.mean()
                std_pred = y_pred_proba.std()
                
                performance_list.append({
                    'snapshot_date': snapshot_date,
                    'sample_count': len(date_data),
                    'actual_default_rate': y_true.mean(),
                    'predicted_default_rate': y_pred_binary.mean(),
                    'auc': auc,
                    'gini': gini,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'mean_prediction': mean_pred,
                    'std_prediction': std_pred,
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn)
                })
                
            except Exception as e:
                print(f"Error calculating metrics for {snapshot_date}: {str(e)}")
                continue
        
        if not performance_list:
            print("No performance metrics could be calculated")
            return None
            
        performance_df = pd.DataFrame(performance_list)
        performance_df['snapshot_date'] = pd.to_datetime(performance_df['snapshot_date'])
        performance_df = performance_df.sort_values('snapshot_date')
        
        print(f"Performance metrics calculated for {len(performance_df)} time periods")
        return performance_df
        
    except Exception as e:
        print(f"Error calculating performance metrics: {str(e)}")
        return None


def analyze_data_stability(monitoring_data: pd.DataFrame, config: Dict[str, Any],
                          baseline_performance: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Analyze data stability and drift over monitoring period
    
    Args:
        monitoring_data: DataFrame containing monitoring data
        config: Configuration dictionary
        baseline_performance: Baseline performance for comparison
        
    Returns:
        Dictionary containing stability analysis results or None if analysis fails
    """
    
    try:
        print(f"Analyzing data stability and drift...")
        
        stability_results = {}
        
        # 1. Prediction distribution stability
        prediction_stability = analyze_prediction_stability(monitoring_data, baseline_performance)
        stability_results['prediction_stability'] = prediction_stability
        
        # 2. Model performance stability  
        model_stability = analyze_model_stability(monitoring_data, config)
        stability_results['model_stability'] = model_stability
        
        # 3. Prediction pattern analysis (as proxy for feature drift)
        # Note: This analyzes prediction distribution patterns, not individual features
        prediction_patterns = analyze_prediction_patterns(monitoring_data)
        stability_results['prediction_patterns'] = prediction_patterns
        
        # 4. Create feature drift summary for compatibility with existing code
        # Extract key metrics from prediction patterns for downstream processing
        feature_drift = {}
        if prediction_patterns and 'prediction_distribution_psi' in prediction_patterns:
            # Create a summary entry for overall prediction drift
            feature_drift['overall_prediction_drift'] = prediction_patterns['prediction_distribution_psi']
            
            # Add summary statistics as pseudo-features for monitoring display
            if 'summary' in prediction_patterns:
                summary = prediction_patterns['summary']
                feature_drift['prediction_stability_score'] = 1 - min(summary.get('avg_prediction_psi', 0), 1)
        
        stability_results['feature_drift'] = feature_drift
        
        # 5. Overall stability assessment
        overall_stability = assess_overall_stability(
            prediction_stability, model_stability, feature_drift, config
        )
        stability_results['overall_assessment'] = overall_stability
        
        print(f"Stability analysis completed")
        return stability_results
        
    except Exception as e:
        print(f"Error in stability analysis: {str(e)}")
        return None


def analyze_prediction_stability(monitoring_data: pd.DataFrame, 
                                baseline_performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze prediction distribution stability over time
    """
    
    try:
        # Calculate prediction statistics over time
        prediction_stats = []
        
        for snapshot_date in sorted(monitoring_data['snapshot_date'].unique()):
            date_data = monitoring_data[monitoring_data['snapshot_date'] == snapshot_date]
            
            prediction_stats.append({
                'date': snapshot_date,
                'mean': date_data['model_prediction_proba'].mean(),
                'std': date_data['model_prediction_proba'].std(),
                'min': date_data['model_prediction_proba'].min(),
                'max': date_data['model_prediction_proba'].max(),
                'q25': date_data['model_prediction_proba'].quantile(0.25),
                'q75': date_data['model_prediction_proba'].quantile(0.75)
            })
        
        stats_df = pd.DataFrame(prediction_stats)
        
        # Calculate drift metrics
        mean_drift = stats_df['mean'].std()
        std_drift = stats_df['std'].std()
        
        # Kolmogorov-Smirnov test between first and last periods
        if len(stats_df) >= 2:
            first_period = monitoring_data[monitoring_data['snapshot_date'] == stats_df['date'].iloc[0]]
            last_period = monitoring_data[monitoring_data['snapshot_date'] == stats_df['date'].iloc[-1]]
            
            ks_statistic, ks_p_value = stats.ks_2samp(
                first_period['model_prediction_proba'],
                last_period['model_prediction_proba']
            )
        else:
            ks_statistic, ks_p_value = 0, 1
        
        # Stability assessment
        is_stable = (mean_drift < 0.05) and (ks_statistic < 0.1)
        
        return {
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            'is_stable': is_stable,
            'prediction_stats': stats_df
        }
        
    except Exception as e:
        print(f"Error in prediction stability analysis: {str(e)}")
        return {}


def analyze_model_stability(monitoring_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze model performance stability over time
    """
    
    try:
        # Calculate performance metrics over time
        performance_over_time = []
        
        for snapshot_date in sorted(monitoring_data['snapshot_date'].unique()):
            date_data = monitoring_data[monitoring_data['snapshot_date'] == snapshot_date]
            
            if len(date_data) >= 10:  # Minimum samples for reliable AUC
                try:
                    auc = roc_auc_score(date_data['actual_label'], date_data['model_prediction_proba'])
                    performance_over_time.append({
                        'date': snapshot_date,
                        'auc': auc
                    })
                except:
                    continue
        
        if len(performance_over_time) < 2:
            return {'performance_volatility': 0, 'consistency_score': 1, 'is_stable': True}
        
        performance_df = pd.DataFrame(performance_over_time)
        
        # Calculate stability metrics
        performance_volatility = performance_df['auc'].std()
        mean_performance = performance_df['auc'].mean()
        consistency_score = 1 - (performance_volatility / mean_performance) if mean_performance > 0 else 0
        
        # Stability assessment
        is_stable = performance_volatility < 0.05  # Less than 5% volatility
        
        return {
            'performance_volatility': performance_volatility,
            'consistency_score': consistency_score,
            'is_stable': is_stable,
            'performance_trend': performance_df
        }
        
    except Exception as e:
        print(f"Error in model stability analysis: {str(e)}")
        return {}


def analyze_prediction_patterns(monitoring_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze prediction distribution patterns over time
    Returns prediction-level drift metrics, not feature-level metrics
    """
    
    try:
        # Analyze prediction distribution changes over time
        pattern_analysis = {}
        
        # Group data by time periods
        monitoring_data_copy = monitoring_data.copy()
        monitoring_data_copy['period'] = pd.to_datetime(monitoring_data_copy['snapshot_date'])
        monitoring_data_copy = monitoring_data_copy.sort_values('period')
        
        # Calculate prediction distribution bins
        prediction_bins = np.linspace(0, 1, 11)  # 10 bins
        
        period_distributions = []
        for snapshot_date in sorted(monitoring_data_copy['snapshot_date'].unique()):
            date_data = monitoring_data_copy[monitoring_data_copy['snapshot_date'] == snapshot_date]
            
            # Calculate histogram for predictions
            hist, _ = np.histogram(date_data['model_prediction_proba'], bins=prediction_bins)
            hist_normalized = hist / hist.sum() if hist.sum() > 0 else hist
            
            period_distributions.append({
                'date': snapshot_date,
                'distribution': hist_normalized
            })
        
        # Calculate PSI-like metric for prediction distribution drift
        if len(period_distributions) >= 2:
            baseline_dist = period_distributions[0]['distribution']
            
            psi_scores = []
            for period in period_distributions[1:]:
                current_dist = period['distribution']
                
                # Calculate PSI for prediction distribution
                psi = 0
                for i in range(len(baseline_dist)):
                    if baseline_dist[i] > 0 and current_dist[i] > 0:
                        psi += (current_dist[i] - baseline_dist[i]) * np.log(current_dist[i] / baseline_dist[i])
                
                psi_scores.append(psi)
            
            avg_psi = np.mean(psi_scores) if psi_scores else 0
        else:
            avg_psi = 0
        
        pattern_analysis['prediction_distribution_psi'] = avg_psi
        pattern_analysis['period_distributions'] = period_distributions
        
        # Add summary metrics for compatibility
        pattern_analysis['summary'] = {
            'avg_prediction_psi': avg_psi,
            'num_periods': len(period_distributions),
            'has_significant_drift': avg_psi > 0.2
        }
        
        return pattern_analysis
        
    except Exception as e:
        print(f"Error in prediction pattern analysis: {str(e)}")
        return {
            'prediction_distribution_psi': 0,
            'period_distributions': [],
            'summary': {'avg_prediction_psi': 0, 'num_periods': 0, 'has_significant_drift': False}
        }


def assess_overall_stability(prediction_stability: Dict, model_stability: Dict, 
                           feature_drift: Dict, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess overall model stability based on all analyses
    """
    
    stability_score = 0
    max_score = 3
    
    # Prediction stability (weight: 1)
    if prediction_stability.get('is_stable', False):
        stability_score += 1
    
    # Model performance stability (weight: 1)
    if model_stability.get('is_stable', False):
        stability_score += 1
    
    # Feature drift assessment (weight: 1)
    # Handle the new feature_drift structure
    drift_psi = feature_drift.get('overall_prediction_drift', 0)
    stability_threshold = config.get('stability_threshold', 0.2)
    
    if drift_psi < stability_threshold:
        stability_score += 1
    
    overall_stability_score = stability_score / max_score
    
    # Overall assessment
    if overall_stability_score >= 0.8:
        assessment = "STABLE"
    elif overall_stability_score >= 0.6:
        assessment = "MODERATE"
    else:
        assessment = "UNSTABLE"
    
    return {
        'stability_score': overall_stability_score,
        'assessment': assessment,
        'prediction_stable': prediction_stability.get('is_stable', False),
        'performance_stable': model_stability.get('is_stable', False),
        'distribution_stable': drift_psi < stability_threshold,
        'drift_metrics': {
            'overall_prediction_drift': drift_psi,
            'stability_threshold': stability_threshold
        }
    }


def generate_monitoring_visualizations(performance_metrics: pd.DataFrame, stability_analysis: Dict,
                                     monitoring_data: pd.DataFrame, baseline_performance: Optional[Dict],
                                     config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Generate visualizations for monitoring report
    
    Args:
        performance_metrics: DataFrame containing performance metrics over time
        stability_analysis: Dictionary containing stability analysis results
        monitoring_data: Raw monitoring data
        baseline_performance: Baseline performance metrics
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping chart names to file paths or None if generation fails
    """
    
    try:
        # Create output directory
        viz_output_path = config["visualizations_output_path"]
        if not os.path.exists(viz_output_path):
            os.makedirs(viz_output_path)
        
        visualization_paths = {}
        
        # 1. Performance Trend Chart
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(pd.to_datetime(performance_metrics['snapshot_date']), 
                performance_metrics['auc'], marker='o', linewidth=2, markersize=6)
        
        # Add baseline line if available
        if baseline_performance and 'auc_oot' in baseline_performance:
            plt.axhline(y=baseline_performance['auc_oot'], color='red', linestyle='--', 
                       label=f'Baseline OOT AUC: {baseline_performance["auc_oot"]:.3f}')
        
        plt.title('Model Performance Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('AUC')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        
        # 2. Gini Trend
        plt.subplot(1, 2, 2)
        plt.plot(pd.to_datetime(performance_metrics['snapshot_date']), 
                performance_metrics['gini'], marker='s', linewidth=2, markersize=6, color='green')
        
        if baseline_performance and 'gini_oot' in baseline_performance:
            plt.axhline(y=baseline_performance['gini_oot'], color='red', linestyle='--',
                       label=f'Baseline OOT Gini: {baseline_performance["gini_oot"]:.3f}')
        
        plt.title('Gini Coefficient Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Gini')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        performance_chart_path = os.path.join(viz_output_path, 'performance_trend.png')
        plt.savefig(performance_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['performance_trend'] = performance_chart_path
        
        # 3. Prediction Distribution Over Time
        plt.figure(figsize=(12, 8))
        
        # Box plot of predictions by date
        dates = sorted(monitoring_data['snapshot_date'].unique())
        prediction_data = [monitoring_data[monitoring_data['snapshot_date'] == date]['model_prediction_proba'] 
                          for date in dates]
        
        plt.boxplot(prediction_data, labels=[str(date)[:10] for date in dates])
        plt.title('Prediction Distribution Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Prediction Probability')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        distribution_chart_path = os.path.join(viz_output_path, 'prediction_distribution.png')
        plt.savefig(distribution_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['prediction_distribution'] = distribution_chart_path
        
        # 4. Performance vs Prediction Volume
        plt.figure(figsize=(10, 6))
        plt.scatter(performance_metrics['sample_count'], performance_metrics['auc'], 
                   s=100, alpha=0.7, c=range(len(performance_metrics)), cmap='viridis')
        plt.colorbar(label='Time Order')
        plt.xlabel('Sample Count')
        plt.ylabel('AUC')
        plt.title('Performance vs Sample Volume', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        volume_chart_path = os.path.join(viz_output_path, 'performance_vs_volume.png')
        plt.savefig(volume_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['performance_vs_volume'] = volume_chart_path
        
        print(f"Generated {len(visualization_paths)} monitoring charts")
        return visualization_paths
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        return None


def detect_performance_alerts(performance_metrics: pd.DataFrame, stability_analysis: Dict,
                            baseline_performance: Optional[Dict], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect performance degradation and generate alerts
    
    Args:
        performance_metrics: DataFrame containing performance metrics
        stability_analysis: Dictionary containing stability analysis
        baseline_performance: Baseline performance for comparison
        config: Configuration dictionary containing alert thresholds
        
    Returns:
        List of alert dictionaries
    """
    
    try:
        alerts = []
        alert_thresholds = config.get('alert_thresholds', {})
        
        # 1. Performance degradation alerts
        if baseline_performance and 'auc_oot' in baseline_performance:
            baseline_auc = baseline_performance['auc_oot']
            
            for _, row in performance_metrics.iterrows():
                current_auc = row['auc']
                auc_drop = (baseline_auc - current_auc) / baseline_auc
                
                if auc_drop >= alert_thresholds.get('auc_critical', 0.10):
                    alerts.append({
                        'type': 'performance_degradation',
                        'severity': 'critical',
                        'metric': 'auc',
                        'date': str(row['snapshot_date'])[:10],
                        'value': current_auc,
                        'baseline': baseline_auc,
                        'degradation_pct': auc_drop * 100,
                        'message': f"Critical AUC degradation: {auc_drop*100:.1f}% drop from baseline"
                    })
                    
                elif auc_drop >= alert_thresholds.get('auc_warning', 0.05):
                    alerts.append({
                        'type': 'performance_degradation',
                        'severity': 'warning',
                        'metric': 'auc',
                        'date': str(row['snapshot_date'])[:10],
                        'value': current_auc,
                        'baseline': baseline_auc,
                        'degradation_pct': auc_drop * 100,
                        'message': f"AUC degradation warning: {auc_drop*100:.1f}% drop from baseline"
                    })
        
        # 2. Stability alerts
        overall_assessment = stability_analysis.get('overall_assessment', {})
        
        if overall_assessment.get('assessment') == 'UNSTABLE':
            alerts.append({
                'type': 'stability',
                'severity': 'critical',
                'metric': 'stability_score',
                'date': str(performance_metrics['snapshot_date'].iloc[-1])[:10],
                'value': overall_assessment.get('stability_score', 0),
                'message': f"Model stability is unstable (score: {overall_assessment.get('stability_score', 0):.2f})"
            })
            
        elif overall_assessment.get('assessment') == 'MODERATE':
            alerts.append({
                'type': 'stability',
                'severity': 'warning',
                'metric': 'stability_score',
                'date': str(performance_metrics['snapshot_date'].iloc[-1])[:10],
                'value': overall_assessment.get('stability_score', 0),
                'message': f"Model stability is moderate (score: {overall_assessment.get('stability_score', 0):.2f})"
            })
        
        # 3. Prediction drift alerts
        prediction_stability = stability_analysis.get('prediction_stability', {})
        ks_statistic = prediction_stability.get('ks_statistic', 0)
        
        if ks_statistic >= 0.15:  # Significant distribution shift
            alerts.append({
                'type': 'distribution_drift',
                'severity': 'warning',
                'metric': 'ks_statistic',
                'date': str(performance_metrics['snapshot_date'].iloc[-1])[:10],
                'value': ks_statistic,
                'message': f"Prediction distribution drift detected (KS: {ks_statistic:.3f})"
            })
        
        # 4. Sample volume alerts
        recent_samples = performance_metrics['sample_count'].iloc[-1] if len(performance_metrics) > 0 else 0
        avg_samples = performance_metrics['sample_count'].mean() if len(performance_metrics) > 0 else 0
        
        if recent_samples < avg_samples * 0.5:  # 50% drop in volume
            alerts.append({
                'type': 'sample_volume',
                'severity': 'warning',
                'metric': 'sample_count',
                'date': str(performance_metrics['snapshot_date'].iloc[-1])[:10],
                'value': recent_samples,
                'baseline': avg_samples,
                'message': f"Low prediction volume: {recent_samples} vs avg {avg_samples:.0f}"
            })
        
        # Sort alerts by severity and date
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        alerts.sort(key=lambda x: (severity_order.get(x['severity'], 3), x['date']))
        
        print(f"Generated {len(alerts)} monitoring alerts")
        return alerts
        
    except Exception as e:
        print(f"Error detecting alerts: {str(e)}")
        return []


def save_monitoring_results(performance_metrics: pd.DataFrame, stability_analysis: Dict,
                          alerts: List[Dict], config: Dict[str, Any], spark) -> Optional[Dict[str, str]]:
    """
    Save monitoring results to datamart following medallion architecture
    
    Args:
        performance_metrics: DataFrame containing performance metrics
        stability_analysis: Dictionary containing stability analysis
        alerts: List of alert dictionaries
        config: Configuration dictionary
        spark: SparkSession object
        
    Returns:
        Dictionary containing output paths or None if saving fails
    """
    
    try:
        output_paths = {}
        
        # 1. Save performance metrics
        performance_output_path = config["performance_output_path"]
        if not os.path.exists(performance_output_path):
            os.makedirs(performance_output_path)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        performance_file = f"performance_metrics_{config['monitoring_start_date']}_{config['monitoring_end_date']}_{timestamp}.parquet"
        performance_filepath = os.path.join(performance_output_path, performance_file)
        
        # Save as Spark DataFrame
        performance_sdf = spark.createDataFrame(performance_metrics)
        performance_sdf.write.mode("overwrite").parquet(performance_filepath)
        output_paths['performance_metrics'] = performance_filepath
        
        # 2. Save stability analysis
        stability_output_path = config["stability_output_path"]
        if not os.path.exists(stability_output_path):
            os.makedirs(stability_output_path)
        
        stability_file = f"stability_analysis_{config['monitoring_start_date']}_{config['monitoring_end_date']}_{timestamp}.pkl"
        stability_filepath = os.path.join(stability_output_path, stability_file)
        
        with open(stability_filepath, 'wb') as f:
            pickle.dump(stability_analysis, f)
        output_paths['stability_analysis'] = stability_filepath
        
        # 3. Save alerts
        if alerts:
            alerts_output_path = config["alerts_output_path"]
            if not os.path.exists(alerts_output_path):
                os.makedirs(alerts_output_path)
            
            alerts_df = pd.DataFrame(alerts)
            alerts_file = f"alerts_{config['monitoring_start_date']}_{config['monitoring_end_date']}_{timestamp}.parquet"
            alerts_filepath = os.path.join(alerts_output_path, alerts_file)
            
            alerts_sdf = spark.createDataFrame(alerts_df)
            alerts_sdf.write.mode("overwrite").parquet(alerts_filepath)
            output_paths['alerts'] = alerts_filepath
        
        print(f"Monitoring results saved successfully")
        return output_paths
        
    except Exception as e:
        print(f"Error saving monitoring results: {str(e)}")
        return None


def save_monitoring_summary(performance_metrics: pd.DataFrame, stability_analysis: Dict,
                          alerts: List[Dict], baseline_performance: Optional[Dict],
                          config: Dict[str, Any]) -> Optional[str]:
    """
    Save monitoring summary report
    """
    
    try:
        reports_output_path = config["reports_output_path"]
        if not os.path.exists(reports_output_path):
            os.makedirs(reports_output_path)
        
        # Create summary
        summary = {
            'model_version': config['model_version'],
            'monitoring_period': {
                'start_date': config['monitoring_start_date'],
                'end_date': config['monitoring_end_date']
            },
            'performance_summary': {
                'periods_analyzed': len(performance_metrics),
                'avg_auc': performance_metrics['auc'].mean(),
                'min_auc': performance_metrics['auc'].min(),
                'max_auc': performance_metrics['auc'].max(),
                'auc_volatility': performance_metrics['auc'].std()
            },
            'stability_summary': stability_analysis.get('overall_assessment', {}),
            'alerts_summary': {
                'total_alerts': len(alerts),
                'critical_alerts': len([a for a in alerts if a.get('severity') == 'critical']),
                'warning_alerts': len([a for a in alerts if a.get('severity') == 'warning'])
            },
            'baseline_comparison': baseline_performance,
            'generated_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save summary
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        summary_file = f"monitoring_summary_{config['monitoring_start_date']}_{config['monitoring_end_date']}_{timestamp}.pkl"
        summary_filepath = os.path.join(reports_output_path, summary_file)
        
        with open(summary_filepath, 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"Monitoring summary saved to: {summary_filepath}")
        return summary_filepath
        
    except Exception as e:
        print(f"Error saving monitoring summary: {str(e)}")
        return None


def run_batch_monitoring(models_list: List[str], monitoring_start_date: str, monitoring_end_date: str,
                        spark, predictions_path: str = "datamart/gold/model_predictions/",
                        labels_path: str = "datamart/gold/label_store/",
                        monitoring_output_path: str = "datamart/gold/model_monitoring/",
                        model_bank_directory: str = "models/") -> List[Dict[str, Any]]:
    """
    Run monitoring for multiple models in batch
    
    Args:
        models_list: List of model versions to monitor
        monitoring_start_date: Start date for monitoring period
        monitoring_end_date: End date for monitoring period
        spark: SparkSession object
        predictions_path: Path to predictions datamart
        labels_path: Path to labels datamart
        monitoring_output_path: Path to save monitoring results
        model_bank_directory: Path to model bank
        
    Returns:
        List of monitoring results for each model
    """
    
    results = []
    
    print(f"Starting batch monitoring for {len(models_list)} models")
    print(f"Monitoring period: {monitoring_start_date} to {monitoring_end_date}")
    
    for i, model_version in enumerate(models_list, 1):
        print(f"\nProcessing model {i}/{len(models_list)}: {model_version}")
        
        try:
            # Build configuration for this model
            config = build_monitoring_config(
                model_version=model_version,
                monitoring_start_date=monitoring_start_date,
                monitoring_end_date=monitoring_end_date,
                predictions_path=predictions_path,
                labels_path=labels_path,
                monitoring_output_path=monitoring_output_path,
                model_bank_directory=model_bank_directory
            )
            
            # Load baseline
            baseline = load_training_baseline(config)
            
            # Load monitoring data
            monitoring_data = load_monitoring_data(config, spark)
            
            if monitoring_data is None or len(monitoring_data) == 0:
                results.append({
                    'model': model_version,
                    'success': False,
                    'error': 'No monitoring data found'
                })
                continue
            
            # Calculate performance metrics
            performance_metrics = calculate_performance_metrics_over_time(monitoring_data, config)
            
            if performance_metrics is None:
                results.append({
                    'model': model_version,
                    'success': False,
                    'error': 'Failed to calculate performance metrics'
                })
                continue
            
            # Analyze stability
            stability_analysis = analyze_data_stability(monitoring_data, config, baseline)
            
            # Detect alerts
            alerts = detect_performance_alerts(performance_metrics, stability_analysis, baseline, config)
            
            # Save results
            output_paths = save_monitoring_results(performance_metrics, stability_analysis, alerts, config, spark)
            
            results.append({
                'model': model_version,
                'success': True,
                'monitoring_periods': len(performance_metrics),
                'avg_auc': performance_metrics['auc'].mean(),
                'alerts': alerts,
                'output_paths': output_paths
            })
            
            print(f"✓ Successfully monitored {model_version}")
            
        except Exception as e:
            print(f"✗ Failed to monitor {model_version}: {str(e)}")
            results.append({
                'model': model_version,
                'success': False,
                'error': str(e)
            })
    
    return results


def generate_monitoring_report(model_version: str, performance_metrics: pd.DataFrame,
                             stability_analysis: Dict, alerts: List[Dict], 
                             baseline_performance: Optional[Dict],
                             visualization_paths: Optional[Dict],
                             config: Dict[str, Any]) -> Optional[str]:
    """
    Generate comprehensive monitoring report
    """
    
    try:
        reports_output_path = config["reports_output_path"]
        if not os.path.exists(reports_output_path):
            os.makedirs(reports_output_path)
        
        # Create report content
        report_content = f"""
# Model Monitoring Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model Version: {model_version}
- Monitoring Period: {config['monitoring_start_date']} to {config['monitoring_end_date']}
- Analysis Periods: {len(performance_metrics)}

## Executive Summary
"""
        
        # Performance summary
        avg_auc = performance_metrics['auc'].mean()
        min_auc = performance_metrics['auc'].min()
        max_auc = performance_metrics['auc'].max()
        
        if baseline_performance and 'auc_oot' in baseline_performance:
            baseline_auc = baseline_performance['auc_oot']
            performance_change = ((avg_auc - baseline_auc) / baseline_auc) * 100
            
            report_content += f"""
- Average AUC: {avg_auc:.4f}
- Performance vs Baseline: {performance_change:+.1f}%
- AUC Range: {min_auc:.4f} - {max_auc:.4f}
"""
        else:
            report_content += f"""
- Average AUC: {avg_auc:.4f}
- AUC Range: {min_auc:.4f} - {max_auc:.4f}
"""
        
        # Stability summary
        overall_assessment = stability_analysis.get('overall_assessment', {})
        stability_status = overall_assessment.get('assessment', 'Unknown')
        
        report_content += f"""

## Stability Assessment
- Overall Status: {stability_status}
- Stability Score: {overall_assessment.get('stability_score', 0):.2f}
"""
        
        # Alerts summary
        critical_alerts = len([a for a in alerts if a.get('severity') == 'critical'])
        warning_alerts = len([a for a in alerts if a.get('severity') == 'warning'])
        
        report_content += f"""

## Alerts Summary
- Total Alerts: {len(alerts)}
- Critical: {critical_alerts}
- Warnings: {warning_alerts}
"""
        
        # Recommendations
        report_content += """

## Recommendations
"""
        
        if critical_alerts > 0:
            report_content += "- **IMMEDIATE ACTION REQUIRED**: Critical performance degradation detected\n"
        
        if stability_status == 'UNSTABLE':
            report_content += "- Consider model retraining due to instability\n"
        
        if stability_status == 'STABLE' and critical_alerts == 0:
            report_content += "- Model performance is stable - continue monitoring\n"
        
        # Save report
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        report_file = f"monitoring_report_{model_version}_{timestamp}.md"
        report_filepath = os.path.join(reports_output_path, report_file)
        
        with open(report_filepath, 'w') as f:
            f.write(report_content)
        
        print(f"Monitoring report generated: {report_filepath}")
        return report_filepath
        
    except Exception as e:
        print(f"Error generating monitoring report: {str(e)}")
        return None


# Airflow-compatible task functions
def monitoring_airflow_task(model_version: str, monitoring_start_date: str, 
                          monitoring_end_date: str, **context) -> str:
    """
    Airflow-compatible monitoring task
    
    Args:
        model_version: Model version to monitor
        monitoring_start_date: Start date for monitoring period
        monitoring_end_date: End date for monitoring period
        context: Airflow context
        
    Returns:
        Success message with monitoring summary
    """
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName(f"Monitoring_{model_version}") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        print(f"[AIRFLOW] Starting monitoring task for {model_version}")
        
        # Build configuration
        config = build_monitoring_config(
            model_version=model_version,
            monitoring_start_date=monitoring_start_date,
            monitoring_end_date=monitoring_end_date,
            predictions_path="/opt/airflow/datamart/gold/model_predictions/",
            labels_path="/opt/airflow/datamart/gold/label_store/",
            monitoring_output_path="/opt/airflow/datamart/gold/model_monitoring/",
            model_bank_directory="/opt/airflow/models/"
        )
        
        # Load baseline and monitoring data
        baseline = load_training_baseline(config)
        monitoring_data = load_monitoring_data(config, spark)
        
        if monitoring_data is None:
            raise ValueError("No monitoring data found")
        
        # Run monitoring analysis
        performance_metrics = calculate_performance_metrics_over_time(monitoring_data, config)
        stability_analysis = analyze_data_stability(monitoring_data, config, baseline)
        alerts = detect_performance_alerts(performance_metrics, stability_analysis, baseline, config)
        
        # Save results
        output_paths = save_monitoring_results(performance_metrics, stability_analysis, alerts, config, spark)
        
        # Create summary message
        critical_alerts = len([a for a in alerts if a.get('severity') == 'critical'])
        avg_auc = performance_metrics['auc'].mean() if len(performance_metrics) > 0 else 0
        
        message = f"Monitoring completed: {len(performance_metrics)} periods analyzed, "
        message += f"AUC avg: {avg_auc:.4f}, {len(alerts)} alerts ({critical_alerts} critical)"
        
        print(f"[AIRFLOW] {message}")
        return message
        
    finally:
        try:
            if 'spark' in locals() and spark is not None:
                print("[SPARK CLEANUP] Stopping Spark session...")
                spark.stop()
                print("[SPARK CLEANUP] ✓ Spark session stopped successfully")
        except Exception as e:
            print(f"[SPARK CLEANUP] Warning: Error stopping Spark session: {e}")
            print("[SPARK CLEANUP] Training completed successfully despite cleanup warning")
            try:
                from pyspark.sql import SparkSession
                SparkSession._instantiatedSession = None
                SparkSession._activeSession = None
                print("[SPARK CLEANUP] ✓ Alternative cleanup completed")
            except:
                print("[SPARK CLEANUP] Alternative cleanup failed, proceeding anyway")



if __name__ == "__main__":
    # Example usage and testing
    print("Model Monitoring Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")
