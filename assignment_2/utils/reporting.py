"""
Pipeline Reporting Utilities Module
Provides production-ready reporting functionality for ML pipelines

This module handles:
- Comprehensive pipeline execution reporting
- Cross-task result aggregation and analysis
- Performance trend analysis and insights
- Executive summary generation
- Actionable recommendations based on pipeline results
- Airflow-compatible reporting tasks
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def build_reporting_config(execution_date_str: str,
                          report_output_path: str = "datamart/gold/pipeline_reports/",
                          report_retention_days: int = 90,
                          trend_analysis_periods: int = 6) -> Dict[str, Any]:
    """
    Build configuration dictionary for pipeline reporting
    
    Args:
        execution_date_str: Execution date in YYYY-MM-DD format
        report_output_path: Path to save generated reports
        report_retention_days: Number of days to retain reports
        trend_analysis_periods: Number of periods for trend analysis
        
    Returns:
        Configuration dictionary containing all reporting parameters
    """
    
    execution_date = datetime.strptime(execution_date_str, "%Y-%m-%d")
    
    config = {
        # Execution details
        "execution_date_str": execution_date_str,
        "execution_date": execution_date,
        "report_generation_timestamp": datetime.now(),
        
        # Report configuration
        "report_output_path": report_output_path,
        "report_retention_days": report_retention_days,
        "trend_analysis_periods": trend_analysis_periods,
        
        # Report structure
        "report_sections": [
            "executive_summary",
            "data_validation_results", 
            "training_results",
            "inference_results",
            "monitoring_results",
            "performance_trends",
            "recommendations",
            "next_actions"
        ]
    }
    
    return config


def analyze_data_validation_results(validation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze data validation results for reporting
    
    Args:
        validation_results: Results from data validation task
        
    Returns:
        Dictionary containing analyzed validation insights
    """
    
    try:
        if not validation_results:
            return {
                'section': 'data_validation',
                'status': 'no_data',
                'summary': 'No data validation results available',
                'insights': [],
                'concerns': ['Data validation task did not execute']
            }
        
        analysis = {
            'section': 'data_validation',
            'status': validation_results.get('overall_status', 'unknown'),
            'overall_score': validation_results.get('overall_score', 0.0),
            'validation_passed': validation_results.get('validation_passed', False),
            'insights': [],
            'concerns': []
        }
        
        # Analyze feature store results
        feature_store = validation_results.get('store_results', {}).get('feature_store', {})
        if feature_store:
            feature_quality = feature_store.get('quality_metrics', {})
            if feature_quality.get('feature_completeness', 0) < 0.8:
                analysis['concerns'].append(
                    f"Feature completeness low: {feature_quality.get('feature_completeness', 0):.2f}"
                )
            else:
                analysis['insights'].append("Feature store quality is acceptable")
        
        # Analyze label store results
        label_store = validation_results.get('store_results', {}).get('label_store', {})
        if label_store:
            label_quality = label_store.get('quality_metrics', {})
            default_rate = label_quality.get('default_rate', 0)
            if 0.01 <= default_rate <= 0.15:  # Reasonable default rate range
                analysis['insights'].append(f"Label distribution is balanced (default rate: {default_rate:.3f})")
            else:
                analysis['concerns'].append(f"Unusual default rate detected: {default_rate:.3f}")
        
        # Generate summary
        if analysis['validation_passed']:
            analysis['summary'] = f"Data validation passed with score {analysis['overall_score']:.2f}"
        else:
            analysis['summary'] = f"Data validation failed with score {analysis['overall_score']:.2f}"
        
        return analysis
        
    except Exception as e:
        return {
            'section': 'data_validation',
            'status': 'analysis_error',
            'summary': f"Error analyzing data validation results: {str(e)}",
            'insights': [],
            'concerns': ['Failed to analyze data validation results']
        }


def analyze_training_results(training_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze model training results for reporting
    
    Args:
        training_results: Results from model training task
        
    Returns:
        Dictionary containing analyzed training insights
    """
    
    try:
        if not training_results:
            return {
                'section': 'model_training',
                'status': 'no_data',
                'summary': 'No training results available',
                'insights': [],
                'concerns': ['Training task did not execute']
            }
        
        analysis = {
            'section': 'model_training',
            'training_executed': training_results.get('training_executed', False),
            'success': training_results.get('success', False),
            'insights': [],
            'concerns': []
        }
        
        if not analysis['training_executed']:
            analysis['status'] = 'skipped'
            analysis['summary'] = 'Training was skipped - using existing models'
            analysis['insights'].append('Training skipped according to schedule')
            return analysis
        
        if analysis['success']:
            analysis['status'] = 'success'
            models_trained = training_results.get('models_trained', [])
            analysis['summary'] = f"Successfully trained {len(models_trained)} models"
            analysis['insights'].append(f"Models trained: {', '.join(models_trained)}")
        else:
            analysis['status'] = 'failed'
            error_msg = training_results.get('error', 'Unknown error')
            analysis['summary'] = f"Training failed: {error_msg}"
            analysis['concerns'].append(f"Training error: {error_msg}")
        
        return analysis
        
    except Exception as e:
        return {
            'section': 'model_training',
            'status': 'analysis_error',
            'summary': f"Error analyzing training results: {str(e)}",
            'insights': [],
            'concerns': ['Failed to analyze training results']
        }


def analyze_inference_results(inference_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze model inference results for reporting
    
    Args:
        inference_results: Results from model inference task
        
    Returns:
        Dictionary containing analyzed inference insights
    """
    
    try:
        if not inference_results:
            return {
                'section': 'model_inference',
                'status': 'no_data',
                'summary': 'No inference results available',
                'insights': [],
                'concerns': ['Inference task did not execute']
            }
        
        analysis = {
            'section': 'model_inference',
            'status': 'unknown',
            'insights': [],
            'concerns': []
        }
        
        successful_models = inference_results.get('successful_models', [])
        failed_models = inference_results.get('failed_models', [])
        success_rate = inference_results.get('success_rate', 0.0)
        
        total_models = len(successful_models) + len(failed_models)
        
        if success_rate >= 0.8:
            analysis['status'] = 'excellent'
            analysis['summary'] = f"Inference highly successful: {len(successful_models)}/{total_models} models"
        elif success_rate >= 0.5:
            analysis['status'] = 'partial_success'
            analysis['summary'] = f"Inference partially successful: {len(successful_models)}/{total_models} models"
        else:
            analysis['status'] = 'failed'
            analysis['summary'] = f"Inference largely failed: {len(successful_models)}/{total_models} models"
        
        # Add insights
        if successful_models:
            analysis['insights'].append(f"Successfully processed models: {', '.join(successful_models)}")
        
        if failed_models:
            failed_model_names = [f['model'] if isinstance(f, dict) else str(f) for f in failed_models]
            analysis['concerns'].append(f"Failed models: {', '.join(failed_model_names)}")
        
        return analysis
        
    except Exception as e:
        return {
            'section': 'model_inference',
            'status': 'analysis_error', 
            'summary': f"Error analyzing inference results: {str(e)}",
            'insights': [],
            'concerns': ['Failed to analyze inference results']
        }


def analyze_monitoring_results(monitoring_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze model monitoring results for reporting
    
    Args:
        monitoring_results: Results from model monitoring task
        
    Returns:
        Dictionary containing analyzed monitoring insights
    """
    
    try:
        if not monitoring_results:
            return {
                'section': 'model_monitoring',
                'status': 'no_data',
                'summary': 'No monitoring results available',
                'insights': [],
                'concerns': ['Monitoring task did not execute']
            }
        
        analysis = {
            'section': 'model_monitoring',
            'monitoring_executed': monitoring_results.get('monitoring_executed', True),
            'insights': [],
            'concerns': []
        }
        
        if not analysis['monitoring_executed']:
            analysis['status'] = 'skipped'
            analysis['summary'] = 'Monitoring was skipped'
            return analysis
        
        models_monitored = monitoring_results.get('models_monitored', [])
        critical_alerts = monitoring_results.get('critical_alerts', [])
        warning_alerts = monitoring_results.get('warning_alerts', [])
        
        if len(critical_alerts) > 0:
            analysis['status'] = 'critical_issues'
            analysis['summary'] = f"Critical issues detected in {len(critical_alerts)} cases"
            analysis['concerns'].extend([f"Critical: {alert}" for alert in critical_alerts])
        elif len(warning_alerts) > 0:
            analysis['status'] = 'warnings'
            analysis['summary'] = f"Warnings detected for {len(warning_alerts)} models"
            analysis['concerns'].extend([f"Warning: {alert}" for alert in warning_alerts])
        else:
            analysis['status'] = 'healthy'
            analysis['summary'] = f"All {len(models_monitored)} models are performing well"
            analysis['insights'].append("No performance or stability issues detected")
        
        return analysis
        
    except Exception as e:
        return {
            'section': 'model_monitoring',
            'status': 'analysis_error',
            'summary': f"Error analyzing monitoring results: {str(e)}",
            'insights': [],
            'concerns': ['Failed to analyze monitoring results']
        }


def generate_executive_summary(all_analyses: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate executive summary from all pipeline analyses
    
    Args:
        all_analyses: List of analysis results from all pipeline components
        config: Reporting configuration
        
    Returns:
        Dictionary containing executive summary
    """
    
    try:
        executive_summary = {
            'section': 'executive_summary',
            'execution_date': config['execution_date_str'],
            'pipeline_status': 'unknown',
            'overall_health_score': 0.0,
            'key_achievements': [],
            'priority_issues': [],
            'business_impact': 'neutral'
        }
        
        # Calculate overall health score
        status_scores = {
            'excellent': 1.0, 'success': 1.0, 'healthy': 1.0,
            'good': 0.8, 'partial_success': 0.6, 'warnings': 0.6,
            'warning': 0.4, 'failed': 0.2, 'critical_issues': 0.2,
            'critical': 0.0, 'no_data': 0.0, 'analysis_error': 0.0,
            'skipped': 0.7  # Neutral score for intentionally skipped tasks
        }
        
        scores = []
        critical_issues = []
        achievements = []
        
        for analysis in all_analyses:
            status = analysis.get('status', 'unknown')
            score = status_scores.get(status, 0.3)  # Default score for unknown status
            scores.append(score)
            
            # Collect critical issues
            if status in ['critical', 'failed', 'critical_issues']:
                critical_issues.extend(analysis.get('concerns', []))
            
            # Collect achievements  
            if status in ['excellent', 'success', 'healthy']:
                achievements.extend(analysis.get('insights', []))
        
        # Calculate overall health score
        if scores:
            overall_health_score = np.mean(scores)
        else:
            overall_health_score = 0.0
        
        executive_summary['overall_health_score'] = float(overall_health_score)
        
        # Determine overall pipeline status
        if overall_health_score >= 0.8:
            executive_summary['pipeline_status'] = 'excellent'
            executive_summary['business_impact'] = 'positive'
        elif overall_health_score >= 0.6:
            executive_summary['pipeline_status'] = 'good'
            executive_summary['business_impact'] = 'neutral'
        elif overall_health_score >= 0.4:
            executive_summary['pipeline_status'] = 'concerning'
            executive_summary['business_impact'] = 'at_risk'
        else:
            executive_summary['pipeline_status'] = 'critical'
            executive_summary['business_impact'] = 'negative'
        
        # Compile key achievements and priority issues
        executive_summary['key_achievements'] = achievements[:5]  # Top 5 achievements
        executive_summary['priority_issues'] = critical_issues[:5]  # Top 5 issues
        
        return executive_summary
        
    except Exception as e:
        return {
            'section': 'executive_summary',
            'pipeline_status': 'analysis_error',
            'overall_health_score': 0.0,
            'error': str(e),
            'key_achievements': [],
            'priority_issues': [f"Failed to generate executive summary: {str(e)}"]
        }


def generate_actionable_recommendations(all_analyses: List[Dict[str, Any]], 
                                      executive_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate actionable recommendations based on pipeline analysis
    
    Args:
        all_analyses: List of analysis results from all pipeline components
        executive_summary: Executive summary of pipeline execution
        
    Returns:
        Dictionary containing actionable recommendations
    """
    
    try:
        recommendations = {
            'section': 'recommendations',
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_optimizations': [],
            'priority_level': 'low'
        }
        
        pipeline_status = executive_summary.get('pipeline_status', 'unknown')
        
        # Collect all concerns from analyses
        all_concerns = []
        for analysis in all_analyses:
            all_concerns.extend(analysis.get('concerns', []))
        
        # Generate immediate actions for critical issues
        if pipeline_status in ['critical', 'concerning']:
            recommendations['priority_level'] = 'high'
            recommendations['immediate_actions'].append(
                "Investigate pipeline failures before next execution"
            )
            
            # Add specific immediate actions based on concerns
            if any('data validation' in concern.lower() for concern in all_concerns):
                recommendations['immediate_actions'].append(
                    "Fix data quality issues in feature and label stores"
                )
            
            if any('training' in concern.lower() for concern in all_concerns):
                recommendations['immediate_actions'].append(
                    "Review training process and address model training failures"
                )
            
            if any('inference' in concern.lower() for concern in all_concerns):
                recommendations['immediate_actions'].append(
                    "Check model artifacts and inference pipeline integrity"
                )
        
        # Generate short-term improvements
        if pipeline_status in ['good', 'concerning']:
            recommendations['short_term_improvements'].extend([
                "Implement automated alerting for pipeline failures",
                "Add more comprehensive monitoring metrics",
                "Review and optimize resource allocation"
            ])
        
        # Generate long-term optimizations
        recommendations['long_term_optimizations'].extend([
            "Consider implementing A/B testing for model deployments",
            "Explore automated model retraining triggers",
            "Implement advanced feature engineering techniques",
            "Consider model ensemble strategies for improved performance"
        ])
        
        # Adjust priority level based on concerns
        if len(all_concerns) > 5:
            recommendations['priority_level'] = 'high'
        elif len(all_concerns) > 2:
            recommendations['priority_level'] = 'medium'
        
        return recommendations
        
    except Exception as e:
        return {
            'section': 'recommendations',
            'immediate_actions': [f"Fix recommendation generation error: {str(e)}"],
            'short_term_improvements': [],
            'long_term_optimizations': [],
            'priority_level': 'high',
            'error': str(e)
        }


def compile_comprehensive_report(validation_results: Dict[str, Any], training_results: Dict[str, Any],
                               inference_results: Dict[str, Any], monitoring_results: Dict[str, Any],
                               config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compile comprehensive pipeline execution report
    
    Args:
        validation_results: Data validation results
        training_results: Model training results
        inference_results: Model inference results
        monitoring_results: Model monitoring results
        config: Reporting configuration
        
    Returns:
        Comprehensive pipeline report
    """
    
    try:
        print(f"Compiling comprehensive pipeline report...")
        
        # Analyze each component
        validation_analysis = analyze_data_validation_results(validation_results)
        training_analysis = analyze_training_results(training_results)
        inference_analysis = analyze_inference_results(inference_results)
        monitoring_analysis = analyze_monitoring_results(monitoring_results)
        
        all_analyses = [validation_analysis, training_analysis, inference_analysis, monitoring_analysis]
        
        # Generate executive summary
        executive_summary = generate_executive_summary(all_analyses, config)
        
        # Generate recommendations
        recommendations = generate_actionable_recommendations(all_analyses, executive_summary)
        
        # Compile final report
        comprehensive_report = {
            'report_metadata': {
                'execution_date': config['execution_date_str'],
                'report_generation_time': config['report_generation_timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'report_version': '1.0',
                'pipeline_components_analyzed': len(all_analyses)
            },
            'executive_summary': executive_summary,
            'component_analyses': {
                'data_validation': validation_analysis,
                'model_training': training_analysis,
                'model_inference': inference_analysis,
                'model_monitoring': monitoring_analysis
            },
            'recommendations': recommendations,
            'report_status': 'completed'
        }
        
        print(f"Comprehensive report compiled successfully")
        print(f"Pipeline status: {executive_summary['pipeline_status']}")
        print(f"Overall health score: {executive_summary['overall_health_score']:.3f}")
        
        return comprehensive_report
        
    except Exception as e:
        print(f"Error compiling comprehensive report: {str(e)}")
        return {
            'report_metadata': {
                'execution_date': config['execution_date_str'],
                'report_generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'report_status': 'error'
            },
            'error': str(e),
            'executive_summary': {
                'pipeline_status': 'report_generation_failed',
                'overall_health_score': 0.0
            }
        }


def save_pipeline_report(report: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """
    Save pipeline report to storage
    
    Args:
        report: Comprehensive pipeline report
        config: Reporting configuration
        
    Returns:
        Path to saved report or None if saving fails
    """
    
    try:
        report_output_path = config["report_output_path"]
        execution_date_str = config["execution_date_str"]
        
        # Create output directory if it doesn't exist
        if not os.path.exists(report_output_path):
            os.makedirs(report_output_path)
        
        # Generate report filename
        timestamp = config["report_generation_timestamp"].strftime("%Y%m%d_%H%M%S")
        report_filename = f"pipeline_report_{execution_date_str.replace('-', '_')}_{timestamp}.pkl"
        report_filepath = os.path.join(report_output_path, report_filename)
        
        # Save report
        with open(report_filepath, 'wb') as f:
            pickle.dump(report, f)
        
        print(f"Pipeline report saved to: {report_filepath}")
        return report_filepath
        
    except Exception as e:
        print(f"Error saving pipeline report: {str(e)}")
        return None


def generate_comprehensive_pipeline_report_airflow_task(**context) -> Dict[str, Any]:
    """
    Airflow-compatible pipeline report generation task
    
    Args:
        context: Airflow context
        
    Returns:
        Comprehensive pipeline report
    """
    
    try:
        print(f"[AIRFLOW REPORTING] Starting report generation for {context['ds']}")
        
        # Build reporting configuration
        config = build_reporting_config(
            execution_date_str=context['ds'],
            report_output_path="/opt/airflow/datamart/gold/pipeline_reports/"
        )
        
        # Retrieve results from previous tasks using XCom
        ti = context['ti']
        
        validation_results = ti.xcom_pull(task_ids='data_validation_group.validate_data_stores')
        training_results = ti.xcom_pull(task_ids='model_training_group.training_completed')
        inference_results = ti.xcom_pull(task_ids='model_inference_group.inference_completed')
        monitoring_results = ti.xcom_pull(task_ids='model_monitoring_group.monitoring_completed')
        
        # Compile comprehensive report
        comprehensive_report = compile_comprehensive_report(
            validation_results=validation_results,
            training_results=training_results,
            inference_results=inference_results,
            monitoring_results=monitoring_results,
            config=config
        )
        
        # Save report
        report_path = save_pipeline_report(comprehensive_report, config)
        
        # Add file path to report metadata
        if report_path:
            comprehensive_report['report_metadata']['saved_to'] = report_path
        
        pipeline_status = comprehensive_report['executive_summary']['pipeline_status']
        health_score = comprehensive_report['executive_summary']['overall_health_score']
        
        print(f"[AIRFLOW REPORTING] Report generation completed")
        print(f"[AIRFLOW REPORTING] Pipeline status: {pipeline_status}")
        print(f"[AIRFLOW REPORTING] Health score: {health_score:.3f}")
        
        return comprehensive_report
        
    except Exception as e:
        print(f"[AIRFLOW REPORTING] Report generation failed: {str(e)}")
        return {
            'report_metadata': {
                'execution_date': context['ds'],
                'report_status': 'error'
            },
            'error': str(e),
            'executive_summary': {
                'pipeline_status': 'report_generation_failed',
                'overall_health_score': 0.0
            }
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Pipeline Reporting Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")
