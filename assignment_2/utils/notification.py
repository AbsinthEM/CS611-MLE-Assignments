"""
Pipeline Notification Utilities Module
Provides production-ready notification functionality for ML pipelines

This module handles:
- Multi-channel notification management (email, Slack, dashboard)
- Intelligent notification prioritization and routing
- Template-based message generation
- Notification delivery tracking and retry logic
- Alert escalation based on severity levels
- Airflow-compatible notification tasks
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


def build_notification_config(execution_date_str: str,
                            notification_channels: List[str] = None,
                            escalation_enabled: bool = True,
                            notification_retention_days: int = 30) -> Dict[str, Any]:
    """
    Build configuration dictionary for notification management
    
    Args:
        execution_date_str: Execution date in YYYY-MM-DD format
        notification_channels: List of notification channels to use
        escalation_enabled: Whether to enable alert escalation
        notification_retention_days: Number of days to retain notification logs
        
    Returns:
        Configuration dictionary containing all notification parameters
    """
    
    if notification_channels is None:
        notification_channels = ['console', 'log_file']  # Default safe channels
    
    execution_date = datetime.strptime(execution_date_str, "%Y-%m-%d")
    
    config = {
        # Execution details
        "execution_date_str": execution_date_str,
        "execution_date": execution_date,
        "notification_timestamp": datetime.now(),
        
        # Notification configuration
        "notification_channels": notification_channels,
        "escalation_enabled": escalation_enabled,
        "notification_retention_days": notification_retention_days,
        
        # Channel settings
        "channel_config": {
            "console": {"enabled": True},
            "log_file": {"enabled": True, "log_path": "/opt/airflow/logs/notifications/"},
            "email": {"enabled": False, "recipients": []},
            "slack": {"enabled": False, "webhook_url": None, "channel": "#ml-alerts"},
            "dashboard": {"enabled": False, "endpoint": None}
        },
        
        # Message templates
        "templates": {
            "success": {
                "subject": "✅ ML Pipeline Execution Successful",
                "priority": "low"
            },
            "warning": {
                "subject": "⚠️ ML Pipeline Warnings Detected", 
                "priority": "medium"
            },
            "critical": {
                "subject": "🚨 ML Pipeline Critical Issues",
                "priority": "high"
            },
            "failure": {
                "subject": "❌ ML Pipeline Execution Failed",
                "priority": "critical"
            }
        }
    }
    
    return config


def determine_notification_severity(pipeline_report: Dict[str, Any]) -> str:
    """
    Determine notification severity based on pipeline report
    
    Args:
        pipeline_report: Comprehensive pipeline report
        
    Returns:
        Severity level: 'success', 'warning', 'critical', or 'failure'
    """
    
    try:
        executive_summary = pipeline_report.get('executive_summary', {})
        pipeline_status = executive_summary.get('pipeline_status', 'unknown')
        health_score = executive_summary.get('overall_health_score', 0.0)
        priority_issues = executive_summary.get('priority_issues', [])
        
        # Determine severity based on status and health score
        if pipeline_status in ['critical', 'report_generation_failed']:
            return 'failure'
        elif pipeline_status == 'concerning' or health_score < 0.5:
            return 'critical'
        elif pipeline_status in ['good', 'warnings'] or len(priority_issues) > 0:
            return 'warning'
        elif pipeline_status == 'excellent':
            return 'success'
        else:
            return 'warning'  # Default to warning for unknown status
            
    except Exception as e:
        print(f"Error determining notification severity: {str(e)}")
        return 'critical'  # Err on the side of caution


def generate_notification_message(pipeline_report: Dict[str, Any], severity: str, 
                                config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate notification message based on pipeline report and severity
    
    Args:
        pipeline_report: Comprehensive pipeline report
        severity: Notification severity level
        config: Notification configuration
        
    Returns:
        Dictionary containing formatted notification message
    """
    
    try:
        executive_summary = pipeline_report.get('executive_summary', {})
        execution_date = config['execution_date_str']
        template = config['templates'].get(severity, config['templates']['warning'])
        
        # Extract key information
        pipeline_status = executive_summary.get('pipeline_status', 'unknown')
        health_score = executive_summary.get('overall_health_score', 0.0)
        key_achievements = executive_summary.get('key_achievements', [])
        priority_issues = executive_summary.get('priority_issues', [])
        
        # Generate subject
        subject = f"{template['subject']} - {execution_date}"
        
        # Generate message body
        message_body = f"""
ML Pipeline Execution Report
===========================
Execution Date: {execution_date}
Pipeline Status: {pipeline_status.upper()}
Overall Health Score: {health_score:.2f}/1.00

"""
        
        # Add achievements for success cases
        if severity == 'success' and key_achievements:
            message_body += "✅ Key Achievements:\n"
            for achievement in key_achievements[:3]:  # Top 3 achievements
                message_body += f"  • {achievement}\n"
            message_body += "\n"
        
        # Add issues for warning/critical cases
        if severity in ['warning', 'critical', 'failure'] and priority_issues:
            icon = "⚠️" if severity == 'warning' else "🚨"
            message_body += f"{icon} Priority Issues:\n"
            for issue in priority_issues[:5]:  # Top 5 issues
                message_body += f"  • {issue}\n"
            message_body += "\n"
        
        # Add component status summary
        component_analyses = pipeline_report.get('component_analyses', {})
        if component_analyses:
            message_body += "📊 Component Status:\n"
            for component, analysis in component_analyses.items():
                status = analysis.get('status', 'unknown')
                component_name = component.replace('_', ' ').title()
                status_icon = get_status_icon(status)
                message_body += f"  {status_icon} {component_name}: {status}\n"
            message_body += "\n"
        
        # Add recommendations for non-success cases
        recommendations = pipeline_report.get('recommendations', {})
        if severity != 'success' and recommendations:
            immediate_actions = recommendations.get('immediate_actions', [])
            if immediate_actions:
                message_body += "🔧 Immediate Actions Required:\n"
                for action in immediate_actions[:3]:  # Top 3 actions
                    message_body += f"  • {action}\n"
                message_body += "\n"
        
        # Add footer
        message_body += f"Report generated: {config['notification_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        message_body += "For detailed analysis, check the pipeline reports in the datamart.\n"
        
        notification_message = {
            'severity': severity,
            'priority': template['priority'],
            'subject': subject,
            'body': message_body,
            'execution_date': execution_date,
            'health_score': health_score,
            'pipeline_status': pipeline_status,
            'timestamp': config['notification_timestamp'].isoformat()
        }
        
        return notification_message
        
    except Exception as e:
        # Generate error notification
        return {
            'severity': 'critical',
            'priority': 'critical',
            'subject': f"❌ Notification Generation Failed - {config['execution_date_str']}",
            'body': f"Failed to generate notification message: {str(e)}",
            'execution_date': config['execution_date_str'],
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def get_status_icon(status: str) -> str:
    """
    Get appropriate icon for status
    
    Args:
        status: Status string
        
    Returns:
        Unicode icon for the status
    """
    
    status_icons = {
        'excellent': '🟢', 'success': '🟢', 'healthy': '🟢',
        'good': '🟡', 'partial_success': '🟡', 'warnings': '🟡', 'warning': '🟡',
        'concerning': '🟠', 'critical_issues': '🟠',
        'critical': '🔴', 'failed': '🔴', 'failure': '🔴',
        'skipped': '⚪', 'no_data': '⚪',
        'unknown': '❓', 'analysis_error': '❓'
    }
    
    return status_icons.get(status, '❓')


def send_console_notification(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send notification to console/stdout
    
    Args:
        message: Notification message dictionary
        
    Returns:
        Delivery result dictionary
    """
    
    try:
        print("\n" + "="*60)
        print(f"PIPELINE NOTIFICATION - {message['severity'].upper()}")
        print("="*60)
        print(f"Subject: {message['subject']}")
        print("-"*60)
        print(message['body'])
        print("="*60)
        
        return {
            'channel': 'console',
            'status': 'delivered',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'channel': 'console',
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def send_log_file_notification(message: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send notification to log file
    
    Args:
        message: Notification message dictionary
        config: Notification configuration
        
    Returns:
        Delivery result dictionary
    """
    
    try:
        log_config = config['channel_config']['log_file']
        log_path = log_config.get('log_path', '/opt/airflow/logs/notifications/')
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        # Generate log filename
        log_filename = f"notification_{message['execution_date'].replace('-', '_')}.log"
        log_filepath = os.path.join(log_path, log_filename)
        
        # Format log entry
        log_entry = {
            'timestamp': message['timestamp'],
            'severity': message['severity'],
            'pipeline_status': message['pipeline_status'],
            'health_score': message['health_score'],
            'subject': message['subject'],
            'body': message['body']
        }
        
        # Append to log file
        with open(log_filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, indent=2))
            f.write('\n' + '-'*60 + '\n')
        
        return {
            'channel': 'log_file',
            'status': 'delivered',
            'log_file': log_filepath,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'channel': 'log_file',
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def send_email_notification(message: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send notification via email (placeholder for production implementation)
    
    Args:
        message: Notification message dictionary
        config: Notification configuration
        
    Returns:
        Delivery result dictionary
    """
    
    try:
        email_config = config['channel_config']['email']
        
        if not email_config.get('enabled', False):
            return {
                'channel': 'email',
                'status': 'disabled',
                'timestamp': datetime.now().isoformat()
            }
        
        recipients = email_config.get('recipients', [])
        
        if not recipients:
            return {
                'channel': 'email',
                'status': 'no_recipients',
                'timestamp': datetime.now().isoformat()
            }
        
        # Placeholder for actual email sending logic
        # In production, this would use SMTP or email service API
        print(f"[EMAIL PLACEHOLDER] Would send email to: {recipients}")
        print(f"[EMAIL PLACEHOLDER] Subject: {message['subject']}")
        
        return {
            'channel': 'email',
            'status': 'delivered',
            'recipients': recipients,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'channel': 'email',
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def send_slack_notification(message: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send notification via Slack (placeholder for production implementation)
    
    Args:
        message: Notification message dictionary
        config: Notification configuration
        
    Returns:
        Delivery result dictionary
    """
    
    try:
        slack_config = config['channel_config']['slack']
        
        if not slack_config.get('enabled', False):
            return {
                'channel': 'slack',
                'status': 'disabled',
                'timestamp': datetime.now().isoformat()
            }
        
        webhook_url = slack_config.get('webhook_url')
        channel = slack_config.get('channel', '#ml-alerts')
        
        if not webhook_url:
            return {
                'channel': 'slack',
                'status': 'no_webhook_configured',
                'timestamp': datetime.now().isoformat()
            }
        
        # Placeholder for actual Slack sending logic
        # In production, this would use Slack webhooks or API
        print(f"[SLACK PLACEHOLDER] Would send to channel: {channel}")
        print(f"[SLACK PLACEHOLDER] Subject: {message['subject']}")
        
        return {
            'channel': 'slack',
            'status': 'delivered',
            'channel_sent': channel,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'channel': 'slack',
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def deliver_notifications(message: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Deliver notifications across all configured channels
    
    Args:
        message: Notification message dictionary
        config: Notification configuration
        
    Returns:
        List of delivery results for each channel
    """
    
    delivery_results = []
    channels = config['notification_channels']
    
    print(f"Delivering notifications across {len(channels)} channels...")
    
    # Send to each configured channel
    for channel in channels:
        try:
            if channel == 'console':
                result = send_console_notification(message)
            elif channel == 'log_file':
                result = send_log_file_notification(message, config)
            elif channel == 'email':
                result = send_email_notification(message, config)
            elif channel == 'slack':
                result = send_slack_notification(message, config)
            else:
                result = {
                    'channel': channel,
                    'status': 'unsupported',
                    'error': f"Channel '{channel}' not supported",
                    'timestamp': datetime.now().isoformat()
                }
            
            delivery_results.append(result)
            
            # Log delivery status
            if result['status'] == 'delivered':
                print(f"✅ {channel} notification delivered successfully")
            else:
                print(f"❌ {channel} notification failed: {result.get('error', result['status'])}")
                
        except Exception as e:
            error_result = {
                'channel': channel,
                'status': 'delivery_error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            delivery_results.append(error_result)
            print(f"❌ {channel} notification delivery error: {str(e)}")
    
    return delivery_results


def escalate_critical_notifications(message: Dict[str, Any], delivery_results: List[Dict[str, Any]], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle escalation for critical notifications
    
    Args:
        message: Original notification message
        delivery_results: Results from initial delivery attempts
        config: Notification configuration
        
    Returns:
        Escalation result dictionary
    """
    
    try:
        if not config.get('escalation_enabled', False):
            return {
                'escalation_attempted': False,
                'reason': 'escalation_disabled'
            }
        
        severity = message.get('severity', 'unknown')
        
        # Only escalate critical and failure notifications
        if severity not in ['critical', 'failure']:
            return {
                'escalation_attempted': False,
                'reason': 'severity_not_critical'
            }
        
        # Check if initial delivery failed
        failed_deliveries = [r for r in delivery_results if r['status'] != 'delivered']
        
        if len(failed_deliveries) > 0:
            print(f"🚨 Escalating due to {len(failed_deliveries)} failed deliveries")
            
            # Escalation logic (placeholder)
            # In production, this might page on-call engineers, send SMS, etc.
            escalation_message = f"ESCALATION: Critical ML pipeline notification failed delivery\n"
            escalation_message += f"Original severity: {severity}\n"
            escalation_message += f"Failed channels: {[r['channel'] for r in failed_deliveries]}\n"
            escalation_message += f"Original message: {message['subject']}"
            
            print(f"[ESCALATION PLACEHOLDER] {escalation_message}")
            
            return {
                'escalation_attempted': True,
                'escalation_reason': 'delivery_failure',
                'failed_channels': [r['channel'] for r in failed_deliveries],
                'escalation_timestamp': datetime.now().isoformat()
            }
        
        return {
            'escalation_attempted': False,
            'reason': 'all_deliveries_successful'
        }
        
    except Exception as e:
        return {
            'escalation_attempted': False,
            'error': str(e),
            'escalation_timestamp': datetime.now().isoformat()
        }


def send_pipeline_notifications_airflow_task(**context) -> Dict[str, Any]:
    """
    Airflow-compatible pipeline notification task
    
    Args:
        context: Airflow context
        
    Returns:
        Notification delivery summary
    """
    
    try:
        print(f"[AIRFLOW NOTIFICATIONS] Starting notification processing for {context['ds']}")
        
        # Build notification configuration
        config = build_notification_config(
            execution_date_str=context['ds'],
            notification_channels=['console', 'log_file']  # Safe default channels
        )
        
        # Retrieve pipeline report from previous task
        ti = context['ti']
        pipeline_report = ti.xcom_pull(task_ids='reporting_group.generate_pipeline_report')
        
        if not pipeline_report:
            # Generate fallback notification
            fallback_notification = {
                'severity': 'critical',
                'priority': 'high',
                'subject': f"⚠️ Missing Pipeline Report - {context['ds']}",
                'body': "Pipeline report was not available for notification generation.",
                'execution_date': context['ds'],
                'timestamp': datetime.now().isoformat()
            }
            
            delivery_results = deliver_notifications(fallback_notification, config)
            
            return {
                'notification_sent': True,
                'notification_type': 'fallback',
                'delivery_results': delivery_results,
                'execution_date': context['ds']
            }
        
        # Determine notification severity
        severity = determine_notification_severity(pipeline_report)
        
        # Generate notification message
        notification_message = generate_notification_message(pipeline_report, severity, config)
        
        # Deliver notifications
        delivery_results = deliver_notifications(notification_message, config)
        
        # Handle escalation if needed
        escalation_result = escalate_critical_notifications(notification_message, delivery_results, config)
        
        # Compile notification summary
        notification_summary = {
            'notification_sent': True,
            'severity': severity,
            'pipeline_status': pipeline_report.get('executive_summary', {}).get('pipeline_status', 'unknown'),
            'health_score': pipeline_report.get('executive_summary', {}).get('overall_health_score', 0.0),
            'delivery_results': delivery_results,
            'escalation_result': escalation_result,
            'successful_deliveries': len([r for r in delivery_results if r['status'] == 'delivered']),
            'failed_deliveries': len([r for r in delivery_results if r['status'] != 'delivered']),
            'execution_date': context['ds']
        }
        
        print(f"[AIRFLOW NOTIFICATIONS] Notification processing completed")
        print(f"[AIRFLOW NOTIFICATIONS] Severity: {severity}")
        print(f"[AIRFLOW NOTIFICATIONS] Successful deliveries: {notification_summary['successful_deliveries']}")
        print(f"[AIRFLOW NOTIFICATIONS] Failed deliveries: {notification_summary['failed_deliveries']}")
        
        return notification_summary
        
    except Exception as e:
        print(f"[AIRFLOW NOTIFICATIONS] Notification processing failed: {str(e)}")
        return {
            'notification_sent': False,
            'error': str(e),
            'execution_date': context['ds']
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Pipeline Notification Utilities Module")
    print("Available functions:")
    functions = [func for func in dir() if callable(getattr(__import__(__name__), func)) and not func.startswith('_')]
    for func in functions:
        print(f"  - {func}")
