"""
Model Management Utilities
Handles model file discovery and version management
"""

import os
import glob
from typing import Optional


def find_latest_trained_model(model_type: str, model_bank_directory: str = "/opt/airflow/models/") -> Optional[str]:
    """
    Find the latest trained model for a given model type
    
    Args:
        model_type: Type of model to find
        model_bank_directory: Directory containing model files
        
    Returns:
        Latest model version name or None if no model found
    """
    
    try:
        # Search for model files matching the pattern, but exclude preprocessor files
        pattern = f"{model_bank_directory}credit_model_{model_type}_*.pkl"
        all_files = glob.glob(pattern)
        
        # Filter out preprocessor files
        model_files = [f for f in all_files if not f.endswith("_preprocessor.pkl")]
        
        if not model_files:
            print(f"No trained models found for {model_type}")
            return None
        
        # Extract dates from filenames and find the latest
        model_versions = []
        for file_path in model_files:
            filename = os.path.basename(file_path)
            # Extract model version without .pkl extension
            model_version = filename.replace(".pkl", "")
            
            # Extract date part for sorting
            try:
                date_part = model_version.replace(f"credit_model_{model_type}_", "")
                # Convert to comparable date format
                date_formatted = date_part.replace("_", "-")
                model_versions.append((date_formatted, model_version))
            except:
                continue
        
        if not model_versions:
            print(f"No valid trained models found for {model_type}")
            return None
        
        # Sort by date and get the latest
        model_versions.sort(key=lambda x: x[0], reverse=True)
        latest_model_version = model_versions[0][1]
        
        print(f"Found latest trained model for {model_type}: {latest_model_version}")
        return latest_model_version
        
    except Exception as e:
        print(f"Error finding latest trained model for {model_type}: {str(e)}")
        return None