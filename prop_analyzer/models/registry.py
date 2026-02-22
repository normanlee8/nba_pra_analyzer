import joblib
import logging
import json
from pathlib import Path
from datetime import datetime

from prop_analyzer import config as cfg

def get_model_path(prop_category, is_latest=True, timestamp=None):
    """
    Returns the standard file path for a prop's model artifacts.
    Supports versioning by saving both a 'latest' pointer and a timestamped snapshot.
    """
    clean_cat = prop_category.replace(' ', '_').upper()
    
    if is_latest:
        return cfg.MODEL_DIR / f"model_{clean_cat}_latest.pkl"
    else:
        versions_dir = cfg.MODEL_DIR / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        return versions_dir / f"model_{clean_cat}_{timestamp}.pkl"

def save_artifacts(prop_category, artifacts):
    """
    Saves the trained model dictionary (scaler, features, models) to disk.
    Also extracts and saves the metadata payload as a readable JSON file.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latest_path = get_model_path(prop_category, is_latest=True)
        versioned_path = get_model_path(prop_category, is_latest=False, timestamp=timestamp)
        
        # 1. Save serialized Pickle objects
        joblib.dump(artifacts, versioned_path)
        joblib.dump(artifacts, latest_path) # Overwrite the active champion model
        
        # 2. Extract and save metadata as JSON for easy tracking/reporting
        if 'metadata' in artifacts:
            meta_path = cfg.MODEL_DIR / "metadata"
            meta_path.mkdir(parents=True, exist_ok=True)
            meta_file = meta_path / f"meta_{prop_category.replace(' ', '_').upper()}_latest.json"
            
            with open(meta_file, 'w') as f:
                json.dump(artifacts['metadata'], f, indent=4)
                
        logging.info(f"Saved versioned artifacts for {prop_category} to {latest_path.name}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save artifacts for {prop_category}: {e}")
        return False

def load_artifacts(prop_category):
    """
    Loads the latest active model for a specific prop.
    """
    path = get_model_path(prop_category, is_latest=True)
    if not path.exists():
        return None
        
    try:
        artifacts = joblib.load(path)
        if not isinstance(artifacts, dict):
            logging.warning(f"Model file for {prop_category} is invalid format.")
            return None
            
        required_keys = ['scaler', 'features', 'model']
        if not all(k in artifacts for k in required_keys):
            logging.warning(f"Model file for {prop_category} missing core keys. Needs retraining.")
            return None
            
        return artifacts
    except Exception as e:
        logging.warning(f"Error loading model for {prop_category}: {e}")
        return None

def load_model_cache(props_to_load=None):
    """
    Loads all available 'latest' models into memory for batch analysis.
    """
    logging.info("Loading latest model cache...")
    
    if props_to_load is None:
        props_to_load = cfg.SUPPORTED_PROPS + ['BASE_MULTI']
        
    cache = {}
    loaded_count = 0
    
    for prop in props_to_load:
        artifacts = load_artifacts(prop)
        if artifacts:
            cache[prop] = artifacts
            loaded_count += 1
            
    if loaded_count == 0:
        logging.warning("No models loaded! Ensure you have run the new training pipeline.")
        
    return cache