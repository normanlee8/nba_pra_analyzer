import joblib
import logging
import pandas as pd
from pathlib import Path
from prop_analyzer import config as cfg

def get_model_path(prop_category):
    """
    Returns the standard file path for a prop's model artifacts.
    """
    # Sanitize prop name just in case (e.g. 3PM -> FG3M is handled in mapping, but safety first)
    clean_cat = prop_category.replace(' ', '_').upper()
    return cfg.MODEL_DIR / f"model_{clean_cat}.pkl"

def save_artifacts(prop_category, artifacts):
    """
    Saves the trained model dictionary (scaler, features, models) to disk.
    """
    try:
        path = get_model_path(prop_category)
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(artifacts, path)
        logging.info(f"Saved artifacts for {prop_category} to {path.name}")
        return True
    except Exception as e:
        logging.error(f"Failed to save artifacts for {prop_category}: {e}")
        return False

def load_artifacts(prop_category):
    """
    Loads a specific prop model. Returns None if not found or corrupt.
    """
    path = get_model_path(prop_category)
    if not path.exists():
        return None
        
    try:
        artifacts = joblib.load(path)
        # Basic validation
        if not isinstance(artifacts, dict):
            logging.warning(f"Model file for {prop_category} is invalid format.")
            return None
            
        required_keys = ['scaler', 'features', 'q20', 'q80', 'clf']
        if not all(k in artifacts for k in required_keys):
            logging.warning(f"Model file for {prop_category} missing keys: {required_keys}")
            # Phase 3 compatibility: We might still return it if it has partials, 
            # but inference checks keys. Let's return it and let inference handle/fail.
            
        return artifacts
    except Exception as e:
        logging.warning(f"Error loading model for {prop_category}: {e}")
        return None

def load_model_cache(props_to_load=None):
    """
    Loads all available models into memory for batch analysis.
    Args:
        props_to_load (list): Optional list of specific props to load. 
                              If None, loads everything in SUPPORTED_PROPS.
    Returns:
        dict: { 'PTS': artifacts_dict, 'REB': ... }
    """
    logging.info("Loading model cache...")
    
    if props_to_load is None:
        props_to_load = cfg.SUPPORTED_PROPS
        
    cache = {}
    loaded_count = 0
    
    for prop in props_to_load:
        artifacts = load_artifacts(prop)
        if artifacts:
            cache[prop] = artifacts
            loaded_count += 1
            
    if loaded_count == 0:
        logging.warning("No models loaded! Ensure you have run training first.")
        
    return cache