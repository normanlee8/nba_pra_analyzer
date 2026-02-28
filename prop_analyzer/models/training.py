import sys
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
import re
from datetime import datetime
from pathlib import Path

# --- NEW: Upgraded to Stacking Regressor and RidgeCV ---
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# --- FIX: Enable metadata routing for StackingRegressor sample weights ---
import sklearn
sklearn.set_config(enable_metadata_routing=True)

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.models import registry
from prop_analyzer.utils import common

class PassThroughScaler:
    """Dummy scaler to bypass imputation/scaling. Lets tree models handle NaNs natively."""
    def fit(self, X, y=None): 
        return self
    def transform(self, X): 
        return X.values if isinstance(X, pd.DataFrame) else X
    def fit_transform(self, X, y=None): 
        return self.transform(X)

def get_feature_cols(prop_cat, all_columns):
    relevant = []
    allowed_prefixes = feat_defs.PROP_FEATURE_MAP.get(prop_cat, [prop_cat])
    
    for base_feat in feat_defs.BASE_FEATURE_COLS:
        if base_feat in all_columns:
            relevant.append(base_feat)
        for prefix in allowed_prefixes:
            prefixed_feat = f"{prefix}_{base_feat}"
            if prefixed_feat in all_columns:
                relevant.append(prefixed_feat)

    # Includes newly engineered punitive features
    vacancy_cols = [
        'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F', 
        'TEAM_MISSING_AST_PCT', 'TEAM_MISSING_REB_PCT', Cols.DAYS_REST, 
        'OPP_DAYS_REST', 'OPP_IS_B2B', 'Games_in_Last_7_Days', 
        'IS_ALTITUDE', 'PACE_PG_INTERACTION', 'BLOWOUT_POTENTIAL', 
        'OPP_FOUL_DRAW_RATE', 'EXPECTED_USG_SHIFT'
    ]
    for vc in vacancy_cols:
        if vc in all_columns:
            relevant.append(vc)

    consistency_keywords = ['_CV', '_HIT_RATE', '_STD_DEV', '_CORR']
    for c in all_columns:
        if any(k in c for k in consistency_keywords):
            relevant.append(c)

    keywords = feat_defs.RELEVANT_KEYWORDS.get(prop_cat, [])
    for c in all_columns:
        if any(x in c for x in ['_RANK', 'TEAM_', 'OPP_', 'DVP_']):
            if any(x in c for x in ['NAME', 'ABBREV', Cols.DATE, 'SEASON_ID', Cols.PLAYER_ID]): continue
            if c in vacancy_cols: continue
            if keywords:
                if any(k in c for k in keywords) or 'PACE' in c or 'EFF' in c or 'DVP_' in c:
                    relevant.append(c)
            else:
                relevant.append(c)
    
    final_features = set(relevant)
    
    always_keep = ['VS_OPP_GAMES_PLAYED', 'VS_OPP_MIN']
    for f in feat_defs.VS_OPP_FEATURES:
        if f in always_keep and f in all_columns: final_features.add(f)
        elif any(f == f"VS_OPP_{s}" for s in allowed_prefixes) and f in all_columns: final_features.add(f)
            
    for f in feat_defs.HIST_FEATURES:
        if f == 'HIST_VS_OPP_GAMES' and f in all_columns: final_features.add(f)
        elif any(f.startswith(f"HIST_VS_OPP_{s}_") for s in allowed_prefixes) and f in all_columns: final_features.add(f)
            
    return [c for c in list(final_features) if c in all_columns]

def backfill_missing_cols(df, cols):
    for col in cols:
        if col not in df.columns: df[col] = np.nan 
    return df

def train_ensemble_model(df, target_col):
    logging.info(f"Training Probability-Optimized Meta-Ensemble for {target_col}...")

    date_col = Cols.DATE if Cols.DATE in df.columns else 'GAME_DATE'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    df = df.dropna(subset=[target_col]).copy()
    
    feature_list = get_feature_cols(target_col, df.columns)
    feature_list = list(set(feature_list)) 
    
    if len(feature_list) < 5:
        logging.warning(f"[{target_col}] Not enough features found. Skipping.")
        return

    df = backfill_missing_cols(df, feature_list)
    sanitized_cols = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in feature_list]
    
    X = df[feature_list].copy()
    X.columns = sanitized_cols
    y = df[target_col]

    # Sample Weighting (Exponential Decay - 45 Day Half-Life)
    days_ago = (pd.Timestamp.now() - df[date_col]).dt.days
    sample_weights = np.exp(-days_ago / 45)
    
    preprocessor = PassThroughScaler()
    X_proc = preprocessor.fit_transform(X)
    X_proc_df = pd.DataFrame(X_proc, columns=sanitized_cols)

    tscv = TimeSeriesSplit(n_splits=3) 

    def optimize_base_models():
        optuna.logging.set_verbosity(optuna.logging.INFO)
        
        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9)
            }
            mod = xgb.XGBRegressor(**params, random_state=42, n_jobs=2, objective='reg:tweedie', tree_method='hist')
            
            maes = []
            for train_idx, val_idx in tscv.split(X_proc_df):
                mod.fit(
                    X_proc_df.iloc[train_idx], y.iloc[train_idx],
                    sample_weight=sample_weights.iloc[train_idx]
                )
                maes.append(mean_absolute_error(y.iloc[val_idx], mod.predict(X_proc_df.iloc[val_idx])))
            return np.mean(maes)

        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(xgb_objective, n_trials=15)
        
        def lgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9)
            }
            mod = lgb.LGBMRegressor(**params, random_state=42, n_jobs=2, objective='tweedie', verbose=-1)
            
            maes = []
            for train_idx, val_idx in tscv.split(X_proc_df):
                mod.fit(
                    X_proc_df.iloc[train_idx], y.iloc[train_idx],
                    sample_weight=sample_weights.iloc[train_idx]
                )
                maes.append(mean_absolute_error(y.iloc[val_idx], mod.predict(X_proc_df.iloc[val_idx])))
            return np.mean(maes)

        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lgb_objective, n_trials=15)

        return study_xgb.best_params, study_lgb.best_params

    logging.info(f"[{target_col}] Running Hyperparameter Optimization (Walk-Forward CV)...")
    xgb_params, lgb_params = optimize_base_models()

    xgb_best = xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1, objective='reg:tweedie', tree_method='hist')
    lgb_best = lgb.LGBMRegressor(**lgb_params, random_state=42, n_jobs=-1, objective='tweedie', verbose=-1)
    
    # --- FIX: Explicitly request sample weights for the base estimators ---
    xgb_best.set_fit_request(sample_weight=True)
    lgb_best.set_fit_request(sample_weight=True)

    # --- UPGRADE: Stacking Regressor ---
    ensemble = StackingRegressor(
        estimators=[('xgb', xgb_best), ('lgb', lgb_best)],
        final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
        n_jobs=-1
    )

    logging.info(f"[{target_col}] Fitting Final Stacking Ensemble on full training data...")
    try:
        # FIX: With metadata routing enabled, pass 'sample_weight' directly. 
        # The routing engine forwards it to the base models automatically.
        ensemble.fit(X_proc_df, y, sample_weight=sample_weights)
    except Exception as e:
        logging.warning(f"Could not apply sample weights to stacking regressor ({e}). Fitting unweighted.")
        ensemble.fit(X_proc_df, y)

    split_idx = int(len(X_proc_df) * 0.85)
    X_val, y_val = X_proc_df.iloc[split_idx:], y.iloc[split_idx:]
    val_preds = ensemble.predict(X_val)

    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    logging.info(f"[{target_col}] Holdout Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

    logging.info(f"[{target_col}] Calculating SHAP Feature Importance...")
    shap_importance = []
    try:
        explainer = shap.TreeExplainer(ensemble.named_estimators_['xgb'])
        X_sample = X_proc_df.sample(n=min(1500, len(X_proc_df)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list): shap_values = shap_values[0]
        
        vals = np.abs(shap_values).mean(0)
        feat_imp_df = pd.DataFrame({'Feature': sanitized_cols, 'Importance': vals}).sort_values(by='Importance', ascending=False)
        shap_importance = feat_imp_df.head(20).to_dict(orient='records')
    except Exception as e:
        logging.warning(f"SHAP extraction failed: {e}")

    # EXPORTING TWEEDIE VARIANCE POWER FOR INFERENCE
    metadata = {
        'training_date': datetime.now().isoformat(),
        'target_col': target_col,
        'metrics': {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)},
        'top_features': shap_importance,
        'tweedie_variance_power': float(xgb_params.get('tweedie_variance_power', 1.5))
    }

    artifacts = {'scaler': preprocessor, 'features': sanitized_cols, 'model': ensemble, 'metadata': metadata}
    registry.save_artifacts(target_col, artifacts)
    logging.info(f"[{target_col}] Successfully trained and saved.")

def main():
    common.setup_logging(name="train_models")
    logging.info(">>> STARTING ADVANCED PROBABILITY MODEL TRAINING PIPELINE")

    train_file = cfg.MASTER_TRAINING_FILE
    if not train_file.exists(): return

    df = pd.read_parquet(train_file)
    if df.empty: return

    training_targets = list(set(cfg.SUPPORTED_PROPS + ['MIN']))

    for prop in training_targets:
        if prop in df.columns:
            logging.info(f"--- Training Engine: {prop} ---")
            train_ensemble_model(df, target_col=prop)

    logging.info("<<< TRAINING COMPLETE.")

if __name__ == "__main__":
    main()