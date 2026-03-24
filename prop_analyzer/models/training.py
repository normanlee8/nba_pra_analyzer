import sys
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
import optuna
import shap
import re
import time
from datetime import datetime
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path so we can run this from anywhere
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.models import registry
from prop_analyzer.utils import common

class PassThroughScaler:
    """
    Dummy scaler to bypass explicit imputation and scaling. 
    LightGBM handles NaNs natively and doesn't require feature scaling.
    This preserves the original distributions for quantile regression.
    """
    def fit(self, X, y=None): return self
    def transform(self, X): return X.values if isinstance(X, pd.DataFrame) else X
    def fit_transform(self, X, y=None): return self.transform(X)


def get_feature_cols(prop_cat, all_columns):
    """
    Identifies all relevant predictive features dynamically, including the newly
    engineered WOWY, Rotation Overlap, and Scheme Synergy features, while 
    strictly avoiding target leakage.
    """
    relevant = []
    allowed_prefixes = feat_defs.PROP_FEATURE_MAP.get(prop_cat, [prop_cat])
    
    # 1. Base statistical rolling features
    for base_feat in feat_defs.BASE_FEATURE_COLS:
        for prefix in allowed_prefixes:
            prefixed_feat = f"{prefix}_{base_feat}"
            if prefixed_feat in all_columns:
                relevant.append(prefixed_feat)

    # 2. Specific engineered context columns
    vacancy_cols = [
        'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F', 
        'TEAM_MISSING_AST_PCT', 'TEAM_MISSING_REB_PCT', Cols.DAYS_REST, 
        'OPP_DAYS_REST', 'OPP_IS_B2B', 'Games_in_Last_7_Days', 
        'IS_ALTITUDE', 'PACE_PG_INTERACTION', 'BLOWOUT_POTENTIAL', 
        'OPP_FOUL_DRAW_RATE', 'EXPECTED_USG_SHIFT', 'WOWY_PLAYER_USG', 
        'WOWY_PLAYER_PER36', 'FOUL_TROUBLE_VULNERABILITY', 'IS_BENCH_ROLE',
        'SYNERGY_PAINT_EDGE', 'SYNERGY_3PT_EDGE', 'SYNERGY_REB_EDGE', 'SCHEME_SYNERGY_SCORE',
        'FLIGHT_MILES', 'TZ_SHIFT', 'TEAM_GAMES_L4', 'TEAM_GAMES_L6', 'TEAM_GAMES_L7',
        'IS_3_IN_4', 'IS_4_IN_6', 'IS_TZ_SHOCK', 'LOST_AST_SHARE', 'WOWY_OFF_EFF', 'WOWY_DEF_EFF'
    ]
    for vc in vacancy_cols:
        if vc in all_columns: relevant.append(vc)

    # 3. Dynamic consistency metrics
    consistency_keywords = ['_CV', '_HIT_RATE', '_STD_DEV', '_CORR', '_FORM_RATIO', '_MEDIAN', 'SPLIT_AVG']
    for c in all_columns:
        if any(k in c for k in consistency_keywords): relevant.append(c)

    # 4. Advanced stat tracking & Defense vs Position
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
    always_keep = ['VS_OPP_GAMES_PLAYED', 'VS_OPP_MIN', 'GAME_PACE', 'OPP_GAME_PACE']
    for f in feat_defs.VS_OPP_FEATURES + always_keep:
        if f in always_keep and f in all_columns: final_features.add(f)
        elif any(f == f"VS_OPP_{s}" for s in allowed_prefixes) and f in all_columns: final_features.add(f)
            
    # TARGET LEAKAGE CHECK:
    # Strictly exclude raw counting stats which represent the final answer in the current row
    raw_leakage_blacklist = {
        'PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'MIN', 'FGA', 'FTA', 
        'TOV', 'STL', 'BLK', 'TS_PCT', 'USG_PROXY', 'USG_PROXY_PER36', 
        'PTS_PER36', 'REB_PER36', 'AST_PER36', 'PRA_PER36'
    }
            
    return [c for c in list(final_features) if c in all_columns and c not in raw_leakage_blacklist]


def train_ensemble_model(df, target_col):
    logging.info(f"Training Exact-Quantile Model for {target_col}...")

    date_col = Cols.DATE if Cols.DATE in df.columns else 'GAME_DATE'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)

    df = df.dropna(subset=[target_col]).copy()
    feature_list = list(set(get_feature_cols(target_col, df.columns)))
    
    if len(feature_list) < 5:
        logging.warning(f"[{target_col}] Not enough features found. Skipping.")
        return

    # Sanitize feature names for LightGBM compatibility (no spaces or weird characters)
    sanitized_cols = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in feature_list]
    
    X = df[feature_list].copy()
    X.columns = sanitized_cols
    
    # SAFETY PATCH: Drop or encode string columns to prevent LightGBM crashes
    for col in list(X.columns):
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except Exception:
                X = X.drop(columns=[col])
                sanitized_cols.remove(col)
                
    X = X.fillna(0.0) # Fallback for safety
    y = df[target_col]
    
    preprocessor = PassThroughScaler()
    X_proc = preprocessor.fit_transform(X)
    X_proc_df = pd.DataFrame(X_proc, columns=sanitized_cols)

    tscv = TimeSeriesSplit(n_splits=3) 

    def get_fold_weights(train_idx):
        # Recency Weighting: Models learn better when recent meta-game trends are prioritized
        fold_dates = df.iloc[train_idx][date_col]
        max_date = fold_dates.max()
        days_ago = (max_date - fold_dates).dt.days
        return np.exp(-days_ago / 45)

    def optimize_quantile_model(alpha_val):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        def lgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 60, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'num_leaves': trial.suggest_int('num_leaves', 15, 50),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'quantile',
                'alpha': alpha_val
            }
            mod = lgb.LGBMRegressor(**params, random_state=42, n_jobs=2, verbose=-1)
            
            # Use Pinball Loss (Quantile Loss) directly to grade performance
            losses = []
            for train_idx, val_idx in tscv.split(X_proc_df):
                fold_weights = get_fold_weights(train_idx)
                mod.fit(X_proc_df.iloc[train_idx], y.iloc[train_idx], sample_weight=fold_weights)
                preds = mod.predict(X_proc_df.iloc[val_idx])
                errors = y.iloc[val_idx] - preds
                
                # Math for exact quantile pinball loss
                loss = np.maximum(alpha_val * errors, (alpha_val - 1) * errors).mean()
                losses.append(loss)
                
            return np.mean(losses)

        study = optuna.create_study(direction='minimize')
        study.optimize(lgb_objective, n_trials=10) # Reduced trials to save time
        return study.best_params

    logging.info(f"[{target_col}] Running Hyperparameter Optimization for Q50...")
    
    max_date_global = df[date_col].max()
    final_sample_weights = np.exp(-(max_date_global - df[date_col]).dt.days / 45)

    final_models = {}
    
    # SPEED PATCH: Only optimize for the median (q50), then share parameters
    best_shared_params = optimize_quantile_model(0.50)
    
    quantiles = [0.10, 0.50, 0.90]
    for q in quantiles:
        mod = lgb.LGBMRegressor(**best_shared_params, random_state=42, n_jobs=-1, objective='quantile', alpha=q, verbose=-1)
        mod.fit(X_proc_df, y, sample_weight=final_sample_weights)
        final_models[f'q{int(q*100)}'] = mod
        logging.info(f"[{target_col}] Trained Q{int(q*100)} distribution bound.")

    # Evaluate Median (Q50) for global accuracy metrics and holdout evaluation
    split_idx = int(len(X_proc_df) * 0.85)
    X_val, y_val = X_proc_df.iloc[split_idx:], y.iloc[split_idx:]
    val_preds = final_models['q50'].predict(X_val)

    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    logging.info(f"[{target_col}] Q50 Median Holdout Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

    # Extract SHAP strictly from the median predictor to understand what features drive output
    shap_importance = []
    try:
        explainer = shap.TreeExplainer(final_models['q50'])
        X_sample = X_proc_df.sample(n=min(1500, len(X_proc_df)), random_state=42)
        shap_values_global = explainer.shap_values(X_sample)
        if isinstance(shap_values_global, list): 
            shap_values_global = shap_values_global[0]
        
        vals_global = np.abs(shap_values_global).mean(0)
        feat_imp_df_global = pd.DataFrame({'Feature': sanitized_cols, 'Importance': vals_global}).sort_values(by='Importance', ascending=False)
        shap_importance = feat_imp_df_global.head(20).to_dict(orient='records')
    except Exception as e:
        logging.warning(f"SHAP extraction failed: {e}")

    # Save artifacts as a dictionary mapping directly to exact models
    metadata = {
        'training_date': datetime.now().isoformat(),
        'target_col': target_col,
        'metrics': {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)},
        'top_features': shap_importance,
        'is_quantile': True
    }

    artifacts = {'scaler': preprocessor, 'features': sanitized_cols, 'model': final_models, 'metadata': metadata}
    registry.save_artifacts(target_col, artifacts)
    logging.info(f"[{target_col}] Successfully trained exact probability distributions.")


def main():
    start_time = time.time()  
    common.setup_logging(name="train_models")
    logging.info(">>> STARTING QUANTILE PROBABILITY MODEL TRAINING PIPELINE")

    train_file = cfg.MASTER_TRAINING_FILE
    if not train_file.exists(): 
        logging.error(f"Training file not found at {train_file}")
        return

    df = pd.read_parquet(train_file)
    if df.empty: 
        logging.error("Training dataset is empty. Cannot run training.")
        return

    training_targets = list(set(cfg.SUPPORTED_PROPS + ['MIN']))

    for prop in training_targets:
        if prop in df.columns:
            logging.info(f"--- Training Engine: {prop} ---")
            train_ensemble_model(df, target_col=prop)

    elapsed = time.time() - start_time  
    logging.info(f"========= QUANTILE MODEL TRAINING FINISHED in {int(elapsed // 60)}:{int(elapsed % 60):02d} minutes =========")


if __name__ == "__main__":
    main()