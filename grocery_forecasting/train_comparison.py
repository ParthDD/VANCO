# d:/Parth/vanco-solution-architecture/grocery_forecasting/train_comparison.py
import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ingest full pipeline steps from our data processor
from data_processor import run_full_processor_pipeline

def calculate_rmsle(y_true_raw, y_pred_raw):
    """Calculates true Target-Space Root Mean Squared Logarithmic Error."""
    # Clip negative predictions to 0 to prevent complex log numbers
    y_pred_bounded = np.clip(y_pred_raw, 0, None)
    log_true = np.log1p(y_true_raw)
    log_pred = np.log1p(y_pred_bounded)
    return np.sqrt(mean_squared_error(log_true, log_pred))

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    # 1. Gather master dataset from processor pipeline
    master_df = run_full_processor_pipeline()
    train_set = master_df[master_df['is_test'] == 0].copy()
    
    # Drop chronological cold start buffer rows safely
    train_set = train_set.dropna(subset=['lag_28', 'rolling_mean_7d'])
    
    ignored_cols = ['id', 'date', 'sales', 'is_test']
    feature_cols = [col for col in train_set.columns if col not in ignored_cols]
    
    cat_cols = ['family', 'city', 'state', 'type', 'cluster']
    for col in cat_cols:
        train_set[col] = train_set[col].astype('category')
        
    # 2. Chronological Validation Partition
    cutoff = pd.to_datetime(cfg['validation']['val_cutoff'])
    train_mask = train_set['date'] < cutoff
    val_mask = train_set['date'] >= cutoff
    
    X_train = train_set[train_mask][feature_cols]
    X_val = train_set[val_mask][feature_cols]
    
    # Isolate targets for both tracking spaces
    y_train_raw = train_set[train_mask]['sales']
    y_val_raw = train_set[val_mask]['sales'].values
    
    y_train_log = np.log1p(y_train_raw)
    y_val_log = np.log1p(y_val_raw)
    
    lgb_params = cfg['model_experiments'][0]['params']
    rounds = cfg['model_experiments'][0]['num_boost_rounds']
    stopping = cfg['validation']['early_stopping_rounds']
    
    # -------------------------------------------------------------
    # MODEL A: Optimize Directly for Raw Unit Scales
    # -------------------------------------------------------------
    print("\n [TRAINING MODEL A] Optimizing directly for Raw Unit Scale (RMSE)...")
    dtrain_raw = lgb.Dataset(X_train, label=y_train_raw)
    dval_raw = lgb.Dataset(X_val, label=y_val_raw, reference=dtrain_raw)
    
    model_raw = lgb.train(
        params=lgb_params,
        train_set=dtrain_raw,
        num_boost_round=rounds,
        valid_sets=[dval_raw],
        callbacks=[lgb.early_stopping(stopping_rounds=stopping), lgb.log_evaluation(period=0)]
    )
    preds_from_raw_model = np.clip(model_raw.predict(X_val), 0, None)
    
    # -------------------------------------------------------------
    # MODEL B: Optimize for Log Space (RMSLE)
    # -------------------------------------------------------------
    print("\n [TRAINING MODEL B] Optimizing for Log-Transformed Space (RMSLE)...")
    dtrain_log = lgb.Dataset(X_train, label=y_train_log)
    dval_log = lgb.Dataset(X_val, label=y_val_log, reference=dtrain_log)
    
    model_log = lgb.train(
        params=lgb_params,
        train_set=dtrain_log,
        num_boost_round=rounds,
        valid_sets=[dval_log],
        callbacks=[lgb.early_stopping(stopping_rounds=stopping), lgb.log_evaluation(period=0)]
    )
    preds_from_log_model = np.expm1(model_log.predict(X_val))
    preds_from_log_model = np.clip(preds_from_log_model, 0, None)

    # -------------------------------------------------------------
    # CROSS-EVALUATION PERFORMANCE MATRIX
    # -------------------------------------------------------------
    # Evaluation metrics for Model A
    mae_A = mean_absolute_error(y_val_raw, preds_from_raw_model)
    rmse_A = np.sqrt(mean_squared_error(y_val_raw, preds_from_raw_model))
    rmsle_A = calculate_rmsle(y_val_raw, preds_from_raw_model)
    
    # Evaluation metrics for Model B
    mae_B = mean_absolute_error(y_val_raw, preds_from_log_model)
    rmse_B = np.sqrt(mean_squared_error(y_val_raw, preds_from_log_model))
    rmsle_B = calculate_rmsle(y_val_raw, preds_from_log_model)
    
    print("\n=====================================================================")
    print(" METRIC OPTIMIZATION CROSS-COMPARISON MATRIX:")
    print("=====================================================================")
    print("  METRIC TRACKED              | MODEL A (Raw Optimizer) | MODEL B (Log Optimizer)")
    print("---------------------------------------------------------------------")
    print(f"  Validation RMSLE            | {rmsle_A:.5f}                 | {rmsle_B:.5f}")
    print(f"  Validation MAE (Units)      | {mae_A:.2f} units           | {mae_B:.2f} units")
    print(f"  Validation RMSE (Units)     | {rmse_A:.2f} units          | {rmse_B:.2f} units")
    print("=====================================================================")
    print("\n Architectural Insight:")
    print("  - Look at RMSLE: The Log Optimizer (Model B) will score significantly better.")
    print("  - Look at RMSE: The Raw Optimizer (Model A) focuses entirely on giant stores,")
    print("    yielding a lower raw RMSE, but sacrificing accuracy on smaller product families.")

if __name__ == "__main__":
    main()