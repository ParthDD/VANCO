# d:/Parth/vanco-solution-architecture/grocery_forecasting/train_raw_metrics.py
import os
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ingest full pipeline steps from our established data processor
from data_processor import run_full_processor_pipeline

def generate_regression_diagnostic_plots(y_true, y_pred, val_dates):
    """Generates clean, raw-scale forecasting diagnostic plots."""
    print(" Generating raw-scale regression diagnostic plots...")
    os.makedirs("eda_plots", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Create a dataframe optimized for aggregation visualizations
    plot_df = pd.DataFrame({
        'date': val_dates,
        'actual_sales': y_true,
        'predicted_sales': y_pred
    }).groupby('date').sum().reset_index() # Summing across all series to see global timeline match
    
    # PLOT 1: Actual vs Predicted Timeline overlay
    plt.figure(figsize=(14, 5))
    plt.plot(plot_df['date'], plot_df['actual_sales'], label='True Actual Sales', color='black', marker='o', linewidth=2)
    plt.plot(plot_df['date'], plot_df['predicted_sales'], label='LightGBM Forecast', color='dodgerblue', marker='x', linestyle='--', linewidth=2)
    plt.title("Validation Profile: Aggregate Daily Sales vs. Model Forecast (Aug 1 - Aug 15, 2017)")
    plt.xlabel("Date")
    plt.ylabel("Total Raw Sales Units")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(r"D:\Parth\vanco-solution-architecture\grocery_forecasting\notebooks\eda_plots\04_forecast_vs_actual_raw.png", dpi=150)
    plt.close()
    
    # PLOT 2: Residual Error Distribution Space
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=100, color='purple', kde=True)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Zero Error Baseline')
    plt.title("Residual Error Distribution Profile (Raw Units Space)")
    plt.xlabel("Residual Error (Actual Sales - Predicted Sales)")
    plt.ylabel("Density Count")
    plt.xlim(-500, 500) # Centering x-axis to look closely at highest concentration
    plt.legend()
    plt.tight_layout()
    plt.savefig(r"D:\Parth\vanco-solution-architecture\grocery_forecasting\notebooks\eda_plots\05_residual_distribution_raw.png", dpi=150)
    plt.close()
    
    print("   Saved: eda_plots/04_forecast_vs_actual_raw.png")
    print("   Saved: eda_plots/05_residual_distribution_raw.png")

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Extract master dataset from pipeline
    master_df = run_full_processor_pipeline()
    
    train_set = master_df[master_df['is_test'] == 0].copy()
    
    # Clear cold start missing rows
    train_set = train_set.dropna(subset=['lag_28', 'rolling_mean_7d'])
    
    ignored_cols = ['id', 'date', 'sales', 'is_test']
    feature_cols = [col for col in train_set.columns if col not in ignored_cols]
    
    cat_cols = ['family', 'city', 'state', 'type', 'cluster']
    for col in cat_cols:
        train_set[col] = train_set[col].astype('category')
        
    cutoff = pd.to_datetime(cfg['validation']['val_cutoff'])
    train_mask = train_set['date'] < cutoff
    val_mask = train_set['date'] >= cutoff
    
    X_train = train_set[train_mask][feature_cols]
    y_train = train_set[train_mask]['sales']  # Raw sales training target (No Log Transform)
    
    X_val = train_set[val_mask][feature_cols]
    y_val = train_set[val_mask]['sales']      # Raw sales validation target
    val_dates = train_set[val_mask]['date']
    
    # Isolate LightGBM config details directly out of our baseline position in YAML
    lgb_exp = cfg['model_experiments'][0]
    
    print("\n Training Primary LightGBM Backbone Engine directly on Raw Sales...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    model = lgb.train(
        params=lgb_exp['params'],
        train_set=lgb_train,
        num_boost_round=lgb_exp['num_boost_rounds'],
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=cfg['validation']['early_stopping_rounds']),
            lgb.log_evaluation(period=200)
        ]
    )
    
    # Generate raw predictions
    y_pred_raw = model.predict(X_val)
    y_pred_raw = np.clip(y_pred_raw, 0, None) # Floor bounding clean-up to prevent negative sales
    y_true_raw = y_val.values
    
    # Calculate target scale metrics
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse_raw = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
    
    print("\n=======================================================")
    print(" RAW SCALE VALIDATION METRICS RESULTS MATRIX:")
    print("=======================================================")
    print(f"   MAE  (Average Unit Deviation):       {mae:.2f} sales units")
    print(f"   RMSE (Root Mean Square Error):         {rmse_raw:.2f} sales units")
    print("=======================================================\n")
    
    # Call the reporting graphics generators
    generate_regression_diagnostic_plots(y_true_raw, y_pred_raw, val_dates)
    print("\n Diagnostics execution completed perfectly.")

if __name__ == "__main__":
    main()