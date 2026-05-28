
# d:/Parth/vanco-solution-architecture/grocery_forecasting/generate_submission.py

import os

import yaml

import numpy as np

import pandas as pd

import lightgbm as lgb



# Ingest full pipeline steps from our established data processor

from data_processor import run_full_processor_pipeline



def main():

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, "r") as f:

        cfg = yaml.safe_load(f)

       

    print(" Initializing Production Submission Generation Engine...")

   

    # 1. Gather master unified dataset from pipeline

    master_df = run_full_processor_pipeline()

   

    # 2. Separate into training and testing rows using our structural pipeline flags

    train_set = master_df[master_df['is_test'] == 0].copy()

    test_set = master_df[master_df['is_test'] == 1].copy()

   

    # Clear cold-start missing rows where lag features haven't aligned yet

    print(" Dropping chronological cold-start row initializations...")

    train_set = train_set.dropna(subset=['lag_28', 'rolling_mean_7d'])

   

    # 3. Establish strict column boundaries for training

    ignored_cols = ['id', 'date', 'sales', 'is_test']

    feature_cols = [col for col in train_set.columns if col not in ignored_cols]

   

    # Map string coordinates into explicit optimized categorical datatypes

    cat_cols = ['family', 'city', 'state', 'type', 'cluster']

    for col in cat_cols:

        train_set[col] = train_set[col].astype('category')

        test_set[col] = test_set[col].astype('category')

       

    # 4. Partition into Local Train and Validation for Early Stopping Protection

    cutoff = pd.to_datetime(cfg['validation']['val_cutoff'])

    train_mask = train_set['date'] < cutoff

    val_mask = train_set['date'] >= cutoff

   

    X_train = train_set[train_mask][feature_cols]

    y_train = np.log1p(train_set[train_mask]['sales'])  # Log space target transformation

   

    X_val = train_set[val_mask][feature_cols]

    y_val = np.log1p(train_set[val_mask]['sales'])      # Log space validation transformation

   

    X_test = test_set[feature_cols]

   

    # Isolate LightGBM configurations directly from central config.yaml file

    lgb_exp = cfg['model_experiments'][0]

   

    print("\n Training winning LightGBM Backbone in Log-Space for final inference...")

    lgb_train = lgb.Dataset(X_train, label=y_train)

    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

   

    # Train the model with early stopping active to catch optimal convergence

    model = lgb.train(

        params=lgb_exp['params'],

        train_set=lgb_train,

        num_boost_round=lgb_exp['num_boost_rounds'],

        valid_sets=[lgb_val],

        callbacks=[

            lgb.early_stopping(stopping_rounds=cfg['validation']['early_stopping_rounds']),

            lgb.log_evaluation(period=100)

        ]

    )

   

    # 5. Run blind test inference matrix projections

    print("\n Running predictions across the blind test matrix horizon...")

    test_preds_log = model.predict(X_test)

   

    # 6. Invert target predictions back out of log-space into raw unit scales

    print(" Reverting log scale transformations via expm1...")

    test_preds_raw = np.expm1(test_preds_log)

    test_preds_raw = np.clip(test_preds_raw, 0, None)  # Strictly enforce zero floor boundary constraint

   

    # 7. Map to exact required competition dimensions and export

    submission = pd.DataFrame({

        'id': test_set['id'].astype(int),

        'sales': test_preds_raw

    })

   

    output_path = cfg['paths']['output_submission']

    submission.to_csv(output_path, index=False)

   

    print("\n=======================================================")

    print(" PRODUCTION SUBMISSION GENERATION SUCCESSFUL!")

    print("=======================================================")

    print(f"   Output File Path:  {output_path}")

    print(f"   Total Test Rows:   {len(submission):,}")

    print(f"   File Verification Columns: {list(submission.columns)}")

    print("=======================================================\n")



if __name__ == "__main__":

    main() 

