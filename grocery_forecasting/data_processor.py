# d:/Parth/vanco-solution-architecture/grocery_forecasting/data_processor.py

import pandas as pd

import numpy as np



def load_and_merge_data(train_path, test_path, stores_path, oil_path):

    print(" Loading base training and test streams...")

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

   

    train['is_test'] = 0

    test['is_test'] = 1

    test['sales'] = 0.0

   

    combined = pd.concat([train, test], axis=0).reset_index(drop=True)

    combined['date'] = pd.to_datetime(combined['date'])

   

    print(" Merging store dimensional metadata...")

    stores = pd.read_csv(stores_path)

    combined = combined.merge(stores, on='store_nbr', how='left')

   

    print(" Aligning and forward-filling macroeconomic oil trend line...")

    oil = pd.read_csv(oil_path)

    oil['date'] = pd.to_datetime(oil['date'])

   

    full_date_range = pd.date_range(start=combined['date'].min(), end=combined['date'].max(), freq='D')

    oil_imputed = oil.set_index('date').reindex(full_date_range).ffill().bfill().reset_index()

    oil_imputed.columns = ['date', 'oil_price']

   

    combined = combined.merge(oil_imputed, on='date', how='left')

    return combined



def engineer_domain_features(df, holidays_path):

    print(" Engineering time primitives and bi-weekly payday flags...")

    df['day_of_week'] = df['date'].dt.dayofweek

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['is_payday'] = ((df['date'].dt.day == 15) | (df['date'].dt.is_month_end)).astype(int)

   

    print(" Modeling 2016 Earthquake shock window...")

    earthquake_date = pd.to_datetime('2016-04-16')

    df['days_since_earthquake'] = (df['date'] - earthquake_date).dt.days

    df['earthquake_shock_active'] = df['days_since_earthquake'].between(0, 30).astype(int)

    df.drop(columns=['days_since_earthquake'], inplace=True)

   

    print(" Cleaning holiday metadata (filtering transferred events)...")

    holidays = pd.read_csv(holidays_path)

    holidays['date'] = pd.to_datetime(holidays['date'])

    true_holidays = holidays[holidays['transferred'] == False]

    national_holiday_dates = true_holidays[true_holidays['locale'] == 'National']['date'].unique()

    df['is_national_holiday'] = df['date'].isin(national_holiday_dates).astype(int)

   

    return df



def generate_leakage_proof_lags(df):

    print(" Constructing target lags utilizing a safe 16-day blind-spot gap...")

    # Sorting is critical to ensure shift operations match the chronological sequence

    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

   

    # Define our lag sequence (all values >= 16-day test horizon)

    lag_windows = [16, 17, 18, 21, 28]

   

    for lag in lag_windows:

        df[f'lag_{lag}'] = df.groupby(['store_nbr', 'family'])['sales'].shift(lag)

       

    # Generate rolling statistics anchored behind the 16-day cutoff boundary

    anchor = 'lag_16'

    df['rolling_mean_7d'] = df.groupby(['store_nbr', 'family'])[anchor].transform(lambda x: x.rolling(7).mean())

    df['rolling_std_7d'] = df.groupby(['store_nbr', 'family'])[anchor].transform(lambda x: x.rolling(7).std())

   

    return df



def run_full_processor_pipeline():

    base_dir = r"D:\Parth\vanco-solution-architecture\grocery_forecasting\data\raw"

   

    master_df = load_and_merge_data(

        train_path=f"{base_dir}\\train.csv",

        test_path=f"{base_dir}\\test.csv",

        stores_path=f"{base_dir}\\stores.csv",

        oil_path=f"{base_dir}\\oil.csv"

    )

   

    master_df = engineer_domain_features(master_df, holidays_path=f"{base_dir}\\holidays_events.csv")

    master_df = generate_leakage_proof_lags(master_df)

   

    print("\n Pipeline run complete!")

    print(f"Final Combined Matrix Shape: {master_df.shape}")

    return master_df



if __name__ == "__main__":

    run_full_processor_pipeline()

