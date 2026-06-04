import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os

def load_data(filepath):
    """Load the raw FPL dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the data: parsing dates and sorting."""
    print("Preprocessing data...")
    # Convert kickoff_time to datetime
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    
    # Sort chronologically by player
    df = df.sort_values(by=['name', 'kickoff_time']).reset_index(drop=True)
    return df

def engineer_features(df):
    """Create lag features, rolling features, and the target variable."""
    print("Engineering features...")
    
    # 1. Create Target Variable (predict next GW's points)
    # We group by player and season, then shift the total_points by -1
    # This means the target for GW n is the total_points from GW n+1
    df['target_next_gw_points'] = df.groupby(['name', 'season_x'])['total_points'].shift(-1)
    
    # 2. Lag Features (Previous GW's performance)
    df['lag_1_total_points'] = df.groupby(['name', 'season_x'])['total_points'].shift(1)
    df['lag_1_minutes'] = df.groupby(['name', 'season_x'])['minutes'].shift(1)
    
    # 3. Rolling Features (Form over the last 3 and 5 GWs)
    # Note: We use closed='left' or shift to avoid target leakage (including current GW in rolling calc)
    rolling_cols = ['total_points', 'minutes', 'bps', 'ict_index', 'threat', 'creativity', 'influence', 'goals_scored', 'assists']
    
    for col in rolling_cols:
        # We need to shift by 1 first so the rolling window doesn't include the current match
        shifted_col = df.groupby(['name', 'season_x'])[col].shift(1)
        df[f'rolling_3_avg_{col}'] = shifted_col.groupby(df['name']).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'rolling_5_avg_{col}'] = shifted_col.groupby(df['name']).rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)

    # 4. Drop rows where target is NaN (i.e., the last match of the season for each player)
    # Since we can't train on examples where we don't know the outcome
    print(f"Shape before dropping NaN targets: {df.shape}")
    df = df.dropna(subset=['target_next_gw_points']).reset_index(drop=True)
    print(f"Shape after dropping NaN targets: {df.shape}")
    
    return df

def save_data(df, output_path):
    """Save the engineered dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving features to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    input_file = "cleaned_merged_seasons.csv"
    output_file = "data/processed/engineered_features.csv"
    
    df = load_data(input_file)
    df = preprocess_data(df)
    df = engineer_features(df)
    save_data(df, output_file)
