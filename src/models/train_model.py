import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import os
import pickle

def load_features(filepath):
    """Load the engineered features dataset."""
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath, low_memory=False)

def split_data(df):
    """Split data into train, validation, and test sets based on seasons."""
    print("Splitting data chronologically...")
    train_seasons = ['2016-17', '2017-18', '2020-21', '2021-22']
    val_seasons = ['2022-23']
    test_seasons = ['2023-24']
    
    train_df = df[df['season_x'].isin(train_seasons)].copy()
    val_df = df[df['season_x'].isin(val_seasons)].copy()
    test_df = df[df['season_x'].isin(test_seasons)].copy()
    
    print(f"Train size: {train_df.shape[0]}, Val size: {val_df.shape[0]}, Test size: {test_df.shape[0]}")
    return train_df, val_df, test_df

def prepare_features(train_df, val_df, test_df):
    """Separate features (X) and target (y)."""
    target = 'target_next_gw_points'
    
    cols_to_drop = [
        target, 'name', 'season_x', 'kickoff_time', 'position', 
        'team_x', 'opponent_team', 'opp_team_name', 'element'
    ]
    
    features = [c for c in train_df.columns if c not in cols_to_drop]
    numeric_features = train_df[features].select_dtypes(include=['number', 'bool']).columns.tolist()
    print(f"Using {len(numeric_features)} numerical features.")

    X_train, y_train = train_df[numeric_features], train_df[target]
    X_val, y_val = val_df[numeric_features], val_df[target]
    X_test, y_test = test_df[numeric_features], test_df[target]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, numeric_features

def train_model(X_train, y_train, X_val, y_val):
    """Train a LightGBM Regressor model."""
    from lightgbm import LGBMRegressor
    print("Training LGBMRegressor model...")
    
    model = LGBMRegressor(
        objective='mae',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=150,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    return model

def save_model(model, features_list, filepath):
    """Save the trained model and feature list."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model, 'features': features_list}, f)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    input_file = "data/processed/engineered_features.csv"
    model_output_file = "models/histgb_model.pkl"
    
    df = load_features(input_file)
    train_df, val_df, test_df = split_data(df)
    
    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_features(train_df, val_df, test_df)
    
    model = train_model(X_train, y_train, X_val, y_val)
    
    save_model(model, features, model_output_file)
    
    # Save the test set for evaluation later
    test_df.to_csv("data/processed/test_data_with_targets.csv", index=False)
