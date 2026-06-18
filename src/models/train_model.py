import pandas as pd
import os
import pickle
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Chronological split: train on older seasons, validate on the most recent
# complete season, hold out the latest season as a true out-of-time test set.
TRAIN_SEASONS = ['2016-17', '2017-18', '2020-21', '2021-22']
VAL_SEASONS = ['2022-23']
TEST_SEASONS = ['2023-24']

TARGET = 'target_next_gw_points'

# Config selected on the 2022-23 validation season (see experiments.py):
# L1/MAE clearly beat Tweedie/Poisson/Huber for this metric, and a slower
# learning rate with more leaves edged out the previous settings.
REG_PARAMS = dict(
    objective='mae',
    learning_rate=0.03,
    num_leaves=63,
    max_depth=6,
    n_estimators=3000,          # upper bound; early stopping picks the best
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=60,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Identifiers / text columns that must never be fed to the model as features.
DROP_COLS = [
    TARGET, 'target_next_gw_minutes',  # targets — never features (leakage)
    'name', 'season_x', 'kickoff_time', 'position', 'team_x',
    'opponent_team', 'opp_team_name', 'element', 'fixture', 'round',
    'was_home',  # superseded by the numeric was_home_int / next_was_home
]


def load_features(filepath):
    """Load the engineered features dataset."""
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath, low_memory=False)


def split_data(df):
    """Split data into train, validation, and test sets based on seasons."""
    print("Splitting data chronologically...")
    train_df = df[df['season_x'].isin(TRAIN_SEASONS)].copy()
    val_df = df[df['season_x'].isin(VAL_SEASONS)].copy()
    test_df = df[df['season_x'].isin(TEST_SEASONS)].copy()

    print(f"Train size: {train_df.shape[0]}, Val size: {val_df.shape[0]}, "
          f"Test size: {test_df.shape[0]}")
    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df):
    """Separate features (X) and target (y)."""
    candidate = [c for c in train_df.columns if c not in DROP_COLS]
    features = train_df[candidate].select_dtypes(
        include=['number', 'bool']).columns.tolist()
    print(f"Using {len(features)} numerical features.")

    X_train, y_train = train_df[features], train_df[TARGET]
    X_val, y_val = val_df[features], val_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]

    return X_train, y_train, X_val, y_val, X_test, y_test, features


def train_model(X_train, y_train, X_val, y_val):
    """Train a LightGBM regressor with early stopping on the validation set."""
    print("Training LGBMRegressor model...")

    model = LGBMRegressor(**REG_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l1',
        callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=0)],
    )
    print(f"Best iteration: {model.best_iteration_} "
          f"(val MAE: {model.best_score_['valid_0']['l1']:.4f})")
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

    X_train, y_train, X_val, y_val, X_test, y_test, features = prepare_features(
        train_df, val_df, test_df)

    model = train_model(X_train, y_train, X_val, y_val)

    save_model(model, features, model_output_file)

    # Save the test set for evaluation later
    test_df.to_csv("data/processed/test_data_with_targets.csv", index=False)
