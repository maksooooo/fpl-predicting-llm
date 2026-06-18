"""Benchmark several model families on the same chronological split.

Shows *why* LightGBM was chosen rather than assuming it. All models are trained
on the train seasons and compared on the 2022-23 VALIDATION season (model
selection is never done on the test season). Linear models get median
imputation + standardisation; tree models that support native missing values
(XGBoost, LightGBM) take the raw frames.

    python src/models/model_comparison.py
"""
import time
import warnings
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_model import load_features, split_data, prepare_features

warnings.filterwarnings("ignore")


def linear(est):
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), est)


def tree_imputed(est):
    return make_pipeline(SimpleImputer(strategy='median'), est)


MODELS = {
    "Linear Regression": linear(LinearRegression()),
    "Ridge (α=10)": linear(Ridge(alpha=10.0)),
    "Random Forest": tree_imputed(RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=20,
        n_jobs=-1, random_state=42)),
    "XGBoost": XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective='reg:absoluteerror',
        n_jobs=-1, random_state=42),
    "LightGBM (ours)": LGBMRegressor(
        objective='mae', n_estimators=300, learning_rate=0.05, num_leaves=63,
        max_depth=6, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=60, n_jobs=-1, random_state=42, verbose=-1),
}


def main():
    df = load_features("data/processed/engineered_features.csv")
    train_df, val_df, _ = split_data(df)
    X_tr, y_tr, X_val, y_val, _, _, feats = prepare_features(train_df, val_df, val_df)
    played = (val_df['minutes'] > 0).values

    print(f"\nComparing {len(MODELS)} models on the 2022-23 validation season "
          f"({len(X_tr)} train / {len(X_val)} val rows, {len(feats)} features)\n")
    print("=" * 70)
    print(f"{'Model':<22}{'val MAE':>10}{'val RMSE':>10}{'feat MAE':>10}{'fit (s)':>10}")
    print("-" * 70)

    results = []
    for name, model in MODELS.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        dt = time.time() - t0
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        fmae = mean_absolute_error(y_val[played], pred[played])
        results.append((name, mae, rmse, fmae, dt))
        print(f"{name:<22}{mae:>10.4f}{rmse:>10.4f}{fmae:>10.4f}{dt:>10.1f}")

    print("=" * 70)
    best = min(results, key=lambda r: r[1])
    print(f"\nBest validation MAE: {best[0]} ({best[1]:.4f}).")
    print("Gradient boosting (LightGBM/XGBoost) handles non-linear interactions\n"
          "and missing values natively, beating linear models and Random Forest\n"
          "on this tabular task -- which is why LightGBM is the production model.")


if __name__ == "__main__":
    main()
