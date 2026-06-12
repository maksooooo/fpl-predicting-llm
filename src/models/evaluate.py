import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_model(filepath):
    """Load the trained model and features list."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['features']


def _metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def evaluate_model(model, features, test_data_path):
    """Evaluate the model and benchmark it against naive baselines."""
    print(f"Evaluating model on test data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)

    X_test = df_test[features]
    y_test = df_test['target_next_gw_points']

    predictions = model.predict(X_test)
    df_test['predicted_points'] = predictions

    # --- Naive baselines a useful model must beat ---------------------------
    # "Last GW" simply repeats the player's most recent score; "Form (avg L3)"
    # repeats their 3-match rolling average. Both are leakage-free.
    baselines = {
        'Last GW points':  df_test['lag_1_total_points'].fillna(0),
        'Form (avg L3)':   df_test['rolling_3_avg_total_points'].fillna(0),
    }

    print("\n" + "=" * 52)
    print(f"{'Model / baseline':<24}{'MAE':>8}{'RMSE':>9}{'vs MAE':>10}")
    print("-" * 52)
    model_mae, model_rmse = _metrics(y_test, predictions)
    print(f"{'LightGBM (ours)':<24}{model_mae:>8.4f}{model_rmse:>9.4f}{'--':>10}")
    for name, preds in baselines.items():
        b_mae, b_rmse = _metrics(y_test, preds)
        delta = (b_mae - model_mae) / b_mae * 100  # % the model improves on it
        print(f"{name:<24}{b_mae:>8.4f}{b_rmse:>9.4f}{delta:>9.1f}%")
    print("=" * 52)

    # --- Segmented metrics --------------------------------------------------
    # The dataset is dominated by players who did not feature, where predicting
    # ~0 is trivial. Reporting the "played" subset shows real-world skill.
    played = df_test['minutes'] > 0
    p_mae, p_rmse = _metrics(y_test[played], predictions[played])
    print(f"\nPlayers who featured (minutes>0, n={played.sum()}):")
    print(f"  MAE: {p_mae:.4f} | RMSE: {p_rmse:.4f}")

    print("\nMAE by position (featured players only):")
    seg = df_test[played].copy()
    seg['abs_err'] = (y_test[played] - predictions[played]).abs()
    by_pos = seg.groupby('position')['abs_err'].agg(['mean', 'count'])
    for pos, r in by_pos.iterrows():
        print(f"  {pos:<4} MAE: {r['mean']:.4f}  (n={int(r['count'])})")

    df_test.to_csv(test_data_path, index=False)
    print(f"\nSaved predictions to {test_data_path}")

    return df_test, model, X_test, y_test


def plot_feature_importance(model, X_test, y_test, features,
                            output_path="reports/feature_importance.png"):
    """Plot permutation feature importance on the featured-player subset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("\nCalculating permutation importance...")

    # Evaluate importance where it matters: rows for players who featured.
    mask = (X_test['minutes'] > 0).values if 'minutes' in X_test else np.ones(len(X_test), bool)
    Xs, ys = X_test[mask].head(2000), y_test[mask].head(2000)
    result = permutation_importance(model, Xs, ys, n_repeats=5,
                                    random_state=42, n_jobs=-1)

    feature_imp = pd.DataFrame(
        {'Value': result.importances_mean, 'Feature': features}
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Value", y="Feature",
                data=feature_imp.sort_values(by="Value", ascending=False).head(20))
    plt.title('Top 20 Features (Permutation Importance)')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved feature importance plot to {output_path}")


if __name__ == "__main__":
    model_file = "models/histgb_model.pkl"
    test_data_file = "data/processed/test_data_with_targets.csv"

    model, features = load_model(model_file)
    results_df, model, X_test, y_test = evaluate_model(model, features, test_data_file)
    plot_feature_importance(model, X_test, y_test, features)

    # Show some sample predictions for players who actually played.
    sample = (results_df[results_df['minutes'] > 0]
              [['name', 'GW', 'position', 'target_next_gw_points', 'predicted_points']]
              .head(15))
    print("\nSample Predictions (featured players):")
    print(sample.to_string(index=False))
