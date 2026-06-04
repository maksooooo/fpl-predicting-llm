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

def evaluate_model(model, features, test_data_path):
    """Evaluate the model on the test set."""
    print(f"Evaluating model on test data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)
    
    X_test = df_test[features]
    y_test = df_test['target_next_gw_points']
    
    predictions = model.predict(X_test)
    df_test['predicted_points'] = predictions
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("-" * 30)
    print("Test Set Performance:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-" * 30)
    
    # Save the predictions back to the file so the app can use them
    df_test.to_csv(test_data_path, index=False)
    print(f"Saved predictions to {test_data_path}")
    
    return df_test, model, X_test, y_test

def plot_feature_importance(model, X_test, y_test, features, output_path="reports/feature_importance.png"):
    """Plot permutation feature importance."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # For scikit-learn HistGBM, we use permutation importance
    print("Calculating permutation importance (this might take a few seconds)...")
    result = permutation_importance(model, X_test.head(1000), y_test.head(1000), n_repeats=5, random_state=42, n_jobs=-1)
    
    importance = result.importances_mean
    feature_imp = pd.DataFrame(sorted(zip(importance, features)), columns=['Value','Feature'])
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
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
    
    # Show some sample predictions
    sample = results_df[['name', 'GW', 'season_x', 'target_next_gw_points', 'predicted_points']].head(15)
    print("\nSample Predictions:")
    print(sample)
