"""Ablation study + error / calibration analysis.

  * Ablation: retrain the model dropping one feature *group* at a time and
    measure how much validation MAE worsens -- a clean way to quantify what each
    family of features contributes. Selection is on the 2022-23 validation set.
  * Error analysis: residual diagnostics, MAE by position and by points
    magnitude, and a calibration curve, computed on the 2023-24 test predictions.

Outputs: a printed report plus reports/ablation.png and reports/error_analysis.png.

    python src/models/analysis.py
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

from train_model import (load_features, split_data, prepare_features, REG_PARAMS)

warnings.filterwarnings("ignore")


# Map each feature to a human-readable group (by name pattern).
def feature_group(f):
    if f.startswith('is_'):
        return 'position'
    if f in ('home_form_points', 'away_form_points', 'form_at_next_venue',
             'next_was_home', 'was_home_int'):
        return 'venue'
    if 'opp_' in f or f in ('team_att_form', 'team_def_form'):
        return 'fixture_difficulty'
    if f in ('ewma_total_points', 'season_ppg', 'form_trend_points',
             'rolling_5_std_total_points', 'rolling_5_points_per_90'):
        return 'form_summary'
    if f in ('rolling_5_start_rate', 'days_since_last', 'games_last_14d'):
        return 'availability'
    if f in ('price_change_5', 'points_per_million'):
        return 'price'
    if f.startswith('rolling_') or f.startswith('lag_'):
        return 'rolling_history'
    return 'current_match'   # raw current-GW stats, transfers, value, GW, etc.


def fit(features, X_tr, y_tr, X_val, y_val):
    m = LGBMRegressor(**REG_PARAMS)
    m.fit(X_tr[features], y_tr, eval_set=[(X_val[features], y_val)],
          eval_metric='l1',
          callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    return m, m.predict(X_val[features])


def run_ablation(X_tr, y_tr, X_val, y_val, feats):
    groups = {}
    for f in feats:
        groups.setdefault(feature_group(f), []).append(f)

    _, base_pred = fit(feats, X_tr, y_tr, X_val, y_val)
    base = mean_absolute_error(y_val, base_pred)

    print("\n" + "=" * 60)
    print("ABLATION STUDY  (validation MAE rise when a group is removed)")
    print("-" * 60)
    print(f"{'Feature group':<22}{'# feats':>8}{'val MAE':>10}{'ΔMAE':>10}")
    print(f"{'FULL MODEL':<22}{len(feats):>8}{base:>10.4f}{'--':>10}")
    rows = []
    for g, cols in groups.items():
        keep = [f for f in feats if f not in cols]
        _, pred = fit(keep, X_tr, y_tr, X_val, y_val)
        mae = mean_absolute_error(y_val, pred)
        rows.append((g, len(cols), mae, mae - base))
    for g, n, mae, d in sorted(rows, key=lambda r: -r[3]):
        print(f"{g:<22}{n:>8}{mae:>10.4f}{d:>+10.4f}")
    print("=" * 60)

    # Bar chart of contributions (bigger ΔMAE = more important group).
    rows.sort(key=lambda r: r[3])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([r[0] for r in rows], [r[3] for r in rows], color="#00b35a")
    ax.set_xlabel("Validation MAE increase when group removed")
    ax.set_title("Feature-group importance (ablation)")
    plt.tight_layout()
    plt.savefig("reports/ablation.png", dpi=120)
    print("Saved reports/ablation.png")


def run_error_analysis(test_path="data/processed/test_data_with_targets.csv"):
    df = pd.read_csv(test_path)
    df = df.dropna(subset=['predicted_points', 'target_next_gw_points'])
    y, p = df['target_next_gw_points'].values, df['predicted_points'].values
    resid = p - y
    played = df['minutes'] > 0

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS  (2023-24 test set)")
    print("-" * 60)
    print(f"Mean residual (bias): {resid.mean():+.4f}  (≈0 ⇒ unbiased)")
    print(f"Overall MAE {mean_absolute_error(y, p):.3f} | "
          f"featured MAE {mean_absolute_error(y[played], p[played]):.3f}")

    print("\nMAE by actual-points bucket (featured players):")
    seg = df[played].copy()
    seg['bucket'] = pd.cut(seg['target_next_gw_points'],
                           [-99, 1, 3, 5, 8, 999],
                           labels=['≤1', '2-3', '4-5', '6-8', '9+'])
    by_b = seg.groupby('bucket').apply(
        lambda g: pd.Series({
            'MAE': mean_absolute_error(g['target_next_gw_points'], g['predicted_points']),
            'n': len(g)}))
    for b, r in by_b.iterrows():
        print(f"  {str(b):<5} MAE {r['MAE']:.3f}  (n={int(r['n'])})")
    print("  ⇒ error grows with the haul — the model is conservative on big returns")

    # Four-panel diagnostic figure.
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].hist(resid, bins=60, color="#37003c")
    ax[0, 0].axvline(0, color="#00b35a", lw=2)
    ax[0, 0].set_title("Residuals (pred - actual)")
    ax[0, 0].set_xlabel("residual")

    # Calibration: bin by prediction, plot mean actual.
    cal = df[played].copy()
    cal['pbin'] = pd.qcut(cal['predicted_points'], 10, duplicates='drop')
    g = cal.groupby('pbin').agg(pred=('predicted_points', 'mean'),
                                actual=('target_next_gw_points', 'mean'))
    ax[0, 1].plot(g['pred'], g['actual'], 'o-', color="#00b35a")
    lim = [0, max(g['pred'].max(), g['actual'].max())]
    ax[0, 1].plot(lim, lim, '--', color="gray")
    ax[0, 1].set_title("Calibration (featured players)")
    ax[0, 1].set_xlabel("predicted"); ax[0, 1].set_ylabel("mean actual")

    pos_mae = (df[played].groupby('position')
               .apply(lambda d: mean_absolute_error(
                   d['target_next_gw_points'], d['predicted_points'])))
    ax[1, 0].bar(pos_mae.index, pos_mae.values, color="#04a5c7")
    ax[1, 0].set_title("Featured MAE by position")

    ax[1, 1].scatter(p[played], y[played], s=4, alpha=0.15, color="#e90052")
    ax[1, 1].plot(lim, lim, '--', color="gray")
    ax[1, 1].set_title("Predicted vs actual (featured)")
    ax[1, 1].set_xlabel("predicted"); ax[1, 1].set_ylabel("actual")

    plt.tight_layout()
    plt.savefig("reports/error_analysis.png", dpi=120)
    print("\nSaved reports/error_analysis.png")


def main():
    df = load_features("data/processed/engineered_features.csv")
    train_df, val_df, _ = split_data(df)
    X_tr, y_tr, X_val, y_val, _, _, feats = prepare_features(train_df, val_df, val_df)
    run_ablation(X_tr, y_tr, X_val, y_val, feats)
    run_error_analysis()


if __name__ == "__main__":
    main()
