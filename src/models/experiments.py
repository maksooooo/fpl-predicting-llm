"""Model-selection experiments.

Everything here is selected on the VALIDATION season (2022-23). The test season
(2023-24) is never touched -- final numbers are produced by evaluate.py once a
winning configuration is locked in. Run with:

    python src/models/experiments.py
"""
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

from train_model import (load_features, split_data, prepare_features,
                         TRAIN_SEASONS, VAL_SEASONS)

warnings.filterwarnings("ignore")

BASE = dict(learning_rate=0.05, num_leaves=63, max_depth=6, n_estimators=3000,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=50, random_state=42, n_jobs=-1, verbose=-1)


def fit_eval(params, objective, X_tr, y_tr, X_val, y_val, mins_val,
             clip_neg=False, tweedie_power=None):
    """Train one config and return validation metrics (overall + featured)."""
    p = dict(BASE, **params, objective=objective)
    if tweedie_power is not None:
        p['tweedie_variance_power'] = tweedie_power
    y_fit = y_tr.clip(lower=0) if clip_neg else y_tr

    model = LGBMRegressor(**p)
    model.fit(X_tr, y_fit, eval_set=[(X_val, y_val.clip(lower=0) if clip_neg else y_val)],
              eval_metric='l1',
              callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])

    pred = model.predict(X_val)
    overall = mean_absolute_error(y_val, pred)
    played = mins_val > 0
    featured = mean_absolute_error(y_val[played], pred[played])
    return overall, featured, model.best_iteration_


def main():
    df = load_features("data/processed/engineered_features.csv")
    train_df, val_df, _ = split_data(df)
    X_tr, y_tr, X_val, y_val, _, _, feats = prepare_features(train_df, val_df, val_df)
    mins_val = val_df['minutes'].values
    print(f"\n{len(feats)} features | train {len(X_tr)} | val {len(X_val)}\n")

    print("=" * 64)
    print(f"{'Experiment':<34}{'val MAE':>10}{'feat MAE':>10}{'iters':>8}")
    print("-" * 64)

    def row(name, objective, params=None, **kw):
        o, f, it = fit_eval(params or {}, objective, X_tr, y_tr, X_val, y_val,
                            mins_val, **kw)
        print(f"{name[:33]:<34}{o:>10.4f}{f:>10.4f}{it:>8}")
        return o, f

    # --- 1. Objective comparison (current default = MAE/l1) ---
    row("l1  (MAE, current)", "l1")
    row("huber", "huber")
    row("l2  (MSE)", "l2")
    row("poisson", "poisson", clip_neg=True)
    row("tweedie p=1.3", "tweedie", clip_neg=True, tweedie_power=1.3)

    # --- 2. Feature ablation: do the team/opponent/EWMA features help? ---
    print("-" * 64)
    context_feats = ['team_att_form', 'team_def_form', 'next_opp_att_form',
                     'next_opp_def_form', 'ewma_total_points', 'season_ppg',
                     'days_since_last']
    keep = [c for c in feats if c not in context_feats]
    Xtr2, Xval2 = X_tr[keep], X_val[keep]
    m = LGBMRegressor(**dict(BASE, objective='l1'))
    m.fit(Xtr2, y_tr, eval_set=[(Xval2, y_val)], eval_metric='l1',
          callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    pred = m.predict(Xval2)
    played = mins_val > 0
    print(f"{'l1 WITHOUT context feats':<34}"
          f"{mean_absolute_error(y_val, pred):>10.4f}"
          f"{mean_absolute_error(y_val[played], pred[played]):>10.4f}"
          f"{m.best_iteration_:>8}")

    # --- 3. l1 hyperparameter search ---
    print("-" * 64)
    grid = [
        dict(learning_rate=0.05, num_leaves=63, min_child_samples=50),  # current
        dict(learning_rate=0.03, num_leaves=31, min_child_samples=80),
        dict(learning_rate=0.03, num_leaves=63, min_child_samples=60),
        dict(learning_rate=0.05, num_leaves=31, min_child_samples=100),
        dict(learning_rate=0.02, num_leaves=63, min_child_samples=80,
             colsample_bytree=0.7, subsample=0.7),
        dict(learning_rate=0.05, num_leaves=15, min_child_samples=120),
    ]
    best = None
    for g in grid:
        o, f, it = fit_eval(g, "l1", X_tr, y_tr, X_val, y_val, mins_val)
        print(f"{('l1 ' + str(g))[:33]:<34}{o:>10.4f}{f:>10.4f}{it:>8}")
        if best is None or o < best[0]:
            best = (o, f, g, it)
    print("=" * 64)
    print(f"\nBest l1 config (val MAE {best[0]:.4f}, feat {best[1]:.4f}, "
          f"iters {best[3]}):\n  {best[2]}")


if __name__ == "__main__":
    main()
