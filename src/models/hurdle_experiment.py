"""Two-stage hurdle model vs the single-regressor baseline.

Hurdle idea (classic for zero-inflated data): split the problem into
  stage 1 -- P(player appears next GW)            [classifier]
  stage 2 -- E[points | the player appears]        [regressor on appeared rows]
and combine as  prediction = P(appear) * E[points | appear].

Both models are trained on the train seasons and compared on the 2022-23
VALIDATION season only. The test season is untouched. Adopt the hurdle only if
it beats the baseline here.

    python src/models/hurdle_experiment.py
"""
import warnings
import numpy as np
from lightgbm import (LGBMRegressor, LGBMClassifier,
                      early_stopping, log_evaluation)
from sklearn.metrics import mean_absolute_error, roc_auc_score

from train_model import load_features, split_data, prepare_features

warnings.filterwarnings("ignore")

REG = dict(objective='mae', learning_rate=0.03, num_leaves=63, max_depth=6,
           n_estimators=3000, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, min_child_samples=60, random_state=42,
           n_jobs=-1, verbose=-1)
CLF = dict(objective='binary', learning_rate=0.03, num_leaves=63, max_depth=6,
           n_estimators=3000, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, min_child_samples=60, random_state=42,
           n_jobs=-1, verbose=-1)


def metrics(name, y, pred, played):
    overall = mean_absolute_error(y, pred)
    feat = mean_absolute_error(y[played], pred[played])
    print(f"{name:<28}{overall:>10.4f}{feat:>10.4f}")
    return overall, feat


def main():
    df = load_features("data/processed/engineered_features.csv")
    train_df, val_df, _ = split_data(df)
    X_tr, y_tr, X_val, y_val, _, _, feats = prepare_features(
        train_df, val_df, val_df)

    # Secondary target: did the player feature in the gameweek we predict?
    appear_tr = (train_df['target_next_gw_minutes'] > 0).astype(int).values
    played_val = (val_df['minutes'] > 0).values  # featured in CURRENT gw (segment)

    print(f"\n{len(feats)} features | train {len(X_tr)} | val {len(X_val)} | "
          f"train appearance rate {appear_tr.mean():.3f}\n")
    print("=" * 48)
    print(f"{'Model':<28}{'val MAE':>10}{'feat MAE':>10}")
    print("-" * 48)

    # --- Baseline: single MAE regressor (current production model) ---
    base = LGBMRegressor(**REG)
    base.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='l1',
             callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    base_pred = base.predict(X_val)
    metrics("Single regressor (baseline)", y_val, base_pred, played_val)

    # --- Hurdle stage 1: P(appear next GW) ---
    # Use a held-out slice of train for the classifier's early stopping so we
    # don't peek at validation.
    n = len(X_tr)
    cut = int(n * 0.85)
    clf = LGBMClassifier(**CLF)
    clf.fit(X_tr.iloc[:cut], appear_tr[:cut],
            eval_set=[(X_tr.iloc[cut:], appear_tr[cut:])], eval_metric='auc',
            callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    p_appear_val = clf.predict_proba(X_val)[:, 1]

    # --- Hurdle stage 2: E[points | appeared] (train on appeared rows only) ---
    mask = appear_tr == 1
    reg = LGBMRegressor(**REG)
    reg.fit(X_tr[mask], y_tr[mask],
            eval_set=[(X_val, y_val)], eval_metric='l1',
            callbacks=[early_stopping(60, verbose=False), log_evaluation(0)])
    pts_given_play_val = reg.predict(X_val)

    hurdle_pred = p_appear_val * pts_given_play_val
    metrics("Hurdle (P(play) x E[pts])", y_val, hurdle_pred, played_val)

    # A simple blend often banks the best of both.
    blend = 0.5 * base_pred + 0.5 * hurdle_pred
    metrics("Blend (50/50)", y_val, blend, played_val)
    print("=" * 48)

    auc = roc_auc_score((val_df['minutes'] > 0).astype(int), p_appear_val)
    print(f"\nStage-1 appearance classifier ROC-AUC (vs current-GW mins>0 proxy): "
          f"{auc:.3f}")
    print(f"Stage-1 best iters: {clf.best_iteration_} | "
          f"Stage-2 best iters: {reg.best_iteration_}")


if __name__ == "__main__":
    main()
