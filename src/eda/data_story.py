"""Problem-analysis / EDA data story.

Generates a single presentation-ready figure (reports/eda_overview.png) plus a
printed summary that frames *why the problem is hard* and *what signal exists* —
the evidence the committee looks for under "analysis of the problem and data".

    python src/eda/data_story.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

GREEN, PURPLE, MAGENTA, CYAN = "#00b35a", "#37003c", "#e90052", "#0480a5"
DATA = "data/processed/engineered_features.csv"


def main():
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv(DATA, low_memory=False)
    y = df['target_next_gw_points']
    played = df['minutes'] > 0

    # ---- Printed data story ------------------------------------------------
    print("=" * 60)
    print("DATA STORY  —  why this problem is hard, and what signal exists")
    print("-" * 60)
    print(f"Rows (player-gameweeks): {len(df):,}")
    print(f"Seasons: {sorted(df['season_x'].unique())}")
    print(f"Share of rows where the player did NOT feature: "
          f"{(~played).mean():.1%}")
    print(f"Share of next-GW targets that are 0 points:     "
          f"{(y == 0).mean():.1%}")
    print(f"Target mean {y.mean():.2f} | median {y.median():.0f} | "
          f"max {y.max():.0f}  (heavy right tail)")
    corr = df['rolling_3_avg_total_points'].corr(y)
    print(f"Correlation of 3-game form with next-GW points: {corr:.3f} "
          f"(real but weak — football is noisy)")
    print("\nMean next-GW points by position (featured players only):")
    print(df[played].groupby('position')['target_next_gw_points']
          .mean().round(2).to_string())
    print("=" * 60)

    # ---- Figure ------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("FPL points prediction — problem & data analysis",
                 fontsize=16, weight='bold')

    # (1) Zero-inflated target distribution.
    ax[0, 0].hist(y.clip(-2, 18), bins=40, color=PURPLE)
    ax[0, 0].set_title("Target is zero-inflated & right-skewed")
    ax[0, 0].set_xlabel("next-GW points"); ax[0, 0].set_ylabel("rows")
    ax[0, 0].axvline(0, color=MAGENTA, lw=2)
    ax[0, 0].text(0.5, 0.85, f"{(y == 0).mean():.0%} are 0",
                  transform=ax[0, 0].transAxes, color=MAGENTA, weight='bold')

    # (2) Most rows are non-players.
    counts = [(~played).sum(), played.sum()]
    ax[0, 1].bar(["did not\nfeature", "featured"], counts,
                 color=[MAGENTA, GREEN])
    ax[0, 1].set_title("Most rows are players who didn't play")
    for i, c in enumerate(counts):
        ax[0, 1].text(i, c, f"{c/len(df):.0%}", ha='center', va='bottom')

    # (3) Scoring differs sharply by position.
    pos = df[played].groupby('position')['target_next_gw_points'].mean()
    pos = pos.reindex(['GK', 'DEF', 'MID', 'FWD']).dropna()
    ax[0, 2].bar(pos.index, pos.values, color=CYAN)
    ax[0, 2].set_title("Mean points by position (featured)")
    ax[0, 2].set_ylabel("mean next-GW points")

    # (4) Form carries signal — but noisy (binned form vs outcome).
    tmp = df[played].copy()
    tmp['form_bin'] = pd.qcut(tmp['rolling_3_avg_total_points'], 8,
                              duplicates='drop')
    g = tmp.groupby('form_bin').agg(
        form=('rolling_3_avg_total_points', 'mean'),
        outcome=('target_next_gw_points', 'mean'))
    ax[1, 0].plot(g['form'], g['outcome'], 'o-', color=GREEN)
    ax[1, 0].set_title(f"Recent form predicts points (r={corr:.2f})")
    ax[1, 0].set_xlabel("3-game form (avg pts)")
    ax[1, 0].set_ylabel("mean next-GW points")

    # (5) Long tail among featured players (variance is the enemy).
    ax[1, 1].hist(y[played].clip(-2, 18), bins=20, color=PURPLE)
    ax[1, 1].set_title("Even featured players: low median, long tail")
    ax[1, 1].set_xlabel("next-GW points (featured)")

    # (6) Which simple signals correlate with the target.
    cand = ['rolling_3_avg_total_points', 'rolling_5_avg_total_points',
            'season_ppg', 'ewma_total_points', 'rolling_5_avg_ict_index',
            'rolling_5_avg_minutes', 'value', 'next_opp_att_form']
    cand = [c for c in cand if c in df.columns]
    cors = df[cand].corrwith(y).sort_values()
    ax[1, 2].barh([c.replace('_', ' ')[:22] for c in cors.index], cors.values,
                  color=CYAN)
    ax[1, 2].set_title("Correlation of key features with target")
    ax[1, 2].axvline(0, color='gray', lw=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "reports/eda_overview.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
