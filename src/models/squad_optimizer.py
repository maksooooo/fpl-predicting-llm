"""Optimal FPL squad selection from the model's projections.

Turns point *predictions* into a *decision*: pick the best 15-player squad for a
gameweek under the real Fantasy Premier League rules, then the best starting XI
and captain — by solving a Mixed-Integer Linear Program (HiGHS via SciPy).

Rules encoded:
  * Squad of 15: exactly 2 GK, 5 DEF, 5 MID, 3 FWD.
  * Budget: total price <= £100.0m.
  * At most 3 players from any single real club.
  * Starting XI of 11 with a legal formation (1 GK; 3-5 DEF; 2-5 MID; 1-3 FWD).
  * Captain (one starter) scores double.
Objective: maximise projected points of the starting XI + the captain bonus.

    python src/models/squad_optimizer.py --gw 10
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

BUDGET = 1000          # £100.0m, prices stored in tenths of a million
SQUAD = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
XI_MIN = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
XI_MAX = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
MAX_PER_CLUB = 3


def load_gameweek(path, gw):
    df = pd.read_csv(path)
    df = df[df['GW'] == gw].dropna(subset=['predicted_points', 'value']).copy()
    df['position'] = df['position'].replace({'GKP': 'GK'})
    df = df[df['position'].isin(SQUAD)]
    # One row per player (guard against any duplicates within a GW).
    df = df.sort_values('predicted_points', ascending=False)
    df = df.drop_duplicates(subset=['name', 'team_x']).reset_index(drop=True)
    return df


def optimize_squad(df):
    """Solve the MILP and return (squad_df, summary dict)."""
    n = len(df)
    pred = df['predicted_points'].to_numpy()
    price = df['value'].to_numpy()
    pos = df['position'].to_numpy()
    teams = df['team_x'].to_numpy()

    # Decision variables: [squad x (n) | starter y (n) | captain c (n)].
    N = 3 * n
    def X(i): return i             # squad
    def Y(i): return n + i         # starter
    def C(i): return 2 * n + i     # captain

    # Objective (maximise → minimise the negative): starter points + captain bonus.
    obj = np.zeros(N)
    obj[n:2 * n] = -pred           # each starter scores its points
    obj[2 * n:] = -pred            # captain scores again (doubling)

    cons = []

    def row(coeffs):
        r = lil_matrix((1, N))
        for idx, v in coeffs:
            r[0, idx] = v
        return r

    # Squad composition by position (== required count).
    for p, k in SQUAD.items():
        cons.append(LinearConstraint(
            row([(X(i), 1) for i in range(n) if pos[i] == p]), k, k))

    # Budget.
    cons.append(LinearConstraint(
        row([(X(i), price[i]) for i in range(n)]), 0, BUDGET))

    # Max players per club.
    for t in np.unique(teams):
        cons.append(LinearConstraint(
            row([(X(i), 1) for i in range(n) if teams[i] == t]), 0, MAX_PER_CLUB))

    # Starting XI: 11 starters, each must be in the squad.
    cons.append(LinearConstraint(row([(Y(i), 1) for i in range(n)]), 11, 11))
    for i in range(n):
        cons.append(LinearConstraint(row([(Y(i), 1), (X(i), -1)]), -np.inf, 0))

    # Legal formation for the starters.
    for p in SQUAD:
        idx = [i for i in range(n) if pos[i] == p]
        cons.append(LinearConstraint(
            row([(Y(i), 1) for i in idx]), XI_MIN[p], XI_MAX[p]))

    # Exactly one captain, who must be a starter.
    cons.append(LinearConstraint(row([(C(i), 1) for i in range(n)]), 1, 1))
    for i in range(n):
        cons.append(LinearConstraint(row([(C(i), 1), (Y(i), -1)]), -np.inf, 0))

    res = milp(c=obj, constraints=cons, integrality=np.ones(N),
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"Optimiser failed: {res.message}")

    sol = res.x
    df = df.copy()
    df['in_squad'] = sol[:n] > 0.5
    df['starter'] = sol[n:2 * n] > 0.5
    df['captain'] = sol[2 * n:] > 0.5
    squad = df[df['in_squad']].copy()

    xi = squad[squad['starter']]
    proj = xi['predicted_points'].sum() + squad.loc[squad['captain'],
                                                    'predicted_points'].sum()
    summary = {
        'projected_points': round(float(proj), 1),
        'cost': squad['value'].sum() / 10,
        'formation': '-'.join(str((xi['position'] == p).sum())
                              for p in ['DEF', 'MID', 'FWD']),
    }
    return squad, summary


def print_squad(squad, summary):
    order = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    squad = squad.sort_values(
        by=['starter', 'position', 'predicted_points'],
        key=lambda s: s.map(order) if s.name == 'position' else s,
        ascending=[False, True, False])

    print(f"\nProjected points: {summary['projected_points']}  |  "
          f"Cost: £{summary['cost']:.1f}m  |  Formation: {summary['formation']}")
    print("-" * 60)
    print("STARTING XI")
    for _, r in squad[squad['starter']].iterrows():
        cap = "  (C)" if r['captain'] else ""
        print(f"  {r['position']:<3} {r['name']:<24} {r['team_x']:<12} "
              f"£{r['value']/10:>4.1f}m  {r['predicted_points']:>4.1f}{cap}")
    print("BENCH")
    for _, r in squad[~squad['starter']].iterrows():
        print(f"  {r['position']:<3} {r['name']:<24} {r['team_x']:<12} "
              f"£{r['value']/10:>4.1f}m  {r['predicted_points']:>4.1f}")


def build_optimal_squad(path="data/processed/test_data_with_targets.csv", gw=10):
    df = load_gameweek(path, gw)
    return optimize_squad(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=10)
    ap.add_argument("--path", default="data/processed/test_data_with_targets.csv")
    args = ap.parse_args()

    squad, summary = build_optimal_squad(args.path, args.gw)
    print(f"\n⚽ Optimal FPL squad for Gameweek {args.gw} (2023-24, by projection)")
    print_squad(squad, summary)
