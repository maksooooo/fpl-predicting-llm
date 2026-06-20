import pandas as pd

def get_player_stats(df, player_name, gameweek):
    """Filter the DataFrame by player name and GW and return a stats dict."""
    player_data = df[(df['name'] == player_name) & (df['GW'] == gameweek)]
    if player_data.empty:
        return None
        
    row = player_data.iloc[0]
    
    # Safely get values with fallbacks
    predicted_points = row.get('predicted_points', 0)
    if pd.isna(predicted_points): predicted_points = 0.0
    
    form = row.get('rolling_3_avg_total_points', 0)
    if pd.isna(form): form = 0.0
    
    xg = row.get('expected_goals', 0)
    if pd.isna(xg): xg = 0.0
    
    xa = row.get('expected_assists', 0)
    if pd.isna(xa): xa = 0.0
    
    price = row.get('value', 0)
    if pd.notna(price): price = price / 10.0
    else: price = 0.0
    
    selected_by = row.get('selected_by_percent', 0)
    if pd.isna(selected_by): selected_by = 0.0
    
    return {
        'name': str(row.get('name', player_name)),
        'team': str(row.get('team_x', 'UNK')),
        'position': str(row.get('position', 'UNK')),
        'predicted_points': float(predicted_points),
        'form': float(form),
        'xg': float(xg),
        'xa': float(xa),
        'price': float(price),
        'selected_by': float(selected_by)
    }
