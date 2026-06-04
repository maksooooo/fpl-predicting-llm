SYSTEM_PROMPT = """
You are an elite Fantasy Premier League (FPL) manager and data scientist.
Your job is to provide concise, actionable advice to users about specific players.
You will be provided with:
1. The player's recent historical stats (e.g., points, minutes played).
2. A Machine Learning model's prediction for their points in the upcoming Gameweek.

Synthesize this information. Consider factors like:
- Is the player nailed on (playing 90 mins)?
- Is their form good?
- Does the ML model predict a high return?

Provide a verdict: BUY, SELL, or HOLD.
Keep your response under 150 words. Be direct and analytical.
"""

def generate_player_prompt(player_name, gw, predicted_points, recent_stats):
    """
    Generates the user prompt injecting the ML predictions and stats.
    """
    prompt = f"Analyze {player_name} for the upcoming Gameweek {gw}.\n\n"
    prompt += f"--- ML Model Prediction ---\n"
    prompt += f"Predicted Points next GW: {predicted_points:.2f}\n\n"
    
    prompt += f"--- Recent Form (Last 3 GWs) ---\n"
    for k, v in recent_stats.items():
        if isinstance(v, float):
            prompt += f"{k}: {v:.2f}\n"
        else:
            prompt += f"{k}: {v}\n"
            
    prompt += "\nBased on this data, what is your verdict (BUY/SELL/HOLD) and brief reasoning?"
    return prompt
