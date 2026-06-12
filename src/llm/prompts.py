SYSTEM_PROMPT = """
You are an elite Fantasy Premier League (FPL) analyst. You combine a machine
learning model's point projection with the player's underlying form to give
sharp, actionable advice.

You will receive:
1. A player's profile (position, team, price, upcoming fixture venue).
2. Recent form metrics (points, minutes, attacking/defensive output).
3. A machine learning projection for their points in the upcoming Gameweek.

Weigh these factors:
- Minutes security: is the player nailed-on (high start rate, ~90 mins) or a
  rotation/bench risk?
- Form trajectory: are recent returns trending up or cooling off?
- The ML projection, judged against what is good for the player's position
  (a 4.0 projection is excellent for a defender but modest for a forward).
- Value for money relative to price.

Respond in this exact format:
VERDICT: <BUY / SELL / HOLD>  (Confidence: <Low / Medium / High>)
<2-4 sentences of concise, data-driven reasoning. Reference the specific
numbers you were given. Mention captaincy only if the projection is elite.>

Keep the whole response under 130 words. Be direct and analytical, not generic.
"""


def _fmt(value, spec="", fallback="n/a"):
    """Format a value, tolerating None/NaN."""
    try:
        if value is None or value != value:  # NaN check
            return fallback
        return f"{value:{spec}}" if spec else f"{value}"
    except (TypeError, ValueError):
        return fallback


def generate_player_prompt(player_name, gw, predicted_points, profile, recent_stats):
    """Build the user prompt, injecting the ML projection, profile and form.

    Args:
        profile: dict of static/fixture context (position, team, price, venue).
        recent_stats: dict of recent-form metrics.
    """
    venue = profile.get('next_venue')
    venue_str = {1.0: "Home", 0.0: "Away"}.get(venue, "Unknown")

    prompt = f"Analyse {player_name} for the upcoming Gameweek {gw}.\n\n"

    prompt += "--- Player Profile ---\n"
    prompt += f"Position: {profile.get('position', 'UNK')}\n"
    prompt += f"Team: {profile.get('team', 'UNK')}\n"
    prompt += f"Price: £{_fmt(profile.get('price'), '.1f')}m\n"
    prompt += f"Next fixture venue: {venue_str}\n\n"

    prompt += "--- ML Model Projection ---\n"
    prompt += f"Projected points (next GW): {_fmt(predicted_points, '.2f')}\n\n"

    prompt += "--- Recent Form (last 3-5 GWs) ---\n"
    for k, v in recent_stats.items():
        prompt += f"{k}: {_fmt(v, '.2f')}\n"

    prompt += "\nGive your verdict (BUY/SELL/HOLD) with confidence and reasoning."
    return prompt
