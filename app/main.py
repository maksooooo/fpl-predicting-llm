# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import altair as alt
import pickle

# Add the parent directory to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.agent import FPLAssistant

# Configure page settings
st.set_page_config(
    page_title="FPL AI Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS reliably
def load_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css('style.css')

# Load data for the UI
@st.cache_data
def load_prediction_data():
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/test_data_with_targets.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()

df = load_prediction_data()

# Load ML Model for Explainability
@st.cache_resource
def load_ml_model():
    model_path = os.path.join(os.path.dirname(__file__), '../models/histgb_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['features']
    return None, None

model, features = load_ml_model()

# Initialize Assistant
@st.cache_resource
def get_assistant():
    return FPLAssistant()

assistant = get_assistant()


def render_verdict(advice: str):
    """Split the LLM reply into a colour-coded BUY/SELL/HOLD stamp + reasoning.

    Returns (stamp_html, body_text). Falls back gracefully if the model didn't
    follow the expected 'VERDICT: ...' format.
    """
    import re

    verdict, confidence, body = None, None, advice
    m = re.search(r'VERDICT:\s*(BUY|SELL|HOLD)', advice, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()
        c = re.search(r'Confidence:\s*(Low|Medium|High)', advice, re.IGNORECASE)
        confidence = c.group(1).title() if c else None
        # Everything after the first line break becomes the reasoning body.
        parts = advice.split('\n', 1)
        body = parts[1].strip() if len(parts) > 1 else ""

    if not verdict:
        return "", advice  # no recognisable verdict; show the raw reply

    icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[verdict]
    conf_html = (f"<span class='verdict-confidence'>{confidence} confidence</span>"
                 if confidence else "")
    stamp = (f"<div class='verdict-stamp verdict-{verdict.lower()}'>"
             f"{icon} {verdict}{conf_html}</div>")
    return stamp, body


# Primary kit colour per club (used for accents + the pitch marker). Covers
# clubs across all seasons in the dataset; unknown clubs fall back to FPL green.
TEAM_COLORS = {
    "Arsenal": "#EF0107", "Aston Villa": "#95BFE5", "Bournemouth": "#DA291C",
    "Brentford": "#E30613", "Brighton": "#0057B8", "Burnley": "#6C1D45",
    "Cardiff": "#0070B5", "Chelsea": "#034694", "Crystal Palace": "#1B458F",
    "Everton": "#1f6fd6", "Fulham": "#E5E5E5", "Huddersfield": "#0073CF",
    "Hull": "#F18A00", "Leeds": "#FFE100", "Leicester": "#1f6fd6",
    "Liverpool": "#C8102E", "Luton": "#F78F1E", "Man City": "#6CABDD",
    "Man Utd": "#DA291C", "Middlesbrough": "#E21C38", "Newcastle": "#E8E8E8",
    "Norwich": "#00A650", "Nott'm Forest": "#DD0000", "Sheffield Utd": "#EE2737",
    "Southampton": "#D71920", "Spurs": "#7E97D6", "Stoke": "#E03A3E",
    "Sunderland": "#EB172B", "Swansea": "#E5E5E5", "Watford": "#FBEE23",
    "West Brom": "#3a6abf", "West Ham": "#A7263A", "Wolves": "#FDB913",
}


def team_color(team_name: str) -> str:
    return TEAM_COLORS.get(str(team_name), "#00ff87")


def pitch_map_svg(position: str, kit: str) -> str:
    """Vertical pitch SVG with a glowing marker placed at the player's zone."""
    # Attacking upwards: FWD near the top, GK in front of own goal at the base.
    y = {"GK": 244, "GKP": 244, "DEF": 188, "MID": 120, "FWD": 52}.get(str(position), 150)
    line = "rgba(255,255,255,0.30)"
    return f"""
<svg class='pitch-wrap' width='118' height='168' viewBox='0 0 200 280' xmlns='http://www.w3.org/2000/svg'>
  <defs>
    <linearGradient id='grass' x1='0' y1='0' x2='0' y2='1'>
      <stop offset='0' stop-color='#0e5a34'/><stop offset='1' stop-color='#0a3f26'/>
    </linearGradient>
  </defs>
  <rect x='3' y='3' width='194' height='274' rx='10' fill='url(#grass)' stroke='{line}' stroke-width='2'/>
  <rect x='0' y='40' width='200' height='40' fill='rgba(255,255,255,0.03)'/>
  <rect x='0' y='120' width='200' height='40' fill='rgba(255,255,255,0.03)'/>
  <rect x='0' y='200' width='200' height='40' fill='rgba(255,255,255,0.03)'/>
  <line x1='3' y1='140' x2='197' y2='140' stroke='{line}' stroke-width='2'/>
  <circle cx='100' cy='140' r='30' fill='none' stroke='{line}' stroke-width='2'/>
  <circle cx='100' cy='140' r='3' fill='{line}'/>
  <rect x='55' y='3' width='90' height='44' fill='none' stroke='{line}' stroke-width='2'/>
  <rect x='80' y='3' width='40' height='18' fill='none' stroke='{line}' stroke-width='2'/>
  <rect x='55' y='233' width='90' height='44' fill='none' stroke='{line}' stroke-width='2'/>
  <rect x='80' y='259' width='40' height='18' fill='none' stroke='{line}' stroke-width='2'/>
  <g class='pitch-marker'>
    <circle cx='100' cy='{y}' r='15' fill='{kit}' fill-opacity='0.25'/>
    <circle cx='100' cy='{y}' r='8' fill='{kit}' stroke='#ffffff' stroke-width='2'/>
  </g>
</svg>
"""


# --- Optimal squad (Dream Team) ------------------------------------------------
SQUAD_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '../data/processed/test_data_with_targets.csv')


@st.cache_data(show_spinner=False)
def compute_optimal_squad(gw):
    """Solve the budget-constrained best-XV MILP for a gameweek (cached)."""
    from src.models.squad_optimizer import build_optimal_squad
    squad, summary = build_optimal_squad(SQUAD_DATA_PATH, gw)
    return squad, summary


def _chip(row):
    cap = "<span class='chip-cap'>C</span>" if row['captain'] else ""
    return (
        f"<div class='player-chip{' is-captain' if row['captain'] else ''}'>"
        f"<div class='chip-name'>{row['name'].split()[-1]}{cap}</div>"
        f"<div class='chip-meta'>{row['team_x']} · £{row['value']/10:.1f}m</div>"
        f"<div class='chip-pts'>{row['predicted_points']:.1f}</div></div>"
    )


def render_squad_pitch(squad):
    """Render the starting XI on a pitch, grouped GK/DEF/MID/FWD."""
    xi = squad[squad['starter']]
    rows = ""
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        line = xi[xi['position'] == pos].sort_values('predicted_points', ascending=False)
        chips = "".join(_chip(r) for _, r in line.iterrows())
        rows += f"<div class='pitch-row'>{chips}</div>"
    return f"<div class='squad-pitch'>{rows}</div>"


# UI Layout Header
st.markdown("<div class='kicker'>⚽ Matchday Intelligence</div>", unsafe_allow_html=True)
st.markdown("<h1>FPL <span class='highlight'>Scout AI</span></h1>", unsafe_allow_html=True)
st.markdown("<div class='lower-third'></div>", unsafe_allow_html=True)
st.markdown("<p style='color:#b9a9c9; font-size:1.1rem; margin-bottom: 2rem;'>Machine learning projections + an AI gaffer's verdict. Pick your XI like a pro.</p>", unsafe_allow_html=True)

if df.empty:
    st.error("Prediction data not found. Please run the ML pipeline first.")
    st.stop()

# Sidebar controls
st.sidebar.markdown("<h2 style='margin-bottom:0;'>📋 Team Sheet</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#b9a9c9; font-size:0.9rem; margin-bottom:1rem;'>Pick a player to scout</p>", unsafe_allow_html=True)

players = sorted(df['name'].unique())
selected_player = st.sidebar.selectbox("Search Player", players, index=players.index("Erling Haaland") if "Erling Haaland" in players else 0)

# Filter GWs for selected player
player_gws = sorted(df[df['name'] == selected_player]['GW'].unique())
selected_gw = st.sidebar.selectbox("Gameweek", player_gws)

st.sidebar.markdown("---")
st.sidebar.markdown("**STARTING XI:**")
st.sidebar.markdown("<div style='font-family:monospace;font-size:0.8rem;color:#00ff87'>⚙️ ML: LightGBM Regressor<br>🎙️ AI: OpenRouter LLM</div>", unsafe_allow_html=True)

# Main Content Area
player_data = df[(df['name'] == selected_player) & (df['GW'] == selected_gw)]

if player_data.empty:
    st.warning("No data available for this player in the selected Gameweek.")
else:
    row = player_data.iloc[0]
    
    # Safe data fetching (handling NaN values which caused crashes)
    predicted_points = row.get('predicted_points', 0)
    if pd.isna(predicted_points): predicted_points = 0.0
    
    avg_points = row.get('rolling_3_avg_total_points', 0)
    if pd.isna(avg_points): avg_points = 0.0
        
    avg_mins = row.get('rolling_3_avg_minutes', 0)
    if pd.isna(avg_mins): avg_mins = 0.0
    
    # Gamification Logic safely calculated
    form_icon = "🔥" if avg_points >= 6 else "🧊" if avg_points <= 2 else "⚡"
    mins_percent = min(100, int((avg_mins / 90) * 100))
    risk_badge = "<span class='player-badge badge-risk'>⚠️ Rotation Risk</span>" if avg_mins < 60 else ""

    # Position-coloured badge (classic FPL colours: GK/DEF/MID/FWD)
    position = str(row.get('position', 'UNK'))
    pos_class = {"GK": "badge-gk", "GKP": "badge-gk", "DEF": "badge-def",
                 "MID": "badge-mid", "FWD": "badge-fwd"}.get(position, "badge-team")

    # Team kit colour: drives the card accent stripe and the pitch marker.
    team = str(row.get('team_x', 'UNK'))
    kit = team_color(team)
    pitch_svg = pitch_map_svg(position, kit)

    # Scout Card Container - Single line to prevent Markdown code block bugs
    info_html = (
        f"<div><span class='player-badge badge-team'><span class='kit-dot' style='background:{kit};'></span>{team}</span> "
        f"<span class='player-badge {pos_class}'>{position}</span> {risk_badge}"
        f"<h2 style='margin-top:10px; margin-bottom:2px; font-size:2.4rem;'>{selected_player}</h2>"
        f"<div class='gw-tag'>⚽ Gameweek {selected_gw} Profile</div></div>"
    )
    scout_html = (
        f"<div class='scout-card' style='border-left:5px solid {kit};'>"
        f"<div class='scout-flex'>{info_html}{pitch_svg}</div></div>"
    )
    st.markdown(scout_html, unsafe_allow_html=True)
    
    # Metrics Grid inside Scout Card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted (Next GW)", value=f"{predicted_points:.1f} pts")
    with col2:
        st.metric(label="Form (Avg L3)", value=f"{avg_points:.1f} {form_icon}")
    with col3:
        st.metric(label="Avg Mins (L3)", value=f"{avg_mins:.0f}'")
        # Inject Gamified Progress bar - Single line
        st.markdown(f"<div class='progress-bg'><div class='progress-fill' style='width: {mins_percent}%;'></div></div>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True) # Close scout-card
    
    # Dual Column Layout for Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Form History Chart (Interesting UI Addition)
        st.markdown("<h3 style='margin-top: 1rem; color: #e2e8f0;'>Historical Form</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Player's points trajectory up to the selected Gameweek.</p>", unsafe_allow_html=True)
        
        historical_data = df[(df['name'] == selected_player) & (df['GW'] <= selected_gw)].sort_values('GW')
        if not historical_data.empty and 'total_points' in historical_data.columns:
            chart_data = historical_data.set_index('GW')[['total_points']].rename(columns={'total_points': 'Points'})
            st.bar_chart(chart_data, color="#00ff85", height=250)
            
    with chart_col2:
        # Model Explainability / Local Feature Importance
        st.markdown("<h3 style='margin-top: 1rem; color: #e2e8f0;'>Why this Prediction?</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Top 5 stats that drove the ML model's decision.</p>", unsafe_allow_html=True)
        
        if model is not None and features is not None:
            # Extract player features
            X_player = df[(df['name'] == selected_player) & (df['GW'] == selected_gw)][features]
            
            # Use LightGBM pred_contrib=True to get exact SHAP values for this prediction
            contributions = model.predict(X_player, pred_contrib=True)[0]
            feature_contribs = contributions[:-1] # Drop the expected bias term at the end
            
            contrib_df = pd.DataFrame({
                'Feature': features,
                'Contribution': feature_contribs
            })
            
            # Find the top 5 most impactful features (absolute magnitude)
            contrib_df['AbsContribution'] = contrib_df['Contribution'].abs()
            top_contribs = contrib_df.sort_values('AbsContribution', ascending=False).head(5)
            
            # Make the Feature names slightly more readable
            top_contribs['Feature'] = top_contribs['Feature'].str.replace('_', ' ').str.title()
            
            # Plot using Altair
            chart = alt.Chart(top_contribs).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('Contribution:Q', title='Impact on Points (Pts)', axis=alt.Axis(grid=False)),
                y=alt.Y('Feature:N', sort=alt.EncodingSortField(field="AbsContribution", op="sum", order='descending'), title=''),
                color=alt.condition(
                    alt.datum.Contribution > 0,
                    alt.value('#00ff85'),  # Positive impact (Green)
                    alt.value('#ef4444')   # Negative impact (Red)
                ),
                tooltip=['Feature', 'Contribution']
            ).properties(height=250)
            
            # Add a vertical rule at 0 to prevent silent Vega-Lite crash
            rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='rgba(255,255,255,0.2)').encode(x='x:Q')
            
            st.altair_chart(chart + rule, use_container_width=True)
        else:
            st.info("Model not found. Please train the model to enable explainability.")

    st.markdown("---")
    
    # AI Verdict Trigger
    st.markdown("<div class='ai-header'><span class='ai-header-icon'>🎙️</span> The Gaffer's Verdict</div>", unsafe_allow_html=True)

    if st.button("⚽ Generate Tactical Verdict"):
        # Custom animated radar loader while the LLM is thinking.
        loader = st.empty()
        loader.markdown(
            "<div class='scanner'><div class='radar'></div><div>"
            "<div class='scanner-text'>Scanning the pitch</div>"
            "<div class='scanner-sub'>Crunching xG, form & fixtures…</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        advice = assistant.generate_advice(selected_player, selected_gw)
        loader.empty()  # clear the radar once we have a verdict

        stamp_html, body = render_verdict(advice)

        # Convert the reasoning markdown to HTML for the custom div
        import markdown
        body_html = markdown.markdown(body)

        verdict_html = f"<div class='ai-verdict-container'>{stamp_html}<div class='ai-verdict-box'>{body_html}</div></div>"
        st.markdown(verdict_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Transfer Comparison
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown("<div class='ai-header'><span class='ai-header-icon'>🔄</span> Transfer Comparison</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#b9a9c9; font-size:0.95rem;'>Compare two players and ask the AI whether you should make the transfer.</p>", unsafe_allow_html=True)

t_col1, t_col2 = st.columns(2)

with t_col1:
    p1_name = st.selectbox("Player 1 (Transfer Out)", players, index=players.index("Mohamed Salah") if "Mohamed Salah" in players else 0, key="transfer_p1")
with t_col2:
    p2_name = st.selectbox("Player 2 (Transfer In)", players, index=players.index("Bukayo Saka") if "Bukayo Saka" in players else 1, key="transfer_p2")

transfer_gw = st.selectbox("Gameweek for Comparison", sorted(df['GW'].unique()), key="transfer_gw")

if st.button("🔄 Compare Players"):
    from src.features.player_lookup import get_player_stats
    
    p1_stats = get_player_stats(df, p1_name, transfer_gw)
    p2_stats = get_player_stats(df, p2_name, transfer_gw)
    
    if not p1_stats or not p2_stats:
        st.error("Could not find data for one or both players in the selected Gameweek.")
    else:
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown(f"**{p1_stats['name']}** ({p1_stats['team']} - {p1_stats['position']})")
            st.metric("Price", f"£{p1_stats['price']:.1f}m")
            st.metric("Predicted Points", f"{p1_stats['predicted_points']:.2f}")
            st.metric("Form (Avg L3)", f"{p1_stats['form']:.2f}")
        with s_col2:
            st.markdown(f"**{p2_stats['name']}** ({p2_stats['team']} - {p2_stats['position']})")
            st.metric("Price", f"£{p2_stats['price']:.1f}m")
            st.metric("Predicted Points", f"{p2_stats['predicted_points']:.2f}")
            st.metric("Form (Avg L3)", f"{p2_stats['form']:.2f}")
            
        with st.spinner("Asking the Gaffer..."):
            verdict = assistant.compare_players(p1_stats, p2_stats)
            st.info(verdict)

# ----------------------------------------------------------------------------
# Dream Team — budget-constrained optimal squad for a gameweek
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown("<div class='ai-header'><span class='ai-header-icon'>🏆</span> Dream Team Optimiser</div>", unsafe_allow_html=True)
st.markdown("<p style='color:#b9a9c9; font-size:0.95rem;'>The best 15-player squad for a gameweek under the real FPL rules — £100m budget, valid formation, max 3 per club — chosen by solving a mixed-integer optimisation over the model's projections.</p>", unsafe_allow_html=True)

squad_gw = st.selectbox("Gameweek", sorted(df['GW'].unique()), key="squad_gw")

if st.button("🏆 Build Optimal Squad"):
    with st.spinner("Optimising 15 picks under budget & formation constraints…"):
        try:
            squad, summary = compute_optimal_squad(int(squad_gw))

            st.markdown(
                f"<div class='squad-summary'>"
                f"<div class='squad-stat'><span class='v'>{summary['projected_points']}</span>"
                f"<span class='l'>Projected pts</span></div>"
                f"<div class='squad-stat'><span class='v'>£{summary['cost']:.1f}m</span>"
                f"<span class='l'>Squad cost</span></div>"
                f"<div class='squad-stat'><span class='v'>{summary['formation']}</span>"
                f"<span class='l'>Formation</span></div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(render_squad_pitch(squad), unsafe_allow_html=True)

            bench = squad[~squad['starter']]
            bench_str = " · ".join(
                f"{r['name'].split()[-1]} ({r['position']}, £{r['value']/10:.1f}m)"
                for _, r in bench.iterrows())
            st.markdown(f"<p style='color:#b9a9c9; font-size:0.85rem;'><b>Bench:</b> {bench_str}</p>",
                        unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not build squad: {e}")
