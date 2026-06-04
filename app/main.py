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

# UI Layout Header
st.markdown("<h1>⚡ FPL <span class='highlight'>AI Predictor</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-bottom: 2rem;'>Leverage Machine Learning and LLMs to dominate your mini-leagues.</p>", unsafe_allow_html=True)

if df.empty:
    st.error("Prediction data not found. Please run the ML pipeline first.")
    st.stop()

# Sidebar controls
st.sidebar.markdown("<h2 style='margin-bottom:0;'>Scout Filter</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#94a3b8; font-size:0.9rem; margin-bottom:1rem;'>Select a target for analysis</p>", unsafe_allow_html=True)

players = sorted(df['name'].unique())
selected_player = st.sidebar.selectbox("Search Player", players, index=players.index("Erling Haaland") if "Erling Haaland" in players else 0)

# Filter GWs for selected player
player_gws = sorted(df[df['name'] == selected_player]['GW'].unique())
selected_gw = st.sidebar.selectbox("Gameweek", player_gws)

st.sidebar.markdown("---")
st.sidebar.markdown("**POWERED BY:**")
st.sidebar.markdown("<div style='font-family:monospace;font-size:0.8rem;color:#00ff85'>> ML: LightGBM Regressor<br>> AI: OpenRouter API</div>", unsafe_allow_html=True)

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
    risk_badge = "<span class='player-badge badge-team' style='background:rgba(239, 68, 68, 0.2);color:#ef4444;border:1px solid rgba(239, 68, 68, 0.5);'>⚠️ Rotation Risk</span>" if avg_mins < 60 else ""
    
    # Scout Card Container - Single line to prevent Markdown code block bugs
    scout_html = f"<div class='scout-card'><div style='margin-bottom:24px;'><span class='player-badge badge-team'>{row.get('team_x', 'UNK')}</span> <span class='player-badge badge-pos'>{row.get('position', 'UNK')}</span> {risk_badge} <h2 style='margin-top:10px; margin-bottom:2px; font-size:2.2rem;'>{selected_player}</h2><div style='color:#00ff85; font-size:0.95rem; font-family:\"Outfit\", sans-serif; letter-spacing:1px; text-transform:uppercase;'>Gameweek {selected_gw} Profile</div></div></div>"
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
        if not historical_data.empty:
            chart_data = historical_data.set_index('GW')[['target_next_gw_points']].rename(columns={'target_next_gw_points': 'Points'})
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
    st.markdown("<div class='ai-header'><span class='ai-header-icon'>🤖</span> Ask the FPL AI Expert</div>", unsafe_allow_html=True)
    
    if st.button("Generate Tactical Verdict"):
        with st.spinner("SCANNING NEURAL NETWORK & ANALYZING METRICS..."):
            advice = assistant.generate_advice(selected_player, selected_gw)
            
            # Convert markdown to HTML so it renders correctly inside the custom div
            import markdown
            advice_html = markdown.markdown(advice)
            
            # Inject the fully formatted HTML into the custom styling box - Single line
            verdict_html = f"<div class='ai-verdict-container'><div class='ai-verdict-box'>{advice_html}</div></div>"
            st.markdown(verdict_html, unsafe_allow_html=True)
