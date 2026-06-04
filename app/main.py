# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.agent import FPLAssistant

# Configure page settings
st.set_page_config(
    page_title="FPL AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css('app/style.css')
except FileNotFoundError:
    pass

# Load data for the UI
@st.cache_data
def load_prediction_data():
    data_path = os.path.join(os.path.dirname(__file__), '../data/processed/test_data_with_targets.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()

df = load_prediction_data()

# Initialize Assistant
@st.cache_resource
def get_assistant():
    return FPLAssistant()

assistant = get_assistant()

# UI Layout
st.title("⚡ FPL AI Predictor")
st.markdown("Leverage Machine Learning and LLMs to dominate your mini-leagues.")

if df.empty:
    st.error("Prediction data not found. Please run the ML pipeline first.")
    st.stop()

# Sidebar controls
st.sidebar.header("Select Parameters")
players = sorted(df['name'].unique())
selected_player = st.sidebar.selectbox("Search Player", players, index=players.index("Erling Haaland") if "Erling Haaland" in players else 0)

# Filter GWs for selected player
player_gws = sorted(df[df['name'] == selected_player]['GW'].unique())
selected_gw = st.sidebar.selectbox("Gameweek", player_gws)

st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("- 🧠 `HistGradientBoostingRegressor`")
st.sidebar.markdown("- 🤖 `gpt-oss-120b` (via OpenAI API)")

# Main Content Area
player_data = df[(df['name'] == selected_player) & (df['GW'] == selected_gw)]

if player_data.empty:
    st.warning("No data available for this player in the selected Gameweek.")
else:
    row = player_data.iloc[0]
    
    st.markdown(f"### {selected_player} | GW {selected_gw}")
    st.caption(f"Team: {row['team_x']} | Position: {row['position']}")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    predicted_points = row.get('predicted_points', 0)
    avg_points = row.get('rolling_3_avg_total_points', 0)
    avg_mins = row.get('rolling_3_avg_minutes', 0)
    
    with col1:
        st.metric(label="Predicted Points (Next GW)", value=f"{predicted_points:.1f} pts")
    with col2:
        st.metric(label="Form (Avg Pts L3)", value=f"{avg_points:.1f} pts")
    with col3:
        st.metric(label="Avg Mins (L3)", value=f"{avg_mins:.0f} mins")

    st.markdown("---")
    
    # AI Advice Trigger
    st.subheader("🤖 Ask the FPL AI Expert")
    st.markdown("Get a definitive BUY/SELL/HOLD verdict based on the ML predictions and recent stats.")
    
    if st.button("Generate AI Verdict"):
        with st.spinner("Analyzing data and generating verdict..."):
            advice = assistant.generate_advice(selected_player, selected_gw)
            
            # Display advice in a styled box
            st.markdown(f"<div class='ai-verdict'>{advice}</div>", unsafe_allow_html=True)
