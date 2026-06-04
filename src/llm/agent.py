import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# We assume this script might be run from the root directory or src/llm
# Adjust path imports as needed
try:
    from .prompts import SYSTEM_PROMPT, generate_player_prompt
except ImportError:
    from prompts import SYSTEM_PROMPT, generate_player_prompt

# Load environment variables from .env
load_dotenv()

class FPLAssistant:
    def __init__(self, model_name="gpt-oss-120b"):
        """
        Initializes the OpenAI client.
        Relies on OPENAI_API_KEY and OPENAI_BASE_URL being set in the environment.
        """
        api_key = os.getenv("OPENAI_API_KEY", "dummy_key")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = os.getenv("MODEL_ID", model_name)
        self.data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/test_data_with_targets.csv')

    def get_player_data(self, player_name, gw):
        """Fetches the latest prediction and form for a player from the test set."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Prediction data not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # Filter for player and GW
        player_df = df[(df['name'] == player_name) & (df['GW'] == gw)]
        
        if player_df.empty:
            return None
            
        row = player_df.iloc[0]
        
        predicted_points = row.get('predicted_points', 0)
        recent_stats = {
            "Avg Points (Last 3)": row.get('rolling_3_avg_total_points', 0),
            "Avg Minutes (Last 3)": row.get('rolling_3_avg_minutes', 0),
            "Avg BPS (Last 3)": row.get('rolling_3_avg_bps', 0),
        }
        
        return predicted_points, recent_stats

    def generate_advice(self, player_name, gw):
        """Queries the LLM for advice based on ML predictions."""
        data = self.get_player_data(player_name, gw)
        if not data:
            return f"Error: Could not find prediction data for {player_name} in GW {gw}."
            
        predicted_points, recent_stats = data
        
        user_prompt = generate_player_prompt(player_name, gw, predicted_points, recent_stats)
        
        print(f"Querying {self.model_name}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM API Error: {str(e)}"

if __name__ == "__main__":
    # Test the agent with a dummy key/endpoint
    assistant = FPLAssistant()
    
    # We pick Erling Haaland and an arbitrary GW from the 2023-24 season test set
    player = "Erling Haaland"
    gw = 1  # Check test data for valid GWs
    
    print(f"Fetching advice for {player} (GW {gw})...")
    advice = assistant.generate_advice(player, gw)
    print("\n--- Advice ---\n")
    print(advice)
