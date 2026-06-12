import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

try:
    from .prompts import SYSTEM_PROMPT, generate_player_prompt
except ImportError:
    from prompts import SYSTEM_PROMPT, generate_player_prompt

# Load environment variables from .env
load_dotenv()


class FPLAssistant:
    def __init__(self, model_name="gpt-oss-120b"):
        """Initialise the OpenAI/OpenRouter client.

        Relies on OPENAI_API_KEY and OPENAI_BASE_URL being set in the
        environment (see .env).
        """
        api_key = os.getenv("OPENAI_API_KEY", "dummy_key")
        base_url = os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = os.getenv("MODEL_ID", model_name)
        self.data_path = os.path.join(
            os.path.dirname(__file__), '../../data/processed/test_data_with_targets.csv')
        self._df = None  # lazily cached so we don't re-read the CSV every call

    def _load(self):
        if self._df is None:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Prediction data not found at {self.data_path}")
            self._df = pd.read_csv(self.data_path)
        return self._df

    def get_player_data(self, player_name, gw):
        """Fetch the projection, profile and recent form for a player/GW."""
        df = self._load()
        player_df = df[(df['name'] == player_name) & (df['GW'] == gw)]
        if player_df.empty:
            return None

        row = player_df.iloc[0]

        predicted_points = row.get('predicted_points', 0)

        profile = {
            "position": row.get('position', 'UNK'),
            "team": row.get('team_x', 'UNK'),
            # FPL prices are stored in tenths of a million (e.g. 55 -> £5.5m).
            "price": row.get('value', float('nan')) / 10
                     if pd.notna(row.get('value')) else float('nan'),
            "next_venue": row.get('next_was_home', float('nan')),
        }

        recent_stats = {
            "Avg Points (L3)": row.get('rolling_3_avg_total_points'),
            "Avg Points (L5)": row.get('rolling_5_avg_total_points'),
            "Avg Minutes (L3)": row.get('rolling_3_avg_minutes'),
            "Start rate (L5)": row.get('rolling_5_start_rate'),
            "Points per 90 (L5)": row.get('rolling_5_points_per_90'),
            "Avg BPS (L3)": row.get('rolling_3_avg_bps'),
            "Avg ICT (L3)": row.get('rolling_3_avg_ict_index'),
        }

        return predicted_points, profile, recent_stats

    def generate_advice(self, player_name, gw):
        """Query the LLM for advice based on the ML projection and form."""
        data = self.get_player_data(player_name, gw)
        if not data:
            return f"Error: Could not find prediction data for {player_name} in GW {gw}."

        predicted_points, profile, recent_stats = data
        user_prompt = generate_player_prompt(
            player_name, gw, predicted_points, profile, recent_stats)

        print(f"Querying {self.model_name}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                max_tokens=260,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM API Error: {str(e)}"


if __name__ == "__main__":
    assistant = FPLAssistant()
    player = "Erling Haaland"
    gw = 1
    print(f"Fetching advice for {player} (GW {gw})...")
    print("\n--- Prompt preview ---\n")
    data = assistant.get_player_data(player, gw)
    if data:
        pp, prof, stats = data
        print(generate_player_prompt(player, gw, pp, prof, stats))
    print("\n--- Advice ---\n")
    print(assistant.generate_advice(player, gw))
