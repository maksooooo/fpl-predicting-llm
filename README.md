# ⚡ FPL AI Predictor

Leverage Machine Learning and Large Language Models (LLMs) to dominate your Fantasy Premier League (FPL) mini-leagues. 

This project combines historical FPL data, a robust Gradient Boosting Machine Learning model, and advanced LLM reasoning to forecast player points and provide definitive "BUY/SELL/HOLD" advice for any given Gameweek.

## 🚀 Features

- **Machine Learning Forecasts:** Predicts expected points for the next Gameweek based on historical performance, recent form, and advanced underlying metrics.
- **AI Expert Verdict:** Uses state-of-the-art Large Language Models (via OpenRouter) to provide personalized, human-readable advice for any player, contextualized with ML predictions.
- **Interactive UI:** A clean, easy-to-use Streamlit web application that lets you search for players, select Gameweeks, and view metrics at a glance.
- **Comprehensive Data Pipeline:** Processes multiple seasons of historical FPL data, engineered with rolling averages (e.g., last 3 matches form) and chronological train/validation/test splits.

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** `LightGBM` (`LGBMRegressor`, gradient-boosted trees) with `scikit-learn` for metrics & permutation importance
- **Data Manipulation:** `pandas`, `numpy`
- **LLM Integration:** `openai` Python SDK (configured for OpenRouter)
- **Visualization:** `matplotlib`, `seaborn` (for feature importance analysis)

## 📂 Project Structure

```text
fpl-predicting-llm/
├── app/
│   ├── main.py                # Main Streamlit application
│   └── style.css              # Custom UI styling
├── data/
│   └── processed/             # Cleaned and engineered feature datasets
├── models/                    # Saved ML models (e.g., .pkl files)
├── notebooks/                 # Jupyter notebooks for EDA and experimentation
├── reports/                   # Generated reports and feature importance plots
├── src/
│   ├── features/              # Feature engineering scripts
│   ├── llm/
│   │   ├── agent.py           # LLM interaction script (FPLAssistant)
│   │   └── prompts.py         # System and User prompt templates
│   └── models/
│       ├── train_model.py     # Script to train the ML model
│       └── evaluate.py        # Script to evaluate model performance and save predictions
├── .env                       # Environment variables (API keys)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ⚙️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fpl-predicting-llm.git
   cd fpl-predicting-llm
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > **macOS note:** LightGBM needs the OpenMP runtime. If you hit a
   > `Library not loaded: @rpath/libomp.dylib` error, install it with
   > `brew install libomp`.

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your OpenRouter (or OpenAI) credentials:
   ```env
   OPENAI_API_KEY=your_openrouter_api_key_here
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   MODEL_ID=openai/gpt-oss-120b:free  # Or any other preferred model
   ```

## 🧠 Machine Learning Pipeline

**Quick start:** run the whole pipeline (features → train → evaluate) with one
command:
```bash
./run_pipeline.sh          # rebuild model + print metrics
./run_pipeline.sh app      # launch the Streamlit web app
```

Or run the three stages individually to rebuild everything from the raw data:

1. **Engineer Features:**
   ```bash
   python src/features/build_features.py
   ```
   *Builds leakage-safe features from `cleaned_merged_seasons.csv`: lag and
   rolling-form metrics (3/5 GW), scoring consistency (std), points-per-90,
   minutes/start reliability, one-hot position flags, and the next fixture's
   home/away venue. Writes `data/processed/engineered_features.csv`.*

2. **Train the Model:**
   ```bash
   python src/models/train_model.py
   ```
   *Trains an `LGBMRegressor` on historical seasons with early stopping on a
   chronological validation split (2022-23), and saves the model to
   `models/histgb_model.pkl` (filename kept for backwards compatibility).*

3. **Evaluate and Generate Predictions:**
   ```bash
   python src/models/evaluate.py
   ```
   *Evaluates on the 2023-24 hold-out season and saves `predicted_points` back
   to `data/processed/test_data_with_targets.csv` for the app.*

### How good is it?

The model is benchmarked against two naive baselines, and metrics are reported
on the **subset of players who actually featured** — overall MAE is otherwise
flattered by the ~56% of rows where a non-playing player trivially scores ~0.

| Predictor | Test MAE | Test RMSE |
|-----------|---------:|----------:|
| **LightGBM (ours)** | **0.80** | **2.03** |
| Baseline: form average (last 3 GW) | 1.05 | 2.20 |
| Baseline: repeat last GW score | 1.14 | 2.59 |

The model improves on the form-average baseline by ~24% and on last-GW by ~30%.
Among players who featured, MAE is ≈1.77 points, and `evaluate.py` also prints a
per-position breakdown (GK / DEF / MID / FWD).

## 🏃‍♂️ Running the Web App

Start the Streamlit interface to interact with the predictions and AI verdicts:

```bash
streamlit run app/main.py
```

Navigate to `http://localhost:8501` in your web browser. Select a player, choose a Gameweek, and click **"Generate AI Verdict"** to get your FPL advice!
