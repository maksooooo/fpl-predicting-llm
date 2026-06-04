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
- **Machine Learning:** `scikit-learn` (`HistGradientBoostingRegressor`)
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

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your OpenRouter (or OpenAI) credentials:
   ```env
   OPENAI_API_KEY=your_openrouter_api_key_here
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   MODEL_ID=openai/gpt-oss-120b:free  # Or any other preferred model
   ```

## 🧠 Machine Learning Pipeline

If you want to train the model from scratch or evaluate it on new data:

1. **Train the Model:**
   ```bash
   python src/models/train_model.py
   ```
   *This trains a `HistGradientBoostingRegressor` on historical seasons and saves the model to `models/histgb_model.pkl`.*

2. **Evaluate and Generate Predictions:**
   ```bash
   python src/models/evaluate.py
   ```
   *This evaluates the test set (2023-24 season), prints the MAE/RMSE, and saves the `predicted_points` back to `data/processed/test_data_with_targets.csv` so the app can use them.*

## 🏃‍♂️ Running the Web App

Start the Streamlit interface to interact with the predictions and AI verdicts:

```bash
streamlit run app/main.py
```

Navigate to `http://localhost:8501` in your web browser. Select a player, choose a Gameweek, and click **"Generate AI Verdict"** to get your FPL advice!
