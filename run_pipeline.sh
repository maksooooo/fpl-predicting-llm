#!/usr/bin/env bash
#
# Rebuild the FPL model end-to-end: features -> train -> evaluate.
#
# Usage:
#   ./run_pipeline.sh          # run the full ML pipeline
#   ./run_pipeline.sh app      # launch the Streamlit web app instead
#
set -euo pipefail

# Always run from the project root (the directory this script lives in).
cd "$(dirname "$0")"

# Prefer the project venv if present, otherwise fall back to python3.
if [ -x "./venv/bin/python" ]; then
    PY="./venv/bin/python"
else
    PY="python3"
fi
echo "Using interpreter: $PY"

if [ "${1:-}" = "app" ]; then
    echo "Launching Streamlit app at http://localhost:8501 ..."
    exec "$PY" -m streamlit run app/main.py
fi

echo ""
echo "==> [1/3] Engineering features"
"$PY" src/features/build_features.py

echo ""
echo "==> [2/3] Training model"
"$PY" src/models/train_model.py

echo ""
echo "==> [3/3] Evaluating model"
"$PY" src/models/evaluate.py

echo ""
echo "Pipeline complete. Predictions written to data/processed/test_data_with_targets.csv"
echo "Start the web app with:  ./run_pipeline.sh app"
