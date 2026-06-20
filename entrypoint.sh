#!/bin/bash
set -e

MODEL="${OLLAMA_MODEL:-mistral:7b}"

echo "▶ Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama to be ready..."
until ollama list > /dev/null 2>&1; do
    sleep 1
done
echo "✔ Ollama is ready."

# Pull the model only if it isn't already in the local store
if ollama list | grep -q "^${MODEL}"; then
    echo "✔ Model '${MODEL}' already available — skipping pull."
else
    echo "⬇ Pulling model '${MODEL}' (this may take a few minutes on first run)..."
    ollama pull "${MODEL}"
    echo "✔ Model '${MODEL}' pulled successfully."
fi

echo "▶ Starting Streamlit app on port 8501..."
exec streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false
