FROM python:3.12-slim

# ── System deps ──────────────────────────────────────────────────────────────
# curl  : needed to install Ollama
# git   : needed for the "Clone from Git repo" feature
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# ── Install Ollama ────────────────────────────────────────────────────────────
RUN curl -fsSL https://ollama.com/install.sh | sh

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model pulled at runtime (not build time) to keep the image lean.
# Override with: docker run -e OLLAMA_MODEL=llama3.2 ...
ENV OLLAMA_MODEL=mistral:7b

# Ollama data dir — mount a volume here to persist pulled models across restarts
VOLUME ["/root/.ollama"]

EXPOSE 8501 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
