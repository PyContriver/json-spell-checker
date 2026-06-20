# JSON Spell Checker

A two-pass spell and grammar checker for JSON files, with an optional AI review powered by a local [Ollama](https://ollama.com) model. Run it from the terminal or through a beautiful Streamlit web UI.

---

## How It Works

Every string value in a JSON file is checked in up to two passes:

| Pass | Tool | What it catches |
|------|------|----------------|
| 1 | [pyspellchecker](https://github.com/barrust/pyspellchecker) | Word-level spelling mistakes (fast, no server required) |
| 2 | Ollama LLM (optional) | Spelling, grammar, and style issues with suggested fixes |

The LLM is prompted with networking and telecommunications domain context, so technical terms like `BGP`, `OSPF`, `MPLS`, `IPsec`, `VXLAN`, and vendor names (Cisco, Juniper, Palo Alto) are treated as correct.

---

## Docker (recommended)

The Docker image is fully self-contained — it installs Ollama, pulls the model on first start, and runs the Streamlit app, all in one container.

```bash
docker compose up
```

On **first run** the container will:
1. Start the Ollama server
2. Pull `mistral:7b` (~4 GB — one-time download)
3. Start the Streamlit UI at **http://localhost:8501**

Subsequent starts skip the pull (model is cached in a Docker volume) and boot in seconds.

### Use a different model

```bash
# via docker compose
OLLAMA_MODEL=llama3.2 docker compose up

# or edit docker-compose.yml
environment:
  - OLLAMA_MODEL=llama3.2
```

### Useful commands

```bash
docker compose up -d              # run in background
docker compose logs -f app        # tail logs
docker compose down               # stop
docker compose down -v            # stop + delete model cache volume
```

### Build and run without Compose

```bash
docker build -t json-spell-checker .
docker run -p 8501:8501 -v ollama_data:/root/.ollama json-spell-checker
```

---

## Requirements (local / without Docker)

- Python 3.10+
- [Ollama](https://ollama.com) running locally (only needed for LLM pass)
1111
Install Python dependencies:

```bash
pip3 install -r requirements.txt
pip3 install streamlit   # only needed for the web UI
```

Pull a model in Ollama (recommended):

```bash
ollama pull mistral
```

---

## CLI Usage

```bash
# Spell check only (no Ollama required)
python3 agent.py sample.json

# Full AI review with mistral
python3 agent.py sample.json --model mistral:7b

# LLM only, skip pyspellchecker
python3 agent.py sample.json --model mistral:7b --no-spellcheck

# Different spell-check language
python3 agent.py sample.json --lang es

# Ignore specific words inline
python3 agent.py sample.json --ignore BGP OSPF MPLS datacenter

# Ignore words from a file (one per line, # comments supported)
python3 agent.py sample.json --model mistral:7b --ignore-file ignore.txt

# Save a machine-readable JSON report
python3 agent.py sample.json --model mistral:7b --output report.json

# List available Ollama models
python3 agent.py --list-models
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `file` | — | Path to the JSON file to check |
| `--model`, `-m` | none | Ollama model for AI review (e.g. `mistral:7b`, `llama3.2`) |
| `--no-spellcheck` | off | Skip pyspellchecker pass |
| `--lang` | `en` | Spell-check language |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL |
| `--ignore WORD …` | — | Words to skip in both passes |
| `--ignore-file FILE` | — | Plain-text file of words to ignore (one per line) |
| `--output`, `-o` | — | Save full report as JSON to this path |
| `--list-models` | — | Print available Ollama models and exit |

---

## Web UI

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### UI Features

- **Upload** a JSON file or **paste** JSON directly
- **Sidebar** to configure the LLM model, spell-check language, and ignore words
- **Live progress bar** during the LLM pass
- **Summary cards** showing total fields, flagged count, and clean count
- **Field-by-field report** — flagged fields expand to show spell issues, LLM analysis table, and a suggested corrected sentence
- **Download** the full report as a JSON file

---

## Ignore Words

You can suppress false positives for domain-specific terms in both passes.

**Inline:**
```bash
python3 agent.py file.json --ignore GigabitEthernet VXLAN QoS
```

**From a file (`ignore.txt`):**
```
# Networking acronyms
BGP
OSPF
MPLS
LDP
TLS
IPsec
SNMP
NMS
VLANs
datacenter
uplink
```

```bash
python3 agent.py file.json --model mistral:7b --ignore-file ignore.txt
```

In the web UI, paste the words (one per line or space-separated) into the **Ignore Words** box in the sidebar.

---

## Sample Files

| File | Description |
|------|-------------|
| `sample.json` | General UI/app strings with intentional errors |
| `networking_sample.json` | Networking config descriptions with intentional errors |

---

## Project Structure

```
json-spell-checker/
├── agent.py               # CLI agent (both passes, rendering, arg parsing)
├── app.py                 # Streamlit web UI
├── sample.json            # General sample with errors
├── networking_sample.json # Networking-domain sample with errors
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Recommended Models

| Model | Best for |
|-------|----------|
| `mistral:7b` | Best JSON schema compliance + good domain vocabulary |
| `llama3.2:latest` | Strong reasoning, accurate grammar corrections |
| `qwen3:0.6b` | Fast, lightweight — good for large files |
