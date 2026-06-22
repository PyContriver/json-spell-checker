"""
Persistent UI settings — saved to settings.json and reloaded on app start.
"""

import json
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

SETTINGS_FILE = Path("settings.json")

DEFAULTS: dict = {
    "selected_model":   "mistral:7b",
    "lang":             "en",
    "no_spellcheck":    True,
    "ignore_raw":       "",
    "workers":          1,
    "llm_timeout":      240,
    "llm_retries":      2,
    "llm_num_ctx":      1024,
    "llm_num_thread":   0,
}


def load() -> dict:
    """Load settings from disk, falling back to defaults for missing keys."""
    if not SETTINGS_FILE.exists():
        log.debug("No settings.json found — using defaults.")
        return dict(DEFAULTS)
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        # Merge: saved values override defaults, new default keys are added
        merged = {**DEFAULTS, **saved}
        log.debug("Settings loaded from %s", SETTINGS_FILE)
        return merged
    except Exception as exc:
        log.warning("Could not read settings.json (%s) — using defaults.", exc)
        return dict(DEFAULTS)


def save(settings: dict) -> None:
    """Persist settings to disk."""
    try:
        SETTINGS_FILE.write_text(
            json.dumps(settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.debug("Settings saved to %s", SETTINGS_FILE)
    except Exception as exc:
        log.warning("Could not save settings.json: %s", exc)
