"""
Module-level run state that survives Streamlit session reconnects.

Streamlit creates a new session (and fresh st.session_state) on every page
refresh, but the server process — and any background threads — keep running.
This module stores the active run at process level so a refreshed session
can pick up where it left off.

Checkpoint writing ensures partial results survive even a full server restart.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

CHECKPOINT_FILE = Path("run_checkpoint.json")
_CHECKPOINT_WRITE_EVERY = 25       # write to disk every N completed fields
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Module-level active run — shared across all Streamlit sessions
# ---------------------------------------------------------------------------

_active: dict | None = None


def start(
    ctx: dict,
    thread: threading.Thread,
    stop_event: threading.Event,
    progress_counter: list,
    llm_results: dict,
    llm_total: int,
) -> None:
    """Register a new run. Called once when the background thread starts."""
    global _active
    with _lock:
        _active = {
            "ctx":              ctx,
            "thread":           thread,
            "stop_event":       stop_event,
            "progress_counter": progress_counter,
            "llm_results":      llm_results,
            "llm_total":        llm_total,
            "started_at":       datetime.now().isoformat(),
            "was_stopped":      False,
        }
    log.info("Run registered in run_manager (total=%d fields).", llm_total)
    _write_checkpoint("in_progress")


def get() -> dict | None:
    """Return the active run dict, or None if no run is registered."""
    return _active


def mark_stopped() -> None:
    global _active
    with _lock:
        if _active:
            _active["was_stopped"] = True
    _write_checkpoint("stopped")


def finish() -> None:
    """Called when the run completes (thread exits normally)."""
    _write_checkpoint("complete")
    global _active
    with _lock:
        _active = None
    log.info("Run finished and cleared from run_manager.")


def clear() -> None:
    """Discard run state without writing a checkpoint (e.g. on explicit clear)."""
    global _active
    with _lock:
        _active = None
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Checkpoint — persists partial results to disk
# ---------------------------------------------------------------------------

def maybe_checkpoint(completed: int) -> None:
    """Write a checkpoint every N completed fields."""
    if completed % _CHECKPOINT_WRITE_EVERY == 0:
        _write_checkpoint("in_progress")


def _write_checkpoint(status: str) -> None:
    run = _active
    if not run:
        return
    ctx = run["ctx"]
    try:
        data = {
            "status":     status,
            "started_at": run.get("started_at"),
            "saved_at":   datetime.now().isoformat(),
            "model":      ctx.get("model", ""),
            "total":      run["llm_total"],
            "completed":  run["progress_counter"][0],
            # Serialise llm_results (keys are "name||field")
            "llm_results": {
                k: v for k, v in run["llm_results"].items()
                if isinstance(k, str)
            },
            # Spell results and file entries so we can render partial results
            "spell_results": _serialise_spell(ctx.get("spell_results", {})),
            "file_entries":  {
                str(k): v for k, v in ctx.get("file_entries", {}).items()
            },
            "ignore_words": list(ctx.get("ignore_words", [])),
            "file_names":   [str(k) for k in ctx.get("file_map", {})],
        }
        CHECKPOINT_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("Could not write checkpoint: %s", exc)


def _serialise_spell(spell_results: dict) -> dict:
    """Convert Path keys to strings for JSON serialisation."""
    return {str(k): v for k, v in spell_results.items()}


def load_checkpoint() -> dict | None:
    """
    Load the last checkpoint from disk.
    Returns None if no checkpoint exists or it is malformed.
    """
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        log.info("Checkpoint loaded: status=%s completed=%s/%s",
                 data.get("status"), data.get("completed"), data.get("total"))
        return data
    except Exception as exc:
        log.warning("Could not read checkpoint: %s", exc)
        return None
