"""
Centralised logging setup for the JSON Spell Checker.

Usage (in any module):
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting check on %s", path)

Configuration:
    LOG_LEVEL  env var  — DEBUG | INFO | WARNING | ERROR  (default: INFO)
    LOG_FILE   env var  — path to log file                (default: logs/app.log)
    LOG_TO_CONSOLE     — set to "0" to suppress console output
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings (all overridable via environment variables)
# ---------------------------------------------------------------------------

_LEVEL_NAME   = os.environ.get("LOG_LEVEL", "INFO").upper()
_LOG_FILE     = os.environ.get("LOG_FILE", "logs/app.log")
_LOG_CONSOLE  = os.environ.get("LOG_TO_CONSOLE", "1") != "0"
_MAX_BYTES    = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 3                  # keep 3 rotated files

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _setup() -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    level = getattr(logging, _LEVEL_NAME, logging.INFO)

    root = logging.getLogger("json_spell_checker")
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # ── File handler (rotating) ──────────────────────────────────────────────
    try:
        log_path = Path(_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except OSError as exc:
        # Can't write log file — fall back to console only
        print(f"[logger] WARNING: cannot open log file {_LOG_FILE}: {exc}", file=sys.stderr)

    # ── Console handler — shows at the configured level (default: INFO) ────
    if _LOG_CONSOLE:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger scoped under the 'json_spell_checker' hierarchy.

    Parameters
    ----------
    name : typically __name__ of the calling module
    """
    _setup()
    # Strip leading "src.utils." for cleaner log lines
    short = name.replace("src.utils.", "").replace("src.", "")
    return logging.getLogger(f"json_spell_checker.{short}")
