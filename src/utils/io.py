import json
import re
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


def _normalise(text: str) -> str:
    """Lowercase, split camelCase/snake_case/kebab-case/dots into words, collapse spaces."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)   # camelCase → camel Case
    text = re.sub(r"[_\-\.\s]+", " ", text)             # separators → space
    return text.strip().lower()


def is_key_echo(field_path: str, value: str) -> bool:
    """
    Return True when the value is just a reformatted version of the field key.

    Examples that return True (false positives to skip):
      path=virus          value="Virus"
      path=top_target_devices  value="Top Target Devices"
      path=source         value="Source"
      path=bgp.status     value="Status"   ← last key segment matches
    """
    # Extract the last key segment from dot-path, strip array indices
    path_clean = re.sub(r"\[\d+\]", "", field_path)
    last_key   = path_clean.split(".")[-1]

    key_norm = _normalise(last_key)
    val_norm = _normalise(value)

    # Exact match after normalisation
    if key_norm == val_norm:
        return True

    # Value is a single word that matches the key (handles "Source" for key "source")
    if len(val_norm.split()) == 1 and val_norm == key_norm:
        return True

    return False


def extract_strings(data: Any, path: str = "") -> list[tuple[str, str]]:
    """Recursively collect (dot-path, string-value) pairs from any JSON structure."""
    results: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child = f"{path}.{key}" if path else key
            results.extend(extract_strings(value, child))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            results.extend(extract_strings(item, f"{path}[{i}]"))
    elif isinstance(data, str) and data.strip():
        results.append((path, data))
    return results


def load_ignore_words(words: list[str], file_path: str | None) -> set[str]:
    """
    Build a lowercase set of words to ignore from an inline list and/or a file.
    Raises FileNotFoundError if file_path is given but does not exist.
    """
    combined = {w.lower() for w in (words or [])}
    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Ignore file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            word = line.strip()
            if word and not word.startswith("#"):
                combined.add(word.lower())
    return combined


def load_json_files_from_dir(
    directory: Path,
    recursive: bool = False,
) -> dict[str, Any]:
    """
    Return {relative_path: parsed_json} for every valid *.json file found.
    Invalid JSON files are skipped with a warning log.
    """
    pattern = "**/*.json" if recursive else "*.json"
    result: dict[str, Any] = {}
    log.info("Scanning directory: %s (recursive=%s)", directory, recursive)
    for p in sorted(directory.glob(pattern)):
        try:
            result[str(p.relative_to(directory))] = json.loads(
                p.read_text(encoding="utf-8")
            )
            log.debug("Loaded: %s", p)
        except json.JSONDecodeError as exc:
            log.warning("Skipping %s — invalid JSON: %s", p, exc)
    log.info("Found %d JSON file(s) in %s", len(result), directory)
    return result
