import json
from pathlib import Path
from typing import Any


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
    Invalid JSON files are silently skipped (caller may log the warning).
    """
    pattern = "**/*.json" if recursive else "*.json"
    result: dict[str, Any] = {}
    for p in sorted(directory.glob(pattern)):
        try:
            result[str(p.relative_to(directory))] = json.loads(
                p.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            pass  # callers can iterate skipped files if needed
    return result
