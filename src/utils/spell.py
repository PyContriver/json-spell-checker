import re

from spellchecker import SpellChecker


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text)


def spell_check(
    checker: SpellChecker,
    text: str,
    ignore: set[str] | None = None,
) -> list[dict]:
    """
    Run pyspellchecker on text and return a list of issue dicts.
    Words in `ignore` (lowercase) are skipped.
    """
    words      = _tokenize(text)
    misspelled = checker.unknown(words)
    if ignore:
        misspelled = {w for w in misspelled if w.lower() not in ignore}
    issues = []
    for word in misspelled:
        candidates = checker.candidates(word) or set()
        issues.append({"word": word, "suggestions": sorted(candidates)[:5]})
    return issues
