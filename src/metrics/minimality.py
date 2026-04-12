from __future__ import annotations

from difflib import SequenceMatcher


def edit_similarity(original: str, changed: str) -> float:
    """Higher is better; 1 means almost unchanged."""
    return float(SequenceMatcher(a=original, b=changed).ratio())

