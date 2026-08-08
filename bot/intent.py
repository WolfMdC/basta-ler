"""
Intent classification: deciding whether a Discord message is a *question*
that should trigger a wiki lookup, as opposed to casual chat.

Two implementations:
  - SimpleHeuristicClassifier (default): pure Portuguese keyword/punctuation
    heuristics, no API key, no ML model. This is what runs out of the box.
  - LLMIntentClassifier: an optional, pluggable stub for a future
    LLM-backed classifier (e.g. for subtler cases the heuristic misses).
    Not wired up or required by default — see the class docstring.

Both implement the same `IntentClassifier` interface so the bot can swap
between them via the INTENT_CLASSIFIER env var without any other code
changes.
"""
from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod


class IntentClassifier(ABC):
    """Interface every intent classifier implementation must satisfy."""

    @abstractmethod
    def is_question(self, message: str) -> bool:
        """Return True if `message` looks like a question seeking wiki info."""
        raise NotImplementedError


# Strong Portuguese interrogative words/phrases. Matched without accents so
# both "é" and "e", "onde" vs "ondé" typos etc. are tolerant of missing
# diacritics (common in casual Discord typing).
_INTERROGATIVES = [
    "como",
    "o que",
    "oque",
    "que",
    "quando",
    "onde",
    "aonde",
    "qual",
    "quais",
    "quanto",
    "quantos",
    "quantas",
    "quem",
    "por que",
    "porque",
    "pra que",
    "para que",
    "cade",
    "cadê",
    "sera que",
    "será que",
    "existe",
    "tem como",
    "sabe",
    "alguem sabe",
    "alguém sabe",
]


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _build_interrogative_pattern() -> re.Pattern:
    # Longest phrases first so e.g. "por que" is checked before bare "que".
    phrases = sorted({_strip_accents(p.lower()) for p in _INTERROGATIVES}, key=len, reverse=True)
    escaped = [re.escape(p) for p in phrases]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


_INTERROGATIVE_PATTERN = _build_interrogative_pattern()


def contains_interrogative(text: str) -> bool:
    normalized = _strip_accents(text.lower())
    return bool(_INTERROGATIVE_PATTERN.search(normalized))


class SimpleHeuristicClassifier(IntentClassifier):
    """No-ML, no-API-key question detector for Portuguese chat.

    A message is treated as a question if either:
      - it ends with "?" (allowing trailing whitespace/emoji-ish punctuation), or
      - it contains a strong Portuguese interrogative word/phrase AND is at
        least `min_chars` characters long (filters out short chatty
        messages like "oq" or "pq" that aren't really asking anything on
        their own).
    """

    def __init__(self, min_chars: int = 12):
        self.min_chars = min_chars

    def is_question(self, message: str) -> bool:
        text = message.strip()
        if not text:
            return False

        if text.rstrip(" !.…").endswith("?"):
            return True

        if len(text) >= self.min_chars and contains_interrogative(text):
            return True

        return False


class LLMIntentClassifier(IntentClassifier):
    """Optional pluggable slot for an LLM-backed intent classifier.

    Not required to run the bot, and not implemented by default — this is a
    stub showing the intended shape so a future contributor (or you, later)
    can wire up a real provider (Anthropic/OpenAI/local LLM/etc.) behind an
    API key without touching the rest of the bot. Selected via
    INTENT_CLASSIFIER=llm, at which point it must be given a real
    implementation or the bot will refuse to start (see `get_intent_classifier`).
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "LLMIntentClassifier requires an API key. Set LLM_API_KEY in .env, "
                "or keep INTENT_CLASSIFIER=simple to use the free heuristic classifier."
            )
        self.api_key = api_key

    def is_question(self, message: str) -> bool:
        raise NotImplementedError(
            "LLM-enhanced intent classification is not implemented yet. "
            "Implement this method to call your LLM provider of choice, "
            "or set INTENT_CLASSIFIER=simple in .env."
        )


def get_intent_classifier(classifier_name: str, min_chars: int, llm_api_key: str = "") -> IntentClassifier:
    if classifier_name == "simple":
        return SimpleHeuristicClassifier(min_chars=min_chars)
    if classifier_name == "llm":
        return LLMIntentClassifier(api_key=llm_api_key)
    raise ValueError(f"Unknown INTENT_CLASSIFIER: {classifier_name!r} (expected 'simple' or 'llm')")
