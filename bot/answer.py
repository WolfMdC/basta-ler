"""
Answer writing: turning retrieval results into the text/description shown
to the user, as an OPTIONAL pluggable module (mirrors bot/intent.py).

  - SimpleAnswerWriter (default): no API key, no LLM call. Reuses the short
    page summary captured at index time (first ~300 chars of the page's
    plain text), and — when the question asks for one — the single infobox
    value that answers it outright (see bot/facts.py).
  - LLMAnswerWriter: stub for a future LLM-generated answer (e.g. "answer
    the user's question using this page's content as context"). Not wired
    up or required by default.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from bot.facts import Fact, find_fact
from bot.retriever import RetrievalResult


@dataclass(frozen=True)
class Answer:
    title: str
    url: str
    description: str
    similarity: float
    # The wiki's own value for what was asked, when the question maps onto an
    # infobox row the page actually has. None for everything else, which is
    # the common case — most questions aren't a request for one number.
    fact: Fact | None = None


class AnswerWriter(ABC):
    """Interface every answer-writing implementation must satisfy."""

    @abstractmethod
    def write(self, question: str, results: list[RetrievalResult]) -> list[Answer]:
        raise NotImplementedError


class SimpleAnswerWriter(AnswerWriter):
    """Reuses the pre-computed page summary as-is — no generation, so it's
    free and deterministic.

    `page_text_lookup` (title -> the page's first indexed chunk) enables the
    direct-answer path: with it, "Quanto de pós-conjuração tem X?" is
    answered with X's pós-conjuração instead of only a link to X. Passing
    None turns the feature off and restores link-only replies.
    """

    def __init__(self, page_text_lookup: Callable[[str], str] | None = None):
        self.page_text_lookup = page_text_lookup

    def write(self, question: str, results: list[RetrievalResult]) -> list[Answer]:
        return [
            Answer(
                title=r.title,
                url=r.url,
                description=r.summary or r.chunk_text[:300],
                similarity=r.similarity,
                fact=self._fact_for(question, r.title),
            )
            for r in results
        ]

    def _fact_for(self, question: str, title: str) -> Fact | None:
        if self.page_text_lookup is None:
            return None
        return find_fact(question, self.page_text_lookup(title), title)


class LLMAnswerWriter(AnswerWriter):
    """Optional pluggable slot for an LLM-generated answer/description.

    Not implemented by default — this is a stub showing the intended shape
    so a future contributor (or you, later) can wire up a real provider
    (Anthropic/OpenAI/local LLM/etc.) behind an API key to produce a
    question-tailored summary instead of the static page snippet. Selected
    via ANSWER_WRITER=llm; requires a real implementation to run.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "LLMAnswerWriter requires an API key. Set LLM_API_KEY in .env, "
                "or keep ANSWER_WRITER=simple to use the free static-summary writer."
            )
        self.api_key = api_key

    def write(self, question: str, results: list[RetrievalResult]) -> list[Answer]:
        raise NotImplementedError(
            "LLM-enhanced answer writing is not implemented yet. Implement this "
            "method to call your LLM provider of choice with `question` + each "
            "result's chunk_text as context, or set ANSWER_WRITER=simple in .env."
        )


def get_answer_writer(
    writer_name: str,
    llm_api_key: str = "",
    page_text_lookup: Callable[[str], str] | None = None,
) -> AnswerWriter:
    if writer_name == "simple":
        return SimpleAnswerWriter(page_text_lookup=page_text_lookup)
    if writer_name == "llm":
        return LLMAnswerWriter(api_key=llm_api_key)
    raise ValueError(f"Unknown ANSWER_WRITER: {writer_name!r} (expected 'simple' or 'llm')")
