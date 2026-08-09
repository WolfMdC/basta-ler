"""
Retrieval against the local Chroma index built by `ingest/build_index.py`.

Loads the same multilingual sentence-transformers model used at index time,
embeds the incoming message, and queries Chroma for the closest chunks.
Chroma's cosine *distance* is converted to a cosine *similarity* score
(1 - distance) so thresholds/logging read naturally as "0 = unrelated,
1 = identical".

Pure vector similarity is a poor fit for a game wiki on its own: page names
like "Rajada Frenética" are proper nouns, and the multilingual MiniLM model
happily rates any other skill page at ~0.5 against them. So the vector hits
are re-ranked with a lexical signal on the page *title*, and any page whose
title literally appears in the message is pulled in directly — if someone
names a page, that page is the answer.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Titles shorter than this are ignored by the exact-title matcher: two- and
# three-letter page names ("SP", "AP", "ATQ") appear inside ordinary
# sentences all the time and would fire constantly.
MIN_TITLE_MATCH_CHARS = 5

# Score floor for a page whose full title was found in the message. Sits
# above anything the vector search realistically produces, so a named page
# always wins.
EXACT_TITLE_SCORE = 0.90

# Maximum bonus for a partial title match (e.g. message mentions "rajada"
# and the page is "Rajada Certeira"), scaled by the fraction matched.
PARTIAL_TITLE_BONUS = 0.15

# Longest page name, in words, we bother looking for in a message. Bounds
# how many word-windows each message is checked against.
MAX_TITLE_WORDS = 8

# Portuguese function words carry no page-identifying signal.
_STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "de", "do", "da", "dos", "das", "em",
    "no", "na", "nos", "nas", "por", "para", "com", "sem", "e", "ou", "que",
    "se", "ao", "aos", "the", "of",
}


@dataclass(frozen=True)
class RetrievalResult:
    title: str
    url: str
    summary: str
    similarity: float  # raw cosine similarity of the page's best chunk
    score: float  # ranking score: similarity plus the title-match bonus
    chunk_text: str
    # True when the message actually mentions the page's name (fully or in
    # part). A match without it rests on embedding similarity alone, which
    # is far more likely to be noise — see SEMANTIC_ONLY_THRESHOLD.
    title_matched: bool = False


def _normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation — so "Frenetica" matches
    "Frenética" and "pos-conjuracao" matches "pós-conjuração"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", stripped)).strip()


def _content_tokens(normalized: str) -> set[str]:
    return {tok for tok in normalized.split() if len(tok) > 2 and tok not in _STOPWORDS}


def _fold(name: str) -> str:
    """Collapse a name to the form used for tolerant comparison.

    Only spacing is ignored, because nobody agrees on "rock ridge" vs
    "Rockridge" or where the hyphens go, and that variation is
    deterministic — folding it can't merge two different pages.

    Two looser rules were tried and rejected against this wiki:

    - *Edit distance.* Portuguese is full of common words sitting one
      character from a page name — "preciso" ("I need") is 0.93 similar to
      the stat page "Precisão". Measured over a 3.8k-word vocabulary drawn
      from the wiki itself, fuzzy matching fired on 35-76 everyday words,
      each of which would become a confidently wrong answer.
    - *Plural stripping.* It matched 61 vocabulary words, and almost all of
      them were hub pages: "habilidade" -> "Habilidades", "monstro" ->
      "Monstros", "carta" -> "Cartas". Those are among the most common words
      in a Ragnarok channel, so the bot answered a generic index page to
      half the conversation. Specific pages — skills, items, cities — are
      singular proper nouns that gain nothing from it.
    """
    return name.replace(" ", "")


class _TitleIndex:
    """Finds page names inside a message, tolerating how people type them.

    Matching runs over *word windows* of the message rather than scanning for
    each title as a substring, which keeps matches aligned to word boundaries
    and makes the tolerant passes cheap dictionary lookups:

      1. exact — "rockridge" == "Rockridge"
      2. folded — "rock ridge" == "Rockridge"

    Without the second pass a single space is the difference between a
    confident answer and silence.
    """

    def __init__(self) -> None:
        self.by_name: dict[str, dict] = {}
        self.by_folded: dict[str, dict] = {}
        self.max_words = 1

    def add(self, normalized_title: str, page: dict) -> None:
        if normalized_title in self.by_name:
            return
        self.by_name[normalized_title] = page
        self.max_words = min(max(self.max_words, len(normalized_title.split())), MAX_TITLE_WORDS)
        # setdefault: if two names fold together, the first keeps the folded
        # key and the other stays reachable through the exact lookup.
        self.by_folded.setdefault(_fold(normalized_title), page)

    def lookup(self, query_norm: str, limit: int = 3) -> list[tuple[dict, float]]:
        """Page names present in the message, most specific first."""
        tokens = query_norm.split()
        found: dict[str, tuple[dict, float, int]] = {}

        for size in range(1, self.max_words + 1):
            for start in range(len(tokens) - size + 1):
                window = " ".join(tokens[start:start + size])
                match = self._match_window(window)
                if match is None:
                    continue
                page, score = match
                previous = found.get(page["title"])
                # Prefer the longer window: "rajada frenetica" beats "rajada".
                if previous is None or (score, len(window)) > (previous[1], previous[2]):
                    found[page["title"]] = (page, score, len(window))

        ranked = sorted(found.values(), key=lambda item: (item[1], item[2]), reverse=True)
        return [(page, score) for page, score, _ in ranked[:limit]]

    def _match_window(self, window: str) -> tuple[dict, float] | None:
        if len(window) < MIN_TITLE_MATCH_CHARS:
            return None

        page = self.by_name.get(window) or self.by_folded.get(_fold(window))
        return (page, EXACT_TITLE_SCORE) if page is not None else None


class Retriever:
    def __init__(self, chroma_db_path: str, collection_name: str, embedding_model: str):
        logger.info("Loading embedding model %s…", embedding_model)
        self.model = SentenceTransformer(embedding_model)

        self.client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        count = self.collection.count()
        if count == 0:
            logger.warning(
                "Chroma collection %r is empty — run `python -m ingest.build_index` first.",
                collection_name,
            )
        else:
            logger.info("Loaded Chroma collection %r with %d chunks", collection_name, count)

        self._titles = self._load_title_index()
        self._page_heads: dict[str, str] = {}

    def page_head(self, title: str) -> str:
        """The page's first chunk — where the infobox is, if it has one.

        Kept separate from `search`, because the chunk a question *matches*
        is often not the chunk that holds the facts: a named page is matched
        through its title card, whose document is only the page blurb.
        Cached because a channel tends to ask about the same few pages.
        """
        cached = self._page_heads.get(title)
        if cached is not None:
            return cached

        try:
            rows = self.collection.get(
                where={"$and": [{"title": title}, {"chunk_index": 0}]},
                include=["documents"],
                limit=1,
            )
            head = next(iter(rows.get("documents") or []), "") or ""
        except Exception:  # pragma: no cover - defensive; never worth a crash
            logger.exception("Could not load the first chunk of page %r", title)
            head = ""

        self._page_heads[title] = head
        return head

    def _load_title_index(self) -> _TitleIndex:
        """Build the page-name index from the collection's metadata.

        Small enough to hold in memory (one entry per page, not per chunk)
        and lets us answer "is this page named in the message?" without a
        vector search.
        """
        index = _TitleIndex()
        try:
            rows = self.collection.get(include=["metadatas"])
        except Exception:  # pragma: no cover - defensive; empty/foreign collection
            logger.exception("Could not load page metadata for title matching")
            return index

        for meta in rows.get("metadatas") or []:
            title = meta.get("title")
            if not title:
                continue
            normalized = _normalize(title)
            if len(normalized) < MIN_TITLE_MATCH_CHARS:
                continue
            index.add(normalized, {"title": title, "url": meta["url"], "summary": meta.get("summary", "")})

        logger.info("Indexed %d distinct page titles for name matching", len(index.by_name))
        return index

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        ids = results.get("ids", [[]])[0]
        if not ids:
            return []

        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        out = []
        for distance, meta, doc in zip(distances, metadatas, documents):
            similarity = 1.0 - distance
            out.append(
                RetrievalResult(
                    title=meta["title"],
                    url=meta["url"],
                    summary=meta.get("summary", ""),
                    similarity=similarity,
                    score=similarity,
                    chunk_text=doc,
                )
            )
        return out

    def best_unique_pages(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Best matching *pages* for a message, ranked by hybrid score.

        Collapses several chunks of the same page down to its best-scoring
        one, re-ranks the vector hits with the title-match bonus, and merges
        in any page explicitly named in the message.
        """
        query_norm = _normalize(query)
        query_tokens = _content_tokens(query_norm)

        # Over-fetch: several of the raw hits are usually different chunks of
        # the same page, and the re-ranking needs candidates to work with.
        raw = self.search(query, top_k=max(top_k * 8, 24))

        best_per_page: dict[str, RetrievalResult] = {}
        for result in raw:
            scored = self._apply_title_bonus(result, query_norm, query_tokens)
            existing = best_per_page.get(scored.title)
            if existing is None or scored.score > existing.score:
                best_per_page[scored.title] = scored

        for named, name_score in self._titles.lookup(query_norm):
            existing = best_per_page.get(named["title"])
            score = max(name_score, existing.score if existing else 0.0)
            best_per_page[named["title"]] = RetrievalResult(
                title=named["title"],
                url=named["url"],
                summary=named["summary"] or (existing.summary if existing else ""),
                similarity=existing.similarity if existing else 0.0,
                score=score,
                chunk_text=existing.chunk_text if existing else "",
                title_matched=True,
            )

        ranked = sorted(best_per_page.values(), key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    def _apply_title_bonus(
        self, result: RetrievalResult, query_norm: str, query_tokens: set[str]
    ) -> RetrievalResult:
        title_tokens = _content_tokens(_normalize(result.title))
        if not title_tokens:
            return result

        matched = len(title_tokens & query_tokens) / len(title_tokens)
        if matched == 0:
            return result

        # Only a *complete* title counts as "the message named this page".
        # One shared word is far too weak a signal to relax the confidence
        # bar on: "kkkkk mano que sorte a sua" shares "sorte" with the skill
        # page "Beijo da Sorte" without being about it at all. Partial hits
        # still get the ranking bonus, they just don't unlock the lower bar.
        bonus = PARTIAL_TITLE_BONUS * matched
        return RetrievalResult(
            title=result.title,
            url=result.url,
            summary=result.summary,
            similarity=result.similarity,
            score=min(result.similarity + bonus, 1.0),
            chunk_text=result.chunk_text,
            title_matched=matched >= 1.0,
        )
