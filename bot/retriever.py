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

"Literally" is tolerant of spacing everywhere, and of number for the
character classes only, since the wiki titles those in the plural while
players name them in the singular. See `_fold` and `_singularize`.

A page also answers to the names the game gives it in other languages — the
"Wild Fire / Fuego Salvaje" row the ingest lifts off each skill page — so
"qual o cast fixo de Wild Fire?" reaches *Fogo de Supressão*. Mixing an
English skill name into a Portuguese sentence is how people type in a LATAM
channel, and to the matcher it is just another name for the page. Which of
those names are safe to match on is the one judgement call: see
MIN_ALIAS_WORDS.

Some answers aren't pages at all. *Sangramento* is a section of *Efeitos
negativos*, and the ingest indexes it as a unit of its own (see
`ingest/build_index.py`), anchor included. Here those sections are matched by
name exactly as pages are, but one rank below them (SECTION_MATCH_SCORE) and
never under a name the infobox uses for a row: "qual o alcance de Bola de
Fogo?" is a question about the skill, not about the stat page's *Alcance*
section. What that leaves is MIN_SECTION_NAME_WORDS, the second judgement
call.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from bot.facts import is_field_label

# Formats written by the ingest and read back here, so each has a single
# definition: categories ride along in the page's text, the other-language
# names in its metadata.
from ingest.html_text import parse_categories, parse_names

logger = logging.getLogger(__name__)

# Titles shorter than this are ignored by the exact-title matcher: two- and
# three-letter page names ("SP", "AP", "ATQ") appear inside ordinary
# sentences all the time and would fire constantly.
MIN_TITLE_MATCH_CHARS = 5

# The singular pass needs its own, lower bar: the whole point is that the
# typed word is shorter than the page name ("Diva" -> "Divas"). The page it
# reaches is still held to MIN_TITLE_MATCH_CHARS.
MIN_SINGULAR_MATCH_CHARS = 4

# Fewest words an other-language name must have to be matched literally.
#
# Two-word names ("Wild Fire", "Earth Strain", "Cuchillo Danzante") are safe:
# a two-word foreign phrase in a Portuguese sentence is a reference to the
# skill, essentially always. One-word names are not, and the wiki has ~130 of
# them per language. Measured against a vocabulary of the words that appear
# lowercase in the wiki's own prose, the single-word names that are also
# ordinary words are few but ruinous: *Faxina* is called "Remover" and
# *Fogo Grego* "Bomba", so "como remover o encanto?" and "onde compro bomba?"
# would both become confident wrong answers. Worst of all, *Resfriamento* is
# called "Cooldown" — the exact word this bot wants to read as a question
# about some *other* skill's cooldown.
#
# The names are still embedded in full (ingest writes a card vector for them),
# so a one-word name can still be found by meaning; it just can't claim a page
# outright. Lower this to 1 to match every name literally.
#
# Two words is not airtight, and the residual risk is Spanish rather than
# English: 26 of the wiki's 1,692 multi-word names are built entirely out of
# ordinary words, and 24 of those are Spanish phrases a Portuguese speaker
# could type by accident — "Escudo Sagrado" is *Escudo Divino*, "Ataque
# Rápido" is *Avanço Ofensivo*. They are kept because the channel is a LATAM
# one and someone really does ask in Spanish.
MIN_ALIAS_WORDS = 2

# Fewest words a section's name must have to be matched literally. One, on
# purpose — see `_load_title_index`. The reason it is a dial rather than a
# constant is that one-word section names are the one place this bot answers
# to ordinary Portuguese words: the wiki points "Sorte", "Força", "Vento" and
# "Visual" at sections of its stat and item pages, so "kkkkk mano que sorte a
# sua" now reaches *Atributos § Sorte* where it used to reach nothing. Of the
# 134 one-word section names on this wiki, ~15 read that way; the rest are
# game jargon ("Sangramento", "Petrificação", "Hipotermia") that no one types
# by accident. Raise this to 2 to keep only the multi-word names ("Sono
# Profundo", "Velocidade de Ataque") and give up the single words entirely.
#
# Nothing measurable separates the two groups: mid-sentence, the wiki's own
# prose capitalises "Sorte" 76% of the time and "Sangramento" 79%, and a
# casual "que sorte a sua" scores no lower against the *Sorte* section than
# "como curar congelamento?" does against *Congelamento*. The choice is
# recall against quiet, not a threshold waiting to be tuned.
MIN_SECTION_NAME_WORDS = 1

# Category marking the pages that may be matched in the singular. See
# `_singularize` for why it is only these. A wiki without this category
# simply gets no singular matching.
CLASS_CATEGORY = "Classes"

# Score floor for a page whose full title was found in the message. Sits
# above anything the vector search realistically produces, so a named page
# always wins.
EXACT_TITLE_SCORE = 0.90

# The same, for a *section* named in the message. Deliberately lower, so that
# a real page always outranks a section of some other page when the message
# names both. The wiki's section names are the ordinary words of the game —
# "Alcance", "Conjuração", "Sorte" are all sections of stat pages — and in
# "qual o alcance de Bola de Fogo?" the subject is the skill, not the stat.
SECTION_MATCH_SCORE = 0.85

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
    url: str  # the section's #anchor included, when this is a section
    summary: str
    similarity: float  # raw cosine similarity of the page's best chunk
    score: float  # ranking score: similarity plus the title-match bonus
    chunk_text: str
    # The page's name in the game's other languages, straight off the wiki.
    # On a section, the titles the wiki redirects to it.
    names: tuple[str, ...] = ()
    # True when the message actually mentions the page's name (fully or in
    # part). A match without it rests on embedding similarity alone, which
    # is far more likely to be noise — see SEMANTIC_ONLY_THRESHOLD.
    title_matched: bool = False
    # The heading, when the match is one named section of a page rather than
    # the page itself. Empty for a page.
    section: str = ""

    @property
    def name(self) -> str:
        """What this match is called — the subject of the answer.

        A section is named by its heading, but the heading alone ("Coma",
        "Sorte") reads as an odd thing to be told, so the page it belongs to
        is kept alongside it.
        """
        return f"{self.title} § {self.section}" if self.section else self.title


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
    - *Plural stripping*, applied to every page. Re-measured over a
      19.3k-word vocabulary drawn from the wiki, it lets 120 everyday words
      reach a title, and the worst of them are the hub pages: "niveis" ->
      *Nível* (a word used on 1,339 pages), "habilidade" -> *Habilidades*
      (955), "quest" -> *Quests* (426), "item" -> *Itens*, "carta" ->
      *Cartas*, "monstro" -> *Monstros*. Those are among the most common
      words in a Ragnarok channel, so the bot answered a generic index page
      to half the conversation.

    `_singularize` keeps the useful half of that second rule by restricting
    it to one category of page.
    """
    return name.replace(" ", "")


def _singularize_word(word: str) -> str:
    """A Portuguese plural reduced to its singular, roughly."""
    if len(word) <= 3 or not word.endswith("s"):
        return word
    for suffix, singular in (
        ("oes", "ao"), ("aes", "ao"),  # "falcoes" -> "falcao" (accents already stripped)
        ("ais", "al"), ("eis", "el"), ("ois", "ol"), ("uis", "ul"),  # "cardeais" -> "cardeal"
        ("ns", "m"),  # "espadachins" -> "espadachim"
        ("res", "r"), ("zes", "z"), ("ses", "s"),  # "mercadores" -> "mercador"
    ):
        if word.endswith(suffix):
            return word[: -len(suffix)] + singular
    return word[:-1]


def _singularize(name: str) -> str:
    """Collapse a name so singular and plural spellings compare equal.

    Applied to *both* sides, so it only has to be self-consistent, not
    linguistically right: "Magus" reduces to the non-word "magu", and a
    message saying "Magus" reduces to the same thing.

    This is deliberately not offered for every page — see `_fold` for what
    happens when it is. It is limited to the character classes, because they
    are the one group the wiki titles in the plural (*Mandraques*, *Divas*)
    while players always name them in the singular: "como eu viro
    Mandraque?" is the normal way to ask, and matching it costs nothing that
    the hub pages cost. Which pages those are is read from the wiki's own
    "Classes" category rather than listed here, so a class added to the wiki
    works after the next ingest with no code change.
    """
    return " ".join(_singularize_word(word) for word in name.split())


def _where(*clauses: dict) -> dict:
    """A Chroma filter from one or more clauses ($and needs at least two)."""
    return clauses[0] if len(clauses) == 1 else {"$and": list(clauses)}


def _name_score(page: dict) -> float:
    return SECTION_MATCH_SCORE if page["section"] else EXACT_TITLE_SCORE


def _is_matchable_section_name(name: str) -> bool:
    """Whether a section may be claimed by this name when a message uses it.

    Held to the same bars a page title is, plus one of its own: a name an
    infobox uses for a row is how questions ask about *other* pages, so it
    can't be a name here. See MIN_SECTION_NAME_WORDS for the rest.
    """
    normalized = _normalize(name)
    return (
        len(normalized) >= MIN_TITLE_MATCH_CHARS
        and len(normalized.split()) >= MIN_SECTION_NAME_WORDS
        and not is_field_label(name)
    )


class _TitleIndex:
    """Finds page names inside a message, tolerating how people type them.

    Matching runs over *word windows* of the message rather than scanning for
    each title as a substring, which keeps matches aligned to word boundaries
    and makes the tolerant passes cheap dictionary lookups:

      1. exact — "rockridge" == "Rockridge"
      2. folded — "rock ridge" == "Rockridge"
      3. singular — "mandraque" == "Mandraques", class pages only

    Without the second pass a single space is the difference between a
    confident answer and silence; without the third, so is a plural.
    """

    def __init__(self) -> None:
        self.by_name: dict[str, dict] = {}
        self.by_folded: dict[str, dict] = {}
        self.by_singular: dict[str, dict] = {}
        self.max_words = 1

    def add(self, normalized_title: str, page: dict, singular_alias: bool = False) -> None:
        if normalized_title in self.by_name:
            return
        self.by_name[normalized_title] = page
        self.max_words = min(max(self.max_words, len(normalized_title.split())), MAX_TITLE_WORDS)
        # setdefault: if two names fold together, the first keeps the folded
        # key and the other stays reachable through the exact lookup.
        self.by_folded.setdefault(_fold(normalized_title), page)
        if singular_alias:
            self.by_singular.setdefault(_fold(_singularize(normalized_title)), page)

    def add_alias(self, normalized_alias: str, page: dict) -> None:
        """Register one of the game's other-language names for a page.

        Matched exactly like a title, minus the singular pass — that rule is
        about Portuguese plurals. A name never displaces a real page title,
        and where two pages share a name the first one keeps it: *Rapto* and
        *Plágio* are both "Intimidate", and a message saying "intimidate"
        gives nothing to tell them apart.
        """
        self.add(normalized_alias, page)

    def lookup(self, query_norm: str, limit: int = 3) -> list[tuple[dict, float]]:
        """Page names present in the message, most specific first."""
        tokens = query_norm.split()
        found: dict[tuple[str, str], tuple[dict, float, int]] = {}

        for size in range(1, self.max_words + 1):
            for start in range(len(tokens) - size + 1):
                window = " ".join(tokens[start:start + size])
                match = self._match_window(window)
                if match is None:
                    continue
                page, score = match
                key = (page["title"], page["section"])
                previous = found.get(key)
                # Prefer the longer window: "rajada frenetica" beats "rajada".
                if previous is None or (score, len(window)) > (previous[1], previous[2]):
                    found[key] = (page, score, len(window))

        ranked = sorted(found.values(), key=lambda item: (item[1], item[2]), reverse=True)
        return [(page, score) for page, score, _ in ranked[:limit]]

    def _match_window(self, window: str) -> tuple[dict, float] | None:
        if len(window) >= MIN_TITLE_MATCH_CHARS:
            page = self.by_name.get(window) or self.by_folded.get(_fold(window))
            if page is not None:
                return page, _name_score(page)

        if len(window) >= MIN_SINGULAR_MATCH_CHARS:
            page = self.by_singular.get(_fold(_singularize(window)))
            if page is not None:
                return page, _name_score(page)

        return None


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

        # Clauses restricting a query to a page's own rows, empty on an index
        # that holds no sections — a wiki that names none, or an index built
        # before they were read. Chroma matches nothing against a field its
        # rows don't have, so filtering unconditionally would quietly turn
        # off every feature below that reads a page's text.
        self._page_only: tuple[dict, ...] = ()
        self._titles = self._load_title_index()
        self._page_heads: dict[str, str] = {}

    def page_head(self, title: str) -> str:
        """The page's first chunk — where the infobox is, if it has one.

        Kept separate from `search`, because the chunk a question *matches*
        is often not the chunk that holds the facts: a named page is matched
        through its title card, whose document is only the page blurb.
        Cached because a channel tends to ask about the same few pages.

        Only the page's own chunks, never a section's: `section` is the field
        that tells them apart in the index (see `ingest/build_index.py`), and
        an infobox belongs to the page.
        """
        cached = self._page_heads.get(title)
        if cached is not None:
            return cached

        try:
            rows = self.collection.get(
                where=_where({"title": title}, *self._page_only, {"chunk_index": 0}),
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
        """Build the name index from the collection's metadata.

        Small enough to hold in memory (one entry per page and per named
        section, not per chunk) and lets us answer "is this named in the
        message?" without a vector search.
        """
        index = _TitleIndex()
        try:
            rows = self.collection.get(include=["metadatas"])
        except Exception:  # pragma: no cover - defensive; empty/foreign collection
            logger.exception("Could not load page metadata for title matching")
            return index

        pages: dict[str, dict] = {}
        sections: dict[tuple[str, str], dict] = {}
        for meta in rows.get("metadatas") or []:
            title = meta.get("title")
            if not title:
                continue
            entry = {
                "title": title,
                "url": meta["url"],
                "summary": meta.get("summary", ""),
                "names": parse_names(meta.get("names", "")),
                "section": meta.get("section", ""),
            }
            if entry["section"]:
                sections.setdefault((title, entry["section"]), entry)
            else:
                pages.setdefault(title, entry)

        if sections:
            self._page_only = ({"section": ""},)

        # Only titles that actually read differently in the singular are worth
        # checking the category of — the other ~1,450 can't gain an alias.
        plural = {
            title for title in pages
            if _singularize(_normalize(title)) != _normalize(title)
        }
        class_titles = self._class_page_titles(plural)

        for title, page in pages.items():
            if len(_normalize(title)) >= MIN_TITLE_MATCH_CHARS:
                index.add(_normalize(title), page, singular_alias=title in class_titles)
        titles_indexed = len(index.by_name)

        # Second pass, after every real title is in: a page named *Vigor*
        # must not be lost to *Determinação*, whose English name is "Vigor".
        for page in pages.values():
            for name in page["names"]:
                normalized = _normalize(name)
                if len(normalized) < MIN_TITLE_MATCH_CHARS:
                    continue
                if len(normalized.split()) < MIN_ALIAS_WORDS:
                    continue
                index.add_alias(normalized, page)
        aliases_indexed = len(index.by_name) - titles_indexed

        # Sections last, so a name shared with a real page always resolves to
        # the page. One-word names are kept, unlike MIN_ALIAS_WORDS: the
        # game's one-word English names are accidents of translation, while a
        # one-word section name is the whole point — "Sangramento",
        # "Cegueira", "Congelamento" are what these things *are* called, and
        # an editor chose to point them at that heading.
        for section in sections.values():
            for name in (section["section"], *section["names"]):
                if _is_matchable_section_name(name):
                    index.add_alias(_normalize(name), section)

        logger.info(
            "Indexed %d names for matching: %d page titles, %d from other languages, "
            "%d for %d named sections (%d titles matchable in the singular)",
            len(index.by_name), titles_indexed, aliases_indexed,
            len(index.by_name) - titles_indexed - aliases_indexed, len(sections),
            len(index.by_singular),
        )
        return index

    def _class_page_titles(self, titles: set[str]) -> set[str]:
        """Which of `titles` are character-class pages.

        Read from the category line the ingest appends to each page's text,
        so the wiki stays the source of truth for what a class is. The line
        goes on the end of the page, so it lands in whichever chunk happens
        to be last — hence the highest `chunk_index` per title, among the
        page's own chunks (a section's chunks don't carry the line).
        """
        if not titles:
            return set()

        try:
            rows = self.collection.get(
                where=_where({"title": {"$in": sorted(titles)}}, *self._page_only),
                include=["documents", "metadatas"],
            )
        except Exception:  # pragma: no cover - defensive; older/foreign index
            logger.exception("Could not load page categories; singular matching is off")
            return set()

        last_chunk: dict[str, tuple[int, str]] = {}
        for meta, document in zip(rows.get("metadatas") or [], rows.get("documents") or []):
            title, chunk_index = meta.get("title"), meta.get("chunk_index", 0)
            if title and (title not in last_chunk or chunk_index > last_chunk[title][0]):
                last_chunk[title] = (chunk_index, document or "")

        return {
            title for title, (_, document) in last_chunk.items()
            if CLASS_CATEGORY in parse_categories(document)
        }

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
                    names=parse_names(meta.get("names", "")),
                    section=meta.get("section", ""),
                )
            )
        return out

    def best_unique_pages(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Best matching *subjects* for a message, ranked by hybrid score.

        Collapses several chunks of the same page down to its best-scoring
        one, re-ranks the vector hits with the title-match bonus, and merges
        in any page explicitly named in the message.

        A named section counts as a subject of its own, so *Efeitos
        negativos* and its *Sangramento* section are two candidates rather
        than one — otherwise the page would swallow the very anchor that
        makes it a useful answer.
        """
        query_norm = _normalize(query)
        query_tokens = _content_tokens(query_norm)

        # Over-fetch: several of the raw hits are usually different chunks of
        # the same page, and the re-ranking needs candidates to work with.
        raw = self.search(query, top_k=max(top_k * 8, 24))

        best_per_page: dict[tuple[str, str], RetrievalResult] = {}
        for result in raw:
            scored = self._apply_title_bonus(result, query_norm, query_tokens)
            existing = best_per_page.get((scored.title, scored.section))
            if existing is None or scored.score > existing.score:
                best_per_page[(scored.title, scored.section)] = scored

        for named, name_score in self._titles.lookup(query_norm):
            key = (named["title"], named["section"])
            existing = best_per_page.get(key)
            score = max(name_score, existing.score if existing else 0.0)
            best_per_page[key] = RetrievalResult(
                title=named["title"],
                url=named["url"],
                summary=named["summary"] or (existing.summary if existing else ""),
                similarity=existing.similarity if existing else 0.0,
                score=score,
                chunk_text=existing.chunk_text if existing else "",
                names=named["names"],
                title_matched=True,
                section=named["section"],
            )

        ranked = sorted(best_per_page.values(), key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    def _apply_title_bonus(
        self, result: RetrievalResult, query_norm: str, query_tokens: set[str]
    ) -> RetrievalResult:
        if result.section and not _is_matchable_section_name(result.section):
            return result  # a name this bot doesn't answer to; see above

        # A section is named by its heading; the page's title is the heading
        # this one happens to live under and says nothing about the subject.
        title_tokens = _content_tokens(_normalize(result.section or result.title))
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
            names=result.names,
            title_matched=matched >= 1.0,
            section=result.section,
        )
