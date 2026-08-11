"""
Crawl a MediaWiki wiki via the Action API and (re)build the local Chroma
vector index used by the Discord bot.

Usage:
    python -m ingest.build_index
    python -m ingest.build_index --api-url https://oldschool.runescape.wiki/api.php \\
                                  --wiki-base-url https://oldschool.runescape.wiki/w/ \\
                                  --limit 25

By default reads MEDIAWIKI_API_URL / WIKI_ARTICLE_BASE_URL from config/.env.
--api-url / --wiki-base-url / --limit let you point at a different wiki
without touching .env — handy for a quick smoke test of the pipeline against
any public MediaWiki install.

Incremental re-indexing: a small JSON state file (INGEST_STATE_PATH) records
each page's title + revision id + index format + the names redirecting to its
sections + the chunk ids it produced. On re-run, pages whose revid and
section names haven't changed *and* whose vectors were built by the current
INDEX_FORMAT are skipped entirely (no re-embedding); pages that changed have
their old chunks deleted from Chroma and replaced; pages that disappeared
from the wiki have their chunks removed too.

That state is written as the crawl goes, not just at the end, so a crawl cut
short by a flaky wiki resumes where it stopped instead of starting over.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CONFIG
from ingest.chunker import Chunk, chunk_text
from ingest.html_text import append_categories, format_names
from ingest.mediawiki_client import MediaWikiClient, PageContent, PageInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")

# What a page's vectors were built from. Bump this whenever the ingest
# changes what it would produce for an *unchanged* page — extraction,
# chunking, metadata, the embedding model — and the next ordinary run
# re-indexes everything, no `--force` to remember. Version 2 added the
# other-language names (see ingest/html_text.py); version 3 the named
# sections (see `_section_rows`).
INDEX_FORMAT = 3

# How often the crawl's progress is written to the state file. Every page
# would mean 1,900 rewrites of the whole file; only at the end means a crawl
# that dies at 60% has nothing to resume from.
STATE_SAVE_EVERY = 50


def _load_state(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: str, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _state_section_names(page_info: PageInfo) -> list[list[str]]:
    """The page's section names in the form the state file stores them.

    A plain JSON structure, because it is compared against what was read back
    out of the file — where tuples have already become lists.
    """
    return sorted([name.anchor, name.name] for name in page_info.section_names)


def _is_current(prev: dict | None, page_info: PageInfo) -> bool:
    """Whether a page's stored vectors are still the ones we'd produce now.

    Three things matter: the wiki hasn't touched the page, nobody has pointed
    a new redirect at one of its sections, and the vectors came from this
    version of the ingest. Without the last one, a change to how pages are
    read leaves every unchanged page looking up to date and silently keeps
    its stale vectors. Without the redirect check, a section named after the
    page was last indexed stays unfindable until the page itself is edited —
    the wiki's own revision id says nothing about pages pointing *at* it.
    """
    if not prev:
        return False
    return (
        prev.get("revid") == page_info.revid
        and prev.get("format") == INDEX_FORMAT
        and prev.get("section_names", []) == _state_section_names(page_info)
    )


def _page_url(base_url: str, title: str) -> str:
    # MediaWiki article paths use underscores for spaces. Percent-encode the
    # rest: Portuguese titles are full of accented characters, and Discord
    # refuses to linkify a URL (or a [text](url) link) containing raw
    # non-ASCII bytes — "https://browiki.org/wiki/Lâminas_Aceleradas" would
    # render as unclickable plain text.
    slug = re.sub(r"\s+", "_", title.strip())
    return base_url.rstrip("/") + "/" + quote(slug, safe="_/:()")


def _section_url(page_url: str, anchor: str) -> str:
    # Percent-encoded like the rest of the URL, for the same reason: Discord
    # won't linkify "…/Efeitos_negativos#Concretação" with the accent raw.
    return f"{page_url}#{quote(anchor, safe='_/:()')}"


def _embedding_text(title: str, chunk_body: str) -> str:
    """Text actually handed to the embedding model.

    The page title is prepended to every chunk because the article body
    usually never repeats it — "Rajada Frenética" opens with "Dispara flechas
    em um único alvo…", so a chunk embedded on its own carries no signal for
    someone asking about the skill *by name*, which is how most questions in
    Discord are phrased.
    """
    return f"{title}\n{chunk_body}"


@dataclass(frozen=True)
class _Row:
    """One vector to store: what it's embedded from, and what it points at."""

    id: str
    embedding_input: str  # text handed to the embedding model
    document: str  # text stored alongside it and read back by the bot
    metadata: dict = field(default_factory=dict)


def _page_rows(page_info: PageInfo, content: PageContent, url: str, chunks: list[Chunk]) -> list[_Row]:
    """The vectors for a page as a whole."""
    summary = content.lead or chunks[0].text
    metadata = {
        "title": page_info.title,
        "url": url,
        "summary": content.lead,
        "names": format_names(content.names),
        "revid": page_info.revid,
        # Empty for a page, the heading for a section — the field that tells
        # the two kinds of row apart when the bot reads them back.
        "section": "",
    }

    # A title-only "card" vector alongside the body chunks. This embedding
    # model dilutes heavily with length: on a stat-heavy page the numbers
    # swamp the name — "Rajada Frenética" scores 0.58 against its own title
    # but only 0.40 against its full chunk, so questions naming a page don't
    # retrieve it. The card keeps one clean by-name vector per page; the body
    # chunks still answer descriptive questions ("qual habilidade acerta 3
    # vezes com arco").
    rows = [
        _Row(
            id=f"{page_info.pageid}::title",
            embedding_input=page_info.title,
            document=summary,
            metadata={**metadata, "chunk_index": -1},
        )
    ]

    # Same idea for the names the game uses in other languages ("Wild Fire /
    # Fuego Salvaje"): its own card rather than a longer title card, so asking
    # by the English name is as clean a vector as asking by the Portuguese
    # one, and neither dilutes the other.
    if content.names:
        rows.append(
            _Row(
                id=f"{page_info.pageid}::names",
                embedding_input=format_names(content.names),
                document=summary,
                metadata={**metadata, "chunk_index": -2},
            )
        )

    rows += [
        _Row(
            id=f"{page_info.pageid}::{chunk.index}",
            embedding_input=_embedding_text(page_info.title, chunk.text),
            document=chunk.text,
            metadata={**metadata, "chunk_index": chunk.index},
        )
        for chunk in chunks
    ]
    return rows


def _section_rows(
    page_info: PageInfo,
    content: PageContent,
    page_url: str,
    chunk_size: int,
    overlap: int,
) -> list[_Row]:
    """The vectors for the sections of a page that the wiki has *named*.

    Some pages are a title with the real subjects underneath it: *Sangramento*
    is not a page on this wiki, it is one section of *Efeitos negativos*, and
    the answer to "o que é sangramento?" is that section and the anchor that
    opens the page on it — not the page, whose own summary is about none of
    its 29 effects in particular.

    Which sections count is read off the wiki rather than guessed at. A
    heading on its own is no evidence: "Notas" is a heading on 1,161 of this
    wiki's pages and names nothing. A *redirect* pointing at one — the editor
    who made `Sangramento` an alias for `Efeitos negativos#Sangramento` — is
    the wiki stating that readers look that section up by name, and what they
    call it. 276 such redirects exist here, covering 138 sections of 33 pages.

    Each named section is then indexed exactly like a small page: a card
    vector for its name, one for the other names redirecting to it, and its
    own body chunks. Sections the wiki hasn't named are left out; they are
    still part of their page's own chunks, as they have always been.
    """
    names_by_anchor: dict[str, list[str]] = {}
    for section_name in page_info.section_names:
        names_by_anchor.setdefault(section_name.anchor, []).append(section_name.name)

    rows: list[_Row] = []
    for section in content.sections:
        names = sorted(names_by_anchor.get(section.anchor, ()))
        if not names:
            continue

        metadata = {
            "title": page_info.title,
            "url": _section_url(page_url, section.anchor),
            "summary": section.lead,
            "names": format_names(names),
            "revid": page_info.revid,
            "section": section.heading,
        }
        rows.append(
            _Row(
                id=f"{page_info.pageid}::{section.anchor}::title",
                embedding_input=section.heading,
                document=section.lead or section.text,
                metadata={**metadata, "chunk_index": -1},
            )
        )
        # The heading is one of the names people use, but rarely the only one:
        # the wiki redirects "ASPD" and "Velocidade de ataque" at *Velocidade
        # de Ataque* too. Same treatment the other-language names get.
        aliases = [name for name in names if name != section.heading]
        if aliases:
            rows.append(
                _Row(
                    id=f"{page_info.pageid}::{section.anchor}::names",
                    embedding_input=format_names(aliases),
                    document=section.lead or section.text,
                    metadata={**metadata, "chunk_index": -2},
                )
            )
        rows += [
            _Row(
                id=f"{page_info.pageid}::{section.anchor}::{chunk.index}",
                embedding_input=_embedding_text(section.heading, chunk.text),
                document=chunk.text,
                metadata={**metadata, "chunk_index": chunk.index},
            )
            for chunk in chunk_text(section.text, chunk_size, overlap)
        ]

    return rows


def build_index(
    api_url: str,
    wiki_base_url: str,
    limit: int | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    force: bool = False,
) -> None:
    chunk_size = chunk_size or CONFIG.chunk_size_tokens
    overlap = overlap or CONFIG.chunk_overlap_tokens

    logger.info("Connecting to MediaWiki API at %s", api_url)
    client = MediaWikiClient(api_url)

    logger.info("Loading embedding model %s (first run downloads it once)…", CONFIG.embedding_model)
    model = SentenceTransformer(CONFIG.embedding_model)

    os.makedirs(CONFIG.chroma_db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=CONFIG.chroma_db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = chroma_client.get_or_create_collection(
        name=CONFIG.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    state = _load_state(CONFIG.ingest_state_path)

    all_pages = client.list_all_pages(limit=limit)
    if limit:
        all_pages = all_pages[:limit]
        logger.info("Dev mode: limiting crawl to first %d pages", limit)

    seen_titles = set()
    pages_skipped = 0
    pages_updated = 0
    pages_new = 0
    total_chunks_written = 0
    sections_written = 0

    pages_removed = 0

    # Whatever happens in here — a wiki that goes down for good, a Ctrl-C —
    # the pages already indexed are recorded before the exception leaves.
    try:
        for page_info in all_pages:
            seen_titles.add(page_info.title)
            prev = state.get(page_info.title)

            if _is_current(prev, page_info) and not force:
                pages_skipped += 1
                continue

            content = client.fetch_page_content(page_info.title)
            time.sleep(client.request_delay_seconds)
            if content is None:
                logger.warning("Skipping %r: no content returned", page_info.title)
                continue

            # Remove any previously-indexed chunks for this page before adding
            # the new ones (covers both "updated" and "first time" cases safely).
            if prev and prev.get("chunk_ids"):
                collection.delete(ids=prev["chunk_ids"])

            page_text = append_categories(content.text, content.categories)

            chunks = chunk_text(page_text, chunk_size, overlap)
            if not chunks:
                logger.info("Page %r produced no chunks (empty content), skipping", page_info.title)
                state.pop(page_info.title, None)
                continue

            url = _page_url(wiki_base_url, page_info.title)
            rows = _page_rows(page_info, content, url, chunks)
            section_rows = _section_rows(page_info, content, url, chunk_size, overlap)
            rows += section_rows
            sections_written += len({row.metadata["section"] for row in section_rows})

            embeddings = model.encode(
                [row.embedding_input for row in rows], show_progress_bar=False
            ).tolist()
            chunk_ids = [row.id for row in rows]

            collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=[row.document for row in rows],
                metadatas=[row.metadata for row in rows],
            )

            state[page_info.title] = {
                "pageid": page_info.pageid,
                "revid": page_info.revid,
                "format": INDEX_FORMAT,
                "section_names": _state_section_names(page_info),
                "chunk_ids": chunk_ids,
            }
            total_chunks_written += len(chunk_ids)

            if prev:
                pages_updated += 1
                logger.info("Updated %r (%d chunks)", page_info.title, len(chunk_ids))
            else:
                pages_new += 1
                logger.info("Indexed %r (%d chunks)", page_info.title, len(chunk_ids))

            if (pages_new + pages_updated) % STATE_SAVE_EVERY == 0:
                _save_state(CONFIG.ingest_state_path, state)

        # Clean up pages that no longer exist on the wiki (only when doing a
        # full, non-limited crawl — a --limit dev run shouldn't be treated as
        # the full page set).
        if not limit:
            removed_titles = set(state.keys()) - seen_titles
            for title in removed_titles:
                chunk_ids = state[title].get("chunk_ids") or []
                if chunk_ids:
                    collection.delete(ids=chunk_ids)
                del state[title]
                pages_removed += 1
                logger.info("Removed %r (no longer on wiki)", title)
    finally:
        _save_state(CONFIG.ingest_state_path, state)
        client.close()

    logger.info(
        "Done. new=%d updated=%d unchanged=%d removed=%d chunks_written=%d "
        "named_sections=%d collection_size=%d",
        pages_new, pages_updated, pages_skipped, pages_removed, total_chunks_written,
        sections_written, collection.count(),
    )


def main():
    parser = argparse.ArgumentParser(description="Build/update the wiki Chroma index.")
    parser.add_argument("--api-url", default=CONFIG.mediawiki_api_url, help="MediaWiki api.php URL")
    parser.add_argument("--wiki-base-url", default=CONFIG.wiki_article_base_url, help="Base URL for article links")
    parser.add_argument("--limit", type=int, default=None, help="Only index the first N pages (for testing)")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and re-embed every page, even ones already indexed by the "
             "current INDEX_FORMAT at their current revision id. Normally "
             "unnecessary: bumping INDEX_FORMAT after changing how content is "
             "extracted, chunked or embedded re-indexes everything by itself, and "
             "an interrupted run resumes on its own.",
    )
    args = parser.parse_args()

    build_index(
        api_url=args.api_url,
        wiki_base_url=args.wiki_base_url,
        limit=args.limit,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        force=args.force,
    )


if __name__ == "__main__":
    main()
