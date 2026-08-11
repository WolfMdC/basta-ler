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
each page's title + revision id + index format + the chunk ids it produced.
On re-run, pages whose revid hasn't changed *and* whose vectors were built by
the current INDEX_FORMAT are skipped entirely (no re-embedding); pages that
changed have their old chunks deleted from Chroma and replaced; pages that
disappeared from the wiki have their chunks removed too.

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
from pathlib import Path
from urllib.parse import quote

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CONFIG
from ingest.chunker import chunk_text
from ingest.html_text import append_categories, format_names
from ingest.mediawiki_client import MediaWikiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")

# What a page's vectors were built from. Bump this whenever the ingest
# changes what it would produce for an *unchanged* page — extraction,
# chunking, metadata, the embedding model — and the next ordinary run
# re-indexes everything, no `--force` to remember. Version 2 added the
# other-language names (see ingest/html_text.py).
INDEX_FORMAT = 2

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


def _is_current(prev: dict | None, revid: int) -> bool:
    """Whether a page's stored vectors are still the ones we'd produce now.

    Both halves matter: the wiki hasn't touched the page *and* the vectors
    came from this version of the ingest. Without the second half, a change
    to how pages are read leaves every unchanged page looking up to date and
    silently keeps its stale vectors.
    """
    if not prev:
        return False
    return prev.get("revid") == revid and prev.get("format") == INDEX_FORMAT


def _page_url(base_url: str, title: str) -> str:
    # MediaWiki article paths use underscores for spaces. Percent-encode the
    # rest: Portuguese titles are full of accented characters, and Discord
    # refuses to linkify a URL (or a [text](url) link) containing raw
    # non-ASCII bytes — "https://browiki.org/wiki/Lâminas_Aceleradas" would
    # render as unclickable plain text.
    slug = re.sub(r"\s+", "_", title.strip())
    return base_url.rstrip("/") + "/" + quote(slug, safe="_/:()")


def _embedding_text(title: str, chunk_body: str) -> str:
    """Text actually handed to the embedding model.

    The page title is prepended to every chunk because the article body
    usually never repeats it — "Rajada Frenética" opens with "Dispara flechas
    em um único alvo…", so a chunk embedded on its own carries no signal for
    someone asking about the skill *by name*, which is how most questions in
    Discord are phrased.
    """
    return f"{title}\n{chunk_body}"


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

    pages_removed = 0

    # Whatever happens in here — a wiki that goes down for good, a Ctrl-C —
    # the pages already indexed are recorded before the exception leaves.
    try:
        for page_info in all_pages:
            seen_titles.add(page_info.title)
            prev = state.get(page_info.title)

            if _is_current(prev, page_info.revid) and not force:
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

            page_summary = content.lead
            url = _page_url(wiki_base_url, page_info.title)

            # A title-only "card" vector alongside the body chunks. This
            # embedding model dilutes heavily with length: on a stat-heavy page
            # the numbers swamp the name — "Rajada Frenética" scores 0.58 against
            # its own title but only 0.40 against its full chunk, so questions
            # naming a page don't retrieve it. The card keeps one clean by-name
            # vector per page; the body chunks still answer descriptive questions
            # ("qual habilidade acerta 3 vezes com arco").
            chunk_ids = [f"{page_info.pageid}::title"]
            embedding_inputs = [page_info.title]
            documents = [page_summary or chunks[0].text]
            chunk_indexes = [-1]

            # Same idea for the names the game uses in other languages ("Wild
            # Fire / Fuego Salvaje"): its own card rather than a longer title
            # card, so asking by the English name is as clean a vector as asking
            # by the Portuguese one, and neither dilutes the other.
            if content.names:
                chunk_ids.append(f"{page_info.pageid}::names")
                embedding_inputs.append(format_names(content.names))
                documents.append(page_summary or chunks[0].text)
                chunk_indexes.append(-2)

            for chunk in chunks:
                chunk_ids.append(f"{page_info.pageid}::{chunk.index}")
                embedding_inputs.append(_embedding_text(page_info.title, chunk.text))
                documents.append(chunk.text)
                chunk_indexes.append(chunk.index)

            embeddings = model.encode(embedding_inputs, show_progress_bar=False).tolist()
            metadatas = [
                {
                    "title": page_info.title,
                    "url": url,
                    "summary": page_summary,
                    "names": format_names(content.names),
                    "revid": page_info.revid,
                    "chunk_index": chunk_index,
                }
                for chunk_index in chunk_indexes
            ]

            collection.upsert(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            state[page_info.title] = {
                "pageid": page_info.pageid,
                "revid": page_info.revid,
                "format": INDEX_FORMAT,
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
        "Done. new=%d updated=%d unchanged=%d removed=%d chunks_written=%d collection_size=%d",
        pages_new, pages_updated, pages_skipped, pages_removed, total_chunks_written, collection.count(),
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
