"""
Central config loader, shared by the ingest scripts and the bot.

Everything is read from environment variables (populated via a `.env` file
in the project root, loaded with python-dotenv). See `.env.example` for the
full list of knobs and their defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_int_list(name: str) -> list[int]:
    raw = os.environ.get(name, "")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class Config:
    # Discord
    discord_bot_token: str = field(default_factory=lambda: _env_str("DISCORD_BOT_TOKEN", ""))
    channel_ids: list[int] = field(default_factory=lambda: _env_int_list("CHANNEL_IDS"))

    # MediaWiki source
    mediawiki_api_url: str = field(default_factory=lambda: _env_str("MEDIAWIKI_API_URL", "https://browiki.org/api.php"))
    wiki_article_base_url: str = field(default_factory=lambda: _env_str("WIKI_ARTICLE_BASE_URL", "https://browiki.org/wiki/"))

    # Embeddings / vector store
    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"))
    chroma_db_path: str = field(default_factory=lambda: _env_str("CHROMA_DB_PATH", "./data/chroma"))
    chroma_collection_name: str = field(default_factory=lambda: _env_str("CHROMA_COLLECTION_NAME", "browiki"))

    # Chunking
    chunk_size_tokens: int = field(default_factory=lambda: _env_int("CHUNK_SIZE_TOKENS", 500))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 50))

    # Retrieval
    top_k_results: int = field(default_factory=lambda: _env_int("TOP_K_RESULTS", 3))
    # Minimum hybrid score (vector similarity + title-match bonus, 0-1) for
    # the bot to answer at all. Raising it makes the bot quieter but more
    # trustworthy; a page named outright in the message scores ~0.9.
    similarity_threshold: float = field(default_factory=lambda: _env_float("SIMILARITY_THRESHOLD", 0.55))
    # Higher bar for a match the message doesn't name at all — one resting
    # purely on embedding similarity. Without this a "bom dia pessoal" greeting
    # matches the item page "Buongiorno" at 0.72 and gets answered.
    semantic_only_threshold: float = field(default_factory=lambda: _env_float("SEMANTIC_ONLY_THRESHOLD", 0.78))

    # Reply formatting: use a Discord embed (nicer) or plain text.
    use_embed_replies: bool = field(default_factory=lambda: _env_str("USE_EMBED_REPLIES", "true").lower() in ("1", "true", "yes"))
    # Quote the wiki's own value when a question asks for a single infobox
    # field ("quanto de pós-conjuração...", "qual o cooldown de..."). Turn
    # off to always reply with just the page link.
    direct_answers: bool = field(default_factory=lambda: _env_str("DIRECT_ANSWERS", "true").lower() in ("1", "true", "yes"))

    # Rate limiting / debounce: minimum seconds between the bot's own
    # replies in a single channel, so a chatty channel doesn't get spammed.
    reply_cooldown_seconds: float = field(default_factory=lambda: _env_float("REPLY_COOLDOWN_SECONDS", 15))

    # Intent classification
    question_min_chars: int = field(default_factory=lambda: _env_int("QUESTION_MIN_CHARS", 12))
    intent_classifier: str = field(default_factory=lambda: _env_str("INTENT_CLASSIFIER", "simple"))
    answer_writer: str = field(default_factory=lambda: _env_str("ANSWER_WRITER", "simple"))
    llm_api_key: str = field(default_factory=lambda: _env_str("LLM_API_KEY", ""))

    # Ingest state (tracks revision ids for incremental re-indexing)
    ingest_state_path: str = field(default_factory=lambda: _env_str("INGEST_STATE_PATH", "./data/ingest_state.json"))


CONFIG = Config()
