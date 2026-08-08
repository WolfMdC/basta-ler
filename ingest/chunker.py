"""
Simple word-count-based chunker.

We approximate "tokens" as whitespace-split words rather than pulling in a
real tokenizer (e.g. tiktoken) — it keeps the dependency list small and the
approximation is fine for chunk-sizing purposes. Chunk size/overlap are
configurable via CHUNK_SIZE_TOKENS / CHUNK_OVERLAP_TOKENS.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    step = chunk_size_tokens - overlap_tokens
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = start + chunk_size_tokens
        chunk_words = words[start:end]
        chunks.append(Chunk(index=index, text=" ".join(chunk_words)))
        index += 1
        if end >= len(words):
            break
        start += step

    return chunks
