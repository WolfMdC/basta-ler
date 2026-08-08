"""
Formats retrieval Answers into a Discord reply.

Deliberately minimal: the page title, its URL, and a note that the answer
comes from the bROWiki. The bot's job is to point at the right page, not to
paste half of it into the channel.

URLs are percent-encoded before being sent — Discord will not turn a link
containing raw non-ASCII characters (very common in Portuguese page titles,
e.g. ".../wiki/Lâminas_Aceleradas") into a clickable link.
"""
from __future__ import annotations

from urllib.parse import quote

import discord

from bot.answer import Answer

EMBED_COLOR = 0x5865F2  # Discord blurple
SOURCE_NOTE = "Informação retirada da bROWiki - a Wiki do Ragnarök LATAM"

# Characters that are already legal in a URL and must not be re-encoded
# ("%" included, so an already-encoded URL survives untouched).
_URL_SAFE = ":/?#[]@!$&'()*+,;=%~_.-"


def safe_url(url: str) -> str:
    return quote(url, safe=_URL_SAFE)


def build_embed(answers: list[Answer]) -> discord.Embed:
    answer = answers[0]
    url = safe_url(answer.url)
    # The embed title links to the page; the bare URL in the description is
    # auto-linked by Discord and keeps the address visible.
    embed = discord.Embed(title=answer.title, url=url, description=url, color=EMBED_COLOR)
    embed.set_footer(text=SOURCE_NOTE)
    return embed


def build_plain_text(answers: list[Answer]) -> str:
    answer = answers[0]
    # Angle brackets keep the link clickable while suppressing Discord's
    # auto-generated link preview, so the reply stays two short lines.
    return f"**{answer.title}**\n<{safe_url(answer.url)}>\n-# {SOURCE_NOTE}"
