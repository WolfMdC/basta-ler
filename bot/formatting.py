"""
Formats retrieval Answers into a Discord reply.

Deliberately minimal: the page title, its URL, and a note that the answer
comes from the bROWiki. The bot's job is to point at the right page, not to
paste half of it into the channel.

The one exception is a question that has a single value for an answer
("Quanto de pós-conjuração tem Impacto Sísmico?"). When bot/facts.py can
pull that value out of the page's infobox it leads the reply, with the page
link kept directly under it as the source — so the value is readable at a
glance and still checkable.

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
    description = url
    if answer.fact is not None:
        description = f"**{answer.fact.label}:** {answer.fact.value}\n\n{url}"
    embed = discord.Embed(title=answer.title, url=url, description=description, color=EMBED_COLOR)
    embed.set_footer(text=SOURCE_NOTE)
    return embed


def build_plain_text(answers: list[Answer]) -> str:
    answer = answers[0]
    # Angle brackets keep the link clickable while suppressing Discord's
    # auto-generated link preview, so the reply stays two short lines.
    link = f"<{safe_url(answer.url)}>"
    if answer.fact is not None:
        return (
            f"**{answer.fact.label}:** {answer.fact.value}\n"
            f"{answer.title} — {link}\n-# {SOURCE_NOTE}"
        )
    return f"**{answer.title}**\n{link}\n-# {SOURCE_NOTE}"
