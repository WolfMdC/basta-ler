"""
Discord bot entry point.

Listens on the configured channel(s), classifies each message as
question-or-not with a Portuguese heuristic classifier, and — for confident
matches — replies with the best-matching bROWiki page URL(s) plus a short
description.

Run with:
    python -m bot.main
"""
from __future__ import annotations

import logging
import time

import discord

from bot.answer import get_answer_writer
from bot.formatting import build_embed, build_plain_text
from bot.intent import get_intent_classifier
from bot.retriever import Retriever
from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bot")


class WikiAnswerBot(discord.Client):
    def __init__(self, *args, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, *args, **kwargs)

        self.channel_ids = set(CONFIG.channel_ids)
        self.classifier = get_intent_classifier(
            CONFIG.intent_classifier, CONFIG.question_min_chars, CONFIG.llm_api_key
        )
        self.retriever = Retriever(
            chroma_db_path=CONFIG.chroma_db_path,
            collection_name=CONFIG.chroma_collection_name,
            embedding_model=CONFIG.embedding_model,
        )
        # The answer writer reads a page's indexed text to quote an infobox
        # value back; without the lookup it falls back to link-only replies.
        self.answer_writer = get_answer_writer(
            CONFIG.answer_writer,
            CONFIG.llm_api_key,
            page_text_lookup=self.retriever.page_head if CONFIG.direct_answers else None,
        )
        self._last_reply_at: dict[int, float] = {}  # channel_id -> monotonic timestamp

    async def on_ready(self):
        logger.info("Logged in as %s (id=%s)", self.user, self.user.id)
        if self.channel_ids:
            logger.info("Listening in channel(s): %s", sorted(self.channel_ids))
        else:
            logger.warning("CHANNEL_IDS is empty — the bot won't respond anywhere. Set it in .env.")

    async def on_message(self, message: discord.Message):
        # Never respond to ourselves or other bots.
        if message.author.bot or message.author.id == (self.user.id if self.user else None):
            return

        if self.channel_ids and message.channel.id not in self.channel_ids:
            return

        content = message.content.strip()
        if not content:
            return

        is_question = self.classifier.is_question(content)
        logger.info(
            "message channel=%s author=%s question=%s content=%r",
            message.channel.id, message.author, is_question, content[:200],
        )
        if not is_question:
            return

        if self._in_cooldown(message.channel.id):
            logger.info("Skipping reply: channel %s is in cooldown", message.channel.id)
            return

        results = self.retriever.best_unique_pages(content, top_k=CONFIG.top_k_results)
        if not results:
            logger.info("No results found for query=%r", content)
            return

        best = results[0]
        confident_results = [r for r in results if self._is_confident(r)]

        if not confident_results:
            logger.info(
                "Low-confidence match, not replying. best_title=%r best_score=%.3f "
                "best_similarity=%.3f title_matched=%s thresholds=%.2f/%.2f query=%r",
                best.title, best.score, best.similarity, best.title_matched,
                CONFIG.similarity_threshold, CONFIG.semantic_only_threshold, content,
            )
            return

        answers = self.answer_writer.write(content, confident_results)
        fact = answers[0].fact
        logger.info(
            "Replying with %r (score=%.3f) fact=%s; other candidates: %s",
            answers[0].title, confident_results[0].score,
            f"{fact.label}={fact.value!r}" if fact else None,
            [(r.title, round(r.score, 3)) for r in confident_results[1:]],
        )

        if CONFIG.use_embed_replies:
            await message.reply(embed=build_embed(answers), mention_author=False)
        else:
            await message.reply(build_plain_text(answers), mention_author=False)

        self._last_reply_at[message.channel.id] = time.monotonic()

    def _is_confident(self, result) -> bool:
        """Whether a match is good enough to post.

        A page the message actually names only has to clear the normal
        threshold. A page matched on embedding similarity alone has to clear
        a higher one: unrelated pages routinely score ~0.6-0.7 against
        ordinary chat (a "bom dia" greeting matches the item "Buongiorno"),
        and answering those is worse than staying quiet.
        """
        if result.title_matched:
            return result.score >= CONFIG.similarity_threshold
        return result.score >= CONFIG.semantic_only_threshold

    def _in_cooldown(self, channel_id: int) -> bool:
        last = self._last_reply_at.get(channel_id)
        if last is None:
            return False
        return (time.monotonic() - last) < CONFIG.reply_cooldown_seconds


def main():
    if not CONFIG.discord_bot_token:
        raise SystemExit("DISCORD_BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")

    client = WikiAnswerBot()
    client.run(CONFIG.discord_bot_token)


if __name__ == "__main__":
    main()
