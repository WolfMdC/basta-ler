# browiki-bot

A Discord bot for a Ragnarok Online (bROWiki) fan server. It watches one or
more channels, figures out whether a message is a *question*, and — if it's
confident it has a good match — replies with the relevant bROWiki page
URL(s) and a short description. Casual chat is ignored, and low-confidence
matches are logged instead of posted.

Everything runs locally and for free:

- **discord.py** for the bot itself
- **sentence-transformers** (`paraphrase-multilingual-MiniLM-L12-v2`, CPU) for
  embeddings — multilingual, so it handles Portuguese content and messages
- **Chroma** (file-based, local) as the vector store — no hosted DB
- A **keyword/heuristic classifier** decides "is this a question?" in
  Portuguese — no LLM call, no API key required for the default path
- LLM-based intent classification / answer writing are wired up as
  **optional, pluggable** interfaces you can implement later (see
  [Optional: LLM-enhanced mode](#optional-llm-enhanced-mode)) — not required
  to run the bot

## How it works

1. **Ingest** (`ingest/build_index.py`): crawls a MediaWiki site via its
   standard Action API (`api.php`), pulls each page as the wiki's own
   *rendered HTML*, flattens it to plain text (`ingest/html_text.py`),
   chunks it, embeds the chunks, and stores them in a local Chroma
   collection with `title`, `url`, and `summary` metadata. Re-running it is
   incremental — pages whose revision id hasn't changed are skipped
   (`--force` re-does everything).
2. **Bot** (`bot/main.py`): listens for messages in the allowed channels. A
   Portuguese heuristic classifier (`bot/intent.py`) decides whether a
   message is a question. If so, the message is embedded and matched
   against the Chroma index, then re-ranked by page title
   (`bot/retriever.py`). If the best match clears the confidence threshold,
   the bot replies with the page title, its URL, and a note that the answer
   came from the bROWiki (`bot/formatting.py`, `bot/answer.py`). Otherwise
   it stays silent and just logs what it saw.

### Why rendered HTML, not wikitext

On a game wiki nearly every answerable fact — cast delay, cooldown, SP cost,
range, per-level damage — lives inside templates (`{{Skill Info}}`) and
tables. In raw wikitext those are opaque markup whose *labels don't exist*:
the wikitext says `| delay = 0,5 segundos`, while the page a player reads
says **"Pós-conjuração: 0,5 segundos"**. Indexing wikitext (and stripping
templates/tables out of it, as any generic stripper must) throws away exactly
the data people ask about, leaving only the prose intro. Asking the wiki to
render the page first gives the human-facing labels — the same words players
type in Discord.

### How matching works

Vector similarity alone is unreliable for page *names*: to a multilingual
embedding model, "Rajada Frenética" and "Rajada Certeira" look nearly
identical. So retrieval combines two signals:

- **Semantic**: each chunk is embedded with its page title prepended, since
  article bodies rarely repeat their own title. Each page also gets one
  extra *title-only* vector, because this embedding model dilutes heavily
  with length — a stat-heavy page's numbers swamp its name, so "Rajada
  Frenética" scores 0.58 against its own title but only 0.40 against its
  full text.
- **Lexical**: candidates whose title words appear in the message get a
  ranking bonus, and any page whose full title appears in the message is
  pulled straight to the top — if someone names a page, that page is the
  answer. Name matching ignores accents and spacing, so "rock ridge",
  "Rock Ridge" and "rockridge" are all one name.

Name matching is deliberately exact under those foldings rather than fuzzy.
Two looser rules were measured against a 3,800-word vocabulary drawn from the
wiki itself and rejected:

- *Edit distance* fired on 35–76 everyday Portuguese words — "preciso" ("I
  need") is 0.93 similar to the stat page "Precisão" — each of which becomes
  a confidently wrong answer.
- *Plural stripping* matched 61 vocabulary words, almost all hub pages
  ("habilidade" → *Habilidades*, "monstro" → *Monstros*). Those words are
  everywhere in a Ragnarok channel, so the bot answered a generic index page
  to half the conversation.

The cost is that genuine typos ("rockrige") fall through to the semantic
path and usually get silence.

Matches are then held to one of two bars. A page the message actually names
only needs `SIMILARITY_THRESHOLD`; a page matched on embedding similarity
alone needs the stricter `SEMANTIC_ONLY_THRESHOLD`, because ordinary chat
scores surprisingly high against unrelated pages — "bom dia pessoal" hits the
item page *Buongiorno* at 0.72, and "que sorte a sua" hits *Beijo da Sorte*.
The cost of that strictness is that a purely descriptive question which never
names its page ("qual habilidade acerta 3 vezes com arco?") usually gets
silence; lower `SEMANTIC_ONLY_THRESHOLD` if you'd rather trade precision for
recall.

## Project layout

```
browiki-bot/
├── bot/
│   ├── main.py          # Discord client, message handling, wiring
│   ├── intent.py         # IntentClassifier interface + simple PT heuristic + LLM stub
│   ├── answer.py          # AnswerWriter interface + simple summary writer + LLM stub
│   ├── retriever.py      # Embeds queries, searches Chroma, re-ranks by title
│   └── formatting.py     # Builds the Discord embed / plain-text reply
├── ingest/
│   ├── mediawiki_client.py  # Generic MediaWiki Action API client
│   ├── html_text.py          # Rendered wiki HTML -> plain text (keeps infoboxes/tables)
│   ├── chunker.py            # Word-count-based chunking
│   └── build_index.py        # CLI: crawl + embed + (incrementally) store in Chroma
├── data/
│   ├── chroma/            # Chroma's persisted vector DB (gitignored)
│   └── ingest_state.json  # Tracks each page's last-indexed revision id (gitignored)
├── config.py              # Central config, reads from .env
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> **Windows note:** Chroma's local vector index depends on `chroma-hnswlib`,
> which has no prebuilt wheel for Windows — pip will compile it from source.
> That needs the "Desktop development with C++" workload from [Visual
> Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> (specifically the MSVC compiler + a Windows SDK). If `pip install` fails
> with a `Microsoft Visual C++ 14.0 or greater is required` error, install
> that workload and try again. macOS/Linux users typically get a prebuilt
> wheel and can skip this.

### 2. Configure

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Purpose |
|---|---|
| `DISCORD_BOT_TOKEN` | Your bot's token (see below) |
| `CHANNEL_IDS` | Comma-separated channel IDs to listen in |
| `MEDIAWIKI_API_URL` | `api.php` endpoint of the wiki to index |
| `WIKI_ARTICLE_BASE_URL` | Base URL for building article links |
| `EMBEDDING_MODEL` | sentence-transformers model name |
| `CHROMA_DB_PATH`, `CHROMA_COLLECTION_NAME` | Where/how the local vector DB persists |
| `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS` | Chunking (approx. tokens ≈ words) |
| `TOP_K_RESULTS` | How many candidate pages to rank internally (the bot replies with the best one) |
| `SIMILARITY_THRESHOLD` | Minimum score (0–1) to reply when the message names the page |
| `SEMANTIC_ONLY_THRESHOLD` | Stricter minimum when the match rests on embedding similarity alone |
| `QUESTION_MIN_CHARS` | Min length for a keyword-only (no `?`) match to count |
| `REPLY_COOLDOWN_SECONDS` | Debounce: min seconds between replies per channel |
| `USE_EMBED_REPLIES` | `true` for a Discord embed reply, `false` for plain text |
| `INTENT_CLASSIFIER`, `ANSWER_WRITER` | `simple` (default, free) or `llm` (stub, see below) |

### 3. Get a Discord bot token & invite it to a server

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications) → **New Application**, give it a name.
2. In the app, open **Bot** (left sidebar) → **Add Bot** (or **Reset Token**) → copy the token into `DISCORD_BOT_TOKEN` in `.env`. Keep this secret.
3. Still under **Bot**, enable the **Message Content Intent** toggle (required — the bot needs to read message text to classify it).
4. Open **OAuth2 → URL Generator**:
   - Scopes: `bot`
   - Bot permissions: `View Channels`, `Send Messages`, `Read Message History`, `Embed Links`
5. Open the generated URL, pick your server, authorize.
6. In Discord, enable **Developer Mode** (User Settings → Advanced), then right-click each channel you want the bot in → **Copy Channel ID** → add to `CHANNEL_IDS` in `.env` (comma-separated). Leaving `CHANNEL_IDS` empty makes the bot listen everywhere it's added, which isn't recommended.

### 4. Build the index

Point `.env` at the wiki you want indexed (these are the defaults):

```
MEDIAWIKI_API_URL=https://browiki.org/api.php
WIKI_ARTICLE_BASE_URL=https://browiki.org/wiki/
```

Then run:

```bash
python -m ingest.build_index
```

This crawls every content page (redirects are skipped, so an aliased title
isn't indexed as a duplicate of its target), embeds it, and writes to
`data/chroma/`. Re-run it any time to pick up new/changed pages — unchanged
pages are skipped automatically (tracked via `data/ingest_state.json`), so
re-indexing is cheap.

A full crawl of bROWiki takes roughly 12 minutes: ~1,900 articles and ~4,200
vectors, all on CPU.

Add `--force` to rebuild everything regardless of revision ids. You need
this after changing how content is extracted, chunked or embedded —
otherwise every page looks "unchanged" and keeps its stale vectors.

**Running it against any other wiki:** the ingestion script targets the
generic MediaWiki Action API with no site-specific assumptions, so it works
against any MediaWiki install — Wikipedia, a self-hosted mirror, or another
game wiki. Nothing in `.env` needs to change for a one-off run:

```bash
python -m ingest.build_index \
  --api-url https://oldschool.runescape.wiki/api.php \
  --wiki-base-url https://oldschool.runescape.wiki/w/ \
  --limit 50
```

`--limit` caps how many pages get crawled — handy for a quick smoke test
against a huge wiki. `--api-url`/`--wiki-base-url` override `.env` for a
single run without touching your config. Omit both to use whatever's in
`.env`.

### 5. Run the bot

```bash
python -m bot.main
```

The bot logs every message it sees in a watched channel, whether it was
classified as a question, what it matched (title + similarity score), and
whether it replied or stayed silent — useful for tuning
`SIMILARITY_THRESHOLD` and `QUESTION_MIN_CHARS`.

## Behavior details

- **Question detection** (`bot/intent.py`): a message counts as a question
  if it ends in `?`, or if it contains a strong Portuguese interrogative
  (como, o que, quando, onde, qual, quanto, quem, por que, cadê, será que,
  …) **and** is at least `QUESTION_MIN_CHARS` characters long — this stops
  short chatty fragments like "oq" from triggering a reply.
- **Confidence threshold**: results below the applicable threshold are
  logged (`Low-confidence match, not replying...`, including whether the
  page name was matched) but never posted.
- **Debounce**: `REPLY_COOLDOWN_SECONDS` enforces a minimum gap between the
  bot's replies *within a single channel*, so a busy channel doesn't get
  spammed even if several questions land in a row.
- **Never replies to itself or other bots.**
- **One answer per reply**: `TOP_K_RESULTS` candidates are retrieved,
  deduplicated to one best chunk per page and re-ranked; only the top page
  is posted. The runners-up are logged so you can see what it considered.
- **Reply shape**: page title, page URL, and a one-line note that the
  information comes from the bROWiki — nothing else. URLs are
  percent-encoded, without which Discord won't linkify addresses containing
  accented characters (`.../wiki/Lâminas_Aceleradas`).

## Optional: LLM-enhanced mode

`bot/intent.py` and `bot/answer.py` each define an abstract interface
(`IntentClassifier`, `AnswerWriter`) with:

- a default `Simple*` implementation (heuristic / static-summary — no API key), and
- an `LLM*` stub that raises `NotImplementedError` until you implement it.

To wire up an LLM later (e.g. for subtler question detection, or a
question-tailored generated answer instead of the static page summary):

1. Implement `LLMIntentClassifier.is_question()` and/or `LLMAnswerWriter.write()` in the respective file, calling whatever provider you like.
2. Set `LLM_API_KEY` in `.env`.
3. Set `INTENT_CLASSIFIER=llm` and/or `ANSWER_WRITER=llm` in `.env`.

The rest of the bot is unaffected — nothing else references these classes
directly, only the interface.

## Tuning tips

- If the bot stays silent too often, lower `SIMILARITY_THRESHOLD` (e.g.
  0.45) or check the logs for the best-match score it's rejecting.
- If it answers with plausible-looking but *wrong* pages, raise
  `SEMANTIC_ONLY_THRESHOLD` first — unrelated pages routinely score ~0.6–0.7
  against ordinary chat, so that's where bad answers come from. Staying
  silent is the better failure mode here.
- After changing chunking, extraction or the embedding model, re-run the
  ingest with `--force`; without it every page looks unchanged and keeps its
  old vectors.
- If it replies to things that aren't really questions, raise
  `QUESTION_MIN_CHARS` or tighten the interrogative list in
  `bot/intent.py`.
- Smaller `CHUNK_SIZE_TOKENS` gives more precise matches on short factual
  pages; larger chunks preserve more context for narrative pages (quest
  walkthroughs, lore, etc.).
