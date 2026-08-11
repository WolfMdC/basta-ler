# basta-ler-bot

A Discord bot for a Ragnarök Online LATAM Discord server. It watches one or
more channels, figures out whether a message is a *question*, and — if it's
confident it has a good match — replies with the relevant bROWiki page
URL(s) and a short description. When the question asks for a single stat
("Quanto de pós-conjuração tem Abalo Sísmico?"), it answers with the value
itself and links the page as the source. Questions come out of a LATAM
channel the way people actually type them — English game terms and English
or Spanish skill names dropped into Portuguese sentences ("qual o cast fixo
de Wild Fire?") — so both the vocabulary and the page names are matched in
all three languages. Casual chat is ignored, and low-confidence matches are
logged instead of posted.

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
   collection with `title`, `url`, `summary`, `names` and `section` metadata
   — `names` being what the game calls the page in English and Spanish,
   lifted off the row under each skill's title, and `section` marking the
   rows that belong to one named section of a page rather than the page
   itself. Re-running it is incremental — pages whose revision id and
   redirects haven't changed are skipped (`--force` re-does everything).
2. **Bot** (`bot/main.py`): listens for messages in the allowed channels. A
   Portuguese heuristic classifier (`bot/intent.py`) decides whether a
   message is a question. If so, the message is embedded and matched
   against the Chroma index, then re-ranked by page title
   (`bot/retriever.py`). If the best match clears the confidence threshold,
   the bot replies with the page title, its URL, and a note that the answer
   came from the bROWiki (`bot/formatting.py`, `bot/answer.py`). Otherwise
   it stays silent and just logs what it saw.
3. **Direct answers** (`bot/facts.py`): if the question asks for one infobox
   field the matched page actually lists, that value leads the reply, with
   the page link underneath it as the source.

### Direct answers

"Quanto de pós-conjuração tem Abalo Sísmico?" has a one-line answer, and a
link to the page is a worse reply than the number:

```
Pós-conjuração: 0,5 segundos
Abalo Sísmico — https://browiki.org/wiki/Abalo_Sísmico
```

Every skill and quest page opens with an infobox of label/value rows, and
those rows are already indexed. `bot/facts.py` matches the question against
the fields people ask for — pós-conjuração, conjuração, recarga/cooldown,
SP, custo de AP, duração, alcance, área, níveis, alvo, tipo, propriedade,
munição, ID — and quotes the page's own value for it, verbatim.

Each field answers to its English name too, because that is half of how the
question gets typed: `cast`/`cast fixo`/`cast time` and `delay`/`aftercast`
reach *Conjuração* and *Pós-conjuração*, `cd`/`cooldown` reaches *Recarga*,
and `range`, `aoe`, `duration`, `target`, `element`, `ammo`, `max level`
reach the rest. One thing it doesn't do is split a row: the wiki writes
conjuração as `1 + 0 seg.`, so "qual o cast fixo" is answered with that
whole cell rather than one half of it — the bot quotes the wiki, it doesn't
do arithmetic on it.

Recovering those rows takes some care, because `ingest/chunker.py` splits on
whitespace and leaves the infobox as one flat run of words with no delimiter
between a label and the next:

```
Tipo Ofensiva Níveis 5 SP 30 + (Nv. da habilidade × 5) Conjuração 0 + 1 seg.
Recarga 2 segundos Alvo Usuário Área 9x9 células …
```

So the labels themselves are the delimiters: a field's value is whatever
sits between its label and the next known label. That works because the
infobox is a contiguous run at the very top of the page — which is also
where it has to *stop*, since past the last row the only "labels" left are
words in the description. The parse therefore ends at the first value that
reads like prose (one running into a sentence break, or cut off on a
dangling "no"/"da"), and values are held to a per-field length and shape
that was measured across every page in the index. Anything that fails just
falls back to the normal link-only reply.

Measured against ground truth — the same pages re-extracted from the wiki
with labels and values still on separate lines — the parser reproduced
**400 of 400** quotable values exactly across a 90-page random sample.

Two things it deliberately doesn't do: guess at a field a page doesn't list
(*Impacto Flamejante* has no pós-conjuração row, so that question gets the
plain link), and read the per-level tables further down the page (asking for
SP gives the formula the infobox shows, not the value at level 7). Set
`DIRECT_ANSWERS=false` to turn the whole path off.

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
  answer, in whichever of the game's languages they named it. Name matching
  ignores accents and spacing, so "rock ridge", "Rock Ridge" and "rockridge"
  are all one name.

Name matching is deliberately exact under those foldings rather than fuzzy.
Two looser rules were measured against a vocabulary drawn from the wiki
itself and rejected:

- *Edit distance* fired on 35–76 everyday Portuguese words — "preciso" ("I
  need") is 0.93 similar to the stat page "Precisão" — each of which becomes
  a confidently wrong answer.
- *Plural stripping, applied to every page*, lets 120 everyday words reach a
  title, and the worst of them are the hub pages: "niveis" → *Nível* (a word
  used on 1,339 pages), "habilidade" → *Habilidades* (955), "quest" →
  *Quests* (426), "item" → *Itens*, "carta" → *Cartas*. Those words are
  everywhere in a Ragnarok channel, so the bot answered a generic index page
  to half the conversation.

The cost is that genuine typos ("rockrige") fall through to the semantic
path and usually get silence.

#### The English and Spanish names

Nobody in a LATAM channel sticks to one language: "qual o cast fixo de Wild
Fire?" is an ordinary sentence, and *Fogo de Supressão* is the page that
answers it. Every skill page carries the game's other names for the skill in
the row under its title —

```
Fogo de Supressão
Wild Fire / Fuego Salvaje
```

— so the ingest lifts that row out (`ingest/html_text.py` recognises it by
its markup, a small+bold cell in the leading rows of the infobox), stores it
as `names` metadata, and gives it its own vector card next to the title
card. The bot then treats those names as page names in their own right: same
exact/folded matching, same score, so naming a page in English pulls it
straight to the top exactly as naming it in Portuguese does. 1,007 of the
wiki's ~1,900 pages have such a row.

**One-word names are matched by meaning only, never literally.** The wiki
has ~130 of them per language and, measured against the words that appear
lowercase in its own prose, the handful that are also ordinary words are the
ones that would hurt most: *Faxina* is called "Remover", *Fogo Grego* is
"Bomba", and — worst of all — *Resfriamento* is called "Cooldown", the exact
word this bot needs to read as a question about some *other* skill's
cooldown. So "onde compro bomba?" stays silent instead of confidently
answering *Fogo Grego*. Two-word names carry that risk far more rarely (26
of 1,692, nearly all Spanish phrases that also read as Portuguese, e.g.
"Escudo Sagrado" → *Escudo Divino*), and are matched. `MIN_ALIAS_WORDS` in
`bot/retriever.py` is the dial.

Reading the names also fixed a parse the bot was getting wrong: on *Aumentar
Recuperação de SP* the "SP" in "Increase **SP** Recovery" was being read as
the infobox's SP row, which cost that page (and *Identificar Item*, and
*Recuperar HP em Movimento*) every direct answer it had. The name row is now
skipped before parsing, exactly as the title always was.

#### Singular names for the character classes

The wiki titles every class in the plural — *Mandraques*, *Divas*,
*Cavaleiros Rúnicos* — but nobody asks that way: "Como eu viro Mandraque?"
is the normal phrasing, and it used to score 0.695 and get silence.

So plural stripping is kept, restricted to the one group of pages it helps.
Which pages those are is read from the wiki's own **"Classes" category**
rather than hardcoded, so a class added to the wiki starts working after the
next ingest. Re-measured with that restriction, the 120 everyday words that
reach a title drop to **59, every one of them a class name** — the hub pages
above are gone, and "quantos níveis tem essa quest?" stays silent as before.

Matching runs singular-insensitively on both sides, so it only has to be
self-consistent, not linguistically correct: *Magus* reduces to the non-word
"magu", and so does a message saying "Magus".

#### Pages whose real subject is a heading

*Sangramento* is not a page on this wiki. It is one of the 29 sections of
*Efeitos negativos*, so "alguém sabe o que faz o sangramento?" got silence,
and the best the bot could have done was link the page as a whole — which
opens on *Alucinação* and says nothing about bleeding until you scroll. The
answer people want is the section, and the anchor that opens the page on it:

```
Efeitos negativos § Sangramento
https://browiki.org/wiki/Efeitos_negativos#Sangramento
```

So the ingest cuts every page along its headings and indexes the named ones
as subjects in their own right — own summary, own vectors, own URL with the
`#anchor` on it — while the page keeps its own chunks exactly as before.

**Which sections count is read off the wiki, not guessed at.** A heading on
its own is no evidence of anything: "Notas" is a heading on 1,161 of this
wiki's pages and names nothing. A *redirect* pointing at one is — the editor
who made `Sangramento` an alias for `Efeitos negativos#Sangramento` is
saying, in the wiki's own data, that readers look that section up by name,
and what they call it. This wiki has 276 such redirects, covering 138
sections of 33 pages, and they carry the abbreviations and alternate
spellings too (`ASPD`, `Velocidade de ataque` → *Atributos § Velocidade de
Ataque*). Redirects are re-read on every crawl, and pointing a new one at a
page re-indexes it — the page's own revision id says nothing about pages
pointing *at* it.

A section is matched by name exactly as a page is, with two restrictions:

- It ranks one step below a real page (`SECTION_MATCH_SCORE` 0.85 against
  `EXACT_TITLE_SCORE` 0.90), so when a message names both, the page wins:
  "qual o alcance de Bola de Fogo?" is a question about the skill.
- **A name an infobox uses for a row is not a name.** *Alcance*,
  *Conjuração*, *Propriedade* and *Munição* are all sections of stat pages
  *and* labels in every skill infobox, which makes them the words people use
  to ask about some *other* page — the same trap "Cooldown" sets above.
  `bot/facts.py` already knows the labels, so the list maintains itself.

What's left is one-word section names, and they are the judgement call this
time. Unlike the game's one-word English names, they can't simply be dropped:
one word is what a status effect *is* called. But the wiki also points
"Sorte", "Força", "Vento" and "Visual" at sections of its stat and item
pages, so "kkkkk mano que sorte a sua" now reaches *Atributos § Sorte* where
it used to reach nothing. Of the 134 one-word section names here, ~15 read
that way; the rest are jargon ("Petrificação", "Hipotermia", "Cristalização")
nobody types by accident.

Nothing measurable separates the two groups — mid-sentence, the wiki's prose
capitalises "Sorte" 76% of the time and "Sangramento" 79%, and a casual "que
sorte a sua" scores no *lower* against the *Sorte* section than "como curar
congelamento?" does against *Congelamento*. It is recall against quiet, not
a threshold waiting to be tuned. `MIN_SECTION_NAME_WORDS` in
`bot/retriever.py` is the dial: raise it to 2 to keep only the multi-word
names (*Sono Profundo*, *Envenenamento Mortal*) and give up the rest.

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
│   ├── facts.py          # Infobox parsing: answers "qual o cooldown de X?" with the value
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

A full crawl of bROWiki takes roughly 12 minutes: ~1,900 articles and ~5,200
vectors (a page's chunks, plus a title card, plus a names card for the 1,007
pages that have other-language names), all on CPU.

The crawl is built to survive a wiki that isn't always up: HTTP 429/5xx and
dropped connections are retried with a widening pause (2s, 4s, 8s, 16s)
before the run gives up, and progress is written to
`data/ingest_state.json` as it goes — a crawl that dies two-thirds through
resumes from where it stopped instead of starting over.

Each page also records the `INDEX_FORMAT` its vectors were built under, so a
change to *how* pages are read doesn't need `--force` to take effect: bump
`INDEX_FORMAT` in `ingest/build_index.py` after changing extraction,
chunking, metadata or the embedding model, and the next ordinary run
re-indexes everything (resumably). Without that stamp an unchanged page
looks up to date and quietly keeps its stale vectors. An index built before
the English/Spanish names is format 1, so one ordinary run brings it up to
date; until then the bot behaves exactly as it did, matching Portuguese
titles only. `--force` remains for re-fetching and re-embedding everything
unconditionally.

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
  information comes from the bROWiki — nothing else, unless the question
  asked for a single infobox value, which is then quoted above the link
  (see [Direct answers](#direct-answers)). URLs are percent-encoded, without
  which Discord won't linkify addresses containing accented characters
  (`.../wiki/Lâminas_Aceleradas`).

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
- To teach direct answers a new field, add a `Field` to `FIELDS` in
  `bot/facts.py` with the infobox label and the phrasings people type. Add
  labels you *don't* want quoted too (with an empty `asked_as`): each one
  recognised is a boundary that stops the row above it from swallowing the
  next — that's why "Arma" is in the list.
- Smaller `CHUNK_SIZE_TOKENS` gives more precise matches on short factual
  pages; larger chunks preserve more context for narrative pages (quest
  walkthroughs, lore, etc.).
