"""
Direct fact answers: pulling a single infobox value out of an indexed page.

When someone asks "Quanto de pós-conjuração tem Impacto Sísmico?", pointing
at the page is a worse answer than the number itself. Every skill/quest page
on the wiki opens with an infobox of label/value pairs, and those pairs are
already in the index — so the value can be answered inline, with the page
URL kept alongside it as the source.

Two steps, both deliberately conservative — anything less than a clean match
returns None and the bot falls back to its normal "here's the page" reply:

  1. `detect_field` — does the question actually ask for a known field?
     Matched against how people *type* it in Discord ("cooldown", "recarga",
     "quanto de sp"), accent-insensitively. Half of that vocabulary is
     English: a Portuguese sentence with the game's English terms dropped
     into it ("qual o cast fixo de Fogo de Supressão?") is normal speech in
     a Ragnarök channel, not a mistake, so both languages point at the same
     row of the wiki's own Portuguese infobox.
  2. `parse_infobox` — recover the label/value pairs from the indexed text.

The parsing is the fiddly half. `ingest/html_text.py` puts every infobox
label and value on its own line, but `ingest/chunker.py` splits on
whitespace, so what is stored is one flat run of words:

    ... Tipo Ofensiva Níveis 5 SP 30 + (Nv. da habilidade × 5) Conjuração
    0 + 1 seg. Recarga 2 segundos Alvo Usuário Área 9x9 células ...

There is no delimiter left, so the labels themselves are the delimiters: a
field's value is whatever sits between its label and the next known label.
That works because the infobox is a contiguous run at the very top of the
page, and it ends where the gap to the next label gets large — the prose
description. Everything past that point (including the per-level table,
which repeats "SP" and "Nv.") is never read.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

# Only the top of the page can hold the infobox; the rest is prose and
# tables. Bounds the parsing work and keeps stray label words out.
MAX_SCAN_CHARS = 1200

# A label further into the page than this doesn't open an infobox — it's a
# word in the article body that happens to match.
INFOBOX_HEAD_WORDS = 40

# Longest gap, in words, between one label and the next that still reads as
# a single field's value. A bigger gap means the infobox ended and the
# description started.
MAX_VALUE_WORDS = 14

# Longest label in words, e.g. "Custo de AP", "Quests anteriores".
MAX_LABEL_WORDS = 4


@dataclass(frozen=True)
class Field:
    """One infobox row the parser knows about.

    `asked_as` empty means the field is a *boundary only*: the parser needs
    to recognise the label so the previous field's value stops there, but
    the bot will never quote it as an answer (either because nobody asks for
    it, or because its value runs into the prose and can't be bounded).
    """

    label: str  # canonical spelling, as shown in the reply
    aliases: tuple[str, ...] = ()  # other spellings used in infoboxes
    asked_as: tuple[str, ...] = ()  # accent-free phrasings people type
    numeric: bool = True  # value must contain a digit to be quoted
    max_words: int = 12  # values longer than this are treated as a misparse


@dataclass(frozen=True)
class Fact:
    label: str
    value: str


# Answerable fields, then labels that exist purely to terminate a value.
# Ordering doesn't matter; matching always prefers the longest label.
# `max_words` is set just above the longest *real* value each field has on
# this wiki, measured over the whole index — a value longer than that has
# run past the end of the infobox and into the description.
FIELDS: tuple[Field, ...] = (
    Field(
        label="Pós-conjuração",
        asked_as=(
            "pos conjuracao", "pos cast", "after cast", "aftercast",
            "cast delay", "delay",
        ),
        max_words=9,
    ),
    Field(
        # The wiki writes this row as "variável + fixo" ("1 + 0 seg."), so a
        # question about either half is answered with the whole row — the bot
        # quotes the wiki, it doesn't do arithmetic on it.
        label="Conjuração",
        asked_as=(
            "tempo de conjuracao", "conjuracao", "tempo de cast", "cast time",
            "casting time", "casting", "cast fixo", "fixed cast",
            "cast variavel", "variable cast", "cast",
        ),
        max_words=13,
    ),
    Field(
        label="Recarga",
        asked_as=("tempo de recarga", "recarga", "cooldown", "cool down", "cd"),
        max_words=10,
    ),
    Field(
        label="SP",
        asked_as=("custo de sp", "gasto de sp", "sp cost", "sp", "mana"),
        max_words=9,
    ),
    Field(
        label="Custo de AP",
        asked_as=("custo de ap", "ap cost", "gasto de ap"),
        max_words=7,
    ),
    Field(
        label="Duração",
        asked_as=("quanto tempo dura", "duracao", "duration"),
        max_words=10,
    ),
    Field(
        label="Alcance",
        asked_as=("alcance", "range"),
        max_words=8,
    ),
    Field(
        label="Área",
        asked_as=("area de efeito", "area", "aoe", "area of effect", "splash"),
        max_words=6,
    ),
    Field(
        label="Níveis",
        asked_as=(
            "quantos niveis", "nivel maximo", "niveis", "max level",
            "level maximo", "lv maximo",
        ),
        max_words=3,
    ),
    Field(
        label="Alvo",
        asked_as=("alvo", "target"),
        numeric=False,
        max_words=6,
    ),
    Field(
        label="Tipo",
        asked_as=("tipo", "type"),
        numeric=False,
        max_words=3,
    ),
    Field(
        label="Propriedade",
        asked_as=("propriedade", "elemento", "element", "property"),
        numeric=False,
        max_words=4,
    ),
    Field(
        label="Munição",
        asked_as=("municao", "ammo", "ammunition"),
        numeric=False,
        max_words=5,
    ),
    Field(
        label="ID",
        asked_as=("id da habilidade", "skill id"),
        numeric=False,
        max_words=3,
    ),
    # Boundary-only labels below this line. Nobody asks for these by name,
    # but each one has to be recognised or the field above it swallows the
    # row that follows: without "Arma", "Níveis 5" on a weapon-restricted
    # skill reads as "5 Arma Espada ou Adaga".
    Field(label="Pré-requisitos"),
    Field(label="Item", aliases=("Itens",)),
    Field(label="Ícone"),
    Field(label="Efeito", aliases=("Efeitos", "Efeito negativo", "Efeito positivo")),
    Field(label="Notas"),
    Field(label="Fórmula"),
    Field(label="Categorias"),
    Field(label="Exclusiva"),
    Field(label="Arma", aliases=("Armas",)),
    Field(label="Requisito", aliases=("Requisitos",)),
    Field(label="Consome"),
    Field(label="Equipado"),
    Field(label="Empurra"),
    Field(label="Regen. de AP"),
    Field(label="Regen. de SP"),
    Field(label="Áudio"),
    Field(label="Equipamento"),
    Field(label="Zeny"),
    Field(label="HP"),
    Field(label="Postura"),
    Field(label="Estado"),
    Field(label="Auréola"),
    Field(label="Espaço Celeste"),
    Field(label="Requisitos mínimos"),
    Field(label="Nv. de base"),
    Field(label="Nv. de classe"),
    Field(label="Classe", aliases=("Classes",)),
    Field(label="Premiação"),
    Field(label="Recompensa"),
    # "Grupo" is deliberately *not* a label: it is a quest-infobox row, but
    # it is also the tail of the common skill value "Usuário e membros do
    # Grupo", and cutting that value there would end the parse mid-infobox.
    Field(label="Retorno"),
    Field(label="Quests anteriores"),
    Field(label="Caça"),
    Field(label="Peso"),
    Field(label="Preço"),
    Field(label="Slots"),
    Field(label="Localização"),
    Field(label="Estilos"),
    Field(label="Anterior"),
    Field(label="Sprite"),
    Field(label="Guia"),
)


def _normalize(text: str) -> str:
    """Lowercase, drop accents, and reduce punctuation to spaces.

    So "Pós-conjuração" and a Discord user's "pos conjuracao" collapse to the
    same string, and "Nv." to "nv".
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", stripped)).strip()


def _build_label_lookup() -> dict[str, Field]:
    lookup: dict[str, Field] = {}
    for field in FIELDS:
        for spelling in (field.label, *field.aliases):
            lookup.setdefault(_normalize(spelling), field)
    return lookup


def _build_question_lookup() -> list[tuple[str, Field]]:
    # Longest first only so the list reads in a useful order; `detect_field`
    # scores every phrase rather than stopping at the first hit.
    phrases = [
        (_normalize(phrase), field)
        for field in FIELDS
        for phrase in field.asked_as
    ]
    return sorted(phrases, key=lambda item: len(item[0]), reverse=True)


_LABELS = _build_label_lookup()
_QUESTION_PHRASES = _build_question_lookup()


def detect_field(question: str) -> Field | None:
    """The infobox field a question asks for, if any.

    Where two phrasings overlap the earlier one in the sentence wins, which
    is what keeps "quanto de pós-conjuração" from being read as a question
    about "conjuração" — the longer label starts first.
    """
    normalized = _normalize(question)
    best: tuple[tuple[int, int], Field] | None = None

    for phrase, field in _QUESTION_PHRASES:
        match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized)
        if match is None:
            continue
        # Earliest match wins; longer phrase breaks a tie.
        key = (match.start(), -len(phrase))
        if best is None or key < best[0]:
            best = (key, field)

    return best[1] if best else None


def _words(text: str) -> tuple[list[str], list[str]]:
    """Split into whitespace-separated words plus their normalized forms.

    A single word can normalize to two ("Pós-conjuração" -> "pos
    conjuracao"), which is exactly why label matching compares joined
    windows rather than word-for-word.
    """
    raw = text.split()
    return raw, [_normalize(word) for word in raw]


def _label_at(normalized: list[str], start: int) -> tuple[Field, int] | None:
    """The longest known label starting at `start`, and its length in words."""
    # Punctuation-only words normalize to nothing, so without this a label
    # would match one word early and swallow whatever precedes it — the "(?)"
    # closing an unknown skill ID, the ")" closing an SP formula.
    if not normalized[start]:
        return None

    limit = min(MAX_LABEL_WORDS, len(normalized) - start)
    for size in range(limit, 0, -1):
        window = " ".join(normalized[start:start + size]).strip()
        field = _LABELS.get(re.sub(r"\s+", " ", window))
        if field is not None:
            return field, size
    return None


def _skip_names(normalized: list[str], start: int, names: Sequence[str]) -> int:
    """Word index just past the page's other-language names.

    Same reason the title is skipped, one line further down: "Increase SP
    Recovery / Aumentar Recuperación" opens *Aumentar Recuperação de SP*, and
    its second word is a field label — read as one, the page's SP row becomes
    "Recovery / Aumentar Recuperación" and the real one is never reached.

    Consumes words only as long as they keep spelling out the names we were
    given, so a page whose names sit somewhere else in the text (or aren't
    there at all) is scanned from the title exactly as before.
    """
    if not names:
        return start

    expected = _normalize(" ".join(names))
    seen = ""
    position = start
    while position < len(normalized):
        word = normalized[position]
        position += 1
        if not word:
            continue  # the "/" between two names, or any other punctuation
        candidate = f"{seen} {word}".strip()
        if not (expected == candidate or expected.startswith(f"{candidate} ")):
            return start
        seen = candidate
        if seen == expected:
            return position
    return start


def parse_infobox(text: str, title: str = "", names: Sequence[str] = ()) -> dict[str, str]:
    """Label -> value pairs from the infobox at the top of a page.

    Returns an empty dict for pages that don't open with one (hub pages,
    class pages, plain prose articles).

    `title` is skipped before scanning, because the page opens with its own
    name and a page can be named after a field — on "Ferir Alvo" the title's
    second word would otherwise be read as the infobox's "Alvo" row. `names`
    — the same page's name in the game's other languages — is skipped right
    after it, for the same reason.
    """
    raw, normalized = _words(text[:MAX_SCAN_CHARS])

    hits: list[tuple[int, int, Field]] = []
    position = _skip_names(normalized, len(title.split()), names)
    while position < len(raw):
        match = _label_at(normalized, position)
        if match is None:
            position += 1
            continue
        field, size = match
        hits.append((position, position + size, field))
        position += size

    if not hits or hits[0][0] > INFOBOX_HEAD_WORDS:
        return {}

    facts: dict[str, str] = {}
    for (_, value_start, field), (next_start, _, _) in zip(hits, hits[1:]):
        gap = next_start - value_start
        if gap > MAX_VALUE_WORDS:
            break  # the infobox ended here; the rest of the page is prose
        if gap <= 0:
            continue  # label with no value of its own, e.g. a bare "Ícone"
        value = " ".join(raw[value_start:next_start]).strip(" :|-–—")
        if not _looks_like_value(value):
            break  # ditto: this "value" is really the start of the description
        if value:
            facts.setdefault(field.label, value)

    # The final label has no following label to bound it, so its value would
    # run into the description — it is deliberately left out.
    return facts


# Words an infobox value never ends on. A value that does was cut out of a
# sentence: the row was the last one before the description, and the label
# that stopped it was a word in the prose ("Alcance 6 células Causa dano
# físico a distância em linha reta, na").
_DANGLING_WORDS = {
    "a", "as", "o", "os", "um", "uma", "de", "do", "da", "dos", "das", "em",
    "no", "na", "nos", "nas", "por", "para", "com", "sem", "e", "ou", "que",
    "se", "ao", "aos", "seu", "sua", "seus", "suas", "mais", "menos", "ate",
    "sob", "sobre", "entre", "contra", "apos", "pelo", "pela", "pelos", "pelas",
}

# Abbreviations whose trailing period doesn't end a sentence.
_ABBREVIATIONS = {"nv.", "seg.", "hab.", "min.", "max.", "máx.", "aprox.", "hrs."}


def _ends_sentence(word: str) -> bool:
    return word.endswith(".") and word.lower() not in _ABBREVIATIONS


def _looks_like_value(value: str) -> bool:
    """Whether a parsed value still reads as an infobox cell, not prose.

    A value's right-hand edge is only as trustworthy as the label that
    stopped it, and past the last infobox row the only "labels" left are
    words in the description — so the first value that reads like a sentence
    fragment is the signal that the infobox has ended.
    """
    words = value.split()
    if not words:
        return True  # an empty cell is odd but not prose
    if all(word.lower() in _ABBREVIATIONS for word in words):
        return False  # a bare "Nv." is a table header, not a value
    if words[-1].lower().strip(".,;:") in _DANGLING_WORDS:
        return False
    if _ends_sentence(words[-1]):
        return False
    # A period mid-value followed by a capitalised word: prose has started.
    return not any(
        _ends_sentence(word) and following[:1].isupper()
        for word, following in zip(words, words[1:])
    )


def _is_quotable(field: Field, value: str) -> bool:
    """Whether a parsed value is safe to quote back as this field's answer.

    `parse_infobox` has already rejected prose; what's left is per-field
    plausibility. Anything that fails just means the bot answers with the
    page link, as it always did.
    """
    words = value.split()
    if not words or len(words) > field.max_words:
        return False
    if not field.numeric or any(char.isdigit() for char in value):
        return True
    # A numeric field can still hold a short written-out value — a Duração of
    # "Permanente", an Área shaped like a "Triângulo". Capitalised, because a
    # fragment sliced out of a sentence starts mid-phrase and lowercase.
    return len(words) <= 2 and words[0][:1].isupper()


def find_fact(
    question: str, page_text: str, title: str = "", names: Sequence[str] = ()
) -> Fact | None:
    """The infobox value answering `question` on this page, if there is one."""
    if not page_text:
        return None

    field = detect_field(question)
    if field is None or not field.asked_as:
        return None

    value = parse_infobox(page_text, title, names).get(field.label)
    if value is None or not _is_quotable(field, value):
        return None

    return Fact(label=field.label, value=value)
