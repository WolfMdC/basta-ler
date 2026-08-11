"""
HTML -> plain text for MediaWiki's *rendered* page output.

We index the rendered HTML (`action=parse&prop=text`) rather than raw
wikitext because on a game wiki almost all the facts people ask about live
inside templates and tables — a skill's cast delay, cooldown, SP cost, range
and per-level numbers are all in the `{{Skill Info}}` infobox and the level
table, none of which exist in the wikitext as readable text. Stripping
wikitext throws that away and leaves only the prose intro; rendering it first
turns `| delay = 0,5 segundos` into the human label the wiki actually shows
("Pós-conjuração | 0,5 segundos"), which is also the wording people type in
Discord.

Table rows are flattened to `cell | cell | cell` on one line so that an
infobox label stays glued to its value in the same chunk.

One row of that infobox is pulled out separately: the line under the page
title that names the skill in the game's other languages ("Wild Fire /
Fuego Salvaje"). Players use those names in Portuguese sentences all the
time, so they are kept as page names in their own right — see `page_names`.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser

# Tags that never have a closing tag, so they must not affect nesting depth.
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}

# Subtrees dropped wholesale — never prose.
DROP_TAGS = {"script", "style", "noscript"}
DROP_CLASSES = {
    "mw-editsection", "toc", "toctitle", "toctogglespan", "navbox",
    "navbox-inner", "catlinks", "printfooter", "noprint", "reference",
    "references", "mw-references-wrap", "mw-jump-link", "metadata",
    "hatnote", "magnify", "thumbcaption", "mw-indicators",
}
DROP_IDS = {"toc", "catlinks", "siteSub", "contentSub", "jump-to-nav", "mw-navigation"}

# Tags that imply a line break in the plain-text rendering.
BLOCK_TAGS = {
    "p", "div", "br", "hr", "tr", "caption", "figcaption", "li", "dt", "dd",
    "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "dl",
    "blockquote", "section", "pre",
}
CELL_TAGS = {"td", "th"}

# The other-language name row carries no label, so it is recognised by the
# markup the wiki gives it: a cell whose entire content is small *and* bold.
# The page title above it is bold but not small; every label below it is bold
# but sits in a two-cell row. See `_TextExtractor.name_rows`.
SMALL_TAGS = {"small"}
BOLD_TAGS = {"b", "strong"}

# Sentinel marking a table-cell boundary; resolved to " | " during cleanup.
_CELL_MARK = "\x00"

_EDIT_LINK_RE = re.compile(r"\[\s*editar[^\]]*\]", re.IGNORECASE)


class _TextExtractor(HTMLParser):
    """Flattens rendered wiki HTML to text, keeping two things on the side.

    `paragraphs` collects the prose outside tables (the page blurb), and
    `name_rows` the small-and-bold cells that open the first table — the
    other-language names on a skill page. Both are things the flattened text
    can no longer be asked for: once the page is one run of lines, a name row
    is indistinguishable from any other short line.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        # Prose paragraphs (<p> outside any table), used for the page blurb.
        self.paragraphs: list[str] = []
        # Candidate other-language name rows, in document order.
        self.name_rows: list[str] = []
        self._depth = 0
        # Depth of the element whose subtree we're currently skipping, if any.
        self._skip_from: int | None = None
        self._table_depth = 0
        self._paragraph: list[str] | None = None
        # Name-row detection: everything below is about the *first* table,
        # up to its first row that has a label and a value in it.
        self._tables_seen = 0
        self._row_cells = 0
        self._reached_labelled_row = False
        self._cell: list[str] | None = None
        self._cell_small_bold: list[str] = []
        self._small_depth = 0
        self._bold_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_TAGS:
            self._depth += 1

        if self._skip_from is not None:
            return

        if tag in DROP_TAGS or _is_chrome(attrs):
            if tag not in VOID_TAGS:
                self._skip_from = self._depth
            return

        if tag == "table":
            self._table_depth += 1
            self._tables_seen += 1
        elif tag == "p" and self._table_depth == 0 and self._paragraph is None:
            self._paragraph = []
        elif tag == "tr":
            self._row_cells = 0
        elif tag in SMALL_TAGS:
            self._small_depth += 1
        elif tag in BOLD_TAGS:
            self._bold_depth += 1

        if tag in CELL_TAGS:
            self._cell = []
            self._cell_small_bold = []
            self.parts.append(_CELL_MARK)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        # Self-closing form, e.g. <br />: no nesting, just the line break.
        if self._skip_from is None and tag in BLOCK_TAGS and not _is_chrome(attrs):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return

        closing_depth = self._depth
        self._depth = max(self._depth - 1, 0)

        if self._skip_from is not None:
            if closing_depth <= self._skip_from:
                self._skip_from = None
            return

        if tag == "table":
            self._table_depth = max(self._table_depth - 1, 0)
        elif tag == "p" and self._paragraph is not None:
            paragraph = re.sub(r"\s+", " ", "".join(self._paragraph)).strip()
            self._paragraph = None
            if paragraph:
                self.paragraphs.append(paragraph)
        elif tag in SMALL_TAGS:
            self._small_depth = max(self._small_depth - 1, 0)
        elif tag in BOLD_TAGS:
            self._bold_depth = max(self._bold_depth - 1, 0)
        elif tag == "tr" and self._row_cells >= 2:
            # A label/value row: the infobox proper has started, so anything
            # after it is a field, not a name.
            self._reached_labelled_row = True

        if tag in CELL_TAGS:
            self._close_cell()

        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_from is not None:
            return
        self.parts.append(data)
        if self._paragraph is not None:
            self._paragraph.append(data)
        if self._cell is not None:
            self._cell.append(data)
            if self._small_depth and self._bold_depth:
                self._cell_small_bold.append(data)

    def _close_cell(self) -> None:
        """Finish the current table cell, recording it if it names the page."""
        cell = re.sub(r"\s+", " ", "".join(self._cell or "")).strip()
        self._cell = None
        if not cell:
            return

        self._row_cells += 1
        small_bold = re.sub(r"\s+", " ", "".join(self._cell_small_bold)).strip()
        # Only in the leading rows of the page's first table, and only when
        # the small+bold run is the whole cell — a value that merely contains
        # a small bold word is a value, not a name.
        if self._tables_seen == 1 and not self._reached_labelled_row and small_bold == cell:
            self.name_rows.append(cell)


def _is_chrome(attrs) -> bool:
    """True for wiki UI furniture (edit links, TOC, navboxes, …)."""
    attr = dict(attrs)
    classes = set((attr.get("class") or "").split())
    if classes & DROP_CLASSES:
        return True
    return (attr.get("id") or "") in DROP_IDS


# A page's categories are appended to its indexed text as one trailing line,
# so they're searchable along with the body ("habilidades de monstro"). The
# bot reads them back out of the index (see bot/retriever.py), so the two
# sides share these helpers rather than each spelling the format out.
CATEGORIES_PREFIX = "Categorias: "
# Unanchored on purpose: the line survives into the index only as far as the
# chunker, which joins everything with spaces (see ingest/chunker.py), so by
# the time the bot reads a chunk back the prefix sits mid-text. The trailing
# `.+` stops at the newline it still has in the un-chunked form.
_CATEGORIES_RE = re.compile(rf"{re.escape(CATEGORIES_PREFIX)}(.+)")


def append_categories(text: str, categories: list[str]) -> str:
    if not categories:
        return text
    return f"{text}\n{CATEGORIES_PREFIX}{', '.join(categories)}"


def parse_categories(text: str) -> list[str]:
    """The categories from a page's indexed text, or [] if it carries none.

    Categories are appended last, so the final match is the real one.
    """
    matches = _CATEGORIES_RE.findall(text)
    if not matches:
        return []
    return [category.strip() for category in matches[-1].split(",") if category.strip()]


# The wiki's own separator between the names in that row, reused when the
# names are stored so they round-trip through the index unchanged. No name
# can contain it: it is what `page_names` splits on.
NAMES_SEPARATOR = " / "

# A name is a page title in another language, so it is short. Anything longer
# came from a row that only looked like a name row.
MAX_NAME_WORDS = 6
MAX_NAME_ROW_WORDS = 14


def page_names(row: str) -> tuple[str, ...]:
    """The individual names in a name row: "Wild Fire / Fuego Salvaje".

    A row holds the English name and the Spanish one, in that order, and
    drops to a single name where the game uses the same one for both.
    """
    if len(row.split()) > MAX_NAME_ROW_WORDS:
        return ()

    names = []
    for part in row.split("/"):
        name = re.sub(r"\s+", " ", part).strip(" |-–—")
        if name and len(name.split()) <= MAX_NAME_WORDS:
            names.append(name)
    return tuple(names)


def format_names(names: Sequence[str]) -> str:
    return NAMES_SEPARATOR.join(names)


def parse_names(value: str) -> tuple[str, ...]:
    """The names back out of the single string they're stored as."""
    return tuple(name.strip() for name in (value or "").split("/") if name.strip())


@dataclass(frozen=True)
class ExtractedPage:
    text: str  # everything, infobox and tables included — this is what we index
    lead: str  # opening prose only — a human-readable blurb for the page
    names: tuple[str, ...] = ()  # the page's name in the game's other languages


def extract_page(html: str, lead_max_chars: int = 300) -> ExtractedPage:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()

    lines: list[str] = []
    for raw_line in "".join(parser.parts).split("\n"):
        cells = [re.sub(r"\s+", " ", cell).strip() for cell in raw_line.split(_CELL_MARK)]
        cells = [cell for cell in cells if cell]
        if not cells:
            continue
        line = _EDIT_LINK_RE.sub("", " | ".join(cells)).strip(" |").strip()
        if line:
            lines.append(line)

    # Only the first candidate row: a page names itself once, and a second
    # small+bold cell before the first field is something else.
    names = page_names(parser.name_rows[0]) if parser.name_rows else ()

    return ExtractedPage(
        text="\n".join(lines),
        lead=_build_lead(parser.paragraphs, lead_max_chars),
        names=names,
    )


def _build_lead(paragraphs: list[str], max_chars: int) -> str:
    """Join the opening prose paragraphs into a short blurb.

    Only <p> elements outside tables count, which is what keeps the infobox
    ("Tipo | Ofensiva | Níveis | 10 | SP | …") out of the summary — on a
    skill page the infobox comes first in document order, so anything
    position-based would pick it up instead of the actual description.
    """
    collected: list[str] = []
    for paragraph in paragraphs:
        if not collected and len(paragraph) < 20:
            continue  # stray one-liners before the real intro
        collected.append(paragraph)
        if sum(len(p) + 1 for p in collected) >= 120:
            break

    blurb = " ".join(collected).strip()
    if len(blurb) <= max_chars:
        return blurb
    return blurb[:max_chars].rsplit(" ", 1)[0] + "…"
