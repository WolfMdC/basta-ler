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
"""
from __future__ import annotations

import re
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

# Sentinel marking a table-cell boundary; resolved to " | " during cleanup.
_CELL_MARK = "\x00"

_EDIT_LINK_RE = re.compile(r"\[\s*editar[^\]]*\]", re.IGNORECASE)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        # Prose paragraphs (<p> outside any table), used for the page blurb.
        self.paragraphs: list[str] = []
        self._depth = 0
        # Depth of the element whose subtree we're currently skipping, if any.
        self._skip_from: int | None = None
        self._table_depth = 0
        self._paragraph: list[str] | None = None

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
        elif tag == "p" and self._table_depth == 0 and self._paragraph is None:
            self._paragraph = []

        if tag in CELL_TAGS:
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

        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_from is not None:
            return
        self.parts.append(data)
        if self._paragraph is not None:
            self._paragraph.append(data)


def _is_chrome(attrs) -> bool:
    """True for wiki UI furniture (edit links, TOC, navboxes, …)."""
    attr = dict(attrs)
    classes = set((attr.get("class") or "").split())
    if classes & DROP_CLASSES:
        return True
    return (attr.get("id") or "") in DROP_IDS


@dataclass(frozen=True)
class ExtractedPage:
    text: str  # everything, infobox and tables included — this is what we index
    lead: str  # opening prose only — a human-readable blurb for the page


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

    return ExtractedPage(
        text="\n".join(lines),
        lead=_build_lead(parser.paragraphs, lead_max_chars),
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
