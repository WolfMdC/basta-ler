"""
Generic client for the MediaWiki Action API (api.php).

Deliberately written against the *standard* Action API only — no assumptions
about extensions that may or may not be installed on a given wiki (e.g.
TextExtracts, which many small wikis including bROWiki are unlikely to have
enabled). Content is fetched as the wiki's own rendered HTML via
`action=parse&prop=text` and reduced to plain text locally, so this works
against any MediaWiki install: Wikipedia, the OSRS Wiki, bROWiki, a
self-hosted mirror, etc.

Rendered HTML rather than raw wikitext is deliberate: on a game wiki the
answerable facts (cast delay, cooldown, SP cost, per-level damage) live in
templates and tables, which only become readable text once the wiki has
expanded them. See ingest/html_text.py.

Two calls are used per full crawl:
  1. `list=allpages` + `prop=revisions` (via a generator) to enumerate every
     content page along with its current revision id — cheap, paginated.
     Redirects are filtered out so an aliased title isn't indexed as a
     second copy of the page it points at.
  2. `action=parse` per page (only for pages whose revision id changed since
     the last run) to fetch rendered content.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from ingest.html_text import extract_page

logger = logging.getLogger(__name__)

USER_AGENT = "browiki-discord-bot/0.1 (personal hobby project; ingestion script)"

# Namespace 0 = main/content namespace. We only want real wiki articles, not
# Talk:, User:, Category:, Template:, etc.
MAIN_NAMESPACE = 0

# A full crawl is thousands of requests over several minutes, and a small
# wiki is not up for all of them — bROWiki answers minutes at a time with a
# Cloudflare 502, then with a 530. Nothing about those failures is specific
# to the request that hit one, so they're waited out rather than thrown at
# the caller.
#
# "Server said it's us, not you": every 5xx, which deliberately includes
# Cloudflare's own 520-530 range for an origin it can't reach, plus 429 for
# being asked to slow down. A 404 or a MediaWiki API error is the request's
# own fault and asking again would fail the same way.
MAX_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 2.0  # doubled after each attempt: 2, 4, 8, 16 → ~30s


def _is_transient(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


@dataclass(frozen=True)
class PageInfo:
    title: str
    pageid: int
    revid: int


@dataclass(frozen=True)
class PageContent:
    title: str
    pageid: int
    revid: int
    text: str  # plain text, rendered markup flattened
    lead: str  # opening prose only, used as the page's short description
    categories: list[str]
    names: tuple[str, ...] = ()  # the page's name in the game's other languages


class MediaWikiClient:
    def __init__(self, api_url: str, request_delay_seconds: float = 0.2, timeout: int = 30):
        self.api_url = api_url
        self.request_delay_seconds = request_delay_seconds
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json"}
        data = self._request(params).json()
        if "error" in data:
            raise RuntimeError(f"MediaWiki API error: {data['error']}")
        return data

    def _request(self, params: dict) -> requests.Response:
        """GET the API, waiting out the failures that are worth waiting out.

        Gives up after MAX_ATTEMPTS by raising the last failure, so a wiki
        that is genuinely down still stops the crawl instead of quietly
        turning every page into a skip.
        """
        delay = RETRY_BACKOFF_SECONDS
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._session.get(self.api_url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                failure: Exception = exc
                problem = type(exc).__name__
            else:
                if not _is_transient(response.status_code):
                    response.raise_for_status()  # a 4xx: asking again won't fix it
                    return response
                failure = requests.HTTPError(
                    f"{response.status_code} Server Error for url: {response.url}",
                    response=response,
                )
                problem = f"HTTP {response.status_code}"

            if attempt == MAX_ATTEMPTS:
                raise failure
            logger.warning(
                "%s from %s (attempt %d/%d); retrying in %.0fs",
                problem, self.api_url, attempt, MAX_ATTEMPTS, delay,
            )
            time.sleep(delay)
            delay *= 2

        raise AssertionError("unreachable")  # pragma: no cover

    def list_all_pages(self, limit: int | None = None) -> list[PageInfo]:
        """Enumerate every main-namespace article with its latest revision id.

        Uses `generator=allpages` combined with `prop=revisions`, which
        returns pages + their current revid in one paginated call — no need
        to hit `action=parse` for pages that haven't changed.

        `gapfilterredir=nonredirects` drops redirect pages: `action=parse`
        follows redirects, so indexing them would store the target's content
        a second time under the alias title (e.g. both "Laminas Aceleradas"
        and "Lâminas Aceleradas"), which then compete in search results.

        If `limit` is given, stops paginating as soon as we have at least
        that many pages (useful for quick dev/test runs against huge wikis).
        """
        pages: dict[int, PageInfo] = {}
        gapcontinue = None

        while True:
            params = {
                "action": "query",
                "generator": "allpages",
                "gapnamespace": MAIN_NAMESPACE,
                "gapfilterredir": "nonredirects",
                "gaplimit": "max",
                "prop": "revisions",
                "rvprop": "ids",
            }
            if gapcontinue:
                params["gapcontinue"] = gapcontinue

            data = self._get(params)
            query = data.get("query", {})
            for page in query.get("pages", {}).values():
                if "missing" in page or not page.get("revisions"):
                    continue
                revid = page["revisions"][0]["revid"]
                pages[page["pageid"]] = PageInfo(
                    title=page["title"], pageid=page["pageid"], revid=revid
                )

            if limit and len(pages) >= limit:
                break

            cont = data.get("continue", {})
            gapcontinue = cont.get("gapcontinue")
            if not gapcontinue:
                break
            time.sleep(self.request_delay_seconds)

        logger.info("Discovered %d pages via allpages", len(pages))
        return list(pages.values())

    def fetch_page_content(self, title: str) -> PageContent | None:
        """Fetch a page's rendered HTML and reduce it to plain text."""
        params = {
            "action": "parse",
            "page": title,
            "prop": "text|revid|categories",
            "redirects": 1,
        }
        try:
            data = self._get(params)
        except RuntimeError as exc:
            # A page deleted/renamed between the allpages listing and now
            # shouldn't abort a crawl that's already minutes deep.
            logger.warning("Could not parse page %r: %s", title, exc)
            return None

        parse = data.get("parse")
        if not parse:
            logger.warning("No parse result for page %r", title)
            return None

        html = parse["text"]["*"] if isinstance(parse["text"], dict) else parse["text"]
        extracted = extract_page(html)
        if not extracted.text.strip():
            return None

        categories = [
            cat["*"].replace("_", " ")
            for cat in parse.get("categories", [])
            if "hidden" not in cat and cat.get("*")
        ]

        return PageContent(
            title=parse.get("title", title),
            pageid=parse["pageid"],
            revid=parse["revid"],
            text=extracted.text,
            lead=extracted.lead,
            categories=categories,
            names=extracted.names,
        )

    def close(self):
        self._session.close()
