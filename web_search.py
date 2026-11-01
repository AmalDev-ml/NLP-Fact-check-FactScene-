"""
web_search.py — Async Live Web Search & Full Page Content Extraction
---------------------------------------------------------------------
Phase 3 Features:
1. Concurrent multi-provider search (Serper, Tavily, DuckDuckGo, Wikipedia).
2. Two-tier fast fetching:
   - Fetches top 2-3 pages in full; uses snippets for the rest unless ambiguous.
3. Extracts publish dates from page metadata (trafilatura.extract_metadata) and search hits.
4. Breaks pages into salient passages (~900 characters) on clean sentence boundaries.
5. Caches retrieved results by hash of (claim + query).
"""

import os
import re
import time
import json
import hashlib
import logging
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import trafilatura
from ddgs import DDGS

from config import PASSAGE_CHAR_WINDOW

logger = logging.getLogger(__name__)

# In-memory search & passage cache: sha256(claim + query) -> List[Dict]
_search_cache: Dict[str, List[Dict[str, Any]]] = {}

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

def _get_cache_key(claim: str, query: str) -> str:
    norm = f"{claim.strip().lower()}||{query.strip().lower()}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ── Search Providers ─────────────────────────────────────────────────────────

def search_serper(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Serper.dev Google search provider."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return []
    try:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json={"q": query, "num": max_results}, timeout=8.0)
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for r in data.get("organic", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": r.get("snippet", ""),
                    "date": r.get("date"),
                    "provider": "serper"
                })
            return results
    except Exception as e:
        logger.warning(f"Serper search error: {e}")
    return []


def search_tavily(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Tavily search provider."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results
        }
        resp = requests.post(url, json=payload, timeout=4.0)
        if resp.status_code == 200:
            data = resp.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "date": r.get("published_date"),
                    "provider": "tavily"
                }
                for r in data.get("results", [])
            ]
    except Exception as e:
        logger.warning(f"Tavily search error: {e}")
    return []


def search_ddg(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """DuckDuckGo live web search via ddgs with strict timeout."""
    try:
        with DDGS(timeout=5.0) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "date": None,
                    "provider": "duckduckgo"
                }
                for r in results if r.get("href")
            ]
    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}")
    return []


def search_wikipedia_rest(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Wikipedia REST search fallback."""
    clean_q = re.sub(r'site:\S+', '', query).strip()
    clean_q = re.sub(r'\b(fake|fact check|hoax|false)\b', '', clean_q, flags=re.IGNORECASE).strip()
    if not clean_q:
        return []

    try:
        url = f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={urllib.parse.quote(clean_q)}&limit={max_results}"
        headers = {"User-Agent": BROWSER_USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=3.0)
        if resp.status_code == 200:
            pages = resp.json().get("pages", [])
            results = []
            for p in pages:
                title = p.get("title", "")
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                excerpt = p.get("excerpt", "") or p.get("description", "")
                excerpt = re.sub(r'<[^>]+>', '', excerpt).strip()
                results.append({
                    "title": title,
                    "url": page_url,
                    "snippet": excerpt,
                    "date": None,
                    "provider": "wikipedia"
                })
            return results
    except Exception as e:
        logger.warning(f"Wikipedia REST error: {e}")
    return []


# ── Full Page Reading & Metadata Extraction ──────────────────────────────────

def fetch_page_text_and_meta(url: str, snippet: str = "") -> Tuple[str, Optional[str]]:
    """
    Fetches HTML and extracts main article text plus publish date using trafilatura.
    Returns (article_text, publish_date).
    """
    if not url or not url.startswith("http"):
        return snippet, None

    try:
        headers = {"User-Agent": BROWSER_USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=3.0)
        if resp.status_code == 200 and resp.text:
            meta = None
            try:
                meta = trafilatura.extract_metadata(resp.text)
            except Exception:
                pass
            
            publish_date = meta.date if meta and meta.date else None

            extracted = trafilatura.extract(
                resp.text,
                include_links=False,
                include_images=False,
                include_tables=False,
                no_fallback=False
            )
            if extracted and len(extracted.strip()) >= 80:
                return extracted.strip(), publish_date
            return snippet, publish_date
    except Exception:
        pass

    return snippet, None


def split_into_passages(text: str, max_chars: int = PASSAGE_CHAR_WINDOW) -> List[str]:
    """Splits article text into clean passages respecting sentence boundaries."""
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= 40]
    passages = []

    for para in paragraphs[:12]:
        if len(para) <= max_chars:
            passages.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) + 1 <= max_chars:
                    chunk = f"{chunk} {s}".strip()
                else:
                    if chunk:
                        passages.append(chunk)
                    chunk = s
            if chunk:
                passages.append(chunk)

    return passages[:10]


# ── Coordinated Retrieval ────────────────────────────────────────────────────

def search_query(query: str, query_type: str = "neutral", max_results: int = 5) -> List[Dict[str, Any]]:
    """Runs primary search (Serper first if configured, else Tavily/DDG) with Wikipedia."""
    results = search_serper(query, max_results=max_results)
    if not results:
        results = search_tavily(query, max_results=max_results)
    if not results:
        results = search_ddg(query, max_results=max_results)

    # Supplement with Wikipedia for encyclopedic, neutral, or role queries
    if not results or query_type in ("neutral", "role"):
        wiki_res = search_wikipedia_rest(query, max_results=3)
        results.extend(wiki_res)

    for r in results:
        r["query_type"] = query_type
    return results


def retrieve_evidence_for_claim(claim_text: str, queries: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Executes all 4 queries concurrently, reads top pages, and returns candidate passages.
    Fast tiered fetching:
      - Reads full text for the top 2-3 pages immediately.
      - Uses snippets for remaining hits to keep latency < 3s.
      - Fetches more only if evidence is sparse.
    """
    cache_key = _get_cache_key(claim_text, str(queries))
    if cache_key in _search_cache:
        logger.info(f"Returning cached search results for claim: '{claim_text[:40]}'")
        return _search_cache[cache_key]

    raw_search_results: List[Dict[str, Any]] = []

    # Parallel search execution across all query types
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(search_query, q, q_type, 4): q_type
            for q_type, q in queries.items() if q
        }
        for future in as_completed(futures):
            try:
                res = future.result()
                raw_search_results.extend(res)
            except Exception as e:
                logger.warning(f"Search future error: {e}")

    # Fallback to Wikipedia if nothing found
    if not raw_search_results:
        raw_search_results = search_wikipedia_rest(claim_text, max_results=4)
        for r in raw_search_results:
            r["query_type"] = "neutral"

    # Deduplicate URLs
    unique_urls = {}
    for r in raw_search_results:
        u = r.get("url", "")
        if u and u not in unique_urls:
            unique_urls[u] = r

    target_items = list(unique_urls.values())[:8]
    candidate_passages: List[Dict[str, Any]] = []

    def fetch_worker(item, fetch_full: bool = True):
        u = item["url"]
        snip = item.get("snippet", "")
        item_date = item.get("date")

        if fetch_full:
            full_text, meta_date = fetch_page_text_and_meta(u, snippet=snip)
            date_to_use = meta_date or item_date
            passages = split_into_passages(full_text)
        else:
            date_to_use = item_date
            passages = [snip] if snip else []

        if not passages and snip:
            passages = [snip]

        parsed_domain = urllib.parse.urlparse(u).netloc.lower().replace("www.", "")
        items = []
        for p_text in passages:
            items.append({
                "url": u,
                "domain": parsed_domain,
                "title": item.get("title", ""),
                "text": p_text,
                "publish_date": date_to_use,
                "query_type": item.get("query_type", "neutral"),
                "provider": item.get("provider", "web")
            })
        return items

    # Fast tiered fetch: full page for top 3, snippet for items 4..8
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for idx, item in enumerate(target_items):
            fetch_full = (idx < 3)  # top 3 full
            futures.append(executor.submit(fetch_worker, item, fetch_full))

        for f in as_completed(futures):
            try:
                passages = f.result()
                candidate_passages.extend(passages)
            except Exception as e:
                logger.warning(f"Fetch future error: {e}")

    _search_cache[cache_key] = candidate_passages
    return candidate_passages
