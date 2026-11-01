"""
rag_pipeline.py — Live Web Evidence Retrieval & Passage Reranking Pipeline
--------------------------------------------------------------------------
Implements:
  1. Multi-query generation (verbatim, keyword, fact-check)
  2. Pluggable live search providers:
     - DuckDuckGo (`ddgs`) as default zero-config web search
     - Wikipedia REST Search Provider (instant encyclopedic live search)
     - Tavily (if TAVILY_API_KEY configured)
     - Google Fact Check Tools API (if GOOGLE_FACTCHECK_API_KEY configured)
  3. Parallel live web search with strict 4.0s timeouts and graceful fallbacks
  4. Concurrent page reading via requests + Trafilatura with snippet fallbacks
  5. Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2) for top 5-8 passages
  6. Domain reputation scoring (gov, fact-checkers, tier-1 news vs social/blogs)
"""

import os
import re
import math
import time
import urllib.parse
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import trafilatura
from ddgs import DDGS
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Cross-Encoder model singleton
_cross_encoder: Optional[CrossEncoder] = None

# Reranking threshold: ms-marco cross-encoder logit threshold
PASSAGE_RERANK_THRESHOLD = -4.5

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ── Domain Reputation & Authority Weights ─────────────────────────────────────
HIGH_AUTHORITY_DOMAINS = {
    # Certified Fact-checkers
    "politifact.com": 1.4,
    "snopes.com": 1.4,
    "factcheck.org": 1.4,
    "fullfact.org": 1.4,
    "leadstories.com": 1.4,
    "altnews.in": 1.4,
    "boomlive.in": 1.4,
    "factly.in": 1.4,
    # Government & Institutional Standards
    ".gov": 1.4,
    ".gov.in": 1.4,
    ".nic.in": 1.4,
    ".mil": 1.3,
    "who.int": 1.4,
    "un.org": 1.3,
    "nasa.gov": 1.4,
    "isro.gov.in": 1.4,
    "cdc.gov": 1.4,
    # Encyclopedic & Scientific Authorities
    "nature.com": 1.35,
    "science.org": 1.35,
    "britannica.com": 1.3,
    "wikipedia.org": 1.25,
    # Major Global & National News Agencies
    "reuters.com": 1.3,
    "apnews.com": 1.3,
    "bbc.com": 1.3,
    "bbc.co.uk": 1.3,
    "thehindu.com": 1.25,
    "indianexpress.com": 1.25,
    "ndtv.com": 1.2,
    "nytimes.com": 1.25,
    "theguardian.com": 1.25,
    "wsj.com": 1.25,
}

LOW_AUTHORITY_DOMAINS = {
    "twitter.com": 0.5,
    "x.com": 0.5,
    "facebook.com": 0.5,
    "instagram.com": 0.5,
    "tiktok.com": 0.4,
    "reddit.com": 0.6,
    "pinterest.com": 0.4,
    "youtube.com": 0.6,
}


def get_domain_reputation(url: str) -> Tuple[str, float]:
    """Extract domain and compute authority multiplier."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        for high_d, weight in HIGH_AUTHORITY_DOMAINS.items():
            if domain == high_d or domain.endswith("." + high_d) or (high_d.startswith(".") and domain.endswith(high_d)):
                return domain, weight

        for low_d, weight in LOW_AUTHORITY_DOMAINS.items():
            if domain == low_d or domain.endswith("." + low_d):
                return domain, weight

        return domain, 1.0
    except Exception:
        return "unknown", 1.0


def get_cross_encoder() -> CrossEncoder:
    """Singleton loader for cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading Cross-Encoder ('cross-encoder/ms-marco-MiniLM-L-6-v2')...")
        try:
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
        except Exception:
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


# ── Step 2: Query Generation ──────────────────────────────────────────────────
def generate_search_queries(claim: str) -> List[str]:
    """
    Produce 2-3 diverse search queries:
      1. Verbatim claim query
      2. Content keyword query
      3. Dedicated 'fact check' query
    """
    clean = claim.strip().rstrip(".!?")
    queries = [clean]

    stopwords = {"is", "are", "was", "were", "the", "a", "an", "of", "in", "at", "by", "for", "with", "about", "that", "this", "to", "and"}
    words = [w for w in re.findall(r'\b[A-Za-z0-9\'-]+\b', clean) if w.lower() not in stopwords]
    if len(words) >= 2:
        keyword_q = " ".join(words[:6])
        if keyword_q.lower() != clean.lower():
            queries.append(keyword_q)

    queries.append(f"{clean} fact check")

    seen = set()
    final_queries = []
    for q in queries:
        norm = q.lower().strip()
        if norm not in seen and len(norm) > 3:
            seen.add(norm)
            final_queries.append(q)

    return final_queries[:3]


# ── Step 3: Pluggable Live Search Providers ───────────────────────────────────
class SearchProvider:
    def search(self, query: str, max_results: int = 4, timeout: float = 4.0) -> List[Dict[str, Any]]:
        raise NotImplementedError


class DuckDuckGoProvider(SearchProvider):
    """Zero-config web search provider using DuckDuckGo (ddgs)."""
    def search(self, query: str, max_results: int = 4, timeout: float = 4.0) -> List[Dict[str, Any]]:
        results = []
        try:
            with DDGS(timeout=timeout) as ddgs:
                hits = ddgs.text(query, max_results=max_results)
                if hits:
                    for h in list(hits):
                        url = h.get("href") or h.get("url")
                        title = h.get("title", "")
                        snippet = h.get("body") or h.get("snippet", "")
                        if url and (title or snippet):
                            results.append({
                                "url": url,
                                "title": title,
                                "snippet": snippet,
                                "source": "DuckDuckGo Live Search"
                            })
        except Exception as e:
            logger.debug(f"DuckDuckGo search error on '{query}': {e}")
        return results


class WikipediaSearchProvider(SearchProvider):
    """Live Wikipedia Search REST API provider."""
    def search(self, query: str, max_results: int = 3, timeout: float = 4.0) -> List[Dict[str, Any]]:
        results = []
        search_url = "https://en.wikipedia.org/w/api.php"
        headers = {"User-Agent": BROWSER_USER_AGENT}
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
        }
        try:
            resp = requests.get(search_url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                hits = resp.json().get("query", {}).get("search", [])
                for h in hits:
                    title = h.get("title", "")
                    snippet = re.sub(r'<[^>]+>', '', h.get("snippet", ""))
                    if title:
                        wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                        results.append({
                            "url": wiki_url,
                            "title": f"Wikipedia: {title}",
                            "snippet": snippet,
                            "source": "Wikipedia Online Knowledge Base"
                        })
        except Exception as e:
            logger.debug(f"Wikipedia search provider error on '{query}': {e}")
        return results


class TavilyProvider(SearchProvider):
    """Tavily search provider when TAVILY_API_KEY is configured."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 4, timeout: float = 4.0) -> List[Dict[str, Any]]:
        results = []
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "max_results": max_results,
            }
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    results.append({
                        "url": item.get("url"),
                        "title": item.get("title", ""),
                        "snippet": item.get("content", ""),
                        "source": "Tavily Live Search"
                    })
        except Exception as e:
            logger.debug(f"Tavily search error on '{query}': {e}")
        return results


class GoogleFactCheckProvider(SearchProvider):
    """Google Fact Check Tools API when GOOGLE_FACTCHECK_API_KEY is configured."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 4, timeout: float = 4.0) -> List[Dict[str, Any]]:
        results = []
        try:
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {"query": query, "key": self.api_key, "pageSize": max_results}
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                claims_data = resp.json().get("claims", [])
                for c in claims_data:
                    for r in c.get("claimReview", []):
                        pub = r.get("publisher", {}).get("name", "FactCheck Organization")
                        review_url = r.get("url")
                        rating = r.get("textualRating", "")
                        title = r.get("title", c.get("text", ""))
                        if review_url:
                            results.append({
                                "url": review_url,
                                "title": f"{pub} Fact Check: {rating}",
                                "snippet": f"Fact-checker {pub} evaluated: '{title}'. Rating: {rating}.",
                                "source": f"Google Fact Check ({pub})"
                            })
        except Exception as e:
            logger.debug(f"Google FactCheck search error on '{query}': {e}")
        return results


def get_active_providers() -> List[SearchProvider]:
    """Initializes available search providers based on environment."""
    providers: List[SearchProvider] = []

    # 1. Tavily if key present
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if tavily_key:
        providers.append(TavilyProvider(tavily_key))

    # 2. DuckDuckGo (zero-config live web search)
    providers.append(DuckDuckGoProvider())

    # 3. Wikipedia Online Search API
    providers.append(WikipediaSearchProvider())

    # 4. Google Fact Check if key present
    gfc_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if gfc_key:
        providers.append(GoogleFactCheckProvider(gfc_key))

    return providers


def execute_parallel_search(claim: str, max_urls: int = 8) -> List[Dict[str, Any]]:
    """
    Runs multi-query search across active providers concurrently.
    Collects top unique URLs with titles and search snippets.
    """
    queries = generate_search_queries(claim)
    providers = get_active_providers()

    raw_items: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=min(len(queries) + len(providers), 5)) as executor:
        futures = []
        # Run primary query across all providers (DuckDuckGo, Wikipedia, Tavily, Google)
        for prov in providers:
            futures.append(executor.submit(prov.search, queries[0], 4, 3.5))

        # Run secondary/fact-check query on fast providers
        if len(queries) > 1:
            for prov in providers:
                if isinstance(prov, (WikipediaSearchProvider, GoogleFactCheckProvider)):
                    futures.append(executor.submit(prov.search, queries[-1], 3, 3.5))

        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=3.8)
                raw_items.extend(res)
            except Exception as e:
                logger.debug(f"Search future error: {e}")

    seen_urls = set()
    unique_items: List[Dict[str, Any]] = []

    for item in raw_items:
        url = item.get("url")
        if not url:
            continue
        clean_url = url.split("#")[0].rstrip("/")
        if clean_url not in seen_urls:
            seen_urls.add(clean_url)
            domain, rep_weight = get_domain_reputation(clean_url)
            unique_items.append({
                **item,
                "url": clean_url,
                "domain": domain,
                "reputation_weight": rep_weight
            })
            if len(unique_items) >= max_urls:
                break

    return unique_items


# ── Step 4: Concurrent Page Reading & Text Chunking ───────────────────────────
def fetch_and_extract_page(item: Dict[str, Any], timeout: float = 2.5) -> List[str]:
    """
    Fetches lead page content via requests + Trafilatura with strict 2.5s timeout.
    Falls back immediately to the search snippet if page download fails or is empty.
    """
    url = item["url"]
    snippet = item.get("snippet", "")
    passages: List[str] = []

    headers = {"User-Agent": BROWSER_USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and len(resp.text) > 100:
            extracted = trafilatura.extract(resp.text, include_comments=False, output_format="txt")
            if extracted and len(extracted.strip()) >= 50:
                paragraphs = [p.strip() for p in extracted.split("\n\n") if len(p.strip()) >= 35]
                # Keep top 3 lead paragraphs (which contain the essential factual summary)
                for p in paragraphs[:3]:
                    passages.append(p[:450])
    except Exception as e:
        logger.debug(f"Page fetch skipped for '{url}': {e}")

    # Fallback / addition of search snippet
    if snippet and len(snippet.strip()) >= 20:
        passages.append(snippet.strip())

    return passages[:3]


def chunk_and_rerank_passages(
    claim: str,
    search_items: List[Dict[str, Any]],
    top_k: int = 6
) -> List[Dict[str, Any]]:
    """
    Fetches pages in parallel, chunks content, and reranks passages
    using cross-encoder/ms-marco-MiniLM-L-6-v2 against the claim.
    """
    if not search_items:
        return []

    page_passages: List[Tuple[Dict[str, Any], str]] = []

    with ThreadPoolExecutor(max_workers=min(len(search_items), 6)) as executor:
        future_map = {
            executor.submit(fetch_and_extract_page, item, 2.5): item
            for item in search_items[:5]  # Top 5 most relevant pages
        }
        for fut in as_completed(future_map):
            item = future_map[fut]
            try:
                passages = fut.result(timeout=3.0)
                for p in passages:
                    page_passages.append((item, p))
            except Exception as e:
                logger.debug(f"Page reading task failed: {e}")
                if item.get("snippet"):
                    page_passages.append((item, item["snippet"]))

    if not page_passages:
        return []

    # Deduplicate passages
    seen_texts = set()
    candidate_records: List[Dict[str, Any]] = []

    for item, text in page_passages:
        norm = re.sub(r'\W+', '', text.lower())[:80]
        if norm and norm not in seen_texts:
            seen_texts.add(norm)
            candidate_records.append({
                "url": item["url"],
                "domain": item["domain"],
                "title": item.get("title", ""),
                "reputation_weight": item.get("reputation_weight", 1.0),
                "text": text.strip(),
            })

    # Fast lexical pre-filtering if more than 12 candidates to keep CPU reranking sub-second
    if len(candidate_records) > 12:
        claim_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower()))
        def overlap_score(rec):
            words = set(re.findall(r'\b[a-zA-Z]{3,}\b', rec["text"].lower()))
            return len(claim_words & words) * rec["reputation_weight"]
        candidate_records.sort(key=overlap_score, reverse=True)
        candidate_records = candidate_records[:12]

    # Cross-Encoder Reranking
    cross_encoder = get_cross_encoder()
    pairs = [[claim, c["text"]] for c in candidate_records]
    scores = cross_encoder.predict(pairs)

    scored_records = []
    for c, sc in zip(candidate_records, scores):
        rel = 1.0 / (1.0 + math.exp(-sc))  # Sigmoid relevance
        weighted_score = float(sc) * c["reputation_weight"]
        scored_records.append({
            **c,
            "score": float(sc),
            "relevance": rel,
            "weighted_score": weighted_score
        })

    valid_records = [r for r in scored_records if r["score"] >= PASSAGE_RERANK_THRESHOLD]
    valid_records.sort(key=lambda x: x["weighted_score"], reverse=True)

    return valid_records[:top_k]


# ── Top-Level Retrieval Interface ─────────────────────────────────────────────
def retrieve_live_evidence_for_claim(claim: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Executes live web search + page reading + cross-encoder reranking for a claim.
    Returns the top 5-8 relevant passages with URL, domain, text, score, and reputation weight.
    """
    search_items = execute_parallel_search(claim, max_urls=8)
    passages = chunk_and_rerank_passages(claim, search_items, top_k=top_k)
    return passages
