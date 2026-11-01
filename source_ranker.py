"""
source_ranker.py — Credibility Tiering & Dense Semantic Passage Reranking
--------------------------------------------------------------------------
Phase 3 Responsibilities:
1. 4-Tier Credibility Hierarchy:
   - Tier 1: Official government / primary issuer sites (.gov, .gov.in, .nic.in, RBI, WHO, etc.)
     (.edu and .ac.in are moved to Tier 3 research/education sources)
   - Tier 2: Certified professional fact-checkers (Alt News, BOOM, Factly, Snopes, PolitiFact, etc.)
   - Tier 3: Major news agencies, established broadsheets, .edu, .ac.in, and encyclopedias
   - Tier 4: Blogs, social media, unverified forums (Twitter, Reddit, Medium, etc.)
2. Encyclopedic Flag:
   Wikipedia and Britannica are marked `is_encyclopedic=True`.
   They serve as background only and can never be the sole basis for SUPPORTED
   on a news, document, or statistic claim.
3. Relevance Filtering:
   Passages are embedded with all-MiniLM-L6-v2.
   Only passages with raw_similarity >= 0.35 are counted as evidence.
   Tier multiplier is removed from ranking relevance.
4. Date Awareness:
   Compares claim time_scope with passage publish_date.
"""

import os
import re
import urllib.parse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import RAW_SIMILARITY_THRESHOLD

# Singleton embedding model
_embedder = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── Credibility Tier Classification ──────────────────────────────────────────

TIER_1_DOMAINS = {
    "pib.gov.in", "finmin.nic.in", "rbi.org.in", "egazette.gov.in", "india.gov.in",
    "dopt.gov.in", "financialservices.gov.in", "sci.gov.in", "supremecourtofindia.nic.in",
    "eci.gov.in", "isro.gov.in", "drdo.gov.in", "who.int", "un.org", "nasa.gov",
    "cdc.gov", "whitehouse.gov", "gov.uk", "europa.eu", "mea.gov.in", "mha.gov.in"
}

TIER_2_FACTCHECKERS = {
    "altnews.in", "boomlive.in", "factly.in", "snopes.com", "politifact.com",
    "factcheck.org", "fullfact.org", "leadstories.com", "vishvasnews.com",
    "newsmobile.in", "thip.media", "logicallyfacts.com", "checkyourfact.com",
    "factcheckni.org", "afp.com/fact-check", "reuters.com/fact-check"
}

TIER_3_MAJOR_NEWS_AND_RESEARCH = {
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk", "thehindu.com",
    "indianexpress.com", "hindustantimes.com", "timesofindia.indiatimes.com",
    "ndtv.com", "indiatoday.in", "business-standard.com", "livemint.com",
    "economictimes.indiatimes.com", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "wsj.com", "bloomberg.com", "wikipedia.org", "britannica.com",
    "nature.com", "science.org"
}

TIER_4_SOCIAL_BLOGS = {
    "twitter.com", "x.com", "facebook.com", "instagram.com", "threads.net",
    "reddit.com", "youtube.com", "quora.com", "medium.com", "wordpress.com",
    "blogspot.com", "tumblr.com", "tiktok.com", "pinterest.com"
}


def classify_source_tier(url: str, domain: str) -> Tuple[int, str, bool]:
    """
    Classifies a source into Tier 1, 2, 3, or 4 and flags encyclopedias.
    Returns: (tier_num, tier_label, is_encyclopedic)
    """
    clean_domain = domain.lower().replace("www.", "")
    clean_url = url.lower()

    is_encyclopedic = "wikipedia.org" in clean_domain or "britannica.com" in clean_domain

    # Fact-check path check on major media or portals (e.g. reuters.com/fact-check, pib.gov.in/factcheck)
    if "factcheck" in clean_url or "fact-check" in clean_url or "webqoof" in clean_url:
        return 2, "Tier 2: Certified Fact-Checker", is_encyclopedic

    # Tier 1: Official government / institutional issuer ONLY (.gov, .gov.in, .nic.in, RBI, WHO, etc.)
    # Note: .edu and .ac.in are moved to Tier 3 research/education
    if (
        clean_domain in TIER_1_DOMAINS or
        clean_domain.endswith(".gov") or
        clean_domain.endswith(".gov.in") or
        clean_domain.endswith(".nic.in") or
        clean_domain.endswith(".mil")
    ):
        return 1, "Tier 1: Official Government / Issuer", False

    # Tier 2: Dedicated Fact-Checkers
    if clean_domain in TIER_2_FACTCHECKERS or any(fc in clean_domain for fc in ["factcheck", "altnews", "boomlive"]):
        return 2, "Tier 2: Certified Fact-Checker", False

    # Tier 3: Major News, Educational/Research (.edu, .ac.in), and Encyclopedias
    if (
        clean_domain in TIER_3_MAJOR_NEWS_AND_RESEARCH or
        clean_domain.endswith(".edu") or
        clean_domain.endswith(".ac.in") or
        is_encyclopedic or
        any(mn in clean_domain for mn in ["reuters", "bbc", "thehindu", "indianexpress"])
    ):
        label = "Tier 3: Encyclopedia / Background" if is_encyclopedic else "Tier 3: Major News / Academic"
        return 3, label, is_encyclopedic

    # Tier 4: Social media & blogs
    if clean_domain in TIER_4_SOCIAL_BLOGS or any(sb in clean_domain for sb in ["reddit", "twitter", "facebook", "blog"]):
        return 4, "Tier 4: Blog / Social Media / Unverified", False

    # Default unknown external web domain
    return 4, "Tier 4: General Web / Unverified", False


# ── Semantic Passage Reranking ───────────────────────────────────────────────

def rank_and_balance_passages(
    claim_text: str,
    passages: List[Dict[str, Any]],
    top_k: int = 8,
    time_scope: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Embeds passages with all-MiniLM-L6-v2, computes raw semantic similarity with claim,
    filters by threshold (raw_similarity >= 0.35), tags credibility tiers,
    and returns top balanced passages.
    """
    if not passages:
        return []

    # Tag tier and encyclopedic status for each passage
    for p in passages:
        tier_num, tier_label, is_encyclopedic = classify_source_tier(p.get("url", ""), p.get("domain", ""))
        p["tier"] = tier_num
        p["tier_name"] = tier_label
        p["is_encyclopedic"] = is_encyclopedic

    embedder = get_embedder()
    claim_emb = embedder.encode(claim_text, normalize_embeddings=True)
    
    passage_texts = [p.get("text", "")[:400] for p in passages]
    p_embs = embedder.encode(passage_texts, normalize_embeddings=True)

    # Compute raw cosine similarities
    sims = np.dot(p_embs, claim_emb)
    
    for idx, p in enumerate(passages):
        raw_sim = float(sims[idx])
        p["raw_similarity"] = round(raw_sim, 3)
        # Requirement: Do NOT use tier multiplier for relevance ranking.
        p["relevance"] = round(raw_sim, 3)

        # Date awareness check
        if time_scope and p.get("publish_date"):
            p["is_outdated"] = _is_passage_outdated(time_scope, p["publish_date"])
        else:
            p["is_outdated"] = False

    # Filter passages by relevance threshold >= 0.35
    relevant_passages = [p for p in passages if p["raw_similarity"] >= RAW_SIMILARITY_THRESHOLD]

    # If none meet threshold, preserve the single best passage to explain lack of evidence
    if not relevant_passages and passages:
        best_p = max(passages, key=lambda x: x["raw_similarity"])
        relevant_passages = [best_p]

    # Group relevant passages by tier
    tier_groups: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}
    for p in relevant_passages:
        tier_groups[p["tier"]].append(p)

    # Sort each tier group by raw relevance
    for t in tier_groups:
        tier_groups[t].sort(key=lambda x: x["raw_similarity"], reverse=True)

    # Balanced selection: prioritize Tier 1 and Tier 2, then Tier 3, then Tier 4
    selected = []
    
    # 1. Grab top Tier 1 (up to 3)
    selected.extend(tier_groups[1][:3])
    # 2. Grab top Tier 2 (up to 3)
    selected.extend(tier_groups[2][:3])
    # 3. Grab top Tier 3 (up to 3)
    selected.extend(tier_groups[3][:3])

    # If space remains, fill with remaining relevant passages
    remaining = [p for p in relevant_passages if p not in selected]
    remaining.sort(key=lambda x: x["raw_similarity"], reverse=True)

    while len(selected) < top_k and remaining:
        selected.append(remaining.pop(0))

    # Final sort of selected passages by raw relevance
    selected.sort(key=lambda x: x["raw_similarity"], reverse=True)
    return selected[:top_k]


def _is_passage_outdated(time_scope: str, publish_date: str) -> bool:
    """Checks if a publish date is significantly older than a current/recent time scope."""
    ts_lower = time_scope.lower()
    if any(w in ts_lower for w in ["currently", "current", "latest", "now", "today", "this year"]):
        match = re.search(r'\b(19\d\d|20\d\d)\b', publish_date)
        if match:
            year = int(match.group(1))
            # If current claim but source is from before 2021
            if year < 2021:
                return True
    return False
