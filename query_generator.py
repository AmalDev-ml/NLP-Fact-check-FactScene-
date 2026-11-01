"""
query_generator.py — Multi-Angle Search Query Generation
--------------------------------------------------------
For each core claim, generates 4 targeted query types:
  a) neutral: The verbatim factual claim or clean entity-predicate search.
  b) debunk: Claim combined with debunk terms ("fake", "fact check", "false", "hoax").
  c) official: Targeted official domains using `site:` syntax
     (pib.gov.in, rbi.org.in, egazette.gov.in, finmin.nic.in, .gov.in, .gov, etc.).
  d) fact_checker: Query targeted at certified fact-checking organizations
     (PIB Fact Check, Alt News, BOOM, Snopes, Reuters, AFP, Factly).
"""

import re
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Standard certified fact checking outlets
FACT_CHECK_ORGANIZATIONS = [
    "PIB Fact Check", "Alt News", "BOOM", "Snopes", "Reuters", "AFP", "Factly"
]

# Official portals mapping by keywords/entities
OFFICIAL_PORTAL_MAP = {
    "finance": ["pib.gov.in", "finmin.nic.in", "rbi.org.in", "egazette.gov.in"],
    "bank": ["rbi.org.in", "pib.gov.in", "financialservices.gov.in"],
    "gazette": ["egazette.gov.in", "pib.gov.in"],
    "holiday": ["pib.gov.in", "rbi.org.in", "dopt.gov.in"],
    "india": ["pib.gov.in", "india.gov.in", "egazette.gov.in"],
    "health": ["who.int", "mohfw.gov.in", "cdc.gov"],
    "space": ["nasa.gov", "isro.gov.in"],
    "election": ["eci.gov.in", "pib.gov.in"],
    "court": ["sci.gov.in", "supremecourtofindia.nic.in"],
}


def clean_query_text(text: str) -> str:
    """Removes quotes, punctuation, and question marks from search query."""
    cleaned = re.sub(r'["\?\!\;]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def generate_queries(claim: Dict[str, Any]) -> Dict[str, str]:
    """
    Generates 4 distinct query types for a claim dictionary:
      Returns:
        {
          "neutral": str,
          "debunk": str,
          "official": str,
          "fact_checker": str
        }
    """
    claim_text = claim.get("text", "")
    claim_type = claim.get("claim_type", "event_fact")
    issuer = claim.get("issuer") or ""
    
    clean_base = clean_query_text(claim_text)

    # 1. Neutral Query
    neutral_query = clean_base

    # Check if claim asserts a specific ordinal, role, or title (e.g. "first Prime Minister of India")
    role_match = re.search(r'\b(first|second|third|longest|earliest|youngest|founder|inventor)\s+([A-Za-z\s]+)', clean_base, re.IGNORECASE)
    role_query = None
    if role_match:
        role_phrase = role_match.group(0).strip().rstrip(".")
        if len(role_phrase.split()) >= 2:
            role_query = role_phrase

    # 2. Debunk Query
    debunk_query = f"{clean_base} fake fact check false hoax"

    # 3. Official Source Query
    # Determine appropriate site filters
    sites: List[str] = []
    text_lower = (claim_text + " " + issuer).lower()
    
    for kw, domains in OFFICIAL_PORTAL_MAP.items():
        if kw in text_lower:
            sites.extend(domains)

    if not sites:
        if "india" in text_lower or ".in" in text_lower:
            sites = ["pib.gov.in", "india.gov.in", "egazette.gov.in"]
        else:
            sites = ["pib.gov.in", "who.int", "nasa.gov", "gov"]

    # Deduplicate and limit to top 2 site filters
    unique_sites = list(dict.fromkeys(sites))[:2]
    site_filter_clause = " OR ".join([f"site:{s}" for s in unique_sites])
    
    # Strip very long sentences down to core topic for official site search
    words = clean_base.split()
    if len(words) > 8:
        substantive = [w for w in words if len(w) > 3 and w.lower() not in {"this", "that", "every", "with", "from", "have", "been"}][:6]
        official_base = " ".join(substantive)
    else:
        official_base = clean_base

    official_query = f"{official_base} {site_filter_clause}"

    # 4. Fact-Checker Query
    if role_query:
        # Use role query for fact checker to find true holder of title
        fact_checker_query = f"{role_query} fact check"
    else:
        fact_checker_targets = " OR ".join([f'"{org}"' for org in FACT_CHECK_ORGANIZATIONS[:3]])
        fact_checker_query = f"{clean_base} ({fact_checker_targets})"

    queries = {
        "neutral": neutral_query,
        "debunk": debunk_query,
        "official": official_query,
        "fact_checker": fact_checker_query
    }
    if role_query:
        queries["role"] = role_query
    return queries
