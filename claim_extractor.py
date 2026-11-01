"""
claim_extractor.py — Context-Aware Claim Extraction, Headline Detection & Normalization
---------------------------------------------------------------------------------------
Phase 3 Features:
1. Ingests full text input (single claim or multi-sentence article).
2. Detects headline (first line under ~25 words with no ending period, or text
   separated by a blank line). If no headline, treats the first sentence as lead claim.
3. Uses LLM structured output (or robust heuristic) to break text into atomic claims.
   Max claims raised to 6, always keeping the headline claim.
4. Tags each claim with:
   - claim_type: event_fact | document_authenticity | circulation_fact | statistic | quote | other
   - is_headline: bool
   - time_scope: Optional[str] ("in 2024", "currently", "yesterday", "in 1947", etc.)
5. Circulation vs Reality Principle:
   Converts viral/purported circulars into document_authenticity queries.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from config import get_llm, extract_text_from_response, MAX_CLAIMS_DEFAULT

logger = logging.getLogger(__name__)

# Heuristic indicator words for viral / circulating claims
VIRAL_KEYWORDS = {
    "circulating", "viral", "purportedly", "purported", "alleged", "allegedly",
    "social media", "whatsapp", "facebook post", "fake circular", "rumor", "rumour"
}

TIME_PATTERNS = [
    r'\b(?:in\s+(?:19\d\d|20\d\d))\b',
    r'\b(?:since\s+(?:19\d\d|20\d\d))\b',
    r'\b(?:in\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+(?:19\d\d|20\d\d))?)\b',
    r'\b(?:currently|current|today|yesterday|latest|now|recently|this year|last year)\b',
    r'\b(?:19\d\d|20\d\d)\b',
]


def extract_time_scope(text: str) -> Optional[str]:
    """Extracts explicit temporal scopes from claim text."""
    for pattern in TIME_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def detect_headline(text: str) -> Tuple[Optional[str], str, bool]:
    """
    Detects if input text begins with a headline.
    Returns (headline_text, remaining_body, has_headline).
    """
    clean = text.strip()
    if not clean:
        return None, "", False

    # Check for blank line separation (paragraph break)
    paragraphs = re.split(r'\n\s*\n', clean, maxsplit=1)
    if len(paragraphs) == 2:
        candidate = paragraphs[0].strip()
        body = paragraphs[1].strip()
        words = candidate.split()
        if len(words) <= 30 and not candidate.endswith('.'):
            return candidate, body, True
        if len(words) <= 25:
            return candidate, body, True

    # Check single newline separation
    lines = [line.strip() for line in clean.split('\n') if line.strip()]
    if len(lines) >= 2:
        first_line = lines[0]
        words = first_line.split()
        if len(words) <= 25 and not first_line.endswith('.'):
            body = "\n".join(lines[1:]).strip()
            return first_line, body, True

    # No explicit headline format: treat the first sentence as lead claim
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    lead_sentence = sentences[0].strip() if sentences else clean
    return lead_sentence, clean, False


def extract_claims_and_headline(input_text: str, max_claims: int = MAX_CLAIMS_DEFAULT) -> Dict[str, Any]:
    """
    Extracts atomic claims and explicitly identifies the headline claim.
    Returns:
      {
        "headline": str or None,
        "has_headline": bool,
        "headline_claim": Dict[str, Any],
        "claims": List[Dict[str, Any]],
        "body_claims": List[Dict[str, Any]]
      }
    """
    clean_text = input_text.strip()
    if not clean_text:
        return {
            "headline": None,
            "has_headline": False,
            "headline_claim": None,
            "claims": [],
            "body_claims": []
        }

    headline_str, body_str, has_headline = detect_headline(clean_text)

    llm = get_llm()
    claims = []
    if llm:
        try:
            claims = _extract_claims_with_llm(clean_text, llm, max_claims=max_claims, detected_headline=headline_str if has_headline else None)
        except Exception as e:
            logger.warning(f"LLM claim extraction failed, falling back to heuristic: {e}")

    if not claims:
        claims = _extract_claims_heuristic(clean_text, max_claims=max_claims, detected_headline=headline_str if has_headline else None)

    # Identify or tag the headline claim
    headline_claim = None
    if has_headline and headline_str:
        # Check if one of the extracted claims matches or represents the headline
        for c in claims:
            if c.get("is_headline") or headline_str.lower() in c["text"].lower() or c["text"].lower() in headline_str.lower():
                c["is_headline"] = True
                headline_claim = c
                break

        if not headline_claim:
            # Prepend headline as claim_0
            headline_claim = {
                "claim_id": "claim_headline",
                "text": headline_str,
                "original_text": headline_str,
                "claim_type": "event_fact",
                "is_core": True,
                "is_headline": True,
                "time_scope": extract_time_scope(headline_str),
                "issuer": None
            }
            claims.insert(0, headline_claim)
    else:
        # First claim is the lead claim
        if claims:
            claims[0]["is_headline"] = True
            headline_claim = claims[0]

    # Enforce time_scope on all claims if not already set
    for c in claims:
        if not c.get("time_scope"):
            c["time_scope"] = extract_time_scope(c.get("text", "") + " " + c.get("original_text", ""))

    body_claims = [c for c in claims if c != headline_claim]

    return {
        "headline": headline_str if has_headline else None,
        "has_headline": has_headline,
        "headline_claim": headline_claim,
        "claims": claims[:max_claims],
        "body_claims": body_claims
    }


def extract_claims(input_text: str, max_claims: int = MAX_CLAIMS_DEFAULT) -> List[Dict[str, Any]]:
    """Backward-compatible entry point returning the list of extracted claims."""
    extracted = extract_claims_and_headline(input_text, max_claims=max_claims)
    return extracted.get("claims", [])


def _extract_claims_with_llm(text: str, llm, max_claims: int = 6, detected_headline: Optional[str] = None) -> List[Dict[str, Any]]:
    """Uses LLM to extract atomic claims with structured JSON schema."""
    headline_clause = f"Detected Headline/Title: \"\"\"{detected_headline}\"\"\"\nAlways include this headline as a primary claim." if detected_headline else ""

    prompt = f"""You are an expert fact-checking claim extraction analyst.
Read the ENTIRE input text provided below. Analyze it as ONE cohesive document with shared context.

{headline_clause}
Input Text:
\"\"\"{text}\"\"\"

CRITICAL INSTRUCTIONS:
1. Identify and verify REAL-WORLD FACTUALITY, not social media circulation:
   - If the text mentions that a document, notification, order, gazette, video, or claim is "circulating on social media", "viral", "purportedly issued", or "allegedly announced", DO NOT verify whether a post or image exists.
   - The CORE claim must test DOCUMENT AUTHENTICITY / REAL-WORLD FACTUALITY:
     e.g., "The Ministry of Finance officially issued a Gazette declaring every Saturday a public holiday for PSU banks."
   - Tag such claims with claim_type: "document_authenticity".
   - If the input says "...is fake" or "...is false", the underlying proposition being checked is still the factual validity of that claim.
2. Break multi-sentence input into atomic, self-contained claims. Rewrite each claim so it can stand completely on its own (resolving pronouns, adding who/what/when/where).
3. If a headline exists, include the headline claim as the first claim (with is_headline: true).
4. Extract date or temporal scope ("in 2024", "currently", "in 1947", "yesterday", or null).
5. If the input contains multiple separate factual statements, extract each as a separate core atomic claim.
6. Limit to the top {max_claims} most important factual claims.

Respond ONLY with valid JSON in this exact structure:
{{
  "claims": [
    {{
      "claim_id": "claim_1",
      "text": "<rewritten standalone factual claim to verify in the real world>",
      "original_text": "<matching sentence from the input>",
      "claim_type": "event_fact" | "document_authenticity" | "circulation_fact" | "statistic" | "quote" | "other",
      "is_core": true,
      "is_headline": true | false,
      "time_scope": "<date or temporal scope or null>",
      "issuer": "<issuer or authority if document_authenticity, else null>"
    }}
  ]
}}
"""
    response = llm.invoke(prompt)
    content = extract_text_from_response(response)

    # Clean markdown json code blocks if present
    content = re.sub(r'^```json\s*', '', content.strip())
    content = re.sub(r'```$', '', content.strip())

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM extraction response")

    data = json.loads(match.group(0))
    raw_claims = data.get("claims", [])
    
    validated_claims = []
    for idx, c in enumerate(raw_claims, 1):
        claim_text = str(c.get("text", "")).strip()
        if not claim_text:
            continue
        
        c_type = str(c.get("claim_type", "event_fact")).lower()
        if c_type not in {"event_fact", "document_authenticity", "circulation_fact", "statistic", "quote", "other"}:
            c_type = "event_fact"

        is_core = bool(c.get("is_core", True))
        if c_type == "circulation_fact":
            is_core = False

        time_scope = c.get("time_scope")
        if not time_scope or time_scope == "null":
            time_scope = extract_time_scope(claim_text + " " + str(c.get("original_text", "")))

        validated_claims.append({
            "claim_id": c.get("claim_id") or f"claim_{idx}",
            "text": claim_text,
            "original_text": str(c.get("original_text", "")).strip() or claim_text,
            "claim_type": c_type,
            "is_core": is_core,
            "is_headline": bool(c.get("is_headline", False)),
            "time_scope": time_scope,
            "issuer": c.get("issuer")
        })

    if validated_claims and not any(c["is_core"] for c in validated_claims):
        validated_claims[0]["is_core"] = True

    return validated_claims[:max_claims]


def _extract_claims_heuristic(text: str, max_claims: int = 6, detected_headline: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fallback deterministic heuristic extraction if LLM is unavailable.
    """
    lower = text.lower()
    has_viral = any(vk in lower for vk in VIRAL_KEYWORDS)

    if has_viral:
        issuer_match = re.search(
            r'(?:issued\s+by|from)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+(?:Ministry|Department|Bank|RBI|Government|Authority))',
            text, re.IGNORECASE
        )
        issuer = issuer_match.group(1).strip() if issuer_match else None
        
        cleaned_core = re.sub(
            r'^(?:a|an)\s+(?:gazette|notification|circular|video|post|order)\s+circulating\s+on\s+social\s+media,?\s*(?:purportedly\s+issued\s+by\s+[^,]+,?)?\s*(?:declares|states|claims|that)?',
            '', text, flags=re.IGNORECASE
        ).strip()
        cleaned_core = re.sub(r'\s+(?:is\s+fake|is\s+false|is\s+a\s+hoax)\.?$', '', cleaned_core, flags=re.IGNORECASE).strip()
        
        if cleaned_core:
            claim_text = cleaned_core
            if issuer and not claim_text.lower().startswith(issuer.lower()):
                claim_text = f"The {issuer} has declared that {cleaned_core}"
        else:
            claim_text = text

        return [{
            "claim_id": "claim_1",
            "text": claim_text,
            "original_text": text,
            "claim_type": "document_authenticity",
            "is_core": True,
            "is_headline": True if detected_headline else False,
            "time_scope": extract_time_scope(text),
            "issuer": issuer
        }]

    # Multi-sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]

    if not sentences:
        sentences = [text]

    claims = []
    for idx, s in enumerate(sentences[:max_claims], 1):
        is_hl = False
        if detected_headline and (detected_headline.lower() in s.lower() or s.lower() in detected_headline.lower()):
            is_hl = True
        elif idx == 1:
            is_hl = True

        claims.append({
            "claim_id": f"claim_{idx}",
            "text": s,
            "original_text": s,
            "claim_type": "event_fact",
            "is_core": True,
            "is_headline": is_hl,
            "time_scope": extract_time_scope(s),
            "issuer": None
        })

    return claims
