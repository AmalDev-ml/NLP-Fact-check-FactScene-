"""
explainer.py — Comprehensive Evidence-Grounded Explanation Synthesizer
----------------------------------------------------------------------
Phase 3 Features:
1. Executes ONE focused LLM call after verdict synthesis using strictly retrieved evidence.
2. Returns a structured JSON payload:
   - verdict_summary: 1-2 sentences in plain language explaining the overall verdict.
   - what_the_claim_says: Precise statement of what the input asserted.
   - what_is_actually_true: The factual real-world scenario with numbers, dates, names,
     and sample sizes, citing evidence sources [Source 1], [Source 2], etc.
   - why_it_is_wrong_or_misleading: Point-by-point breakdown per failing claim.
   - headline_vs_body: Contrast between what the headline implies vs body/evidence shows.
   - evidence: Clean array of cited sources with tier, stance, quote, url, and date.
   - could_not_verify: Elements that could not be settled from credible sources.
   - confidence_reason: Clear justification for the confidence score.
3. If evidence does not contain the real facts, explicitly states:
   "Could not find an authoritative source for the true figure" rather than hallucinating.
4. Robust heuristic fallback when no LLM key is configured.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional

from config import get_llm, extract_text_from_response

logger = logging.getLogger(__name__)


def generate_detailed_explanation(
    overall_verdict: str,
    confidence: int,
    raw_input_text: str,
    per_claim_results: List[Dict[str, Any]],
    framing_data: Dict[str, Any],
    top_evidence_passages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Synthesizes the comprehensive, multi-section factual explanation.
    """
    clean_input = raw_input_text.strip()
    llm = get_llm()

    if llm:
        try:
            return _generate_with_llm(
                overall_verdict,
                confidence,
                clean_input,
                per_claim_results,
                framing_data,
                top_evidence_passages,
                llm
            )
        except Exception as e:
            logger.warning(f"LLM detailed explanation failed: {e}. Using structured fallback.")

    return _generate_fallback(
        overall_verdict,
        confidence,
        clean_input,
        per_claim_results,
        framing_data,
        top_evidence_passages
    )


def _generate_with_llm(
    overall_verdict: str,
    confidence: int,
    raw_input_text: str,
    per_claim_results: List[Dict[str, Any]],
    framing_data: Dict[str, Any],
    top_evidence_passages: List[Dict[str, Any]],
    llm
) -> Dict[str, Any]:
    """Uses LLM with strict grounding prompt to generate detailed explanation."""
    evidence_items = []
    for idx, p in enumerate(top_evidence_passages[:8], 1):
        domain = p.get("domain", "web")
        tier = p.get("tier", 4)
        stance = p.get("stance", "neutral")
        quote = p.get("quote") or p.get("text", "")[:180]
        url = p.get("url", "#")
        date = p.get("publish_date") or p.get("date") or "Date unspecified"
        evidence_items.append(
            f"[Source {idx}] {domain} (Tier {tier}, {date}) - Stance: {stance.upper()}\n"
            f"Quote: \"{quote}\"\nURL: {url}"
        )
    evidence_block = "\n\n".join(evidence_items) if evidence_items else "No authoritative external evidence retrieved."

    claims_summary = []
    for c in per_claim_results:
        c_text = c.get("claim") or c.get("text", "")
        c_v = c.get("verdict", "UNKNOWN")
        c_r = c.get("reason", "")
        claims_summary.append(f"- Claim: '{c_text}' -> Verdict: {c_v} (Reason: {c_r})")
    claims_block = "\n".join(claims_summary)

    framing_context = ""
    if framing_data.get("has_headline"):
        framing_context = (
            f"Headline: \"{framing_data.get('what_headline_implies')}\"\n"
            f"Body reality: \"{framing_data.get('what_body_actually_says')}\"\n"
            f"Framing issues: {', '.join(framing_data.get('framing_issues', []))}\n"
            f"Framing severity: {framing_data.get('severity')}"
        )

    prompt = f"""You are a senior investigative fact-checking editor.
Synthesize a comprehensive, plain-language, evidence-grounded explanation for the fact-checking verdict below.

OVERALL VERDICT: {overall_verdict} ({confidence}% confidence)

ORIGINAL USER INPUT:
\"\"\"{raw_input_text}\"\"\"

PER-CLAIM EVALUATIONS:
{claims_block}

FRAMING ANALYSIS:
{framing_context if framing_context else "Single statement / No separate headline."}

RETRIEVED LIVE EVIDENCE SOURCES:
{evidence_block}

CRITICAL GROUNDING RULES:
1. Rely ONLY on the provided retrieved live evidence. Do NOT invent figures, dates, or study names not in the evidence.
2. If the evidence does NOT contain the real figures or true facts, explicitly state:
   "Could not find an authoritative source for the true figure" or "The official records do not specify the alternative figure."
3. In "what_is_actually_true", cite the source IDs (e.g. "[Source 1]", "[Source 2]") for each assertion.
4. In "why_it_is_wrong_or_misleading", provide clear bullet points per failing claim: wrong number / wrong source / missing context / overstated / unsupported.

Respond ONLY with valid JSON in this exact structure:
{{
  "verdict_summary": "<1-2 sentences in plain language summarizing why this verdict was reached>",
  "what_the_claim_says": "<concise summary of the core claim/article assertion>",
  "what_is_actually_true": "<the real scenario with specific numbers, dates, names, citing [Source X] where available>",
  "why_it_is_wrong_or_misleading": [
    "<point 1 explaining error, exaggeration, or lack of context>",
    "<point 2 if applicable>"
  ],
  "headline_vs_body": "<contrast what headline implies vs body/sources show, or null if no separate headline>",
  "could_not_verify": "<aspects that could not be settled from credible sources, or null>",
  "confidence_reason": "<1 sentence explaining why confidence is high, moderate, or capped>"
}}
"""
    response = llm.invoke(prompt)
    content = extract_text_from_response(response)

    content = re.sub(r'^```json\s*', '', content.strip())
    content = re.sub(r'```$', '', content.strip())

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError("Could not parse JSON from explanation response")

    data = json.loads(match.group(0))

    # Assemble cleaned evidence array
    evidence_clean = []
    for p in top_evidence_passages[:8]:
        evidence_clean.append({
            "source": p.get("domain", "Unknown Source"),
            "tier": p.get("tier", 4),
            "tier_name": p.get("tier_name", "Tier 4: General Web"),
            "url": p.get("url", "#"),
            "short_quote": (p.get("quote") or p.get("text", ""))[:180].strip(),
            "stance": p.get("stance", "neutral"),
            "date": p.get("publish_date") or p.get("date")
        })

    return {
        "verdict_summary": str(data.get("verdict_summary", "")).strip(),
        "what_the_claim_says": str(data.get("what_the_claim_says", "")).strip(),
        "what_is_actually_true": str(data.get("what_is_actually_true", "")).strip(),
        "why_it_is_wrong_or_misleading": data.get("why_it_is_wrong_or_misleading", []),
        "headline_vs_body": data.get("headline_vs_body"),
        "evidence": evidence_clean,
        "could_not_verify": data.get("could_not_verify"),
        "confidence_reason": str(data.get("confidence_reason", "")).strip()
    }


def _generate_fallback(
    overall_verdict: str,
    confidence: int,
    raw_input_text: str,
    per_claim_results: List[Dict[str, Any]],
    framing_data: Dict[str, Any],
    top_evidence_passages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generates structured explanation when LLM is unavailable."""
    failed_claims = [c for c in per_claim_results if c.get("verdict") in ("REFUTED", "MISLEADING")]
    supported_claims = [c for c in per_claim_results if c.get("verdict") == "SUPPORTED"]

    why_reasons = []
    for fc in failed_claims:
        reason = fc.get("reason", "Contradicted by authoritative records.")
        why_reasons.append(f"Claim '{fc.get('claim', '')[:60]}': {reason}")

    if framing_data.get("framing_issues"):
        for issue in framing_data["framing_issues"]:
            why_reasons.append(f"Framing issue: {issue}")

    if overall_verdict == "SUPPORTED":
        summary = "All central assertions are corroborated by verified live web sources and official records."
        actual_true = "Official documents and reputable news agencies confirm the factual truth of this claim."
    elif overall_verdict == "REFUTED":
        summary = "The central claims were refuted or exposed as false by official authorities or professional fact-checkers."
        actual_true = "Official records or verified fact-checks explicitly deny the authenticity or truth of the statement."
    elif overall_verdict == "MISLEADING":
        summary = "The input mixes verified elements with false, exaggerated, or out-of-context assertions."
        actual_true = "While some background details may be accurate, the primary conclusion or headline is distorted or unproven."
    else:
        summary = "No credible official sources or verified reports exist to either confirm or refute this claim."
        actual_true = "Could not find an authoritative source for the true figure or factual claim."

    evidence_clean = []
    for p in top_evidence_passages[:8]:
        evidence_clean.append({
            "source": p.get("domain", "Unknown Source"),
            "tier": p.get("tier", 4),
            "tier_name": p.get("tier_name", "Tier 4: General Web"),
            "url": p.get("url", "#"),
            "short_quote": (p.get("quote") or p.get("text", ""))[:180].strip(),
            "stance": p.get("stance", "neutral"),
            "date": p.get("publish_date") or p.get("date")
        })

    hl_vs_body = None
    if framing_data.get("has_headline"):
        hl_vs_body = (
            f"Headline implies: {framing_data.get('what_headline_implies')}. "
            f"Body and evidence show: {framing_data.get('what_body_actually_says')}."
        )

    return {
        "verdict_summary": summary,
        "what_the_claim_says": raw_input_text[:140],
        "what_is_actually_true": actual_true,
        "why_it_is_wrong_or_misleading": why_reasons if why_reasons else ["No specific factual contradictions detected."],
        "headline_vs_body": hl_vs_body,
        "evidence": evidence_clean,
        "could_not_verify": "Peripheral details not covered in the retrieved passages." if overall_verdict != "SUPPORTED" else None,
        "confidence_reason": f"Computed from source tier authority, relevance scores, and multi-source agreement ({confidence}% confidence)."
    }

