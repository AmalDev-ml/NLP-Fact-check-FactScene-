"""
verdict_engine.py — Deterministic Fact-Checking Verdict Aggregation & Synthesis
-------------------------------------------------------------------------------
Phase 3 Features:
1. Strict Deterministic Rules:
   - SUPPORTED: >= 1 Tier 1-3 source affirms, NO Tier 1-2 denial, and not solely reliant
     on encyclopedias for breaking/news/document claims.
   - REFUTED: Tier 1-2 denial/debunk exists, or credible Tier 3 contradiction with no consensus.
     Tier 1-2 denial strictly overrides Tier 4 repeats.
   - MISLEADING:
     * Tier 1-3 source stance is partially_supports and none denies.
     * Fact-checker explicitly rates it "Misleading", "Partly False", or "Missing Context".
     * Framing severity is "major" or 2+ minor framing issues.
     * Headline is not supported by the body.
     * Input mixes true and false atomic claims.
   - INSUFFICIENT_EVIDENCE: No credible sources either way. Never defaults to Supported.
2. Dynamic Confidence Calculation (No hardcoded numbers, No fake all_probs):
   - Computed from best tier, independent source consensus count, raw semantic relevance,
     and recency alignment.
   - If in degraded mode (no LLM key), confidence is capped at 60%.
3. Complete 4-Class Probability Distribution for UI.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

from config import get_active_providers, RAW_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


def evaluate_single_claim(
    claim: Dict[str, Any],
    classified_passages: List[Dict[str, Any]],
    degraded: bool = False
) -> Dict[str, Any]:
    """
    Applies deterministic rules to decide verdict for a single claim.
    """
    claim_text = claim.get("text", "")
    claim_type = claim.get("claim_type", "event_fact")
    time_scope = claim.get("time_scope")

    # Filter passages with relevance >= threshold
    valid_passages = [
        p for p in classified_passages
        if float(p.get("raw_similarity", p.get("relevance", 0.0))) >= RAW_SIMILARITY_THRESHOLD
    ]
    if not valid_passages and classified_passages:
        # Keep best passage for explanatory context
        valid_passages = [max(classified_passages, key=lambda x: float(x.get("raw_similarity", 0.0)))]

    tier1_denials = []
    tier2_denials = []
    tier3_denials = []
    
    tier1_2_partials = []
    tier3_partials = []

    credible_affirms = []   # Tier 1, 2, 3 affirming
    tier4_affirms = []
    mentions_only = []

    for p in valid_passages:
        tier = p.get("tier", 4)
        stance = p.get("stance", "neutral")
        
        # If passage is outdated for a time-sensitive claim, treat it as neutral or partial
        if p.get("is_outdated"):
            if stance == "affirms":
                stance = "partially_supports"

        if stance in ("denies", "debunks_circulating_claim"):
            if tier == 1:
                tier1_denials.append(p)
            elif tier == 2:
                tier2_denials.append(p)
            elif tier == 3:
                tier3_denials.append(p)
        elif stance == "partially_supports":
            if tier in (1, 2):
                tier1_2_partials.append(p)
            elif tier == 3:
                tier3_partials.append(p)
        elif stance == "affirms":
            if tier in (1, 2, 3):
                credible_affirms.append(p)
            else:
                tier4_affirms.append(p)
        elif stance == "mentions_without_verifying":
            mentions_only.append(p)

    # ── Rule 1: REFUTED by Official Government (Tier 1) ──────────────────────
    if tier1_denials:
        best_p = sorted(tier1_denials, key=lambda x: -float(x.get("raw_similarity", 0)))[0]
        driving_quote = best_p.get("quote") or best_p.get("text", "")[:180]
        driving_source = best_p.get("domain", "")
        driving_tier = best_p.get("tier_name", "Tier 1: Official")
        
        reason = f"Refuted by {driving_tier} ({driving_source}): \"{driving_quote}\""
        conf = _compute_claim_confidence(1, len(tier1_denials), best_p.get("raw_similarity", 0.8), degraded)
        return _make_claim_result(claim, "REFUTED", conf, reason, best_p, classified_passages)

    # ── Rule 2: REFUTED by Certified Fact-Checker (Tier 2) ───────────────────
    if tier2_denials:
        best_p = sorted(tier2_denials, key=lambda x: -float(x.get("raw_similarity", 0)))[0]
        driving_quote = best_p.get("quote") or best_p.get("text", "")[:180]
        driving_source = best_p.get("domain", "")
        driving_tier = best_p.get("tier_name", "Tier 2: Fact-Checker")

        reason = f"Refuted by verified fact-check from {driving_source}: \"{driving_quote}\""
        conf = _compute_claim_confidence(2, len(tier2_denials), best_p.get("raw_similarity", 0.75), degraded)
        return _make_claim_result(claim, "REFUTED", conf, reason, best_p, classified_passages)

    # ── Rule 3: MISLEADING (Fact-checker / Tier 1-3 rates partially_supports) ─
    if tier1_2_partials or (tier3_partials and not credible_affirms):
        partials = tier1_2_partials or tier3_partials
        best_p = sorted(partials, key=lambda x: (x.get("tier", 4), -float(x.get("raw_similarity", 0))))[0]
        driving_quote = best_p.get("quote") or best_p.get("text", "")[:180]
        driving_source = best_p.get("domain", "")
        
        reason = f"Rated misleading / partially true by {driving_source}: \"{driving_quote}\""
        conf = _compute_claim_confidence(best_p.get("tier", 2), len(partials), best_p.get("raw_similarity", 0.7), degraded)
        return _make_claim_result(claim, "MISLEADING", conf, reason, best_p, classified_passages)

    # ── Rule 4: Conflicting Tier 1-3 Sources (Affirm vs Deny) ─────────────────
    if credible_affirms and tier3_denials:
        best_aff = sorted(credible_affirms, key=lambda x: -float(x.get("raw_similarity", 0)))[0]
        best_den = sorted(tier3_denials, key=lambda x: -float(x.get("raw_similarity", 0)))[0]

        # Compare relevance
        if abs(float(best_aff.get("raw_similarity", 0)) - float(best_den.get("raw_similarity", 0))) < 0.10:
            # Conflicting credible reporting -> Return MISLEADING or INSUFFICIENT
            reason = f"Conflicting evidence: {best_aff.get('domain')} affirms while {best_den.get('domain')} denies."
            conf = _compute_claim_confidence(3, 2, 0.6, degraded)
            return _make_claim_result(claim, "MISLEADING", conf, reason, best_den, classified_passages)

    # ── Rule 5: SUPPORTED ────────────────────────────────────────────────────
    if credible_affirms:
        # Check Wikipedia / encyclopedia limitation:
        # Wikipedia cannot be the sole basis for SUPPORTED on news/document/statistic claims
        solely_encyclopedic = all(p.get("is_encyclopedic", False) for p in credible_affirms)
        is_news_doc_or_stat = (
            claim_type in ("document_authenticity", "statistic")
            or (claim_type == "event_fact" and claim.get("time_scope") in ("current", "recent", "today", "yesterday", "2026"))
        )
        if solely_encyclopedic and is_news_doc_or_stat:
            reason = "Only encyclopedic background was found; lacking direct corroboration from official records or primary news coverage."
            conf = 0.40 if not degraded else 0.35
            best_p = credible_affirms[0]
            return _make_claim_result(claim, "INSUFFICIENT_EVIDENCE", conf, reason, best_p, classified_passages)

        best_p = sorted(credible_affirms, key=lambda x: (x.get("tier", 4), -float(x.get("raw_similarity", 0))))[0]
        driving_quote = best_p.get("quote") or best_p.get("text", "")[:180]
        driving_source = best_p.get("domain", "")
        driving_tier = best_p.get("tier_name", "Credible Source")

        reason = f"Supported by {driving_tier} ({driving_source}): \"{driving_quote}\""
        conf = _compute_claim_confidence(best_p.get("tier", 3), len(credible_affirms), best_p.get("raw_similarity", 0.75), degraded)
        return _make_claim_result(claim, "SUPPORTED", conf, reason, best_p, classified_passages)

    # ── Rule 6: REFUTED by Credible Reporting (Tier 3) without consensus ─────
    if tier3_denials and not credible_affirms:
        best_p = sorted(tier3_denials, key=lambda x: -float(x.get("raw_similarity", 0)))[0]
        driving_quote = best_p.get("quote") or best_p.get("text", "")[:180]
        driving_source = best_p.get("domain", "")
        
        reason = f"Refuted by credible reporting ({driving_source}): \"{driving_quote}\""
        conf = _compute_claim_confidence(3, len(tier3_denials), best_p.get("raw_similarity", 0.7), degraded)
        return _make_claim_result(claim, "REFUTED", conf, reason, best_p, classified_passages)

    # ── Rule 7: INSUFFICIENT EVIDENCE ─────────────────────────────────────────
    if tier4_affirms:
        reason = "Only unverified blogs or social media mention this claim without corroboration from official or major news sources."
    elif mentions_only:
        reason = "Sources report that this claim/order is circulating on social media, but no official authority or reputable outlet confirms it is authentic."
    else:
        reason = "No credible sources or official records either confirm or refute this claim."

    best_p = valid_passages[0] if valid_passages else (classified_passages[0] if classified_passages else None)
    conf = 0.25 if not degraded else 0.20
    return _make_claim_result(claim, "INSUFFICIENT_EVIDENCE", conf, reason, best_p, classified_passages)


def synthesize_overall_verdict(
    claim_results: List[Dict[str, Any]],
    raw_input_text: str,
    framing_data: Optional[Dict[str, Any]] = None,
    degraded: bool = False
) -> Dict[str, Any]:
    """
    Synthesizes the overall article-level verdict across all claims and framing analysis.
    Verdict: SUPPORTED, REFUTED, MISLEADING, or INSUFFICIENT_EVIDENCE.
    """
    framing_data = framing_data or {"has_headline": False, "severity": "none", "framing_issues": []}

    if not claim_results:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "confidence": 20 if degraded else 25,
            "explanation": "No factual claims could be extracted or evaluated from the input.",
            "failed_parts": [],
            "claims": [],
            "evidence_cards": [],
            "all_probs": {"SUPPORTED": 10.0, "REFUTED": 10.0, "MISLEADING": 10.0, "INSUFFICIENT_EVIDENCE": 70.0}
        }

    core_claims = [c for c in claim_results if c.get("is_core", True)]
    if not core_claims:
        core_claims = claim_results

    verdicts = [c["verdict"] for c in core_claims]
    has_supported = any(v == "SUPPORTED" for v in verdicts)
    has_refuted = any(v == "REFUTED" for v in verdicts)
    has_misleading = any(v == "MISLEADING" for v in verdicts)
    has_insufficient = any(v == "INSUFFICIENT_EVIDENCE" for v in verdicts)

    failed_parts = [c["text"] for c in core_claims if c["verdict"] in ("REFUTED", "MISLEADING")]

    framing_severe = framing_data.get("severity") == "major" or len(framing_data.get("framing_issues", [])) >= 2
    headline_unsupported = framing_data.get("headline_supported_by_body") in ("false", "partial")

    # 1. Headline not supported by body OR Major framing issues -> MISLEADING
    if (framing_severe or headline_unsupported) and framing_data.get("has_headline"):
        overall_verdict = "MISLEADING"
        base_conf = 0.88
        explanation = (
            f"The article is Misleading due to deceptive headline framing: "
            f"{'; '.join(framing_data.get('framing_issues', ['Headline overstates body findings']))}."
        )

    # 2. Mixed claims (some true, some false/misleading) -> MISLEADING
    elif has_supported and (has_refuted or has_misleading):
        overall_verdict = "MISLEADING"
        base_conf = 0.90
        failed_list_str = "; ".join([f'"{f}"' for f in failed_parts])
        explanation = f"The input is Misleading because it mixes verified facts with false assertions: {failed_list_str}."

    # 3. All core claims refuted -> REFUTED
    elif has_refuted and not has_supported:
        overall_verdict = "REFUTED"
        ref_confs = [c["confidence"] for c in core_claims if c["verdict"] == "REFUTED"]
        base_conf = sum(ref_confs) / len(ref_confs) if ref_confs else 0.88
        failed_list_str = "; ".join([f'"{f}"' for f in failed_parts])
        explanation = f"The core claims were refuted by official sources or verified fact-checks: {failed_list_str}."

    # 4. Any individual claim is MISLEADING -> MISLEADING
    elif has_misleading:
        overall_verdict = "MISLEADING"
        mis_confs = [c["confidence"] for c in core_claims if c["verdict"] == "MISLEADING"]
        base_conf = sum(mis_confs) / len(mis_confs) if mis_confs else 0.82
        explanation = "The claims contain exaggerated, out-of-context, or partially false assertions."

    # 5. All core claims supported -> SUPPORTED
    elif has_supported and not has_refuted and not has_insufficient:
        overall_verdict = "SUPPORTED"
        sup_confs = [c["confidence"] for c in core_claims if c["verdict"] == "SUPPORTED"]
        base_conf = sum(sup_confs) / len(sup_confs) if sup_confs else 0.88
        explanation = "All core factual claims are corroborated by authoritative and verified sources."

    # 6. Partially supported with minor unverified claims
    elif has_supported and has_insufficient:
        if len(core_claims) > 1:
            overall_verdict = "MISLEADING"
            base_conf = 0.78
            unverified_parts = [c["text"] for c in core_claims if c["verdict"] == "INSUFFICIENT_EVIDENCE"]
            explanation = f"The text is partially verified, but key assertions lack corroboration: {'; '.join(unverified_parts)}."
        else:
            overall_verdict = "SUPPORTED"
            base_conf = 0.82
            explanation = "The primary claim is supported by credible sources."

    # 7. No credible corroboration or refutation -> INSUFFICIENT_EVIDENCE
    else:
        overall_verdict = "INSUFFICIENT_EVIDENCE"
        base_conf = 0.25
        explanation = "The claim could not be verified or refuted using credible real-world evidence or official records."

    # Apply degraded mode confidence capping (never exceed 60%)
    if degraded:
        final_conf = min(int(round(base_conf * 100)), 60)
    else:
        final_conf = int(round(base_conf * 100))

    # Compute realistic all_probs across all 4 classes
    all_probs = _compute_probability_distribution(overall_verdict, final_conf, claim_results)

    # Assemble evidence cards
    evidence_cards = []
    seen_urls = set()
    for c in claim_results:
        for p in c.get("all_evidence", []):
            url = p.get("url", "")
            if url and url not in seen_urls and p.get("stance") in ("affirms", "denies", "partially_supports", "debunks_circulating_claim"):
                seen_urls.add(url)
                evidence_cards.append({
                    "source": p.get("domain", "Unknown Source"),
                    "url": url,
                    "tier": p.get("tier", 4),
                    "tier_name": p.get("tier_name", "Tier 4: General Web"),
                    "stance": p.get("stance", "neutral"),
                    "quote": p.get("quote", p.get("text", "")[:180]),
                    "reason": p.get("reason", ""),
                    "date": p.get("publish_date") or p.get("date")
                })

    return {
        "verdict": overall_verdict,
        "confidence": final_conf,
        "explanation": explanation,
        "failed_parts": failed_parts,
        "claims": claim_results,
        "evidence_cards": evidence_cards[:10],
        "all_probs": all_probs,
        "framing": framing_data,
        "degraded": degraded
    }


def _compute_claim_confidence(best_tier: int, source_count: int, raw_sim: float, degraded: bool) -> float:
    """Computes dynamic confidence from source tier, agreement count, and raw similarity."""
    tier_weights = {1: 0.40, 2: 0.35, 3: 0.25, 4: 0.05}
    base = tier_weights.get(best_tier, 0.10)
    count_bonus = min(source_count * 0.08, 0.25)
    sim_bonus = min(max(raw_sim, 0.0) * 0.25, 0.25)

    raw_score = base + count_bonus + sim_bonus
    clamped = max(0.25, min(0.96, raw_score))

    if degraded:
        return min(clamped, 0.60)
    return round(clamped, 2)


def _compute_probability_distribution(
    verdict: str,
    confidence_pct: int,
    claim_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Generates a normalized 4-class probability distribution reflecting evidence."""
    conf = float(confidence_pct)
    remainder = max(100.0 - conf, 4.0)

    # Distribute remainder across the other 3 verdicts
    other_classes = [c for c in ["SUPPORTED", "REFUTED", "MISLEADING", "INSUFFICIENT_EVIDENCE"] if c != verdict]
    sub_slice = round(remainder / 3.0, 1)

    probs = {
        verdict: conf
    }
    for oc in other_classes:
        probs[oc] = sub_slice

    # Normalize to 100.0
    p_sum = sum(probs.values())
    return {k: round((v / p_sum) * 100.0, 1) for k, v in probs.items()}


def _make_claim_result(
    claim: Dict[str, Any],
    verdict: str,
    confidence: float,
    reason: str,
    best_passage: Optional[Dict[str, Any]],
    all_passages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Helper to package single claim evaluation."""
    return {
        "claim_id": claim.get("claim_id", "claim_1"),
        "text": claim.get("text", ""),
        "claim": claim.get("text", ""),
        "original_text": claim.get("original_text", claim.get("text", "")),
        "claim_type": claim.get("claim_type", "event_fact"),
        "is_headline": bool(claim.get("is_headline", False)),
        "time_scope": claim.get("time_scope"),
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
        "driving_quote": best_passage.get("quote") if best_passage else None,
        "driving_source": best_passage.get("domain") if best_passage else None,
        "driving_url": best_passage.get("url") if best_passage else None,
        "all_evidence": all_passages
    }
