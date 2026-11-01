"""
detector.py — Orchestrator for FactSense Phase 3 Fact Verification Pipeline
---------------------------------------------------------------------------
Phase 3 Features:
1. Ingests full text with headline detection (claim_extractor.py).
2. Parallel live search and tiered page reading (web_search.py).
3. 4-Tier credibility ranking and raw similarity filtering (source_ranker.py).
4. Batched pairwise stance classification including "partially_supports" (stance_classifier.py).
5. Headline vs body framing analysis with checklist (framing_checker.py).
6. Deterministic verdict synthesis and dynamic confidence computation (verdict_engine.py).
7. Comprehensive factual explanation (explainer.py).
8. Emits SSE stages with early 'verdict' event before 'complete'.
9. Degraded mode detection and confidence capping if no LLM key is configured.
"""

import os
import re
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import get_active_providers, RAW_SIMILARITY_THRESHOLD
from claim_extractor import extract_claims_and_headline
from query_generator import generate_queries
from web_search import retrieve_evidence_for_claim
from source_ranker import rank_and_balance_passages, get_embedder
from stance_classifier import batch_classify_stances
from framing_checker import check_framing
from verdict_engine import evaluate_single_claim, synthesize_overall_verdict
from explainer import generate_detailed_explanation

logger = logging.getLogger(__name__)

CACHE_TTL_SEC = 3600  # 1 hour cache TTL
_result_cache: Dict[str, Dict[str, Any]] = {}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "were", "weren't", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "won't", "would", "you", "your", "yours", "yourself", "yourselves"
}


def get_embed_model():
    return get_embedder()


def compute_salient_highlights(claim: str, text: str, max_words: int = 8) -> List[Dict[str, Any]]:
    """Generates XAI word highlights between claim and driving evidence quote."""
    if not text:
        return []
    claim_words = re.findall(r'\b[a-zA-Z]{3,}\b', claim.lower())
    ev_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))

    seen = set()
    highlights = []
    for w in claim_words:
        if w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        if w in ev_words:
            highlights.append({"word": w, "weight": 1.0, "direction": "supports_verdict", "raw": 1.0})
        else:
            highlights.append({"word": w, "weight": 0.4, "direction": "against_verdict", "raw": -0.4})

    highlights.sort(key=lambda x: x["weight"], reverse=True)
    return highlights[:max_words]


def get_verdict_color(verdict: str) -> str:
    """Returns CSS color name for verdicts."""
    if verdict == "SUPPORTED":
        return "green"
    elif verdict == "REFUTED":
        return "red"
    elif verdict == "MISLEADING":
        return "purple"
    return "amber"


def verify_article(
    article_text: str,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    degraded: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Executes the modular FactSense Phase 3 pipeline with stage streaming.
    """
    start_time = time.time()
    stage_timings = {}
    active_providers = get_active_providers()
    if degraded is None:
        degraded = active_providers.get("degraded", True)

    # ── Stage 1: Headline & Claim Extraction ─────────────────────────────────
    if progress_callback:
        progress_callback("extracting_claims", "Extracting atomic claims and checking headline...")
    t0 = time.time()

    extraction_data = extract_claims_and_headline(article_text, max_claims=6)
    extracted_claims = extraction_data.get("claims", [])
    headline_text = extraction_data.get("headline")
    has_headline = extraction_data.get("has_headline", False)

    if not extracted_claims:
        extracted_claims = [{
            "claim_id": "claim_1",
            "text": article_text.strip(),
            "original_text": article_text.strip(),
            "claim_type": "event_fact",
            "is_core": True,
            "is_headline": True,
            "time_scope": None,
            "issuer": None
        }]

    core_claims = [c for c in extracted_claims if c.get("is_core", True)]
    if not core_claims:
        core_claims = extracted_claims

    stage_timings["claim_extraction_sec"] = round(time.time() - t0, 3)

    # ── Stage 2: Query Generation ────────────────────────────────────────────
    t0 = time.time()
    claim_queries = {}
    for c in core_claims:
        claim_queries[c["claim_id"]] = generate_queries(c)
    stage_timings["query_generation_sec"] = round(time.time() - t0, 3)

    # ── Stage 3: Live Parallel Web Retrieval & Reading ───────────────────────
    if progress_callback:
        progress_callback("searching", f"Searching live web & official portals across {len(core_claims)} claim(s)...")
    t0 = time.time()

    per_claim_retrieval: Dict[str, List[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=min(len(core_claims), 4)) as executor:
        future_map = {
            executor.submit(retrieve_evidence_for_claim, c["text"], claim_queries[c["claim_id"]]): c["claim_id"]
            for c in core_claims
        }
        for fut in as_completed(future_map):
            c_id = future_map[fut]
            try:
                passages = fut.result(timeout=14.0)
                per_claim_retrieval[c_id] = passages
            except Exception as e:
                logger.warning(f"Retrieval failed for claim {c_id}: {e}")
                per_claim_retrieval[c_id] = []

    stage_timings["web_search_and_reading_sec"] = round(time.time() - t0, 3)

    # ── Stage 4: Credibility Tiering & Relevance Ranking ─────────────────────
    if progress_callback:
        progress_callback("reading_pages", "Reading web sources and ranking credibility tiers...")
    t0 = time.time()

    claim_passage_pairs = []
    for c in core_claims:
        c_id = c["claim_id"]
        raw_passages = per_claim_retrieval.get(c_id, [])
        ranked_passages = rank_and_balance_passages(
            c["text"],
            raw_passages,
            top_k=7,
            time_scope=c.get("time_scope")
        )
        claim_passage_pairs.append((c, ranked_passages))

    stage_timings["source_ranking_sec"] = round(time.time() - t0, 3)

    # ── Stage 5: Batched Stance Classification ───────────────────────────────
    if progress_callback:
        progress_callback("judging", "Classifying evidence stances across credibility tiers...")
    t0 = time.time()

    classified_grouped = batch_classify_stances(claim_passage_pairs)

    per_claim_results = []
    all_top_evidence = []
    for (c, _), classified_passages in zip(claim_passage_pairs, classified_grouped):
        claim_eval = evaluate_single_claim(c, classified_passages, degraded=degraded)
        claim_eval["color"] = get_verdict_color(claim_eval["verdict"])
        claim_eval["original_text"] = c.get("original_text", c["text"])
        
        driving_q = claim_eval.get("driving_quote") or ""
        claim_eval["highlights"] = compute_salient_highlights(c["text"], driving_q)
        per_claim_results.append(claim_eval)
        all_top_evidence.extend(classified_passages)

    stage_timings["stance_classification_sec"] = round(time.time() - t0, 3)

    # ── Stage 6: Headline vs Body Framing Analysis ───────────────────────────
    t0 = time.time()
    framing_result = check_framing(headline_text if has_headline else None, article_text, per_claim_results)
    stage_timings["framing_check_sec"] = round(time.time() - t0, 3)

    # ── Stage 7: Deterministic Verdict Synthesis ─────────────────────────────
    synthesis = synthesize_overall_verdict(
        per_claim_results,
        article_text,
        framing_data=framing_result,
        degraded=degraded
    )

    overall_verdict = synthesis["verdict"]
    overall_color = get_verdict_color(overall_verdict)
    confidence = synthesis["confidence"]

    # Deduplicate top evidence passages across claims
    seen_urls = set()
    unique_evidence_passages = []
    for p in all_top_evidence:
        u = p.get("url")
        if u and u not in seen_urls:
            seen_urls.add(u)
            unique_evidence_passages.append(p)

    primary_claim = per_claim_results[0]
    for c in per_claim_results:
        if c["verdict"] == overall_verdict:
            primary_claim = c
            break

    total_count = len(per_claim_results)
    sup_count = sum(1 for c in per_claim_results if c["verdict"] == "SUPPORTED")
    ref_count = sum(1 for c in per_claim_results if c["verdict"] == "REFUTED")
    mis_count = sum(1 for c in per_claim_results if c["verdict"] == "MISLEADING")
    neu_count = sum(1 for c in per_claim_results if c["verdict"] == "INSUFFICIENT_EVIDENCE")

    ui_claims = []
    for c in per_claim_results:
        ui_claims.append({
            "claim": c["text"],
            "original_text": c.get("original_text", c["text"]),
            "claim_type": c.get("claim_type", "event_fact"),
            "is_headline": c.get("is_headline", False),
            "verdict": c["verdict"],
            "color": c["color"],
            "confidence": round(c["confidence"] * 100, 1) if isinstance(c["confidence"], float) and c["confidence"] <= 1.0 else c["confidence"],
            "reasoning": c["reason"],
            "evidence_quote": c.get("driving_quote") or "",
            "evidence_source": c.get("driving_source") or "",
            "evidence_url": c.get("driving_url") or "",
            "all_evidence": c.get("all_evidence") or [],
            "highlights": c.get("highlights", [])
        })

    intermediate_result = {
        "verdict": overall_verdict,
        "color": overall_color,
        "confidence": confidence,
        "degraded": degraded,
        "failed_parts": synthesis.get("failed_parts", []),
        "all_probs": synthesis.get("all_probs", {}),
        "framing": framing_result,
        "claims": ui_claims,
        "claim": primary_claim["text"],
        "evidence_text": primary_claim.get("driving_quote") or "",
        "evidence_source": primary_claim.get("driving_source") or "",
        "evidence_url": primary_claim.get("driving_url") or "",
        "lime_words": primary_claim.get("highlights", []),
        "evidence_cards": synthesis.get("evidence_cards", []),
        "total_claims": total_count,
        "supported_claims": sup_count,
        "refuted_claims": ref_count,
        "misleading_claims": mis_count,
        "insufficient_claims": neu_count,
        "stage_timings": stage_timings
    }

    # Requirement E.5: Send the verdict event BEFORE the explanation event!
    if event_callback:
        event_callback({"type": "verdict", "result": intermediate_result})

    # ── Stage 8: Detailed Factual Explanation ────────────────────────────────
    if progress_callback:
        progress_callback("explaining", "Writing detailed factual explanation and grounding quotes...")
    t0 = time.time()

    explanation_details = generate_detailed_explanation(
        overall_verdict=overall_verdict,
        confidence=confidence,
        raw_input_text=article_text,
        per_claim_results=per_claim_results,
        framing_data=framing_result,
        top_evidence_passages=unique_evidence_passages
    )
    stage_timings["explanation_sec"] = round(time.time() - t0, 3)

    total_elapsed = round(time.time() - start_time, 2)
    stage_timings["total_elapsed_sec"] = total_elapsed

    final_result = dict(intermediate_result)
    final_result["explanation"] = explanation_details.get("verdict_summary") or synthesis.get("explanation")
    final_result["explanation_details"] = explanation_details
    final_result["elapsed_sec"] = total_elapsed
    final_result["stage_timings"] = stage_timings
    final_result["cached"] = False

    logger.info(
        f"FactSense Phase 3 complete: Verdict={overall_verdict} ({confidence}%) in {total_elapsed}s. "
        f"Timings: {stage_timings}"
    )

    return final_result


def detect(
    article_text: str,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    degraded: Optional[bool] = None
) -> Dict[str, Any]:
    """Entry point with MD5 caching."""
    clean_text = article_text.strip()
    if len(clean_text) < 4 or len(clean_text.split()) < 2:
        return {"error": "Input text is too short. Please provide at least a short factual claim or article."}

    cache_key = hashlib.md5(f"{clean_text}||deg={degraded}".encode("utf-8")).hexdigest()
    now = time.time()

    if cache_key in _result_cache:
        entry = _result_cache[cache_key]
        if now - entry["timestamp"] < CACHE_TTL_SEC:
            logger.info("TTL Cache hit — returning cached result instantly.")
            cached_data = dict(entry["result"])
            cached_data["elapsed_sec"] = 0.0
            cached_data["cached"] = True
            if event_callback:
                event_callback({"type": "verdict", "result": cached_data})
            return cached_data

    result = verify_article(clean_text, progress_callback=progress_callback, event_callback=event_callback, degraded=degraded)
    _result_cache[cache_key] = {"result": result, "timestamp": now}
    return result
