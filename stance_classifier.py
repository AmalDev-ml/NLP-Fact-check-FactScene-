"""
stance_classifier.py — Pairwise Stance & Evidence Classification
----------------------------------------------------------------
Phase 3 Features:
1. Classifies (claim, passage) pairs into discrete stances:
   - "affirms": Confirms that the claim is true in the real world.
   - "denies": Directly contradicts the claim or asserts it is false.
   - "partially_supports": Source says the claim is partly true, lacks context,
     is exaggerated, outdated, or misattributed (including fact-checker ratings
     such as "Misleading", "Partly False", "Missing Context").
   - "debunks_circulating_claim": Explicitly states that a viral circular, social media
     claim, fake gazette, or rumor is FALSE/FAKE/HOAX.
   - "mentions_without_verifying": Mentions the claim or rumor without validation.
   - "neutral": Irrelevant or unrelated background context.
2. Passage window raised to ~900 characters, respecting sentence boundaries.
3. Batched evaluation across claims and passages for high performance.
4. Optional DeBERTa NLI cross-check loaded lazily only if USE_DEBERTA_CROSSCHECK=true.
5. Prints structured stance tables in logs for debugging.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from config import (
    get_llm,
    extract_text_from_response,
    USE_DEBERTA_CROSSCHECK,
    DEFAULT_NLI_MODEL,
    PASSAGE_CHAR_WINDOW,
    RAW_SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

# Auxiliary NLI model singletons (only loaded if USE_DEBERTA_CROSSCHECK is True)
_nli_tokenizer = None
_nli_model = None
_nli_device = None

VALID_STANCES = {
    "affirms",
    "denies",
    "partially_supports",
    "debunks_circulating_claim",
    "mentions_without_verifying",
    "neutral"
}


def get_nli_model():
    """Lazy loader for DeBERTa NLI model; never loaded unless USE_DEBERTA_CROSSCHECK=true."""
    global _nli_tokenizer, _nli_model, _nli_device
    if _nli_model is None:
        logger.info(f"Loading auxiliary NLI model: {DEFAULT_NLI_MODEL}")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        try:
            _nli_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_NLI_MODEL, local_files_only=True)
            _nli_model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_NLI_MODEL, local_files_only=True)
        except Exception:
            _nli_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_NLI_MODEL)
            _nli_model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_NLI_MODEL)
        
        _nli_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _nli_model.to(_nli_device)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model, _nli_device


def clean_passage_window(text: str, max_chars: int = PASSAGE_CHAR_WINDOW) -> str:
    """Cuts text on sentence boundary within max_chars window."""
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    
    truncated = clean[:max_chars]
    # Find last sentence-ending punctuation
    last_punct = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    if last_punct > max_chars * 0.5:
        return truncated[:last_punct + 1].strip()
    return truncated.strip()


def classify_stances(
    claim_dict: Dict[str, Any],
    passages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Classifies the stance of each passage toward the single claim.
    """
    if not passages:
        return []

    results = batch_classify_stances([(claim_dict, passages)])
    return results[0] if results else []


def batch_classify_stances(
    claim_passage_pairs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]
) -> List[List[Dict[str, Any]]]:
    """
    Batches all claims and passages into a single structured LLM call for high speed.
    """
    if not claim_passage_pairs:
        return []

    llm = get_llm()
    if llm:
        try:
            batched_results = _classify_batch_with_llm(claim_passage_pairs, llm)
            if batched_results:
                if USE_DEBERTA_CROSSCHECK:
                    for (c_dict, _), res in zip(claim_passage_pairs, batched_results):
                        _attach_nli_crosscheck(c_dict.get("text", ""), res)
                _log_stance_tables(claim_passage_pairs, batched_results)
                return batched_results
        except Exception as e:
            logger.warning(f"Batched LLM stance classification failed: {e}. Falling back to heuristic/NLI.")

    # Fallback path
    fallback_results = []
    for claim_dict, passages in claim_passage_pairs:
        claim_text = claim_dict.get("text", "")
        classified = _classify_with_heuristic(claim_dict, passages)
        if USE_DEBERTA_CROSSCHECK:
            _attach_nli_crosscheck(claim_text, classified)
        fallback_results.append(classified)

    _log_stance_tables(claim_passage_pairs, fallback_results)
    return fallback_results


def _classify_batch_with_llm(
    claim_passage_pairs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
    llm
) -> List[List[Dict[str, Any]]]:
    """Executes a single LLM prompt classifying all claims and their candidate passages."""
    evaluation_requests = []
    pair_mapping = []  # keeps track of (pair_idx, passage_idx)

    item_id = 1
    for pair_idx, (claim_dict, passages) in enumerate(claim_passage_pairs):
        claim_text = claim_dict.get("text", "")
        claim_type = claim_dict.get("claim_type", "event_fact")
        time_scope = claim_dict.get("time_scope") or "unspecified"

        for p_idx, p in enumerate(passages):
            domain = p.get("domain", "web")
            tier = p.get("tier", 4)
            pub_date = p.get("publish_date") or p.get("date") or "unknown"
            windowed_text = clean_passage_window(p.get("text", ""))

            evaluation_requests.append(
                f"[Item {item_id}]\n"
                f"Claim: \"{claim_text}\" (Type: {claim_type}, Time Scope: {time_scope})\n"
                f"Source: {domain} (Tier {tier}, Date: {pub_date})\n"
                f"Passage: \"{windowed_text}\""
            )
            pair_mapping.append((pair_idx, p_idx, p))
            item_id += 1

    if not evaluation_requests:
        return [passages for _, passages in claim_passage_pairs]

    prompt = f"""You are an objective, rigorous evidence stance evaluator.
For each Item below, evaluate whether the Passage affirms, denies, partially supports, debunks, or merely mentions the Claim.

{chr(10).join(evaluation_requests)}

STANCE TAXONOMY:
- "affirms": The passage confirms that the claim is true in the real world.
- "denies": The passage states that the claim is false, incorrect, unapproved, or directly contradicts it with contrary facts.
- "partially_supports": The passage indicates the claim is partly true, lacks critical context, is exaggerated, outdated, or misattributed (including fact-checker ratings such as "Misleading", "Partly False", "Missing Context").
- "debunks_circulating_claim": The passage specifically states that a viral circular, social media rumor, fake gazette, or hoax is FAKE/FALSE (e.g. "PIB Fact Check confirmed the circular is fake").
- "mentions_without_verifying": The passage only reports that people are talking about the claim or that an image is circulating on social media, WITHOUT validating whether it is true.
- "neutral": The passage is off-topic, unrelated, or provides only generic background.

CRITICAL RULES:
1. Fact-checker labels like "Misleading", "Partly False", or "Missing Context" MUST receive stance "partially_supports".
2. If the passage debunks a viral circular or order as fake, assign "debunks_circulating_claim".
3. Quote the EXACT verbatim sentence (<25 words) from the passage that determined your stance.

Respond ONLY with valid JSON in this exact structure:
{{
  "evaluations": [
    {{
      "item_id": 1,
      "stance": "affirms" | "denies" | "partially_supports" | "debunks_circulating_claim" | "mentions_without_verifying" | "neutral",
      "quote": "<exact verbatim sentence from passage>",
      "reason": "<1-line concise reason>"
    }}
  ]
}}
"""
    response = llm.invoke(prompt)
    raw_str = extract_text_from_response(response)

    raw_str = re.sub(r'^```json\s*', '', raw_str.strip())
    raw_str = re.sub(r'```$', '', raw_str.strip())

    match = re.search(r'\{.*\}', raw_str, re.DOTALL)
    if not match:
        raise ValueError("Could not parse JSON from stance response")

    data = json.loads(match.group(0))
    evals = {e.get("item_id"): e for e in data.get("evaluations", [])}

    # Group back by claim
    grouped_results: List[List[Dict[str, Any]]] = [[] for _ in claim_passage_pairs]

    for item_idx, (pair_idx, p_idx, orig_p) in enumerate(pair_mapping, 1):
        p_copy = dict(orig_p)
        ev = evals.get(item_idx, {})
        stance = str(ev.get("stance", "neutral")).lower()
        if stance not in VALID_STANCES:
            stance = "neutral"

        quote = str(ev.get("quote", "")).strip()
        if not quote:
            sents = re.split(r'(?<=[.!?])\s+', p_copy.get("text", ""))
            quote = sents[0] if sents else p_copy.get("text", "")[:150]

        reason = str(ev.get("reason", "")).strip() or f"Source {stance} the core claim."

        p_copy["stance"] = stance
        p_copy["quote"] = quote
        p_copy["reason"] = reason
        grouped_results[pair_idx].append(p_copy)

    return grouped_results


def _classify_with_heuristic(
    claim_dict: Dict[str, Any],
    passages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Rule-based heuristic stance classifier when LLM is unavailable."""
    claim_text = claim_dict.get("text", "").lower()
    claim_type = claim_dict.get("claim_type", "event_fact")

    results = []
    debunk_words = {"fake", "hoax", "false", "misleading", "fact check", "debunk", "untrue", "bogus", "fraudulent"}
    partial_words = {"partly false", "partially true", "missing context", "exaggerated", "misleading", "lacks context", "outdated"}

    for p in passages:
        p_copy = dict(p)
        text = p.get("text", "")
        text_lower = text.lower()
        raw_sim = float(p.get("raw_similarity", p.get("relevance", 0.0)))

        stance = "neutral"
        quote = ""
        reason = ""

        # Check for debunk keywords
        has_debunk = any(dw in text_lower for dw in debunk_words)
        has_partial = any(pw in text_lower for pw in partial_words)

        if has_partial:
            stance = "partially_supports"
            reason = "Source explicitly rates claim as misleading, partly false, or missing context."
        elif has_debunk and raw_sim >= RAW_SIMILARITY_THRESHOLD:
            if claim_type == "document_authenticity" or "circular" in text_lower or "gazette" in text_lower:
                stance = "debunks_circulating_claim"
                reason = "Source fact-check debunks the circulating circular/document as fake."
            else:
                stance = "denies"
                reason = "Source directly contradicts or debunks the factual claim."
        elif raw_sim >= 0.50:
            stance = "affirms"
            reason = "High semantic alignment corroborates claim in real-world context."
        elif "circulating" in text_lower or "viral" in text_lower:
            stance = "mentions_without_verifying"
            reason = "Source mentions social media virality without authenticating the claim."

        # Extract quote
        sents = re.split(r'(?<=[.!?])\s+', text)
        quote = sents[0] if sents else text[:120]

        p_copy["stance"] = stance
        p_copy["quote"] = quote
        p_copy["reason"] = reason
        results.append(p_copy)

    return results


def _attach_nli_crosscheck(claim_text: str, passages: List[Dict[str, Any]]):
    """Computes DeBERTa NLI cross-check probabilities only when enabled."""
    if not USE_DEBERTA_CROSSCHECK:
        return

    try:
        import torch
        tokenizer, model, device = get_nli_model()
        pairs = [(clean_passage_window(p.get("text", ""), 300), claim_text) for p in passages[:5]]
        if not pairs:
            return

        premises = [p[0] for p in pairs]
        hypotheses = [p[1] for p in pairs]

        inputs = tokenizer(premises, hypotheses, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        id2label = model.config.id2label
        entail_idx = next((i for i, l in id2label.items() if "entail" in l.lower()), 0)
        contra_idx = next((i for i, l in id2label.items() if "contra" in l.lower()), 2)
        neutral_idx = next((i for i, l in id2label.items() if "neut" in l.lower()), 1)

        for idx, p in enumerate(passages[:5]):
            p_ent = float(probs[idx][entail_idx])
            p_cnt = float(probs[idx][contra_idx])
            p_neu = float(probs[idx][neutral_idx])

            nli_verdict = "NEUTRAL"
            if p_cnt >= 0.70:
                nli_verdict = "REFUTED"
            elif p_ent >= 0.65:
                nli_verdict = "SUPPORTED"

            p["nli_cross_check"] = {
                "verdict": nli_verdict,
                "confidence": max(p_ent, p_cnt, p_neu),
                "probs": {"entailment": p_ent, "contradiction": p_cnt, "neutral": p_neu}
            }
    except Exception as e:
        logger.warning(f"DeBERTa NLI cross-check failed: {e}")


def _log_stance_tables(
    claim_passage_pairs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
    grouped_results: List[List[Dict[str, Any]]]
):
    """Prints a formatted stance evaluation table to logs for debugging."""
    for (claim_dict, _), passages in zip(claim_passage_pairs, grouped_results):
        c_text = claim_dict.get("text", "")[:60]
        logger.info(f"=== STANCE TABLE FOR CLAIM: '{c_text}' ===")
        for p in passages:
            domain = p.get("domain", "web")
            tier = p.get("tier", 4)
            stance = p.get("stance", "neutral")
            sim = round(float(p.get("raw_similarity", p.get("relevance", 0.0))), 3)
            quote = (p.get("quote") or "")[:50].replace("\n", " ")
            logger.info(f"  [{domain} | Tier {tier} | Sim: {sim}] -> {stance.upper()} | \"{quote}...\"")
