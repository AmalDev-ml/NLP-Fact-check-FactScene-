"""
framing_checker.py — Headline vs Body Framing & Misleading Detection
---------------------------------------------------------------------
Phase 3 Responsibilities:
1. Compares the article's headline with the article body and per-claim verdicts.
2. Identifies discrepancies, exaggerations, sample bias, or deceptive framing.
3. Checks against the standardized framing issue checklist:
   - Headline overstates magnitude/certainty/scope ("proves", "all", "doubles") vs body ("may", "some", "one study")
   - Correlation presented as causation
   - Tiny, self-selected, or unnamed sample presented as general fact
   - Cherry-picked statistic or missing baseline
   - Outdated information presented as current
   - Real image/quote/video with wrong date, place, or attribution
   - Satire/opinion presented as objective news
   - Claim attributed to a source that did not say it
   - True fact used to imply a false conclusion
4. Computes framing severity: "none", "minor", or "major".
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional

from config import get_llm, extract_text_from_response

logger = logging.getLogger(__name__)

FRAMING_CHECKLIST = [
    "Headline overstates magnitude, certainty, or scope ('proves', 'all', 'doubles', 'cures') vs body qualifiers ('may', 'some', 'one study')",
    "Correlation presented as direct causation",
    "Tiny, self-selected, or unnamed sample presented as a general fact",
    "Cherry-picked statistic or missing baseline comparison",
    "Outdated information presented as current news",
    "Real image, quote, or video with incorrect date, location, or attribution",
    "Satire, parody, or opinion presented as factual news",
    "Claim attributed to an authority or institution that did not say it",
    "True factual premise used to imply an unsubstantiated or false conclusion"
]


def check_framing(
    headline: Optional[str],
    body: str,
    per_claim_verdicts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Executes a structured check evaluating whether the headline is supported by the body.
    Returns:
      {
        "has_headline": bool,
        "headline_supported_by_body": "true" | "false" | "partial",
        "what_headline_implies": str,
        "what_body_actually_says": str,
        "framing_issues": List[str],
        "severity": "none" | "minor" | "major"
      }
    """
    if not headline or not headline.strip():
        return {
            "has_headline": False,
            "headline_supported_by_body": "true",
            "what_headline_implies": "",
            "what_body_actually_says": "",
            "framing_issues": [],
            "severity": "none"
        }

    clean_headline = headline.strip()
    clean_body = body.strip()

    llm = get_llm()
    if llm:
        try:
            result = _check_framing_llm(clean_headline, clean_body, per_claim_verdicts, llm)
            logger.info(f"Framing Check Output: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            logger.warning(f"LLM framing check failed: {e}. Using heuristic fallback.")

    result = _check_framing_heuristic(clean_headline, clean_body, per_claim_verdicts)
    logger.info(f"Heuristic Framing Check Output: {json.dumps(result, indent=2)}")
    return result


def _check_framing_llm(
    headline: str,
    body: str,
    per_claim_verdicts: List[Dict[str, Any]],
    llm
) -> Dict[str, Any]:
    """Uses LLM to evaluate headline vs body framing."""
    claims_summary = []
    for c in per_claim_verdicts:
        c_text = c.get("claim") or c.get("text", "")
        c_v = c.get("verdict", "UNKNOWN")
        c_r = c.get("reason") or c.get("reasoning", "")
        claims_summary.append(f"- Claim: '{c_text}' -> Verdict: {c_v} (Reason: {c_r})")
    claims_str = "\n".join(claims_summary) if claims_summary else "No claim breakdown available."

    prompt = f"""You are an expert news media framing and deceptive headline analyst.
Evaluate whether the headline of the following article accurately reflects the body text and evidence.

HEADLINE:
\"\"\"{headline}\"\"\"

BODY TEXT:
\"\"\"{body}\"\"\"

PER-CLAIM FACT-CHECKING EVIDENCE:
{claims_str}

CHECKLIST OF COMMON FRAMING FLAWS:
1. Headline overstates magnitude, certainty, or scope (e.g. "proves", "all", "doubles", "cures", "40% higher") vs body qualifiers (e.g. "may", "some", "one study", "small survey", "in preliminary trials").
2. Correlation presented as direct causation.
3. Tiny, self-selected, or unnamed sample presented as a broad, general fact.
4. Cherry-picked statistic or missing baseline comparison.
5. Outdated information presented as current news.
6. Real image, quote, or video with incorrect date, location, or attribution.
7. Satire, parody, or opinion presented as factual news.
8. Claim attributed to an authority or institution that did not say it.
9. True factual premise used to imply an unsubstantiated or false conclusion.

INSTRUCTIONS:
- Determine if the headline is supported by the body: "true", "false", or "partial".
- State clearly in 1 sentence what the headline implies to a casual reader.
- State clearly in 1 sentence what the body text / evidence actually says.
- Identify all matching framing issues from the checklist (return as an array of short descriptions).
- Rate the severity:
  * "none": Headline is fully accurate and proportional to body.
  * "minor": Slight exaggeration or missing minor context, but substantially aligned.
  * "major": Headline is misleading, sensationalized, asserts a broad claim based only on a tiny/pilot sample, or directly contradicts the body findings.

Respond ONLY with valid JSON:
{{
  "has_headline": true,
  "headline_supported_by_body": "true" | "false" | "partial",
  "what_headline_implies": "<what reader naturally assumes>",
  "what_body_actually_says": "<what the text/evidence actually substantiates>",
  "framing_issues": ["<issue 1>", "<issue 2>"],
  "severity": "none" | "minor" | "major"
}}
"""
    response = llm.invoke(prompt)
    content = extract_text_from_response(response)

    # Clean markdown json code blocks if present
    content = re.sub(r'^```json\s*', '', content.strip())
    content = re.sub(r'```$', '', content.strip())

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM framing response")

    data = json.loads(match.group(0))
    
    sup = str(data.get("headline_supported_by_body", "true")).lower()
    if sup not in ("true", "false", "partial"):
        sup = "partial" if data.get("framing_issues") else "true"

    sev = str(data.get("severity", "none")).lower()
    if sev not in ("none", "minor", "major"):
        sev = "major" if sup == "false" else ("minor" if sup == "partial" else "none")

    return {
        "has_headline": True,
        "headline_supported_by_body": sup,
        "what_headline_implies": str(data.get("what_headline_implies", "")).strip(),
        "what_body_actually_says": str(data.get("what_body_actually_says", "")).strip(),
        "framing_issues": [str(x).strip() for x in data.get("framing_issues", []) if str(x).strip()],
        "severity": sev
    }


def _check_framing_heuristic(
    headline: str,
    body: str,
    per_claim_verdicts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Fallback heuristic check when no LLM is configured."""
    hl_lower = headline.lower()
    body_lower = body.lower()

    issues = []
    
    # 1. Overstatement / Magnitude check
    absolutes = ["all", "every", "proves", "cures", "doubles", "100%", "miracle", "scores higher", "makes students"]
    qualifiers = ["small survey", "pilot study", "self-reported", "one course", "may", "might", "suggests", "preliminary", "some"]

    has_absolute = any(w in hl_lower for w in absolutes)
    has_qualifier = any(q in body_lower for q in qualifiers)

    if has_absolute and has_qualifier:
        issues.append("Headline presents general assertion while body specifies a small, self-reported sample or pilot study")

    # 2. Check if headline claim itself was refuted in evidence
    hl_refuted = False
    for c in per_claim_verdicts:
        if c.get("is_headline") and c.get("verdict") in ("REFUTED", "MISLEADING"):
            hl_refuted = True
            break

    if hl_refuted:
        issues.append("The central assertion made in the headline was refuted or contradicted by evidence")

    # 3. Check for correlation vs causation
    if "causes" in hl_lower or "makes" in hl_lower:
        if "associated" in body_lower or "correlated" in body_lower:
            issues.append("Correlation in body presented as direct causation in headline")

    severity = "none"
    sup = "true"
    if len(issues) >= 2 or hl_refuted or (has_absolute and has_qualifier):
        severity = "major"
        sup = "false"
    elif len(issues) == 1:
        severity = "minor"
        sup = "partial"

    return {
        "has_headline": True,
        "headline_supported_by_body": sup,
        "what_headline_implies": headline,
        "what_body_actually_says": body[:180] + ("..." if len(body) > 180 else ""),
        "framing_issues": issues,
        "severity": severity
    }

