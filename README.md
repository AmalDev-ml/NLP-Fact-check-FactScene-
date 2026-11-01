# FactSense 3.0 — Live Web Real-World Claim & News Verification System

FactSense is an evidence-grounded claim verification and fake news detection engine. Given a news article or single claim, it verifies whether the claim is **TRUE IN THE REAL WORLD**, never whether it is merely circulating or viral on social media.

The system determines one of four strict verdicts:
* 🟢 **SUPPORTED**
* 🔴 **REFUTED**
* 🟣 **MISLEADING**
* 🟡 **INSUFFICIENT_EVIDENCE**

---

## 🏛️ System Architecture (Phase 3)

FactSense operates a modular 8-stage pipeline with real-time SSE streaming, headline framing analysis, date awareness, and grounded multi-part explanations:

```
                       User Input (Full Article or Claim)
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 1. claim_extractor.py                                                         │
│    • Analyzes the ENTIRE input (never selected lines)                         │
│    • Extracts headline_claim vs body_claims (up to 6 claims, preserves head)  │
│    • Resolves core authenticity if text contains "circulating" or "viral"     │
│    • Extracts time_scope per claim (e.g., "current", "historical", "2026")    │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 2. query_generator.py                                                         │
│    • Generates 4 query types: neutral, debunk, official-source, fact-checker  │
│    • Time-aware queries incorporating relevant years/dates                    │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 3. web_search.py                                                              │
│    • Parallel retrieval across Serper (Google), DuckDuckGo, and Wikipedia     │
│    • Tiered page reading: fetches full text for top 3 hits, snippets for rest │
│    • Extracts publish dates via Trafilatura metadata                          │
│    • Passages chunked to ~900 chars on clean sentence boundaries              │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 4. source_ranker.py                                                           │
│    • Credibility Hierarchy:                                                   │
│      - Tier 1: Official government portals (.gov, .nic, RBI, whitehouse.gov)  │
│      - Tier 2: Certified fact-checkers (Alt News, BOOM, Snopes, etc.)         │
│      - Tier 3: Reputable news agencies, academic institutions (.edu, .ac.in)  │
│      - Tier 4: General web, blogs, unverified social media                    │
│    • Strict threshold: raw_similarity >= 0.35 (tier multiplier does not inflate)│
│    • Encyclopedic flag: Wikipedia is background only, cannot sole-support news│
│    • Stale source penalty for claims requiring current time scope             │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 5. stance_classifier.py                                                       │
│    • Stances: affirms, denies, debunks_circulating_claim, partially_supports, │
│      mentions_without_verifying, neutral                                      │
│    • Batched LLM classification with DeBERTa NLI cross-check (lazy-loaded)    │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 6. framing_checker.py                                                         │
│    • Evaluates headline vs. body text across 9 framing discrepancy checks:    │
│      exaggeration, correlation as causation, small sample, cherry-picked data,│
│      outdated info, wrong attribution/date/place, satire as news, etc.        │
│    • Outputs: {headline_supported_by_body, framing_issues, severity}          │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 7. verdict_engine.py                                                          │
│    • Synthesizes overall verdict & dynamically computed confidence score      │
│    • Fact-checker override: "Partly False" ratings trigger MISLEADING         │
│    • Realistic 4-class probability distribution (all_probs)                   │
│    • Emits early 'verdict' SSE event so UI renders instantly                  │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ 8. explainer.py & Dashboard (app.py)                                          │
│    • Grounded explanation with 8 structured sections: verdict summary,        │
│      what is actually true, why it is wrong/misleading, unverified aspects    │
│    • Side-by-side headline vs body framing card in UI                         │
│    • Degraded Mode alert banner (capped at 60% confidence when no LLM key)    │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Modular File Structure

```
NLP project/
├── config.py               # Unified LLM provider configurations & active runtime logger
├── app.py                  # Flask web server, SSE streaming endpoint (/analyze_stream) & health status
├── detector.py             # Phase 3 coordinator, stage latencies & early verdict emitter
├── claim_extractor.py      # Step 1: Headline detection, atomic claims & time_scope
├── query_generator.py      # Step 2: 4-type query generator with date awareness
├── web_search.py           # Step 3: Serper/DuckDuckGo/Wikipedia with tiered Trafilatura page reading
├── source_ranker.py        # Step 4: 4-tier credibility classifier, date recency & Wikipedia rule
├── stance_classifier.py    # Step 5: Batched stance classification (affirms, denies, partially_supports)
├── framing_checker.py      # Step 6: Headline vs. Body framing discrepancy checklist
├── verdict_engine.py       # Step 7: Deterministic verdict rules, dynamic confidence & all_probs
├── explainer.py            # Step 8: Multi-part grounded explanation generator
├── evaluate.py             # Benchmark evaluation on LIAR/FEVER datasets across 4 verdict classes
├── .env.example            # Environment configuration template
├── requirements.txt        # Python library dependencies
├── tests/
│   ├── test_phase3.py       # Comprehensive 10-test suite for Phase 3 requirements
│   └── test_new_pipeline.py # Baseline pipeline tests
├── static/
│   ├── script.js           # Frontend logic, early verdict rendering, framing & explainer cards
│   └── style.css           # Glassmorphism stylesheet with degraded banner & framing layouts
└── templates/
    └── index.html          # Web dashboard with headline-body comparison and explainer blocks
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
* Python 3.10+ (tested on Windows/Linux/macOS)
* pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your preferred LLM and search provider:
```ini
# Recommended LLM (Gemini, Groq, or OpenAI)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest

# Live Web Search
SERPER_API_KEY=your_serper_api_key_here

# Speed Flags
USE_DEBERTA_CROSSCHECK=false
```
*Note: If no LLM API key is configured, FactSense runs in **Degraded Mode** (local heuristic fallback with a visible banner and confidence capped at 60%).*

### 4. Run the Web Dashboard
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

---

## 🧪 Automated Test Suite

Run the full Phase 3 test suite covering all 10 specifications:

```bash
python -m unittest tests/test_phase3.py
```

### Tested Phase 3 Scenarios:
1. **Headline Exaggeration** ("AI tools make students score 40% higher") ➔ `MISLEADING` with framing severity.
2. **Accurate Article** ("Water boils at 100 degrees Celsius at sea level") ➔ `SUPPORTED` (>= 70% confidence).
3. **Fake Bank Holiday Gazette** (with & without "is fake") ➔ `REFUTED` / `INSUFFICIENT_EVIDENCE` (never `SUPPORTED`).
4. **True Statistic Implying False Conclusion** (Ice cream & shark attacks) ➔ `MISLEADING`.
5. **Real Event with Wrong Date/Place** ➔ `REFUTED` or `MISLEADING`.
6. **Fact-Checker Override** ("Partly False" rating) ➔ `MISLEADING`.
7. **Wikipedia Background Restriction** (News claim only on Wikipedia + blogs) ➔ `INSUFFICIENT_EVIDENCE`.
8. **Nonsense / Gibberish Claim** ➔ `INSUFFICIENT_EVIDENCE`.
9. **Article with Headline Removed** ➔ Handled cleanly, extracts lead claim.
10. **Degraded Mode (No LLM Key)** ➔ `degraded=True` and confidence capped at 60%.

---

## 👥 Authors & Academic Info

* **Course:** NLP Academic Project
* **Batch:** R7A
* **Students:**
  * Amal Dev M L (Roll No: 215)
  * A S Adityanarayanan (Roll No: 220)
  * Rohit Ajoy (Roll No: 255)
