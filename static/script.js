/* ============================================================
   script.js — FactSense Live Web Verification Logic
   ============================================================ */

// ── Model status polling ──────────────────────────────────────────────────────
const statusDot = document.getElementById('model-status');

async function pollModelStatus() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.model_ready) {
      statusDot.classList.remove('loading');
      statusDot.classList.add('ready');
      statusDot.title = `FactSense Ready (LLM: ${data.llm_provider || 'Active'}, Search: ${data.search_provider || 'Active'})`;
      const degBanner = document.getElementById('degraded-banner');
      if (degBanner) {
        if (data.degraded) {
          degBanner.classList.remove('hidden');
        } else {
          degBanner.classList.add('hidden');
        }
      }
    } else {
      setTimeout(pollModelStatus, 2000);
    }
  } catch {
    setTimeout(pollModelStatus, 3000);
  }
}
pollModelStatus();

// ── Char counter ──────────────────────────────────────────────────────────────
const textarea    = document.getElementById('article-input');
const charCounter = document.getElementById('char-count');

textarea.addEventListener('input', () => {
  const len = textarea.value.length;
  charCounter.textContent = `${len.toLocaleString()} / 8,000`;
  charCounter.style.color = len > 7500 ? 'var(--amber)' : 'var(--text-d)';
});

// ── UI state management ───────────────────────────────────────────────────────
function showSection(id) {
  ['loading-section', 'error-section', 'results-section'].forEach(s => {
    const el = document.getElementById(s);
    if (el) el.classList.add('hidden');
  });
  if (id) {
    const target = document.getElementById(id);
    if (target) target.classList.remove('hidden');
  }
}

function resetUI() {
  showSection(null);
  const btn = document.getElementById('analyze-btn');
  if (btn) btn.disabled = false;
  const btnText = document.getElementById('btn-text');
  if (btnText) btnText.textContent = 'Verify Claim';
  const spinner = document.getElementById('btn-spinner');
  if (spinner) spinner.classList.add('hidden');
  const icon = document.getElementById('btn-icon');
  if (icon) icon.classList.remove('hidden');
  if (textarea) textarea.focus();
}

function setExample(text) {
  if (!textarea) return;
  textarea.value = text;
  textarea.dispatchEvent(new Event('input'));
  analyzeArticle();
}

// ── Stage Progress Stepper ───────────────────────────────────────────────────
const STAGE_STEP_MAP = {
  'extracting_claims': 1,
  'searching': 2,
  'reading_pages': 3,
  'judging': 4,
  'explaining': 5,
};

function updateActiveStage(stageKey, message) {
  const activeStepNum = STAGE_STEP_MAP[stageKey] || 1;
  for (let i = 1; i <= 5; i++) {
    const stepEl = document.getElementById(`step-${i}`);
    if (!stepEl) continue;
    if (i < activeStepNum) {
      stepEl.classList.remove('step-active');
      stepEl.classList.add('step-done');
    } else if (i === activeStepNum) {
      stepEl.classList.remove('step-done');
      stepEl.classList.add('step-active');
      if (message) {
        const textEl = document.getElementById(`step-text-${i}`);
        if (textEl) textEl.textContent = `${i}. ${message}`;
      }
    } else {
      stepEl.classList.remove('step-active', 'step-done');
    }
  }
}

function resetStageIndicators() {
  for (let i = 1; i <= 5; i++) {
    const stepEl = document.getElementById(`step-${i}`);
    if (stepEl) {
      stepEl.classList.remove('step-done');
      if (i === 1) stepEl.classList.add('step-active');
      else stepEl.classList.remove('step-active');
    }
  }
}

// ── Build LIME / Token Attribution Chart ──────────────────────────────────────
function renderLimeChart(words) {
  const container = document.getElementById('lime-chart');
  if (!container) return;
  container.innerHTML = '';

  if (!Array.isArray(words) || words.length === 0) {
    container.innerHTML = '<p style="color:var(--text-d);font-size:14px;">No token importance data available.</p>';
    return;
  }

  const validWeights = words.map(w => Number(w.weight) || 0);
  const maxW = Math.max(...validWeights, 0.0001);

  words.forEach((w, i) => {
    const rawWeight = Number(w.weight) || 0;
    const pct     = Math.min(100, Math.max(8, Math.round((rawWeight / maxW) * 100)));
    const isPos   = w.direction === 'supports_verdict';
    const wordStr = String(w.word ?? '');
    const row     = document.createElement('div');
    row.className = 'lime-row';
    row.style.animationDelay = `${i * 35}ms`;
    row.innerHTML = `
      <span class="lime-word" title="${escHtml(wordStr)}">${escHtml(wordStr)}</span>
      <div class="lime-bar-track">
        <div class="lime-bar-fill ${isPos ? 'pos' : 'neg'}" style="width:${pct}%"></div>
      </div>
      <span class="lime-weight">${rawWeight.toFixed(2)}</span>
    `;
    container.appendChild(row);
  });
}

// ── Render confidence distribution bars ───────────────────────────────────────
function renderConfidenceBars(probs) {
  if (!probs) return;
  const map = {
    'SUPPORTED':             { bar: 'bar-supported',   prob: 'prob-supported' },
    'REFUTED':               { bar: 'bar-refuted',     prob: 'prob-refuted' },
    'MISLEADING':            { bar: 'bar-misleading',  prob: 'prob-misleading' },
    'INSUFFICIENT_EVIDENCE': { bar: 'bar-insufficient', prob: 'prob-insufficient' },
    'INSUFFICIENT EVIDENCE': { bar: 'bar-insufficient', prob: 'prob-insufficient' },
  };

  Object.entries(map).forEach(([label, ids]) => {
    const barEl = document.getElementById(ids.bar);
    const probEl = document.getElementById(ids.prob);
    if (!barEl || !probEl) return;
    const val = probs[label] ?? 0;
    probEl.textContent = `${val}%`;
    setTimeout(() => {
      barEl.style.width = `${val}%`;
    }, 100);
  });
}

// ── Render Per-Claim Table ────────────────────────────────────────────────────
function renderClaimsTable(claims) {
  const tbody = document.getElementById('claims-table-body');
  const countLabel = document.getElementById('claims-summary-count');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!Array.isArray(claims) || claims.length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--text-d);padding:24px;">No claims analyzed.</td></tr>';
    return;
  }

  if (countLabel) {
    countLabel.textContent = `${claims.length} Claim${claims.length > 1 ? 's' : ''} Analyzed`;
  }

  claims.forEach((item) => {
    const tr = document.createElement('tr');

    let pillCls = 'pill-amber';
    if (item.verdict === 'SUPPORTED') pillCls = 'pill-green';
    else if (item.verdict === 'REFUTED') pillCls = 'pill-red';
    else if (item.verdict === 'MISLEADING') pillCls = 'pill-purple';

    // Highlight tags
    let tagsHtml = '';
    if (item.highlights && item.highlights.length > 0) {
      tagsHtml = '<div class="xai-tags">' + item.highlights.slice(0, 5).map(h => {
        const cls = h.direction === 'supports_verdict' ? 'xai-tag-pos' : 'xai-tag-neg';
        return `<span class="xai-tag ${cls}">${escHtml(h.word)}</span>`;
      }).join('') + '</div>';
    }

    // Cited evidence & sources block
    let evidenceHtml = '';
    if (item.all_evidence && item.all_evidence.length > 0) {
      evidenceHtml = item.all_evidence.slice(0, 3).map(ev => {
        const quote = ev.quote || ev.text || '';
        const url = ev.url || '#';
        let domain = ev.domain || 'Live Web';
        if (!ev.domain) {
          try {
            domain = new URL(url).hostname.replace(/^www\./, '');
          } catch (_) {}
        }
        const tierNum = ev.tier || 4;
        const tierLabel = ev.tier_name ? ev.tier_name.split(':')[0] : `Tier ${tierNum}`;
        const stanceStr = ev.stance || 'neutral';

        return `
          <div style="margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;flex-wrap:wrap;">
              <span class="pill-tier pill-tier${tierNum}">${escHtml(tierLabel)}</span>
              <span class="stance-badge stance-${escHtml(stanceStr)}">${escHtml(stanceStr.replace(/_/g, ' '))}</span>
            </div>
            <div class="claim-evidence-text">
              "${escHtml(quote)}"
            </div>
            <a class="claim-source-tag" href="${escHtml(url)}" target="_blank" rel="noopener">
              🔗 ${escHtml(domain)} →
            </a>
            ${ev.publish_date ? `<span class="evidence-card-date" style="margin-left:8px;">📅 ${escHtml(ev.publish_date)}</span>` : ''}
          </div>
        `;
      }).join('');
    } else if (item.evidence_quote) {
      const srcUrl = item.evidence_url || '#';
      const srcName = item.evidence_source || 'Live Web Source';
      evidenceHtml = `
        <div class="claim-evidence-text">
          "${escHtml(item.evidence_quote)}"
        </div>
        <a class="claim-source-tag" href="${escHtml(srcUrl)}" target="_blank" rel="noopener">
          🔗 ${escHtml(srcName)} →
        </a>
      `;
    } else {
      evidenceHtml = `
        <div class="claim-evidence-text" style="border-left-color:var(--amber);">
          ⚠️ No conclusive evidence found across search engines or fact-checking databases.
        </div>
      `;
    }

    // Auxiliary DeBERTa Cross-check badge
    let nliBadgeHtml = '';
    if (item.nli_cross_check && item.nli_cross_check.verdict) {
      const nliVerdict = item.nli_cross_check.verdict;
      const nliConf = Math.round((item.nli_cross_check.confidence || 0) * 100);
      nliBadgeHtml = `<div class="cross-check-chip">NLI Cross-Check: ${nliVerdict} (${nliConf}%)</div>`;
    }

    const reasoningHtml = item.reasoning ? `<div class="claim-rationale">${escHtml(item.reasoning)}</div>` : '';

    tr.innerHTML = `
      <td>
        <div class="claim-item-text">${escHtml(item.claim)}</div>
        ${tagsHtml}
      </td>
      <td>
        <span class="verdict-pill ${pillCls}">${escHtml(item.verdict)}</span>
        <div style="font-size:12px;color:var(--text-m);margin-top:6px;font-family:var(--mono);">
          ${item.confidence}% conf.
        </div>
        ${nliBadgeHtml}
      </td>
      <td>
        ${evidenceHtml}
        ${reasoningHtml}
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Verdict Configuration ─────────────────────────────────────────────────────
const VERDICT_CONFIG = {
  'SUPPORTED': {
    icon: '✅',
    cls:  'verdict-green',
  },
  'REFUTED': {
    icon: '❌',
    cls:  'verdict-red',
  },
  'MISLEADING': {
    icon: '⚡',
    cls:  'verdict-purple',
  },
  'INSUFFICIENT_EVIDENCE': {
    icon: '⚠️',
    cls:  'verdict-amber',
  },
  'INSUFFICIENT EVIDENCE': {
    icon: '⚠️',
    cls:  'verdict-amber',
  },
};

// ── Main Results Renderer ─────────────────────────────────────────────────────
function renderResults(data, isInterim = false) {
  try {
    const verdictKey = data.verdict || 'INSUFFICIENT_EVIDENCE';
    const cfg = VERDICT_CONFIG[verdictKey] || VERDICT_CONFIG['INSUFFICIENT_EVIDENCE'];

    // Degraded Mode Alert Banner
    const degBanner = document.getElementById('degraded-banner');
    if (degBanner) {
      if (data.degraded) degBanner.classList.remove('hidden');
      else degBanner.classList.add('hidden');
    }

    const vCard = document.getElementById('verdict-card');
    if (vCard) vCard.className = `card verdict-card ${cfg.cls}`;

    const iconEl = document.getElementById('verdict-icon');
    if (iconEl) iconEl.textContent = cfg.icon;

    const textEl = document.getElementById('verdict-text');
    if (textEl) textEl.textContent = data.verdict;

    // Stats chips
    const statsContainer = document.getElementById('verdict-stats-chips');
    if (statsContainer) {
      const isMisleading = data.verdict === 'MISLEADING';
      const misChip = isMisleading ? `<span class="stats-chip" style="color:var(--purple)">Status: <b>Misleading Mixture</b></span>` : '';
      statsContainer.innerHTML = `
        <span class="stats-chip">Total Claims: <b>${data.total_claims ?? (data.claims ? data.claims.length : 1)}</b></span>
        <span class="stats-chip" style="color:var(--green)">Supported: <b>${data.supported_claims ?? (data.verdict === 'SUPPORTED' ? 1 : 0)}</b></span>
        <span class="stats-chip" style="color:var(--red)">Refuted: <b>${data.refuted_claims ?? (data.verdict === 'REFUTED' ? 1 : 0)}</b></span>
        <span class="stats-chip" style="color:var(--amber)">Insufficient: <b>${data.insufficient_claims ?? (data.verdict.startsWith('INSUF') ? 1 : 0)}</b></span>
        ${misChip}
      `;
    }

    // Failed Parts Banner (for MISLEADING / REFUTED articles)
    const fpContainer = document.getElementById('failed-parts-container');
    if (fpContainer) {
      if (Array.isArray(data.failed_parts) && data.failed_parts.length > 0) {
        fpContainer.className = 'failed-parts-banner';
        fpContainer.innerHTML = `
          <div class="failed-parts-title">
            <span>⚠️ Refuted or False Statements in this Input:</span>
          </div>
          <ul class="failed-parts-list">
            ${data.failed_parts.map(p => `<li>${escHtml(p)}</li>`).join('')}
          </ul>
        `;
      } else {
        fpContainer.className = 'hidden';
        fpContainer.innerHTML = '';
      }
    }

    // Framing Analysis Card
    const framingCard = document.getElementById('framing-card');
    if (framingCard) {
      const framing = data.framing;
      if (framing && framing.has_headline && framing.severity && framing.severity !== 'none') {
        framingCard.classList.remove('hidden');
        const badge = document.getElementById('framing-severity-badge');
        if (badge) {
          badge.className = `severity-badge severity-${framing.severity}`;
          badge.textContent = `${framing.severity.toUpperCase()} Discrepancy`;
        }
        const hText = document.getElementById('framing-headline-text');
        if (hText) hText.textContent = framing.what_headline_implies || '–';
        const bText = document.getElementById('framing-body-text');
        if (bText) bText.textContent = framing.what_body_actually_says || '–';

        const issuesCont = document.getElementById('framing-issues-container');
        const issuesList = document.getElementById('framing-issues-list');
        if (issuesCont && issuesList) {
          if (Array.isArray(framing.framing_issues) && framing.framing_issues.length > 0) {
            issuesCont.classList.remove('hidden');
            issuesList.innerHTML = framing.framing_issues.map(iss => `
              <div class="framing-issue-item">
                <span>⚠️</span>
                <span>${escHtml(iss)}</span>
              </div>
            `).join('');
          } else {
            issuesCont.classList.add('hidden');
          }
        }
      } else {
        framingCard.classList.add('hidden');
      }
    }

    const confBadge = document.getElementById('confidence-badge');
    if (confBadge) {
      if (data.confidence !== null && data.confidence !== undefined) {
        confBadge.textContent = `${data.confidence}% confidence`;
        confBadge.style.display = '';
      } else {
        confBadge.style.display = 'none';
      }
    }

    if (data.all_probs) {
      renderConfidenceBars(data.all_probs);
    }

    // Render Per-Claim Table
    if (data.claims && data.claims.length > 0) {
      renderClaimsTable(data.claims);
    }

    // Focus Claim Card
    const claimEl = document.getElementById('claim-text');
    if (claimEl) claimEl.textContent = data.claim || '–';

    const evText = document.getElementById('evidence-text');
    const evLink = document.getElementById('evidence-link');
    const evChip = document.getElementById('evidence-badge-chip');

    if (data.evidence_text) {
      if (evText) {
        evText.textContent = data.evidence_text;
        evText.style.color = '';
      }
      if (evChip) {
        evChip.style.display = 'inline-block';
        evChip.textContent = `🌐 ${data.evidence_source || 'Live Web'}`;
        evChip.style.background = 'rgba(129, 140, 248, 0.15)';
        evChip.style.color = 'var(--accent)';
        evChip.style.border = '1px solid rgba(129, 140, 248, 0.3)';
      }
      if (evLink) {
        if (data.evidence_url) {
          evLink.href = data.evidence_url;
          const linkSpan = evLink.querySelector('span');
          if (linkSpan) linkSpan.textContent = `Source: ${data.evidence_source || 'Live Web Page'} →`;
          evLink.style.display = '';
        } else {
          evLink.style.display = 'none';
        }
      }
    } else {
      if (evText) {
        evText.textContent = '⚠️ No matching conclusive evidence was retrieved from live web sources.';
        evText.style.color = 'var(--amber)';
      }
      if (evChip) {
        evChip.style.display = 'inline-block';
        evChip.textContent = 'Inconclusive';
        evChip.style.background = 'rgba(245, 158, 11, 0.15)';
        evChip.style.color = 'var(--amber)';
        evChip.style.border = '1px solid rgba(245, 158, 11, 0.3)';
      }
      if (evLink) evLink.style.display = 'none';
    }

    // Structured Detailed Explainer
    const expEl = document.getElementById('explanation-text');
    const trueBox = document.getElementById('explainer-true-box');
    const trueText = document.getElementById('explainer-true-text');
    const whyBox = document.getElementById('explainer-why-box');
    const whyList = document.getElementById('explainer-why-list');
    const unvBox = document.getElementById('explainer-unverified-box');
    const unvText = document.getElementById('explainer-unverified-text');
    const confBox = document.getElementById('explainer-confidence-box');
    const confText = document.getElementById('explainer-confidence-text');

    if (data.explanation_details) {
      const d = data.explanation_details;
      if (expEl) expEl.textContent = d.verdict_summary || data.explanation || '–';

      if (trueBox && trueText) {
        if (d.what_is_actually_true && d.what_is_actually_true.trim()) {
          trueText.textContent = d.what_is_actually_true;
          trueBox.classList.remove('hidden');
        } else {
          trueBox.classList.add('hidden');
        }
      }

      if (whyBox && whyList) {
        if (Array.isArray(d.why_it_is_wrong_or_misleading) && d.why_it_is_wrong_or_misleading.length > 0) {
          whyList.innerHTML = d.why_it_is_wrong_or_misleading.map(item => `<li><span>${escHtml(item)}</span></li>`).join('');
          whyBox.classList.remove('hidden');
        } else {
          whyBox.classList.add('hidden');
        }
      }

      if (unvBox && unvText) {
        if (d.could_not_verify && d.could_not_verify.trim()) {
          unvText.textContent = d.could_not_verify;
          unvBox.classList.remove('hidden');
        } else {
          unvBox.classList.add('hidden');
        }
      }

      if (confBox && confText) {
        if (d.confidence_reason && d.confidence_reason.trim()) {
          confText.textContent = `Confidence rationale: ${d.confidence_reason}`;
          confBox.classList.remove('hidden');
        } else {
          confBox.classList.add('hidden');
        }
      }
    } else {
      if (expEl) {
        if (isInterim) {
          expEl.textContent = '⚡ Synthesizing comprehensive grounded explanation and cross-referencing citations...';
        } else {
          expEl.textContent = data.explanation || '–';
        }
      }
      if (trueBox) trueBox.classList.add('hidden');
      if (whyBox) whyBox.classList.add('hidden');
      if (unvBox) unvBox.classList.add('hidden');
      if (confBox) confBox.classList.add('hidden');
    }

    // Multi-Source Evidence Cards across Tiers
    const evSection = document.getElementById('all-evidence-section');
    const evGrid = document.getElementById('evidence-cards-container');
    const evCount = document.getElementById('evidence-cards-count');
    if (evSection && evGrid) {
      const cards = data.evidence_cards || [];
      if (cards.length > 0) {
        evSection.style.display = 'block';
        if (evCount) evCount.textContent = `${cards.length} Sources Analyzed`;
        evGrid.innerHTML = cards.map(c => {
          const tierNum = c.tier || 4;
          const tierLabel = c.tier_name ? c.tier_name.split(':')[0] : `Tier ${tierNum}`;
          const stanceStr = c.stance || 'neutral';
          return `
            <div class="evidence-card-item">
              <div class="evidence-card-header">
                <span class="pill-tier pill-tier${tierNum}">${escHtml(tierLabel)}</span>
                <span class="stance-badge stance-${escHtml(stanceStr)}">${escHtml(stanceStr.replace(/_/g, ' '))}</span>
              </div>
              <div class="evidence-card-quote">
                "${escHtml(c.quote || '')}"
              </div>
              ${c.reason ? `<div class="evidence-card-reason">${escHtml(c.reason)}</div>` : ''}
              <div class="evidence-card-footer">
                <a class="claim-source-tag" href="${escHtml(c.url || '#')}" target="_blank" rel="noopener">
                  🔗 ${escHtml(c.source || 'Web Source')} →
                </a>
                ${c.publish_date ? `<span class="evidence-card-date">📅 ${escHtml(c.publish_date)}</span>` : ''}
              </div>
            </div>
          `;
        }).join('');
      } else {
        evSection.style.display = 'none';
      }
    }

    renderLimeChart(data.lime_words || []);

    // Execution Stage Latencies
    const timingsGroup = document.getElementById('timings-group');
    if (timingsGroup && data.stage_timings) {
      timingsGroup.innerHTML = `
        <span class="timing-badge">Claims: ${data.stage_timings.claim_extraction_sec ?? 0}s</span>
        <span class="timing-badge">Live Search & Reading: ${data.stage_timings.live_search_and_reading_sec ?? 0}s</span>
        <span class="timing-badge">Verification: ${data.stage_timings.verification_sec ?? 0}s</span>
        <span class="timing-badge" style="color:var(--green);border-color:rgba(34,211,160,0.3)">Total: ${data.elapsed_sec}s</span>
      `;
    }

    const timeEl = document.getElementById('elapsed-time');
    if (timeEl && data.elapsed_sec) timeEl.textContent = `${data.elapsed_sec}s`;

    if (!isInterim) {
      showSection('results-section');
      const resSec = document.getElementById('results-section');
      if (resSec) resSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

  } catch (renderErr) {
    console.error('Error rendering results UI:', renderErr);
    showError(`Error displaying results: ${renderErr.message}`);
  }
}

// ── Button state helper ───────────────────────────────────────────────────────
function restoreButton() {
  const btn = document.getElementById('analyze-btn');
  if (btn) btn.disabled = false;
  const btnText = document.getElementById('btn-text');
  if (btnText) btnText.textContent = 'Verify Claim';
  const icon = document.getElementById('btn-icon');
  if (icon) icon.classList.remove('hidden');
  const spinner = document.getElementById('btn-spinner');
  if (spinner) spinner.classList.add('hidden');
}

// ── SSE / Streamed Analysis Execution ─────────────────────────────────────────
async function analyzeArticle() {
  const article = textarea ? textarea.value.trim() : '';

  if (!article) {
    if (textarea) {
      textarea.focus();
      textarea.style.borderColor = 'var(--red)';
      setTimeout(() => { textarea.style.borderColor = ''; }, 1500);
    }
    showError('Please paste an article or claim before verifying.');
    return;
  }

  const btn = document.getElementById('analyze-btn');
  if (btn) btn.disabled = true;
  const btnText = document.getElementById('btn-text');
  if (btnText) btnText.textContent = 'Verifying…';
  const icon = document.getElementById('btn-icon');
  if (icon) icon.classList.add('hidden');
  const spinner = document.getElementById('btn-spinner');
  if (spinner) spinner.classList.remove('hidden');

  showSection('loading-section');
  resetStageIndicators();

  ['bar-supported', 'bar-refuted', 'bar-misleading', 'bar-insufficient'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.width = '0%';
  });

  let finished = false;

  try {
    const url = `/analyze_stream?article=${encodeURIComponent(article)}`;
    const eventSource = new EventSource(url);

    eventSource.onmessage = (e) => {
      if (!e.data) return;
      try {
        const payload = JSON.parse(e.data);
        if (payload.type === 'stage') {
          updateActiveStage(payload.stage, payload.message);
        } else if (payload.type === 'verdict') {
          // Requirement E.5: Render verdict card immediately before explanation finishes
          renderResults(payload.result, true);
          const resSec = document.getElementById('results-section');
          if (resSec) resSec.classList.remove('hidden');
        } else if (payload.type === 'complete') {
          finished = true;
          eventSource.close();
          restoreButton();
          renderResults(payload.result, false);
        } else if (payload.type === 'error') {
          finished = true;
          eventSource.close();
          restoreButton();
          showError(payload.error || 'Verification error');
        }
      } catch (parseErr) {
        console.error('Error parsing SSE event:', parseErr);
      }
    };

    eventSource.onerror = async () => {
      eventSource.close();
      if (!finished) {
        console.warn('SSE stream closed, falling back to POST /analyze...');
        await analyzeArticleStandard(article);
      }
    };

  } catch (streamErr) {
    console.warn('EventSource setup failed, falling back to POST /analyze:', streamErr);
    await analyzeArticleStandard(article);
  }
}

async function analyzeArticleStandard(article) {
  try {
    const res = await fetch('/analyze', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ article }),
    });

    const data = await res.json();
    if (!res.ok || data.error) {
      showError(data.error || `Server error (${res.status})`);
      return;
    }
    renderResults(data, false);
  } catch (err) {
    console.error('Fetch error:', err);
    showError(`Network error: ${err.message}. Make sure the Flask server is running.`);
  } finally {
    restoreButton();
  }
}

function showError(msg) {
  const errEl = document.getElementById('error-message');
  if (errEl) errEl.textContent = msg;
  showSection('error-section');
  const errSec = document.getElementById('error-section');
  if (errSec) errSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

if (textarea) {
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      analyzeArticle();
    }
  });
}

function escHtml(str) {
  return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
