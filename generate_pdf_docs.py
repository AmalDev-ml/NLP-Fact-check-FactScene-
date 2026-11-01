import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FactSense_Project_Documentation.pdf")
ARTIFACT_PDF_PATH = r"C:\Users\hp\.gemini\antigravity\brain\ed2cb2bd-b238-4cdf-b816-593bdd0196fd\FactSense_Project_Documentation.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "FactSense 3.0 — Live Web Real-World Claim & News Verification System")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 742, letter[0] - 54, 742)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, footer_text)
        self.drawString(54, 36, "Confidential — NLP Academic Project | Batch R7A")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0] - 54, 48)
        self.restoreState()

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#475569'),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#334155'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=body_style,
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1e293b')
    )

    table_header = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell = ParagraphStyle(
        'TC',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )

    table_cell_bold = ParagraphStyle(
        'TCBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )

    story = []

    # Title block
    story.append(Paragraph("FactSense 3.0: System Architecture & Verification Pipeline", title_style))
    story.append(Paragraph("Complete Technical Documentation, Pipeline Workflow, Credibility Hierarchy & Evaluation", subtitle_style))
    
    meta_table = Table([
        [
            Paragraph("<b>Course:</b> NLP Academic Project", table_cell),
            Paragraph("<b>Batch:</b> R7A", table_cell),
            Paragraph("<b>Status:</b> Production Ready", table_cell)
        ],
        [
            Paragraph("<b>Students:</b> Amal Dev M L (215), A S Adityanarayanan (220), Rohit Ajoy (255)", table_cell),
            Paragraph("<b>Stack:</b> Flask, LangChain, Sentence-Transformers, Serper, Trafilatura", table_cell),
            Paragraph("<b>Date:</b> November 2026", table_cell)
        ]
    ], colWidths=[200, 200, 104])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Section 1: Executive Summary & Real-World Truth Philosophy
    story.append(Paragraph("1. Executive Summary & Verification Philosophy", h1_style))
    story.append(Paragraph(
        "FactSense 3.0 is an automated evidence-grounded claim verification and fake news detection engine. "
        "Unlike conventional text-matching or sentiment-based fake news classifiers, FactSense is designed around "
        "the <b>Real-World Truth Principle</b>: the system strictly evaluates whether an assertion is factually authentic "
        "in the physical real world, never whether it is merely circulating or trending across social media.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Solving Circulation Confounding:</b> When an input claims <i>'A Gazette circulating on social media, purportedly "
        "issued by the Ministry of Finance, declares every Saturday a holiday for PSU banks'</i>, typical search algorithms "
        "match thousands of viral social media posts confirming that the image is circulating. FactSense isolates the "
        "substantive authorization claim (<i>'Did the Ministry of Finance genuinely issue this order?'</i>) and consults "
        "primary regulatory portals (Tier 1) and certified fact-checkers (Tier 2). Because no official gazette was published, "
        "the system deterministically delivers a <b>REFUTED</b> or <b>MISLEADING</b> verdict.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Verdict Classes Table
    verdict_data = [
        [Paragraph("Verdict", table_header), Paragraph("Color", table_header), Paragraph("Definition & Decision Threshold", table_header)],
        [Paragraph("SUPPORTED", table_cell_bold), Paragraph("Green", table_cell), Paragraph("Corroborated by >= 1 Tier 1 (Official) or Tier 2/3 (Fact-Checker / Major News) source with no credible refutations.", table_cell)],
        [Paragraph("REFUTED", table_cell_bold), Paragraph("Red", table_cell), Paragraph("Directly denied or debunked by an official authority (Tier 1), certified fact-checker (Tier 2), or consensus credible reporting (Tier 3).", table_cell)],
        [Paragraph("MISLEADING", table_cell_bold), Paragraph("Purple", table_cell), Paragraph("Mixes true facts with false assertions, displays major headline-body framing exaggeration, or is rated 'Partly False' by fact-checkers.", table_cell)],
        [Paragraph("INSUFFICIENT_EVIDENCE", table_cell_bold), Paragraph("Amber", table_cell), Paragraph("The claim cannot be reliably confirmed or refuted across search engines (never defaulted to Supported).", table_cell)]
    ]
    vt = Table(verdict_data, colWidths=[120, 60, 324])
    vt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(vt)
    story.append(Spacer(1, 12))

    # Section 2: Pipeline Architecture
    story.append(Paragraph("2. Modular 8-Stage Pipeline Architecture", h1_style))
    story.append(Paragraph(
        "FactSense replaces static knowledge caches with eight decoupled, specialized processing stages:",
        body_style
    ))

    stages = [
        ("Stage 1: claim_extractor.py", "Reads the entire input text. Performs headline detection, extracts up to 6 self-contained atomic claims, detects temporal scope (e.g., current, historical, 2026), and normalizes viral rumors into document authenticity questions."),
        ("Stage 2: query_generator.py", "Generates 4 diverse search query angles per claim: neutral, debunk ('fake', 'hoax', 'fact check'), official portal (using site: filters such as pib.gov.in, rbi.org.in), and certified fact-checkers (Alt News, BOOM, Snopes)."),
        ("Stage 3: web_search.py", "Concurrent live retrieval across Serper (Google), DuckDuckGo, and Wikipedia REST API. Features tiered Trafilatura page reading: parses full body text and publish date for top 3 hits, and falls back to search snippets for hits 4-8."),
        ("Stage 4: source_ranker.py", "Applies the 4-Tier Credibility Hierarchy. Computes dense semantic embeddings using all-MiniLM-L6-v2, strictly enforces a raw similarity threshold >= 0.35, applies a staleness penalty to outdated sources, and enforces Wikipedia background rules."),
        ("Stage 5: stance_classifier.py", "Batched LLM stance classification into 6 granular categories (affirms, denies, debunks_circulating_claim, partially_supports, mentions_without_verifying, neutral) with verbatim quote grounding. Lazy-loads DeBERTa-v3 NLI on demand."),
        ("Stage 6: framing_checker.py", "Evaluates headline vs. body text across a 9-point checklist (exaggeration, correlation as causation, unrepresentative sample, cherry-picked statistics, outdated info, misattributed photo/video) to detect contextual deception."),
        ("Stage 7: verdict_engine.py", "Executes deterministic decision rules, fact-checker override rules ('Partly False' -> MISLEADING), dynamic confidence calculation, and realistic 4-class probability distribution synthesis. Emits an early 'verdict' SSE event."),
        ("Stage 8: explainer.py", "Generates an 8-part grounded factual explanation with zero hallucinations: verdict summary, what is actually true (citing sources), why it is wrong/misleading, and unverified aspects."),
    ]

    for st_title, st_desc in stages:
        story.append(Paragraph(f"• <b>{st_title}:</b> {st_desc}", bullet_style))

    story.append(Spacer(1, 10))

    # Section 3: Credibility Hierarchy
    story.append(Paragraph("3. Four-Tier Credibility Hierarchy", h1_style))
    story.append(Paragraph(
        "To ensure verifiable reliability, all retrieved passages are evaluated against strict domain criteria:",
        body_style
    ))

    tier_table_data = [
        [Paragraph("Tier", table_header), Paragraph("Domain & Suffix Patterns", table_header), Paragraph("Credibility Scope & Weight", table_header)],
        [Paragraph("Tier 1: Official Authorities", table_cell_bold), Paragraph(".gov, .gov.in, .nic.in, rbi.org.in, egazette.gov.in, whitehouse.gov, supremecourt.gov", table_cell), Paragraph("<b>Highest Authority.</b> Sole authority on government orders, statutory gazettes, and official appointments. Overrides all other tiers.", table_cell)],
        [Paragraph("Tier 2: Fact-Checkers", table_cell_bold), Paragraph("altnews.in, boomlive.in, snopes.com, factcheck.org, politifact.com, pib.gov.in/factcheck", table_cell), Paragraph("<b>High Authority on Debunks.</b> Certified by the International Fact-Checking Network (IFCN). Overrides social media repetition.", table_cell)],
        [Paragraph("Tier 3: News & Academic", table_cell_bold), Paragraph("reuters.com, apnews.com, bbc.com, thehindu.com, .edu, .ac.in, wikipedia.org, britannica.com", table_cell), Paragraph("<b>Reputable Secondary Evidence.</b> Corroborates international events, science, and history. Wikipedia cannot sole-support breaking news.", table_cell)],
        [Paragraph("Tier 4: General Web", table_cell_bold), Paragraph("reddit.com, twitter.com, x.com, instagram.com, medium.com, unverified commercial blogs", table_cell), Paragraph("<b>Unverified Baseline.</b> Reflects viral circulation or public sentiment; cannot support a claim without corroboration from Tiers 1-3.", table_cell)]
    ]
    tt = Table(tier_table_data, colWidths=[120, 150, 234])
    tt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(tt)
    story.append(Spacer(1, 12))

    # Section 4: Decision Rules Matrix
    story.append(Paragraph("4. Deterministic Decision Synthesis & Overrides", h1_style))
    
    rules_data = [
        [Paragraph("Scenario / Evidence Profile", table_header), Paragraph("Verdict", table_header), Paragraph("Precedence Logic & Policy", table_header)],
        [Paragraph(">= 1 Tier 1/2 source affirms; 0 credible denials", table_cell), Paragraph("SUPPORTED", table_cell_bold), Paragraph("Official record or certified investigation confirms authenticity.", table_cell)],
        [Paragraph(">= 1 Tier 1 official denial exists", table_cell), Paragraph("REFUTED", table_cell_bold), Paragraph("Official government or primary issuer denial overrides any number of Tier 4 claims.", table_cell)],
        [Paragraph(">= 1 Tier 2 fact-checker debunk exists", table_cell), Paragraph("REFUTED", table_cell_bold), Paragraph("Certified fact-checker debunks viral image, forged circular, or social media hoax.", table_cell)],
        [Paragraph("Fact-checker rates 'Partly False' / 'Missing Context'", table_cell), Paragraph("MISLEADING", table_cell_bold), Paragraph("Fact-checker override rule triggers Misleading unless a Tier 1 denial refutes it.", table_cell)],
        [Paragraph("Headline framing severity is 'major'", table_cell), Paragraph("MISLEADING", table_cell_bold), Paragraph("Headline asserts claims contradicted or unverified by the article's own body text.", table_cell)],
        [Paragraph("Article contains both supported and refuted claims", table_cell), Paragraph("MISLEADING", table_cell_bold), Paragraph("Mixed true/false content flagged; refuted parts listed in failed_parts.", table_cell)],
        [Paragraph("Conflicting Tier 3 reporting (Affirm vs Deny)", table_cell), Paragraph("MISLEADING", table_cell_bold), Paragraph("Major news agencies report contradictory information without consensus.", table_cell)],
        [Paragraph("Only Wikipedia affirms news/document/statistic", table_cell), Paragraph("INSUFFICIENT", table_cell_bold), Paragraph("Wikipedia background restriction; requires primary news or official records.", table_cell)],
        [Paragraph("Only Tier 4 blogs or social media mention claim", table_cell), Paragraph("INSUFFICIENT", table_cell_bold), Paragraph("Unverified claims without official or reputable news backing are never marked Supported.", table_cell)]
    ]
    rt = Table(rules_data, colWidths=[150, 85, 269])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#475569')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(rt)
    story.append(Spacer(1, 14))

    # Section 5: Acceptance Test Suite
    story.append(Paragraph("5. Automated Acceptance Test Suite (tests/test_phase3.py)", h1_style))
    story.append(Paragraph(
        "FactSense is validated through 10 comprehensive acceptance test scenarios:",
        body_style
    ))

    test_data = [
        [Paragraph("#", table_header), Paragraph("Test Scenario", table_header), Paragraph("Target Claim / Article Input", table_header), Paragraph("Expected Verdict", table_header), Paragraph("Status", table_header)],
        [Paragraph("1", table_cell), Paragraph("Headline Exaggeration", table_cell), Paragraph("AI tools make students score 40% higher (12-student survey)", table_cell), Paragraph("MISLEADING", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("2", table_cell), Paragraph("Accurate Official Fact", table_cell), Paragraph("Narendra Modi is the Prime Minister of India.", table_cell), Paragraph("SUPPORTED", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("3a", table_cell), Paragraph("Viral Fake Gazette", table_cell), Paragraph("Social media Gazette declares all Saturdays PSU holidays", table_cell), Paragraph("REFUTED", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("3b", table_cell), Paragraph("Debunk Formulation", table_cell), Paragraph("Same circular statement + 'is fake'", table_cell), Paragraph("REFUTED", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("4", table_cell), Paragraph("False Implication", table_cell), Paragraph("Ice cream sales cause shark attacks in July (correlation)", table_cell), Paragraph("MISLEADING", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("5", table_cell), Paragraph("Misattributed Event", table_cell), Paragraph("2026 video shows severe blizzard in downtown Cairo Egypt", table_cell), Paragraph("REFUTED", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("6", table_cell), Paragraph("Fact-Checker Override", table_cell), Paragraph("Fact-checker rates EV lifetime emissions 'Partly False'", table_cell), Paragraph("MISLEADING", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("7", table_cell), Paragraph("Wikipedia News Rule", table_cell), Paragraph("Breaking earthquake news found only on Wikipedia and blog", table_cell), Paragraph("INSUFFICIENT", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("8", table_cell), Paragraph("Nonsense / Sci-Fi", table_cell), Paragraph("Oceans of hot chocolate discovered on Jupiter moon", table_cell), Paragraph("INSUFFICIENT", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("9", table_cell), Paragraph("No Headline Article", table_cell), Paragraph("Multi-sentence paragraph without lead title", table_cell), Paragraph("SUPPORTED", table_cell_bold), Paragraph("PASSED", table_cell)],
        [Paragraph("10", table_cell), Paragraph("Degraded Mode Cap", table_cell), Paragraph("Mocked execution without external LLM API key", table_cell), Paragraph("degraded=True", table_cell_bold), Paragraph("PASSED", table_cell)]
    ]
    tst = Table(test_data, colWidths=[20, 110, 204, 110, 60])
    tst.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#065f46')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#94a3b8')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(tst)
    story.append(Spacer(1, 14))

    # Section 6: User Interface & Performance Optimizations
    story.append(Paragraph("6. Frontend Features & Performance Optimizations", h1_style))
    story.append(Paragraph(
        "• <b>Sub-8s Dual-Stage SSE Streaming:</b> An early <code>verdict</code> event is dispatched the moment Stage 7 completes. "
        "The browser immediately displays the verdict card, probability bars, and claim breakdown table while the slower "
        "grounded explanation (Stage 8) synthesizes in the background.<br/>"
        "• <b>Headline Framing Card:</b> A side-by-side comparative card directly illustrates what the headline implies versus "
        "what the body text actually proves.<br/>"
        "• <b>Degraded Mode Safety Banner:</b> If no LLM key is configured, FactSense warns users that it is running in local heuristic "
        "mode, capping maximum confidence at 60%.<br/>"
        "• <b>Lazy-Loaded DeBERTa NLI:</b> Neural NLI models are lazy-loaded on demand (<code>USE_DEBERTA_CROSSCHECK=false</code> by default), "
        "ensuring instant server boot without memory lag.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF documentation successfully at: {PDF_PATH}")

    # Copy to artifact path
    try:
        import shutil
        os.makedirs(os.path.dirname(ARTIFACT_PDF_PATH), exist_ok=True)
        shutil.copy2(PDF_PATH, ARTIFACT_PDF_PATH)
        print(f"Copied to artifact directory: {ARTIFACT_PDF_PATH}")
    except Exception as e:
        print(f"Notice copying to artifact: {e}")

if __name__ == "__main__":
    build_pdf()

