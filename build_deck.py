"""Builds the executive decision deck: own model vs paid LLM API for email intent classification.

Grids are composed from individual shapes rather than native PPTX tables — the
adxppt converter's `render_text_body` looks for `a:p` directly under `a:tc`,
but PowerPoint nests them inside `a:txBody`, so native table cells render empty.
Shapes also let us colour cells, which the converter's table path doesn't support.

Every paragraph sets alignment and anchor explicitly: the converter defaults
both to "center" when the attribute is absent.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

W, H = 13.333, 7.5

NAVY   = RGBColor(0x0B, 0x24, 0x47)
BLUE   = RGBColor(0x19, 0x37, 0x6D)
ACCENT = RGBColor(0x57, 0x6C, 0xBC)
SKY    = RGBColor(0xA5, 0xD7, 0xE8)
GREEN  = RGBColor(0x1E, 0x7A, 0x46)
GREENL = RGBColor(0xE3, 0xF3, 0xE9)
RED    = RGBColor(0xB3, 0x26, 0x1E)
REDL   = RGBColor(0xFB, 0xE9, 0xE7)
AMBER  = RGBColor(0x9A, 0x66, 0x00)
AMBERL = RGBColor(0xFF, 0xF4, 0xDF)
GREY   = RGBColor(0x5A, 0x64, 0x72)
DARK   = RGBColor(0x1A, 0x1A, 0x1A)
BGL    = RGBColor(0xF5, 0xF7, 0xFA)
BGL2   = RGBColor(0xEA, 0xEF, 0xF5)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)


def blank():
    """Title Only layout — the converter reads the title placeholder for index.html."""
    return prs.slides.add_slide(prs.slide_layouts[5])


def rect(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, line_w=1.0):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    if s.has_text_frame:
        s.text_frame.word_wrap = True
    return s


def write(shape, lines, anchor=MSO_ANCHOR.MIDDLE, margin=0.06):
    """lines: list of (text, size_pt, bold, color, align)."""
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    for i, (text, size, bold, color, align) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return shape


def textbox(slide, x, y, w, h, lines, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    write(tb, lines, anchor=anchor, margin=0.0)
    return tb


def header(slide, title, kicker=None, sub=None):
    rect(slide, 0, 0, W, H, fill=WHITE)
    rect(slide, 0, 0, W, 1.02, fill=NAVY)
    rect(slide, 0, 1.02, W, 0.055, fill=ACCENT)

    ph = slide.shapes.title
    ph.left, ph.top, ph.width, ph.height = Inches(0.55), Inches(0.20), Inches(9.6), Inches(0.50)
    write(ph, [(title, 25, True, WHITE, PP_ALIGN.LEFT)], anchor=MSO_ANCHOR.MIDDLE, margin=0.0)

    if kicker:
        textbox(slide, 0.55, 0.66, 9.6, 0.28,
                [(kicker, 12, False, SKY, PP_ALIGN.LEFT)])
    if sub:
        textbox(slide, 0.55, 1.22, 12.2, 0.38,
                [(sub, 13, False, GREY, PP_ALIGN.LEFT)])


def pagenum(slide, n, total=9):
    textbox(slide, W - 1.15, H - 0.42, 0.7, 0.28,
            [(f"{n} / {total}", 10, False, GREY, PP_ALIGN.RIGHT)])
    # The title placeholder is created first and would be painted over by the
    # background shapes added after it; re-append it so it sits on top.
    el = slide.shapes.title._element
    el.getparent().append(el)


def grid(slide, x, y, widths, rows, row_h=0.44, head_h=0.44,
         head_fill=BLUE, head_color=WHITE, sizes=None, zebra=True,
         cell_colors=None, bolds=None, head_size=11, body_size=10.5):
    """rows[0] is the header row. cell_colors/bolds: dict {(r,c): value}."""
    cell_colors = cell_colors or {}
    bolds = bolds or {}
    sizes = sizes or {}
    cy = y
    for r, row in enumerate(rows):
        cx = x
        h = head_h if r == 0 else row_h
        for c, val in enumerate(row):
            w = widths[c]
            if r == 0:
                fill, color, bold = head_fill, head_color, True
                size = head_size
            else:
                fill = cell_colors.get((r, c))
                if fill is None:
                    fill = BGL if (zebra and r % 2 == 0) else WHITE
                color = DARK
                bold = bolds.get((r, c), False)
                size = sizes.get((r, c), body_size)
            align = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            sh = rect(slide, cx, cy, w, h, fill=fill,
                      line=RGBColor(0xD8, 0xDE, 0xE6) if r > 0 else None)
            if isinstance(val, tuple):
                txt, color = val[0], val[1]
                bold = val[2] if len(val) > 2 else bold
            else:
                txt = val
            write(sh, [(txt, size, bold, color, align)])
            cx += w
        cy += h
    return cy


def bullet_card(slide, x, y, w, h, title, body, accent=ACCENT, fill=BGL, tsize=13, bsize=11):
    rect(slide, x, y, w, h, fill=fill)
    rect(slide, x, y, 0.06, h, fill=accent)
    lines = [(title, tsize, True, NAVY, PP_ALIGN.LEFT)]
    for b in body:
        lines.append((b, bsize, False, DARK, PP_ALIGN.LEFT))
    tb = slide.shapes.add_textbox(Inches(x + 0.22), Inches(y + 0.12),
                                  Inches(w - 0.38), Inches(h - 0.24))
    write(tb, lines, anchor=MSO_ANCHOR.TOP, margin=0.0)


# ─────────────────────────────────────────────────────────── 1. Title
s = blank()
rect(s, 0, 0, W, H, fill=NAVY)
rect(s, 0, 0, 0.28, H, fill=ACCENT)
rect(s, 7.9, 0, 5.43, H, fill=BLUE)

ph = s.shapes.title
ph.left, ph.top, ph.width, ph.height = Inches(0.95), Inches(2.05), Inches(6.7), Inches(1.5)
write(ph, [("Email Intent Classification", 38, True, WHITE, PP_ALIGN.LEFT),
           ("Build our own model, or rent an LLM API?", 21, False, SKY, PP_ALIGN.LEFT)],
      anchor=MSO_ANCHOR.MIDDLE, margin=0.0)

textbox(s, 0.95, 1.45, 6.7, 0.35,
        [("DECISION BRIEFING", 12, True, ACCENT, PP_ALIGN.LEFT)])
textbox(s, 0.95, 3.75, 6.7, 0.9,
        [("Automating triage of ~10,000 procurement emails per month", 14, False, WHITE, PP_ALIGN.LEFT),
         ("across shared supplier inboxes.", 14, False, WHITE, PP_ALIGN.LEFT)])
textbox(s, 0.95, 6.55, 6.7, 0.3,
        [("Prepared by Data & ML Engineering  ·  September 2026", 11, False, GREY, PP_ALIGN.LEFT)])

rect(s, 8.5, 2.25, 4.2, 2.55, fill=WHITE)
rect(s, 8.5, 2.25, 4.2, 0.5, fill=GREEN)
write(rect(s, 8.5, 2.25, 4.2, 0.5, fill=GREEN),
      [("RECOMMENDATION", 12, True, WHITE, PP_ALIGN.CENTER)])
textbox(s, 8.75, 2.95, 3.7, 1.75,
        [("Build our own model —", 16, True, NAVY, PP_ALIGN.LEFT),
         ("using a paid LLM as a", 16, True, NAVY, PP_ALIGN.LEFT),
         ("one-time teacher, not as", 16, True, NAVY, PP_ALIGN.LEFT),
         ("the runtime engine.", 16, True, NAVY, PP_ALIGN.LEFT),
         ("", 7, False, DARK, PP_ALIGN.LEFT),
         ("Already validated on 4,499 real emails.", 10.5, False, GREEN, PP_ALIGN.LEFT)])
pagenum(s, 1)

# ─────────────────────────────────────────────────────────── 2. Problem
s = blank()
header(s, "The problem we are solving", "THE OPPORTUNITY",
       "Shared procurement inboxes are triaged by hand. Every email is read by a person before anything happens.")

grid(s, 0.55, 1.85, [3.0, 4.6, 4.6],
     [["", "Today", "With intent classification"],
      ["Triage", "Analyst reads every email", "Auto-classified on arrival"],
      ["Routing", "Manual forward to the right team", "Routed by intent, automatically"],
      ["Prioritisation", "First-in, first-out", "Urgent and high-value first"],
      ["Downstream systems", "Re-keyed into SAP by hand", "Structured trigger for SAP actions"],
      ["Volume handled", "~500/day, people-limited", "~500/day, no added headcount"]],
     row_h=0.52,
     cell_colors={(r, 2): GREENL for r in range(1, 6)},
     bolds={(r, 0): True for r in range(1, 6)})

bullet_card(s, 0.55, 5.35, 3.95, 1.55, "Scale",
            ["~10,000 emails / month", "~400–500 per day", "Multiple shared inboxes"])
bullet_card(s, 4.72, 5.35, 3.95, 1.55, "Content",
            ["Supplier bank details", "Contract pricing", "Invoices, POs, disputes"], accent=AMBER, fill=AMBERL)
bullet_card(s, 8.89, 5.35, 3.89, 1.55, "Implication",
            ["Highly sensitive data", "Financial consequences", "Auditability is mandatory"], accent=RED, fill=REDL)
pagenum(s, 2)

# ─────────────────────────────────────────────────────────── 3. Options
s = blank()
header(s, "Four ways to do this", "THE OPTIONS",
       "All four were evaluated. The real choice is between C and D — a model we own, or a service we rent.")

grid(s, 0.55, 1.85, [2.5, 4.0, 1.9, 2.0, 1.85],
     [["Approach", "What it is", "Accuracy", "Cost / month", "Data leaves us?"],
      ["A. Keyword / TF-IDF", "Hand-written rules and word counts", "Low", "~$0", "No"],
      ["B. Embeddings + head", "Frozen embeddings, small classifier", "Medium", "Low", "No"],
      ["C. Fine-tuned ModernBERT", "Our own model, trained on our mail", "High", "Fixed, ~$0 marginal", "No"],
      ["D. Paid LLM API", "Commercial LLM called per email", "High", "Grows with volume", "Yes"]],
     row_h=0.58,
     cell_colors={**{(3, c): GREENL for c in range(5)},
                  **{(4, c): AMBERL for c in range(5)}},
     bolds={(3, 0): True, (4, 0): True, (1, 0): True, (2, 0): True})

rect(s, 0.55, 4.62, 6.05, 2.25, fill=GREENL)
rect(s, 0.55, 4.62, 0.06, 2.25, fill=GREEN)
textbox(s, 0.8, 4.78, 5.6, 2.0,
        [("C — Our own model  ✓ recommended", 14, True, GREEN, PP_ALIGN.LEFT),
         ("", 6, False, DARK, PP_ALIGN.LEFT),
         ("A compact 150M-parameter model, fine-tuned on our own", 11, False, DARK, PP_ALIGN.LEFT),
         ("emails and our own intent taxonomy.", 11, False, DARK, PP_ALIGN.LEFT),
         ("", 6, False, DARK, PP_ALIGN.LEFT),
         ("Runs inside our network. Costs the same whether we", 11, False, DARK, PP_ALIGN.LEFT),
         ("classify 10,000 or 1,000,000 emails. Version-frozen,", 11, False, DARK, PP_ALIGN.LEFT),
         ("so an audit can reproduce any decision we made.", 11, False, DARK, PP_ALIGN.LEFT)])

rect(s, 6.98, 4.62, 5.8, 2.25, fill=AMBERL)
rect(s, 6.98, 4.62, 0.06, 2.25, fill=AMBER)
textbox(s, 7.23, 4.78, 5.35, 2.0,
        [("D — Paid LLM API", 14, True, AMBER, PP_ALIGN.LEFT),
         ("", 6, False, DARK, PP_ALIGN.LEFT),
         ("Fastest to a first result, and genuinely strong at", 11, False, DARK, PP_ALIGN.LEFT),
         ("understanding language it has never seen.", 11, False, DARK, PP_ALIGN.LEFT),
         ("", 6, False, DARK, PP_ALIGN.LEFT),
         ("But every email — including supplier bank details —", 11, False, DARK, PP_ALIGN.LEFT),
         ("leaves our network, on every single call, forever.", 11, False, DARK, PP_ALIGN.LEFT),
         ("The vendor can change the model underneath us.", 11, False, DARK, PP_ALIGN.LEFT)])
pagenum(s, 3)

# ─────────────────────────────────────────────────────────── 4. Comparison
s = blank()
header(s, "Own model vs. paid LLM API", "HEAD TO HEAD",
       "Eight dimensions that matter to us. Green marks the stronger option on each row.")

grid(s, 0.55, 1.8, [3.15, 4.55, 4.55],
     [["Dimension", "Our own fine-tuned model", "Paid LLM API"],
      ["Accuracy on our taxonomy", "89% measured — learns our 20 intents", "Strong generally; must be re-told our intents every call"],
      ["Latency per email", "~15 milliseconds", "2–5 seconds (100–300× slower)"],
      ["Marginal cost per email", "~$0 once built", "$0.006 – $0.05 (measured)"],
      ["Sensitive data residency", "Never leaves our network", "Bank details and pricing sent to a third party"],
      ["Reproducibility / audit", "Version frozen — any decision replayable", "Vendor updates the model without notice"],
      ["Availability", "No rate limits, no external outage", "Rate limits; dependent on vendor uptime"],
      ["Vendor lock-in", "None — the asset is ours", "High — pricing and terms can change"],
      ["Time to first result", "6–8 weeks", "Days"]],
     row_h=0.47,
     cell_colors={(1, 1): GREENL, (2, 1): GREENL, (3, 1): GREENL, (4, 1): GREENL,
                  (5, 1): GREENL, (6, 1): GREENL, (7, 1): GREENL, (8, 2): GREENL},
     bolds={(r, 0): True for r in range(1, 9)})

rect(s, 0.55, 6.28, 12.23, 0.72, fill=NAVY)
textbox(s, 0.85, 6.42, 11.7, 0.5,
        [("The LLM API wins on exactly one dimension — how fast we can start. We can have that too, by using it as the teacher (slide 7).",
          12.5, True, WHITE, PP_ALIGN.LEFT)])
pagenum(s, 4)

# ─────────────────────────────────────────────────────────── 5. Cost
s = blank()
header(s, "What it actually costs", "THE HONEST NUMBERS",
       "Token counts below are measured on our own 4,499-email corpus, not estimated. Pricing at frontier-model rates.")

textbox(s, 0.55, 1.72, 6.05, 0.3,
        [("Paid API run-rate  ·  120,000 emails / year", 12.5, True, NAVY, PP_ALIGN.LEFT)])
grid(s, 0.55, 2.08, [3.15, 1.35, 1.55],
     [["Prompt scenario", "Tok / call", "$ / yr"],
      ["Truncated + prompt caching", "~730", "$0.4k"],
      ["As piloted (truncated)", "1,368", "$0.7k"],
      ["Full email, untruncated", "5,238", "$2.1k"],
      ["Full thread + 3× consistency", "~15,700", "$6.2k"],
      ["Enterprise-wide (10× volume)", "—", "$4k – 62k"]],
     row_h=0.46, head_h=0.44, body_size=10,
     bolds={(r, 0): True for r in range(1, 6)})

textbox(s, 6.98, 1.72, 5.8, 0.3,
        [("Three-year total cost of ownership", 12.5, True, NAVY, PP_ALIGN.LEFT)])
grid(s, 6.98, 2.08, [2.9, 1.45, 1.45],
     [["", "Paid API", "Own model"],
      ["Engineering build", "$0", "$20k – 35k"],
      ["One-time LLM labelling", "$0", "~$0.5k"],
      ["Hosting / run, 3 yrs", "included", "~$6k"],
      ["3-yr total — today", "$1k – 19k", "$26k – 41k"],
      ["3-yr total — 10× volume", "$12k – 186k", "$28k – 44k"]],
     row_h=0.46, head_h=0.44, body_size=10,
     cell_colors={(4, 1): GREENL, (5, 2): GREENL},
     bolds={(4, 0): True, (5, 0): True, (4, 1): True, (5, 2): True,
            (1, 0): True, (2, 0): True, (3, 0): True})

textbox(s, 0.55, 4.95, 12.2, 0.28,
        [("Green marks the cheaper option in each row. Engineering estimated at one fully-loaded engineer for 6–8 weeks.",
          9.5, False, GREY, PP_ALIGN.LEFT)])

rect(s, 0.55, 5.38, 12.23, 1.5, fill=AMBERL)
rect(s, 0.55, 5.38, 0.06, 1.5, fill=AMBER)
textbox(s, 0.85, 5.52, 11.7, 1.3,
        [("Stated plainly: at today's volume, building our own model is the more expensive option.", 13, True, AMBER, PP_ALIGN.LEFT),
         ("Break-even sits at roughly $10–12k / year of API spend — about ten times our current volume, or any scenario using full", 11.5, False, DARK, PP_ALIGN.LEFT),
         ("thread context with self-consistency checks. Below that line, the API is cheaper and we should not pretend otherwise.", 11.5, False, DARK, PP_ALIGN.LEFT),
         ("We are recommending the build for control, auditability and data residency — not to save money in year one.", 11.5, True, NAVY, PP_ALIGN.LEFT)])
pagenum(s, 5)

# ─────────────────────────────────────────────────────────── 6. Evidence
s = blank()
header(s, "We have already proven it works", "EVIDENCE — WORKING PROTOTYPE",
       "Not a proposal on paper. An end-to-end pipeline was built and run on 4,499 real emails.")

for i, (x, big, cap) in enumerate([
        (0.55, "4,499", "emails extracted\nand cleaned"),
        (3.12, "15", "intents discovered\nand validated"),
        (5.69, "89.1%", "accuracy on held-out\ntest data"),
        (8.26, "84.3%", "macro-F1 across\nall 15 classes"),
        (10.83, "~1 day", "from raw mailbox\nto trained model")]):
    w = 2.4 if i < 4 else 1.95
    rect(s, x, 1.8, w, 1.5, fill=BGL)
    rect(s, x, 1.8, w, 0.06, fill=ACCENT)
    textbox(s, x + 0.1, 2.0, w - 0.2, 0.6,
            [(big, 26, True, NAVY, PP_ALIGN.CENTER)])
    textbox(s, x + 0.1, 2.62, w - 0.2, 0.6,
            [(line, 10, False, GREY, PP_ALIGN.CENTER) for line in cap.split("\n")])

textbox(s, 0.55, 3.52, 12.2, 0.3,
        [("How it was done", 13, True, NAVY, PP_ALIGN.LEFT)])
grid(s, 0.55, 3.88, [0.62, 4.3, 7.31],
     [["#", "Step", "Result"],
      ["1", "Extract history from the mailbox", "4,499 emails, cleaned of HTML, quotes and footers"],
      ["2", "Discover the intent taxonomy", "15 intents generated from the real corpus, not guessed"],
      ["3", "Label the corpus with an LLM (one-time)", "100% labelled; only 5.4% needed human attention"],
      ["4", "Fine-tune our own model", "ModernBERT-base, on a single laptop GPU, in minutes"],
      ["5", "Measure on held-out data", "89.1% accuracy / 84.3% macro-F1 on unseen emails"]],
     row_h=0.44, bolds={(r, 0): True for r in range(1, 6)})

rect(s, 0.55, 6.6, 12.23, 0.55, fill=GREENL)
rect(s, 0.55, 6.6, 0.06, 0.55, fill=GREEN)
textbox(s, 0.85, 6.69, 11.7, 0.4,
        [("The hardware was a laptop GPU. The procurement version needs better data and business review — not better technology.",
          12, True, GREEN, PP_ALIGN.LEFT)])
pagenum(s, 6)

# ─────────────────────────────────────────────────────────── 7. Hybrid
s = blank()
header(s, "The winning move: use both, in the right roles", "RECOMMENDED APPROACH",
       "This is not build-or-buy. We buy the LLM once for what it is best at, then own the asset it produces.")

rect(s, 0.55, 1.85, 12.23, 1.9, fill=BGL)
for x, w, title, sub, fill, color in [
        (0.85, 2.55, "Paid LLM", "Labels our corpus\nONE TIME  ·  ~$500", AMBERL, AMBER),
        (4.05, 2.55, "Labelled corpus", "Our data, our intents\nOwned by us", BGL2, BLUE),
        (7.25, 2.55, "Our own model", "Fine-tuned ModernBERT\nTrained on our corpus", GREENL, GREEN),
        (10.45, 2.05, "Runtime", "Classifies every email\n~$0 per email", GREENL, GREEN)]:
    rect(s, x, 2.12, w, 1.35, fill=fill)
    rect(s, x, 2.12, w, 0.055, fill=color)
    textbox(s, x + 0.12, 2.32, w - 0.24, 0.35,
            [(title, 14, True, color, PP_ALIGN.CENTER)])
    textbox(s, x + 0.12, 2.72, w - 0.24, 0.7,
            [(ln, 10, False, DARK, PP_ALIGN.CENTER) for ln in sub.split("\n")])

for ax in (3.52, 6.72, 9.92):
    textbox(s, ax, 2.58, 0.45, 0.4, [("→", 20, True, ACCENT, PP_ALIGN.CENTER)])

textbox(s, 0.55, 4.0, 12.2, 0.3,
        [("Why this is the strongest position", 13, True, NAVY, PP_ALIGN.LEFT)])
bullet_card(s, 0.55, 4.38, 3.95, 1.5, "We get the speed",
            ["The LLM does the slow part —", "labelling — in days, not months."], accent=GREEN, fill=GREENL)
bullet_card(s, 4.72, 4.38, 3.95, 1.5, "We keep the asset",
            ["The labelled corpus and the model", "are ours permanently."], accent=GREEN, fill=GREENL)
bullet_card(s, 8.89, 4.38, 3.89, 1.5, "We stop paying",
            ["The teacher is paid once.", "The runtime costs us nothing."], accent=GREEN, fill=GREENL)

rect(s, 0.55, 6.1, 12.23, 0.88, fill=NAVY)
textbox(s, 0.85, 6.23, 11.7, 0.65,
        [("Renting an LLM per email means paying forever for a capability we never own, while our most sensitive data leaves the network on every call.",
          12, False, WHITE, PP_ALIGN.LEFT),
         ("Paying once to train our own model turns that same spend into an asset on our balance sheet.",
          12, True, SKY, PP_ALIGN.LEFT)])
pagenum(s, 7)

# ─────────────────────────────────────────────────────────── 8. Risks
s = blank()
header(s, "What could go wrong, and how we handle it", "RISKS — STATED UP FRONT",
       "These are real. All are manageable, and all are cheaper to address than an unbounded API dependency.")

grid(s, 0.55, 1.85, [3.5, 4.6, 4.13],
     [["Risk", "Why it matters", "How we handle it"],
      ["Needs labelled data", "A model is only as good as its examples", "LLM labels the bulk; people check the uncertain 5%"],
      ["Labels may be noisy", "Bad labels quietly cap accuracy", "600-email human gold set measures the truth"],
      ["Taxonomy may not be learnable", "Business categories can overlap in text", "Validated against real mail before we spend"],
      ["Language drifts over time", "New suppliers, new systems, new wording", "Weekly retraining, with a quality gate"],
      ["ML skills needed in-house", "We must be able to maintain it", "Standard open tooling; pipeline already built"],
      ["Multi-topic emails", "One email can carry two requests", "Detected and escalated to a person"]],
     row_h=0.53,
     cell_colors={(r, 2): GREENL for r in range(1, 7)},
     bolds={(r, 0): True for r in range(1, 7)})

rect(s, 0.55, 5.5, 6.05, 1.45, fill=REDL)
rect(s, 0.55, 5.5, 0.06, 1.45, fill=RED)
textbox(s, 0.8, 5.65, 5.6, 1.2,
        [("The risk of doing nothing", 13, True, RED, PP_ALIGN.LEFT),
         ("Triage stays people-limited. Volume grows,", 11, False, DARK, PP_ALIGN.LEFT),
         ("headcount has to grow with it, and nothing", 11, False, DARK, PP_ALIGN.LEFT),
         ("downstream can be automated.", 11, False, DARK, PP_ALIGN.LEFT)])

rect(s, 6.98, 5.5, 5.8, 1.45, fill=AMBERL)
rect(s, 6.98, 5.5, 0.06, 1.45, fill=AMBER)
textbox(s, 7.23, 5.65, 5.35, 1.2,
        [("The risk of renting instead", 13, True, AMBER, PP_ALIGN.LEFT),
         ("Sensitive supplier data leaves our network on", 11, False, DARK, PP_ALIGN.LEFT),
         ("every call. Costs rise with every new inbox.", 11, False, DARK, PP_ALIGN.LEFT),
         ("We own nothing at the end of it.", 11, False, DARK, PP_ALIGN.LEFT)])
pagenum(s, 8)

# ─────────────────────────────────────────────────────────── 9. Ask
s = blank()
header(s, "The decision, and what we need", "THE ASK",
       "One decision today; the rest is already specified and partly built.")

rect(s, 0.55, 1.8, 12.23, 0.95, fill=GREEN)
textbox(s, 0.85, 1.95, 11.7, 0.7,
        [("Approve building our own email intent model, with a paid LLM used once as a teacher.", 16, True, WHITE, PP_ALIGN.LEFT),
         ("Not approving means committing to a permanent per-email fee and sending supplier data outside our network.", 11.5, False, SKY, PP_ALIGN.LEFT)])

textbox(s, 0.55, 3.0, 5.9, 0.3, [("What we need from the business", 13, True, NAVY, PP_ALIGN.LEFT)])
grid(s, 0.55, 3.36, [3.4, 2.5],
     [["Input", "Effort"],
      ["Agreed list of intents", "Workshop"],
      ["~100 example emails per intent", "Collection"],
      ["Review of uncertain labels", "~30 hours"],
      ["Access to shared inboxes", "IT setup"]],
     row_h=0.46, bolds={(r, 0): True for r in range(1, 5)})

textbox(s, 6.9, 3.0, 5.9, 0.3, [("What we deliver", 13, True, NAVY, PP_ALIGN.LEFT)])
grid(s, 6.9, 3.36, [3.4, 2.48],
     [["Deliverable", "When"],
      ["Validated intent taxonomy", "Week 2"],
      ["Labelled corpus (ours to keep)", "Week 4"],
      ["Trained, measured model", "Week 6"],
      ["Benchmark report vs. LLM", "Week 8"]],
     row_h=0.46,
     cell_colors={(r, 1): GREENL for r in range(1, 5)},
     bolds={(r, 0): True for r in range(1, 5)})

rect(s, 0.55, 5.85, 12.23, 1.1, fill=BGL)
rect(s, 0.55, 5.85, 0.06, 1.1, fill=ACCENT)
textbox(s, 0.85, 5.98, 11.7, 0.9,
        [("At the end of eight weeks we will own a measured, auditable classification model that runs inside our network at effectively zero",
          12, False, DARK, PP_ALIGN.LEFT),
         ("cost per email — and a benchmark showing exactly how it compares to the paid alternative. If it loses, we will say so.",
          12, False, DARK, PP_ALIGN.LEFT),
         ("Detailed technical specification is already written and ready for review.", 11, True, NAVY, PP_ALIGN.LEFT)])
pagenum(s, 9)

prs.save("docs/email_intent_briefing.pptx")
print(f"saved: {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
