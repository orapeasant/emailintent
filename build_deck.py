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

#: Legibility uplift. The deck is read on a projector and on laptops, where
#: 10–11pt body text is too small. Sizes are mapped centrally here rather than
#: edited at ~90 call sites, so the visual hierarchy stays consistent.
#: Anchors requested: 11 -> 16 and 10.5 -> 14.
FONT_SCALE = {
    6: 7, 7: 8,          # blank spacer lines
    9.5: 13, 10: 13.5, 10.5: 14,
    11: 16, 11.5: 16, 12: 16.5, 12.5: 17, 13: 17.5,
    14: 18, 15: 19, 16: 20,
    20: 22, 21: 23, 25: 26, 26: 30, 38: 40,
}


def fs(size):
    return FONT_SCALE.get(size, size)


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
        r.font.size = Pt(fs(size))
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


def pagenum(slide, n, total=12):
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
ph.left, ph.top, ph.width, ph.height = Inches(0.95), Inches(1.82), Inches(6.8), Inches(2.0)
write(ph, [("Email Intent Classification", 38, True, WHITE, PP_ALIGN.LEFT),
           ("Build our own model, or rent an LLM API?", 21, False, SKY, PP_ALIGN.LEFT)],
      anchor=MSO_ANCHOR.MIDDLE, margin=0.0)

textbox(s, 0.95, 1.45, 6.7, 0.35,
        [("DECISION BRIEFING", 12, True, ACCENT, PP_ALIGN.LEFT)])
textbox(s, 0.95, 4.05, 6.7, 1.0,
        [("Automating triage of ~10,000 procurement emails per month across shared supplier inboxes.", 14, False, WHITE, PP_ALIGN.LEFT)])
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

grid(s, 0.55, 1.85, [2.9, 3.55, 1.9, 2.0, 1.85],
     [["Approach", "What it is", "Accuracy", "Cost / month", "Data leaves us?"],
      ["A. Keyword / TF-IDF", "Hand-written rules and word counts", "Low", "~$0", "No"],
      ["B. Embeddings + head", "Frozen embeddings, small classifier", "Medium", "Low", "No"],
      ["C. Fine-tuned ModernBERT", "Our own model, trained on our mail", "High", "Fixed, ~$0 marginal", "No"],
      ["D. Paid LLM API", "Commercial LLM called per email", "High", "Grows with volume", "Yes"]],
     row_h=0.58,
     cell_colors={**{(3, c): GREENL for c in range(5)},
                  **{(4, c): AMBERL for c in range(5)}},
     bolds={(3, 0): True, (4, 0): True, (1, 0): True, (2, 0): True})

rect(s, 0.55, 4.68, 6.05, 2.2, fill=GREENL)
rect(s, 0.55, 4.68, 0.06, 2.2, fill=GREEN)
textbox(s, 0.8, 4.82, 5.6, 1.92,
        [("C — Our own model  ✓ recommended", 14, True, GREEN, PP_ALIGN.LEFT),
         ("", 6, False, DARK, PP_ALIGN.LEFT),
         ("A compact 150M-parameter model, fine-tuned on our own emails and our own intent taxonomy.", 11, False, DARK, PP_ALIGN.LEFT),
         ("Runs in our network. Version-frozen, so any past decision can be reproduced for an audit.", 11, False, DARK, PP_ALIGN.LEFT)])

rect(s, 6.98, 4.68, 5.8, 2.2, fill=AMBERL)
rect(s, 6.98, 4.68, 0.06, 2.2, fill=AMBER)
textbox(s, 7.23, 4.82, 5.35, 1.92,
        [("D — Paid LLM API", 14, True, AMBER, PP_ALIGN.LEFT),
         ("", 6, False, DARK, PP_ALIGN.LEFT),
         ("Fastest to a first result, and strong at language it has never seen.", 11, False, DARK, PP_ALIGN.LEFT),
         ("But every email — including bank details — leaves our network on every call, forever.", 11, False, DARK, PP_ALIGN.LEFT)])
pagenum(s, 3)

# ─────────────────────────────────────────────────────────── 4. Comparison
s = blank()
header(s, "Own model vs. paid LLM API", "HEAD TO HEAD",
       "Eight dimensions that matter to us. Green marks the stronger option on each row.")

grid(s, 0.55, 1.8, [3.15, 4.55, 4.55],
     [["Dimension", "Our own fine-tuned model", "Paid LLM API"],
      ["Accuracy on our taxonomy", "89% measured on our 20 intents", "Must be re-told our intents every call"],
      ["Latency per email", "~15 milliseconds", "2–5 seconds (100–300× slower)"],
      ["Marginal cost per email", "~$0 once built", "$0.006 – $0.05 (measured)"],
      ["Sensitive data residency", "Never leaves our network", "Bank details sent to a third party"],
      ["Reproducibility / audit", "Version frozen, replayable", "Vendor updates without notice"],
      ["Availability", "No rate limits, no outage risk", "Rate limits; vendor uptime"],
      ["Vendor lock-in", "None — the asset is ours", "High — pricing and terms can change"],
      ["Time to first result", "6–8 weeks", "Days"]],
     row_h=0.47,
     cell_colors={(1, 1): GREENL, (2, 1): GREENL, (3, 1): GREENL, (4, 1): GREENL,
                  (5, 1): GREENL, (6, 1): GREENL, (7, 1): GREENL, (8, 2): GREENL},
     bolds={(r, 0): True for r in range(1, 9)})

rect(s, 0.55, 6.18, 12.23, 0.85, fill=NAVY)
textbox(s, 0.85, 6.28, 11.7, 0.68,
        [("The LLM API wins on exactly one dimension — time to first result. We can have that too, by using it as the teacher (slide 7).",
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

textbox(s, 0.55, 4.92, 12.2, 0.28,
        [("Green marks the cheaper option in each row. Engineering estimated at one fully-loaded engineer for 6–8 weeks.",
          9.5, False, GREY, PP_ALIGN.LEFT)])

rect(s, 0.55, 5.25, 12.23, 1.72, fill=AMBERL)
rect(s, 0.55, 5.25, 0.06, 1.72, fill=AMBER)
textbox(s, 0.85, 5.36, 11.7, 1.55,
        [("Stated plainly: at today's volume, building our own model is the more expensive option.", 13, True, AMBER, PP_ALIGN.LEFT),
         ("Break-even is about $10–12k / year of API spend — roughly ten times our current volume, or any full-thread scenario with self-consistency checks.", 11.5, False, DARK, PP_ALIGN.LEFT),
         ("We recommend building for control, auditability and data residency — not to save money in year one.", 11.5, True, NAVY, PP_ALIGN.LEFT)])
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

textbox(s, 0.55, 3.45, 12.2, 0.3,
        [("How it was done", 13, True, NAVY, PP_ALIGN.LEFT)])
grid(s, 0.55, 3.8, [0.62, 4.3, 7.31],
     [["#", "Step", "Result"],
      ["1", "Extract history from the mailbox", "4,499 emails, cleaned of HTML and quotes"],
      ["2", "Discover the intent taxonomy", "15 intents from the real corpus, not guessed"],
      ["3", "Label the corpus with an LLM", "100% labelled; only 5.4% needed attention"],
      ["4", "Fine-tune our own model", "ModernBERT-base, one laptop GPU, minutes"],
      ["5", "Measure on held-out data", "89.1% accuracy / 84.3% macro-F1, unseen mail"]],
     row_h=0.42, bolds={(r, 0): True for r in range(1, 6)})

rect(s, 0.55, 6.42, 12.23, 0.6, fill=GREENL)
rect(s, 0.55, 6.42, 0.06, 0.6, fill=GREEN)
textbox(s, 0.85, 6.52, 11.7, 0.42,
        [("Hardware was a laptop GPU — what we need is better data and review, not better technology.",
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
        (10.45, 2.05, "Runtime", "Classifies all mail\n~$0 per email", GREENL, GREEN)]:
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

rect(s, 0.55, 6.0, 12.23, 1.0, fill=NAVY)
textbox(s, 0.85, 6.1, 11.7, 0.85,
        [("Renting an LLM per email means paying forever for a capability we never own, while our most sensitive data leaves the network on every call.",
          12, False, WHITE, PP_ALIGN.LEFT),
         ("Paying once to train our own model turns that same spend into an asset we keep.",
          12, True, SKY, PP_ALIGN.LEFT)])
pagenum(s, 7)

# ─────────────────────────────────────────────────────────── 8. End-to-end process
s = blank()
header(s, "How it works, end to end", "THE PIPELINE",
       "Ten steps from mailbox to routed intent. Steps 1–5 build the model once; steps 6–10 run continuously.")


def pipeline_row(slide, y, label, label_color, box_fill, steps):
    textbox(slide, 0.55, y, 8.0, 0.28,
            [(label, 11.5, True, label_color, PP_ALIGN.LEFT)])
    bx = 0.55
    for i, (num, title, sub) in enumerate(steps):
        rect(slide, bx, y + 0.34, 2.15, 1.3, fill=box_fill)
        rect(slide, bx, y + 0.34, 2.15, 0.055, fill=label_color)
        textbox(slide, bx + 0.07, y + 0.48, 2.01, 0.26,
                [(f"{num}. {title}", 11.5, True, NAVY, PP_ALIGN.CENTER)])
        textbox(slide, bx + 0.07, y + 0.79, 2.01, 0.75,
                [(ln, 9.5, False, GREY, PP_ALIGN.CENTER) for ln in sub.split("\n")])
        if i < len(steps) - 1:
            textbox(slide, bx + 2.17, y + 0.84, 0.32, 0.34,
                    [("→", 16, True, label_color, PP_ALIGN.CENTER)])
        bx += 2.51


pipeline_row(s, 1.78, "BUILD ONCE  —  6 to 8 weeks", ACCENT, BGL, [
    ("1", "Retrieve", "Graph API delta sync\nfrom shared mailboxes"),
    ("2", "Clean", "Strip HTML, quotes,\nsignatures, footers"),
    ("3", "Label", "LLM assigns intent\n+ confidence, one time"),
    ("4", "Review", "Business corrects the\nuncertain and sampled"),
    ("5", "Train", "Fine-tune ModernBERT\non our own corpus"),
])

pipeline_row(s, 3.66, "RUN CONTINUOUSLY  —  every day thereafter", GREEN, GREENL, [
    ("6", "Evaluate", "Gold set decides\nwhether it ships"),
    ("7", "Deploy", "Versioned model,\nfrozen for audit"),
    ("8", "Classify", "New mail scored\nin ~15 milliseconds"),
    ("9", "Route", "Confident cases auto,\nrest to a person"),
    ("10", "Improve", "Corrections feed\nweekly retraining"),
])

rect(s, 0.55, 5.42, 12.23, 1.6, fill=BGL)
rect(s, 0.55, 5.42, 0.06, 1.6, fill=ACCENT)
textbox(s, 0.85, 5.52, 11.7, 1.45,
        [("Two things hold this together", 12.5, True, NAVY, PP_ALIGN.LEFT),
         ("Step 2 uses identical cleaning code in training and production. Divergence there is the most common silent killer of accuracy.", 11, False, DARK, PP_ALIGN.LEFT),
         ("Step 3 is the only point where email content leaves our network — once, and never again after training.", 11, True, GREEN, PP_ALIGN.LEFT)])
pagenum(s, 8)

# ─────────────────────────────────────────────────────────── 9. Graph API access
s = blank()
header(s, "What access we need — Microsoft Graph", "THE ACCESS REQUEST",
       "Read-only, scoped to the named procurement mailboxes, granted to a service identity rather than to a person.")

textbox(s, 0.55, 1.8, 5.9, 0.3,
        [("What we are asking for", 12.5, True, GREEN, PP_ALIGN.LEFT)])
grid(s, 0.55, 2.14, [2.6, 3.3],
     [["Permission / control", "What it allows"],
      ["Mail.Read  (application)", "Read message bodies"],
      ["Application Access Policy", "Limited to named mailboxes"],
      ["Certificate credential", "No shared passwords"],
      ["Delta query", "Incremental sync only"]],
     row_h=0.44, head_h=0.44, body_size=10, head_fill=GREEN,
     bolds={(r, 0): True for r in range(1, 5)})

textbox(s, 6.9, 1.8, 5.88, 0.3,
        [("What we are deliberately not asking for", 12.5, True, RED, PP_ALIGN.LEFT)])
grid(s, 6.9, 2.14, [2.6, 3.28],
     [["Permission", "Why we do not want it"],
      ["Mail.Send", "We never send email"],
      ["Mail.ReadWrite", "We never edit or delete"],
      ["Unscoped Mail.Read", "Would expose every mailbox"],
      ["Directory / User.Read.All", "Not needed at all"]],
     row_h=0.44, head_h=0.44, body_size=10, head_fill=RED,
     bolds={(r, 0): True for r in range(1, 5)})

textbox(s, 0.55, 4.46, 8.0, 0.3,
        [("One-time setup, by IT", 12.5, True, NAVY, PP_ALIGN.LEFT)])
bx = 0.55
for i, (title, sub) in enumerate([
        ("Register app", "Entra ID app\nregistration"),
        ("Admin consent", "Tenant admin\ngrants Mail.Read"),
        ("Scope it", "Scoped to named\nmailboxes"),
        ("Certificate", "Cert in Key Vault,\nrotated"),
        ("Verify", "No other mailbox\nis reachable")]):
    rect(s, bx, 4.78, 2.15, 1.05, fill=BGL2)
    textbox(s, bx + 0.07, 4.89, 2.01, 0.28,
            [(title, 11, True, NAVY, PP_ALIGN.CENTER)])
    textbox(s, bx + 0.07, 5.2, 2.01, 0.58,
            [(ln, 9.5, False, GREY, PP_ALIGN.CENTER) for ln in sub.split("\n")])
    if i < 4:
        textbox(s, bx + 2.17, 5.18, 0.32, 0.3, [("→", 15, True, ACCENT, PP_ALIGN.CENTER)])
    bx += 2.51

rect(s, 0.55, 5.95, 12.23, 0.85, fill=GREENL)
rect(s, 0.55, 5.95, 0.06, 0.85, fill=GREEN)
textbox(s, 0.85, 6.05, 11.7, 0.7,
        [("One application, read-only, with an Exchange policy that makes every other mailbox in the tenant unreachable — not merely off-limits by convention. Revoking it is a single switch in Entra ID.",
          11.5, False, DARK, PP_ALIGN.LEFT)])
pagenum(s, 9)

# ─────────────────────────────────────────────────────────── 10. Access control safety
s = blank()
header(s, "Access control and safety", "PROTECTING THE SHARED MAILBOXES",
       "Personal login is right for a prototype and wrong for a running system.")

grid(s, 0.55, 1.82, [3.1, 4.5, 4.63],
     [["", "Personal login (today)", "Service principal (recommended)"],
      ["Identity", "A named employee", "Purpose-built, non-human"],
      ["Graph permission", "Mail.Read.Shared", "Mail.Read + access policy"],
      ["Reach", "Any mailbox they can open", "Only the named mailboxes"],
      ["Audit trail", "Attributed to a person", "Attributed to the service"],
      ["If they change role or leave", "Pipeline breaks, access follows them", "Unaffected"],
      ["MFA / password rotation", "Interrupts the automation", "Certificate, rotated on schedule"],
      ["Revoking access", "Disable an employee's account", "Disable one app registration"]],
     row_h=0.43, head_h=0.44, body_size=10,
     cell_colors={(r, 2): GREENL for r in range(1, 8)},
     bolds={(r, 0): True for r in range(1, 8)})

bullet_card(s, 0.55, 5.4, 2.93, 0.95, "Read-only",
            ["No send, no edit"], accent=GREEN, fill=GREENL, tsize=12, bsize=10)
bullet_card(s, 3.65, 5.4, 2.93, 0.95, "Scoped",
            ["Named mailboxes only"], accent=GREEN, fill=GREENL, tsize=12, bsize=10)
bullet_card(s, 6.75, 5.4, 2.93, 0.95, "Audited",
            ["Purview logs all reads"], accent=GREEN, fill=GREENL, tsize=12, bsize=10)
bullet_card(s, 9.85, 5.4, 2.93, 0.95, "Minimised",
            ["Only fields we need"], accent=GREEN, fill=GREENL, tsize=12, bsize=10)

rect(s, 0.55, 6.42, 12.23, 0.68, fill=AMBERL)
rect(s, 0.55, 6.42, 0.06, 0.68, fill=AMBER)
textbox(s, 0.85, 6.5, 11.7, 0.56,
        [("The one-time labelling step does send email content to an external LLM — it needs its own approval, and it is exactly what owning the model ends.",
          11, False, DARK, PP_ALIGN.LEFT)])
pagenum(s, 10)

# ─────────────────────────────────────────────────────────── 11. Risks
s = blank()
header(s, "What could go wrong, and how we handle it", "RISKS — STATED UP FRONT",
       "These are real. All are manageable, and all are cheaper to address than an unbounded API dependency.")

grid(s, 0.55, 1.85, [3.5, 4.6, 4.13],
     [["Risk", "Why it matters", "How we handle it"],
      ["Needs labelled data", "A model is only as good as its examples", "LLM labels bulk; people check the 5%"],
      ["Labels may be noisy", "Bad labels quietly cap accuracy", "600-email human gold set"],
      ["Taxonomy may not be learnable", "Business categories overlap in text", "Validated on real mail first"],
      ["Language drifts over time", "New suppliers, systems, wording", "Weekly retraining with a gate"],
      ["ML skills needed in-house", "We must be able to maintain it", "Open tooling; pipeline built"],
      ["Multi-topic emails", "One email can carry two requests", "Detected and escalated"]],
     row_h=0.5,
     cell_colors={(r, 2): GREENL for r in range(1, 7)},
     bolds={(r, 0): True for r in range(1, 7)})

rect(s, 0.55, 5.38, 6.05, 1.55, fill=REDL)
rect(s, 0.55, 5.38, 0.06, 1.55, fill=RED)
textbox(s, 0.8, 5.5, 5.6, 1.31,
        [("The risk of doing nothing", 13, True, RED, PP_ALIGN.LEFT),
         ("Triage stays people-limited. Volume grows, headcount grows with it, and nothing downstream can be automated.", 11, False, DARK, PP_ALIGN.LEFT)])

rect(s, 6.98, 5.38, 5.8, 1.55, fill=AMBERL)
rect(s, 6.98, 5.38, 0.06, 1.55, fill=AMBER)
textbox(s, 7.23, 5.5, 5.35, 1.31,
        [("The risk of renting instead", 13, True, AMBER, PP_ALIGN.LEFT),
         ("Sensitive supplier data leaves our network on every call. Costs rise with every new inbox, and we own nothing at the end.", 11, False, DARK, PP_ALIGN.LEFT)])
pagenum(s, 11)

# ─────────────────────────────────────────────────────────── 12. Ask
s = blank()
header(s, "The decision, and what we need", "THE ASK",
       "One decision today; the rest is already specified and partly built.")

rect(s, 0.55, 1.8, 12.23, 0.95, fill=GREEN)
textbox(s, 0.85, 1.95, 11.7, 0.75,
        [("Approve building our own model, with a paid LLM used once as a teacher.", 16, True, WHITE, PP_ALIGN.LEFT),
         ("Not approving means a permanent per-email fee and supplier data leaving our network.", 11.5, False, SKY, PP_ALIGN.LEFT)])

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

rect(s, 0.55, 5.8, 12.23, 1.1, fill=BGL)
rect(s, 0.55, 5.8, 0.06, 1.1, fill=ACCENT)
textbox(s, 0.85, 5.9, 11.7, 0.95,
        [("In eight weeks we will own a measured, auditable model running inside our network at effectively zero cost per email — plus a benchmark against the paid alternative. If it loses, we will say so.",
          12, False, DARK, PP_ALIGN.LEFT),
         ("Detailed technical specification is already written and ready for review.", 11, True, NAVY, PP_ALIGN.LEFT)])
pagenum(s, 12)

prs.save("docs/email_intent_briefing.pptx")
print(f"saved: {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
