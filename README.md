# Email Intent Classification

Training pipeline and design work for a **procurement email intent classification**
system: extract email, label it, fine-tune a compact transformer, and classify
incoming mail by intent.

The code here is a **working pilot** built against a personal Gmail account to
prove the approach end to end. The production design for procurement is
specified in [`email_spec.md`](email_spec.md) but not yet implemented.

---

## Status

| Area | State |
|---|---|
| Pilot pipeline (extract → label → train → classify) | **Working**, run on 4,499 emails |
| `email_spec.md` — production design | **Written** (~89 KB, 16 sections), not signed off |
| Executive decision deck | **Built** — `docs/email_intent_briefing.html` |
| `intentpipe/` — production rewrite | **Scaffolding only**, no implementation |

### Pilot results

Fine-tuned `answerdotai/ModernBERT-base` on 4,499 consumer emails across 15
auto-discovered intents, on a single RTX 4060 Laptop GPU (8 GB):

- **89.1%** accuracy / **84.3%** macro-F1 on held-out test data
- **94.0%** accuracy / **90.8%** macro-F1 on validation
- **97.5%** agreement with the LLM silver labels across the full corpus

These are consumer-mail numbers. They demonstrate the method works; they are
**not** a forecast of procurement accuracy.

---

## Layout

```
├── email_spec.md              # Production design spec — the main design artifact
├── intents.json               # Pilot taxonomy (15 intents), generated from the corpus
│
├── extract_historical_emails.py   # 1. IMAP → dataset/historical_extract.jsonl
├── generate_intent_taxonomy.py    # 2. Corpus sample → intents.json
├── label_with_llm.py              # 3. LLM silver labels → dataset/labeled.jsonl
├── train_modernbert.py            # 4. Fine-tune → models/modernbert-intent-v1/
├── classify_dataset.py            # 5. Batch classify + intent distribution summary
├── test_random_inference.py       #    Spot-check on a random live email
│
├── build_deck.py              # Builds the executive deck (.pptx)
├── bundle_deck_html.py        # Merges converter output into one navigable .html
│
├── dataset/                   # JSONL corpora            (gitignored — real email)
├── models/                    # Trained weights, ~4 GB   (gitignored)
├── docs/                      # Deck deliverables
└── intentpipe/                # Production rewrite target (empty)
```

---

## Setup

Two environments are needed, and **this trips people up**:

| Environment | Has | Used by |
|---|---|---|
| `.venv` (Python 3.13) | `torch`, `transformers`, `datasets`, `scikit-learn` | training, classification |
| `gyrfalcon` package | IMAP + LLM clients (no torch) | extraction, taxonomy, labeling |

`gyrfalcon` is **not installed** in `.venv`. Scripts that touch IMAP or call an
LLM need it on `PYTHONPATH`:

```bash
export PYTHONPATH=/home/ubuntu/agent/gyrfalcon
```

Mailbox credentials come from the gyrfalcon config plus:

```bash
export GYRFALCON_IMAP_PASSWORD=...
```

---

## Running the pipeline

Each step writes a file the next step reads, so you can stop and resume anywhere.

```bash
cd /home/ubuntu/gf/email
export PYTHONPATH=/home/ubuntu/agent/gyrfalcon

# 1. Extract mail over IMAP  (default folder: "[Gmail]/All Mail")
.venv/bin/python extract_historical_emails.py --limit 5000

# 2. Discover an intent taxonomy from a sample of the corpus
.venv/bin/python generate_intent_taxonomy.py --sample-size 150

# 3. Label every email with an LLM (resumable; skips already-labeled ids)
.venv/bin/python label_with_llm.py --workers 8

# 4. Fine-tune ModernBERT  (~minutes on an 8 GB GPU)
.venv/bin/python train_modernbert.py --epochs 4 --batch-size 8

# 5. Classify the whole corpus and print the intent distribution
.venv/bin/python classify_dataset.py

# Spot-check a random live email against the trained model
.venv/bin/python test_random_inference.py --top-k 3
```

Every script takes `--help`. Useful flags: `--limit`, `--model`, `--output`,
`--seed`, `--base-model`, `--max-seq-length`.

---

## Building the decision deck

Three steps — build the `.pptx`, convert it to per-slide HTML, then bundle it
into one self-contained file:

```bash
.venv/bin/python build_deck.py
python3 ~/.copilot/skills/adxppt/scripts/potx_to_html.py \
        docs/email_intent_briefing.pptx docs/deck_html
.venv/bin/python bundle_deck_html.py
```

Output: **`docs/email_intent_briefing.html`** — 9 slides, no external assets.
Navigate by clicking the slide (right side forward, left edge back), the
Back/Next buttons, the numbered jump buttons, or arrow keys. `#5` in the URL
deep-links to a slide.

> **Converter caveat:** the adxppt converter renders native PowerPoint tables
> with **empty cells** — it looks for `a:p` directly under `a:tc`, but PowerPoint
> nests them inside `a:txBody`. `build_deck.py` therefore composes all grids from
> individual shapes. If you edit the `.pptx` in PowerPoint and add a real table,
> its text will silently vanish on re-conversion.

---

## Design notes

The full reasoning lives in `email_spec.md`. Two points matter most:

**The two regimes.** Training and production are deliberately different, and
`email_spec.md §1.2` is the section to read first:

- **Regime A — model building:** one email, one intent, softmax + cross-entropy.
  Multi-intent emails are *excluded* from training, not discarded.
- **Regime B — production:** thread-level, multi-intent, with orchestration,
  decision revision and action idempotency layered *around* the model.

Multi-intent behaviour is an emergent property of the orchestration layer, not
of the model. That keeps training simple: 512 tokens, batch 32, standard
stratification, no per-class threshold matrix.

**The train/serve gap** (`§2.2`) is the principal technical risk. Cleaning and
formatting code must be shared *verbatim* between training and production;
divergence is a silent, brutal-to-diagnose accuracy killer.

---

## Privacy

`dataset/` contains **real email bodies, subjects and sender addresses**. It is
gitignored and must stay that way. Do not commit corpora, and do not move them
outside this machine without checking what is in them first.
