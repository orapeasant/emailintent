# Procurement Email Intent Classification — Solution Specification

**Status:** Draft for sign-off
**Version:** 1.0
**Date:** 2026-09-12
**Owner:** Data / ML Engineering
**Supersedes:** the consumer-email pilot in this repo (retained as a reference implementation only)

---

## 1. Purpose

Automatically classify inbound email in shared procurement mailboxes by **business intent**, so that mail can be routed, prioritised, and actioned without an analyst reading every message first.

The system must:

1. Learn from a taxonomy defined by the procurement business team, validated against real mail.
2. Bootstrap its training labels using an LLM, corrected by human reviewers where it matters most.
3. Produce a self-hosted fine-tuned model (no per-inference LLM cost or data egress at run time).
4. Be measurable against a **human-labeled gold set**, not against its own teacher.
5. Improve continuously from human corrections, without self-reinforcing its own errors.

### 1.1 Scope of this specification

| Area | In scope to build here | Notes |
|---|---|---|
| Taxonomy definition & validation | ✅ | Incl. bottom-up clustering reconciliation |
| Email data collection | ✅ | IMAP now, Graph/Exchange later |
| Single-label LLM silver labeling | ✅ | With self-consistency confidence |
| Gold evaluation set tooling | ✅ | Sampling, dual annotation, IAA |
| Human review export/import | ✅ | File-based; UI is a separate project |
| ModernBERT single-label training | ✅ | |
| Benchmark evaluation & threshold tuning | ✅ | |
| Inference testing harness | ✅ | |
| **Production serving / thread orchestration** | ❌ design only | Implemented in a separate platform; §12 is the recommended approach |
| **Reviewer web UI** | ❌ design only | Separate project; §9.4 defines the contract it must satisfy |

### 1.2 The two regimes — read this first

This specification describes **two deliberately different regimes**. Conflating them is the most common way this kind of programme goes wrong, so they are kept structurally separate throughout.

| | **Regime A — Model building** (§4–§11) | **Regime B — Production inference** (§12) |
|---|---|---|
| **Unit** | **One email** | **One thread** (living, growing) |
| **Cardinality** | **Exactly one intent** | **Multiple intents**, accumulating over time |
| **Data source** | Business-curated single-intent emails + reviewed historical mail | Live shared-inbox traffic |
| **Multi-intent emails** | **Excluded from training data** | Must be handled (§12.8) |
| **Model head** | Softmax + cross-entropy | *(same model)* |
| **Decision** | A label | A label **set**, plus actions, revisions, SAP dispatch |
| **Revisable?** | N/A — a static labeled corpus | Yes — continuous re-classification (§12.7) |
| **Built here?** | ✅ Yes | ❌ Design only |

> **The model is a single-label, single-email classifier.** Multi-intent behaviour at thread level is **not** a model property — it is an *emergent property of the orchestration layer*, which applies the single-label model per message and accumulates the results into a thread-level intent set (§12.8).

This is why §12's complexity (accumulation, supersession, invalidation, action lifecycle) creates **no** complexity in training: those concerns live entirely above the model.

---

## 2. Locked decisions

These were agreed during design review and drive everything downstream. Changing any of them after Phase 2 begins is expensive.

Decisions are tagged **[A]** model building, **[B]** production inference, or **[A+B]** both.

| # | Scope | Decision | Value | Rationale |
|---|---|---|---|---|
| D1 | **[A]** | Training label cardinality | **Single-label — one email, one intent** | Business supplies curated single-intent examples. Multi-intent emails are **excluded from the training corpus** (D13). Removes ambiguity from the learning signal entirely. |
| D2 | **[A]** | Training unit | **Single email** — not thread | Threads are a production construct. The model learns the intent of one message. |
| D2b | **[B]** | Production unit | **Thread**, classified **per inbound message** | Each message is scored by the single-label model; results accumulate into the thread's intent set (§12.8). |
| D3 | **[A+B]** | Gold evaluation set | **Yes — required**, drawn from **real production-like mail**, not curated examples | Without it, accuracy is unmeasurable and retraining cannot be gated. Must reflect the deployment distribution, not the training distribution (§8.2). |
| D4 | **[A]** | LLM data governance | **Cleared** for real procurement mail | Enterprise endpoint approved. Redaction optional, not mandatory. |
| D5 | **[A+B]** | Mail source | **IMAP first**, Graph/Exchange later | Connector abstracted behind an interface so the second source is additive. |
| D6 | **[A]** | Retraining cadence | **Weekly**, gated | Daily is too frequent for expected volume and risks instability. |
| D7 | **[A]** | Model head | **Softmax + cross-entropy**, class-weighted, with an **abstention threshold** | Follows directly from D1. Not sigmoid/BCE — there is no multi-label training signal to learn from. See §10.2. |
| D8 | **[B]** | Production decision model | Calibrated model + **abstention** to MISC | Coverage-at-precision is the business KPI, not raw accuracy. |
| D9 | **[B]** | Classification lifecycle | **Continuous re-classification** — a thread is re-classified on every inbound message | "Classify once, never again" was evaluated and **rejected**: it freezes a decision on incomplete evidence and cannot represent a thread whose intent genuinely evolves. See §12.7. |
| D10 | **[B]** | Intent accumulation | **Additive by default, with supersession/invalidation/retraction as exception paths** | Newly detected intents append to the thread's intent set and spawn actions. See §12.7.2, §12.7.8. |
| D11 | **[B]** | Classification trigger | **On arrival** of each inbound counterparty message | Requires per-thread serialisation (§12.7.6 F1–F3). Agent-authored and internal messages never trigger. |
| D12 | **[B]** | Action dispatch | One action per newly-activated intent; **SAP call keyed `(thread_id, action_id)`** with success/failure written back and retry on failure | Needs **dual idempotency** (§12.7.4). |
| D13 | **[A]** | Training data purity | **Multi-intent emails are excluded from training**, not force-labeled | A forced single label on a two-ask email teaches the model to ignore one of them. Exclusion is cleaner than noise. Such emails are logged for §12.8 design instead. |
| D14 | **[B]** | Multi-intent recovery at inference | **Segmentation + margin detection**, with abstention | A single-label model cannot emit two intents for one message. Production recovers them by classifying message *segments* and by treating a low top-2 margin as a multi-intent signal. See §12.8. |

### 2.1 Volume assumptions

- Training corpus: **5,000–10,000 single-intent emails**
- Steady-state inbound: **~10,000 emails/month**, ~400–500/day
- Expected thread volume: ~3,000 threads/week reaching the classifier

### 2.2 The train/serve gap — the principal technical risk of this design

D1/D2/D13 buy a clean learning signal, and that is genuinely worth having. But they create a **distribution gap** between what the model is trained on and what it will see in production. This must be managed explicitly, not discovered later.

| | Training distribution (Regime A) | Production distribution (Regime B) |
|---|---|---|
| Curation | Business-selected, prototypical | Whatever arrives |
| Intents per email | Exactly 1 (by construction) | 1, 2, sometimes 3 |
| Message type | Mostly thread-opening asks | ~60%+ replies |
| Short replies (*"Yes, approved"*) | Rare — nobody submits these as examples | Common |
| Quoted history / forwards | Usually clean | Pervasive |
| Ambiguity | Filtered out | Concentrated here |

Three consequences, each with a required mitigation:

1. **Business-curated data is optimistic by selection.** Reviewers naturally supply clear, textbook examples — precisely *not* the mail that causes production errors. Evaluating on curated data would show excellent numbers that do not survive deployment.
   → **Mitigation: the gold set is drawn from real production-like mail, never from the curated training pool (D3, §8.2).**

2. **Multi-intent emails exist in production even though they are excluded from training.** The model can only ever emit one label per message.
   → **Mitigation: segmentation + margin detection at inference (D14, §12.8).** A genuine two-ask email produces a low top-2 margin, which is detectable even though it was never trained on.

3. **Short replies and quoted-context messages are under-represented.** *"Yes, approved."* carries no standalone signal.
   → **Mitigation: Phase 2 training augmentation (§10.6)** — after the first model ships, add real single-intent messages sampled from production traffic, including replies, keeping one-intent-per-email. This closes the gap without violating D1 or D13.

> The single-label/single-email regime is the right call for **shipping a first model quickly with a clean signal**. It is not a permanent ceiling: §10.6 defines how the corpus grows toward the production distribution while preserving the one-email-one-intent rule.

---

## 3. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — TAXONOMY                                                      │
│  Business draft intents ──┐                                              │
│  Seed sentences per intent├─► separability check ─► reconcile ─► v1      │
│  Bottom-up clustering ────┘    (embeddings)         (merge/split)  FROZEN │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────┐
│  PHASE 2 — DATA COLLECTION  (single emails, not threads)                 │
│  IMAP/Graph ─► message fetch ─► clean ─► SQLite `emails`                 │
│                (quote/sig strip)                                         │
│  + business-curated single-intent examples                               │
│  + stratified sampling (time / inbox / sender-cap) + dedup               │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────┐
│  PHASE 3 — LABELING  (one email → one intent)                            │
│  LLM single-label (k=3 self-consistency) ─► confidence bands             │
│         │                                                                │
│         ├─► HIGH  ─► stratified 1-in-N sample ──┐                        │
│         ├─► MED   ─► full review ───────────────┼─► human corrections    │
│         ├─► LOW   ─► full review ───────────────┘                        │
│         └─► MULTI-INTENT DETECTED ─► EXCLUDED from training (D13),       │
│                                      logged for §12.8 design             │
│                                                                          │
│  PARALLEL: random 600 *production-like* emails ─► 2 annotators           │
│            ─► adjudicate ─► GOLD SET  (multi-intent allowed here)        │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────┐
│  PHASE 4 — TRAINING  (Regime A)                                          │
│  ModernBERT-base + softmax head ─► calibrate ─► abstention threshold     │
│  Prototype embedding model (cold-start / disagreement signal)            │
│  Rules engine (PO/invoice regex, sender domain) — decorrelated signal    │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼─────────────────────────────────────┐
│  PHASE 5 — EVALUATION (on frozen gold test split)                        │
│  macro-F1 · per-class P/R · coverage@precision                           │
│  + production-mode eval: does the predicted label fall in the gold set?  │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
═══════════════════ REGIME BOUNDARY — model ends here ═══════════════════
                                     │
┌────────────────────────────────────▼─────────────────────────────────────┐
│  PHASE 6 — PRODUCTION (separate platform — design only, §12)             │
│                                                                          │
│  per inbound message ─► single-label model ─► segmentation + margin      │
│                                               check (§12.8)              │
│                              │                                           │
│                              ▼                                           │
│           accumulate into THREAD intent set (§12.7)                      │
│           ├─ new intent      → spawn action → SAP dispatch → retry       │
│           ├─ conflicting     → supersede  → compensate                   │
│           └─ contradicted    → invalidate → compensate + fix label       │
│                              │                                           │
│                              ▼                                           │
│            abstain ─► HITL UI ─► weekly gated retrain                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Taxonomy

### 4.1 Definition process

The business taxonomy is a **draft until validated against real mail**. Business-authored taxonomies mirror process ownership rather than linguistic separability, producing intents that no model can distinguish from text alone.

Known failure patterns to actively screen for:

- **Synonym pairs** — *"Vendor registration"* vs *"Supplier onboarding"*: identical text, different owning team.
- **Metadata masquerading as intent** — *"Urgent PO change"* vs *"PO change"*: urgency is a priority attribute, not an intent. Model it as a separate field.
- **Outcome-based intents** — *"Approved invoice"* vs *"Rejected invoice"*: the inbound email often does not contain the outcome.

### 4.2 Reconciliation procedure

1. Business team drafts intent list + **10–20 seed sentences per intent** (§4.3).
2. Embed seed sentences; compute inter-intent centroid cosine similarity.
   - Any pair with centroid similarity **> 0.85** → flagged as a merge candidate.
3. Embed a 2,000-thread sample of real mail; cluster (HDBSCAN, fallback k-means).
4. Map clusters ↔ draft intents:
   - Two intents dominating one cluster → **merge**, or move the distinction to a metadata field.
   - A large cluster (>2% of volume) mapping to no intent → **missing intent**, add it.
   - An intent matching no cluster → either genuinely rare (keep, expect low support) or not text-detectable (drop).
5. Business signs off → **taxonomy v1 frozen** before any labeling spend.

### 4.3 Seed sentences — purpose

Seed sentences are **not** training data for a deployed classifier; 10–20 examples per class cannot train a head without severe overfitting. Their two real uses:

1. **Separability diagnostic** (step 2 above) — cheap, catches a broken taxonomy before the labeling budget is spent.
2. **Few-shot retrieval for the LLM labeler** — inject the nearest seed examples into each labeling prompt. Measurably improves silver-label quality.

A nearest-centroid **prototype classifier** (no training) built from these seeds serves as the day-1 cold-start model and, later, as a decorrelated disagreement signal.

### 4.4 Structure & governance

```json
{
  "version": 1,
  "frozen_at": "2026-…",
  "intents": [
    {
      "id": "po_confirmation",
      "label": "PO Confirmation",
      "description": "Supplier acknowledges or confirms receipt/acceptance of a purchase order.",
      "business_owner": "Buying Team",
      "priority": "high",
      "misroute_cost": "high",
      "seed_sentences": ["...", "..."],
      "example_subjects": ["...", "..."]
    }
  ]
}
```

- `MISC` is a **first-class intent in the taxonomy**, not merely a threshold artefact — so the models learn to predict "none of the above" directly, in addition to the abstention path in §12.3.
- `misroute_cost` drives per-class precision targets (§11.4). Misrouting an invoice ≠ misrouting a newsletter.
- Every stored row carries `taxonomy_version`. Adding or splitting an intent increments the version and requires a documented migration for existing labels.

### 4.5 Co-occurrence diagnostic

Once silver labels exist, build the label co-occurrence matrix. **Any pair co-occurring in >60% of the threads where either appears is a merge candidate** — strong evidence of two names for one intent. Free validation of the business taxonomy on real data.

---

## 5. Data collection

### 5.1 Connector abstraction

```python
@dataclass(frozen=True)
class RawMessage:
    message_id: str
    thread_hint: str | None      # X-GM-THRID, Exchange ConversationId
    in_reply_to: str | None
    references: list[str]
    from_addr: str
    to_addrs: list[str]
    cc_addrs: list[str]
    sent_at: datetime
    subject: str
    body_raw: str
    body_is_html: bool
    has_attachments: bool
    attachment_types: list[str]
    source_inbox: str
    folder_labels: list[str]

class MailConnector(Protocol):
    def iter_messages(self, *, folder: str, since: date | None,
                      before: date | None, limit: int) -> Iterator[RawMessage]: ...
```

- `ImapConnector` — Phase 2 (this build). Must fetch `Message-ID`, `In-Reply-To`, `References`, and `X-GM-THRID` where available. **Regime A does not use these**, but Regime B's thread assembly depends on them entirely, so they are captured from day one — re-extracting later is expensive.
- `GraphConnector` — later, additive. Uses `conversationId`, which is more reliable than header-chain reconstruction.

Mailbox names containing spaces or brackets must be explicitly quoted — `imaplib` does not auto-quote (e.g. `"[Gmail]/All Mail"`).

### 5.2 Training corpus composition (Regime A)

The training corpus is built from **single emails**, each carrying **exactly one intent** (D1, D2). It has two sources:

| Source | Role | Volume |
|---|---|---|
| **Business-curated examples** | Canonical, unambiguous examples per intent, supplied by the procurement team | ≥100 per intent where possible |
| **Sampled historical mail** | Real-world variation, LLM-labeled then human-reviewed | Remainder to 5–10k |

**Business-curation brief** — what to ask for, and what to warn against:

- Provide emails that express **one intent only**. If an email contains two asks, **do not submit it** — submit it to the multi-intent log instead (D13).
- Provide **hard and boring examples**, not just clear ones. Curated corpora skew optimistic (§2.2), and the model learns nothing from twenty near-identical textbook cases.
- Include **replies and short messages** where they are genuinely single-intent.
- Cover **rare intents deliberately** — natural sampling will not find them.

### 5.2.1 Multi-intent emails — excluded, not discarded

Emails found to carry more than one intent are **excluded from training** (D13) but **retained in a separate table**. They are not waste; they are the design input for §12.8 and a required test set for the production orchestration layer.

```sql
CREATE TABLE multi_intent_log (
  email_id     TEXT PRIMARY KEY REFERENCES emails(email_id),
  intents_json TEXT NOT NULL,       -- the several intents observed
  detected_by  TEXT,                -- llm | human
  segmentable  INTEGER,             -- can the asks be split into separate spans? (§12.8)
  notes        TEXT,
  created_at   TEXT
);
```

Their **rate** is also a required business input: if 30% of real mail is multi-intent, §12.8's segmentation is critical-path; if it is 3%, abstention alone may suffice.

### 5.3 Sampling strategy

Taking the most recent N messages produces a deceptively easy, unrepresentative dataset. Sampling must be **stratified**:

| Dimension | Rule | Reason |
|---|---|---|
| Time | Even allocation across ≥12 months | Quarter-end and fiscal-year-end mail looks nothing like mid-quarter mail |
| Inbox | Proportional across all shared inboxes, min 200 emails each | Inboxes have different intent mixes |
| Sender | **Cap any single sender domain at 5%** of the corpus | A few automated senders would otherwise dominate |
| Message position | Ensure **≥30% replies** (not thread-openers) | Production is reply-heavy; a corpus of opening asks alone widens the §2.2 gap |
| Intent | Oversample rare intents | Natural frequency leaves rare classes untrainable |

### 5.4 Cleaning

Applied **per email**, identically in both regimes — training/serving consistency depends on it:

1. **HTML → text** — strip `script`/`style`, convert block tags to newlines, unescape entities.
2. **Quote-chain stripping — mandatory.** Un-stripped replies are largely duplicated quoted text; the model would learn boilerplate instead of intent, and would classify *someone else's earlier ask* (A11). Patterns handled:
   - `>` -prefixed lines
   - `On <date>, <person> wrote:`
   - `-----Original Message-----`
   - Outlook `From:/Sent:/To:/Subject:` blocks and `____` separators
3. **Signature & disclaimer removal** — `-- ` delimiter, `Sent from my …`, and legal footers (`CONFIDENTIALITY NOTICE`, `This email and any attachments…`, `If you are not the intended recipient…`). Procurement footers are large and would dominate short messages.
4. **Invisible character stripping** — zero-width joiners, BOM, soft hyphens, combining marks.
5. **Whitespace artefact collapse** — separator runs (`| | |`), repeated blank lines.
6. **URL normalisation** — replace long tracking URLs with `<URL:domain.com>`, retaining the domain as signal while discarding token noise.
7. **Minimum length filter** — below 20 characters after cleaning, an email carries no signal.

> **This cleaning code is shared, not duplicated.** Any divergence between the training-time and inference-time cleaner silently degrades production accuracy — one of the most common and hardest-to-diagnose ML defects.

### 5.5 Model input format (Regime A and B — identical)

The text presented to the LLM labeler, to ModernBERT at training time, and to ModernBERT at inference time:

```
Subject: <subject, Re:/Fwd: stripped>
From: <sender domain only>

<cleaned body — this message's new content only>
```

- **One email. No thread history.** Identical in both regimes, which is what keeps training and serving consistent.
- Sender reduced to **domain** — identity is noise; supplier-vs-internal is signal.
- `max_seq_length` 512 is sufficient for a single cleaned message (vs the 1024–2048 a thread would have required — a useful side benefit of D2).

### 5.6 Thread assembly (Regime B only)

Not used in training. Specified here because the connector must capture the headers it needs (§5.1).

1. Group by connector-provided conversation id (`X-GM-THRID` / `conversationId`) where present — authoritative.
2. Otherwise union-find over `Message-ID` ↔ `In-Reply-To` / `References` edges.
3. Fallback: normalised subject (strip `Re:`/`Fwd:`/ticket numbers) **+ participant-set overlap ≥ 50%** **+ within a 30-day window**. Subject alone over-merges (`"Invoice"` would collapse into one giant thread).

Exchange `conversationId` can span months of unrelated replies to a recurring subject — apply a **30-day silence split** on top of it.

### 5.7 Deduplication

- Exact: SHA-256 of normalised email text.
- Near-duplicate: MinHash/LSH at Jaccard ≥ 0.9 (templated automated mail). Retain one exemplar, record `duplicate_of`.

### 5.8 Training schema (Regime A) — SQLite → PostgreSQL at scale

Email-centric and single-label. Threads do not appear here at all; they are a Regime B construct (§5.9).

```sql
CREATE TABLE emails (
  email_id         TEXT PRIMARY KEY,        -- Message-ID
  source           TEXT NOT NULL,           -- curated | sampled
  source_inbox     TEXT,
  connector        TEXT,                    -- imap | graph
  from_domain      TEXT,
  sent_at          TEXT,
  subject_norm     TEXT,
  body_clean       TEXT NOT NULL,
  model_input_text TEXT NOT NULL,           -- exactly what the model sees (§5.5)
  text_hash        TEXT NOT NULL,
  duplicate_of     TEXT REFERENCES emails(email_id),
  is_reply         INTEGER,                 -- track reply share (§5.3)
  has_attachments  INTEGER,
  attachment_types TEXT,
  -- Regime B threading headers, captured now, unused in training
  thread_hint      TEXT,
  in_reply_to      TEXT,
  references_json  TEXT,
  taxonomy_version INTEGER,
  split            TEXT,                    -- train | val | test | gold_dev | gold_test | excluded
  exclusion_reason TEXT,                    -- multi_intent (D13) | empty | duplicate
  review_status    TEXT DEFAULT 'unreviewed',
  created_at TEXT, updated_at TEXT
);

-- ONE intent per email per source (D1). Multiple rows only where sources differ,
-- which is what allows model-vs-LLM-vs-human comparison.
CREATE TABLE email_labels (
  email_id      TEXT NOT NULL REFERENCES emails(email_id),
  intent        TEXT NOT NULL,
  source        TEXT NOT NULL,   -- llm | human | bert | prototype | rule | revision
  confidence    REAL,
  rationale     TEXT,
  model_version TEXT,
  annotator     TEXT,
  created_at    TEXT,
  PRIMARY KEY (email_id, source, model_version, annotator)   -- one intent per source
);

CREATE TABLE email_confidence (
  email_id      TEXT PRIMARY KEY REFERENCES emails(email_id),
  band          TEXT,      -- high | medium | low
  consistency   REAL,      -- agreement rate across self-consistency runs
  k_runs        INTEGER,
  model_version TEXT
);

CREATE TABLE gold_annotations (
  email_id   TEXT NOT NULL REFERENCES emails(email_id),
  annotator  TEXT NOT NULL,
  intent     TEXT NOT NULL,
  is_primary INTEGER DEFAULT 0,   -- gold MAY be multi-intent (§8.2); training may not
  round      TEXT,                -- initial | adjudication
  created_at TEXT,
  PRIMARY KEY (email_id, annotator, intent, round)
);

CREATE TABLE model_registry (
  model_version TEXT PRIMARY KEY,
  model_type    TEXT,        -- bert | prototype
  trained_at    TEXT,
  train_rows    INTEGER,
  taxonomy_version INTEGER,
  metrics_json  TEXT,
  thresholds_json TEXT,
  promoted      INTEGER DEFAULT 0,
  promoted_at   TEXT,
  notes         TEXT
);
```

### 5.9 Runtime schema (Regime B) — threads, classification, intent lifecycle, actions

Production only; not built here. `threads` and `messages` are reconstructed from `emails` via §5.6.

```sql
CREATE TABLE threads (
  thread_id          TEXT PRIMARY KEY,
  source_inbox       TEXT NOT NULL,
  subject_norm       TEXT,
  first_at TEXT, last_at TEXT,
  msg_count          INTEGER,
  participant_domains TEXT,
  correction_budget_used INTEGER DEFAULT 0,   -- A16
  state              TEXT,   -- active | quiet | frozen_for_human
  created_at TEXT, updated_at TEXT
);

CREATE TABLE messages (
  message_id     TEXT PRIMARY KEY,
  thread_id      TEXT NOT NULL REFERENCES threads(thread_id),
  seq            INTEGER,
  from_domain    TEXT, sent_at TEXT,
  body_clean     TEXT,
  model_input_text TEXT,          -- §5.5 — same format as training
  is_reply       INTEGER,
  is_agent       INTEGER DEFAULT 0,   -- B6/R17: never triggers, never trains
  has_attachments INTEGER
);
```

Training/curation uses §5.8. Production additionally requires the tables below. `email_labels` records *what a model said about one message*; `thread_intents` records *what the thread currently means* — deliberately separate, because the second has a lifecycle and drives side effects.

```sql
-- One immutable row per classification run (one per inbound message).
-- Append-only: re-classification adds a row, never edits one. Audit spine.
CREATE TABLE classifications (
  classification_id  TEXT PRIMARY KEY,
  thread_id          TEXT NOT NULL REFERENCES threads(thread_id),
  trigger_message_id TEXT NOT NULL REFERENCES messages(message_id),
  message_text_snapshot TEXT NOT NULL,  -- G1: audit must not depend on live mail
  segment_index      INTEGER,           -- §12.8 segmentation; NULL = whole message
  predicted_intent   TEXT NOT NULL,     -- SINGLE label — the model is single-label (D1)
  predicted_prob     REAL NOT NULL,
  runner_up_intent   TEXT,              -- D14 margin check
  runner_up_prob     REAL,
  margin             REAL,              -- predicted_prob - runner_up_prob
  multi_intent_suspected INTEGER DEFAULT 0,   -- §12.8
  model_version      TEXT NOT NULL,
  taxonomy_version   INTEGER NOT NULL,
  decided_at         TEXT NOT NULL
);

-- The thread's *current* intent set, with lifecycle. One row per intent
-- ever activated on the thread; state changes, rows are never deleted.
CREATE TABLE thread_intents (
  thread_id        TEXT NOT NULL REFERENCES threads(thread_id),
  intent           TEXT NOT NULL,
  state            TEXT NOT NULL,   -- detected|active|actioned|superseded|retracted
  first_detected_classification_id TEXT REFERENCES classifications(classification_id),
  confidence       REAL,
  source           TEXT,            -- bert | ensemble | rule | human
  superseded_by    TEXT,            -- intent id that invalidated this one (A3)
  retracted_reason TEXT,            -- human_correction | conflict | stale
  taxonomy_version INTEGER,
  model_version    TEXT,
  created_at TEXT, updated_at TEXT,
  PRIMARY KEY (thread_id, intent)   -- A1: dedupe re-expressed intents
);

-- Declares which intents invalidate which. Drives A3, E1, E3.
CREATE TABLE intent_conflicts (
  intent           TEXT NOT NULL,
  supersedes       TEXT NOT NULL,
  mode             TEXT NOT NULL,   -- supersede | mutually_exclusive | requires
  PRIMARY KEY (intent, supersedes)
);

-- One row per action spawned by a newly-activated intent.
CREATE TABLE actions (
  action_id        TEXT PRIMARY KEY,
  thread_id        TEXT NOT NULL REFERENCES threads(thread_id),
  intent           TEXT NOT NULL,
  action_type      TEXT NOT NULL,
  risk_tier        INTEGER NOT NULL,      -- 0..3 (§12.7.3)
  state            TEXT NOT NULL,
    -- pending|blocked|ready|awaiting_approval|dispatched
    -- |succeeded|failed|retrying|dead_letter|compensating|compensated|cancelled
  payload_json     TEXT NOT NULL,         -- FROZEN at creation (D1/§12.7.5)
  business_key     TEXT,                  -- invoice no. / PO no. — 2nd idempotency level (C10)
  depends_on       TEXT REFERENCES actions(action_id),  -- C5 ordering
  attempt_count    INTEGER DEFAULT 0,
  next_attempt_at  TEXT,
  last_error_code  TEXT,
  last_error_class TEXT,                  -- transient | terminal | data_missing (C6/C7)
  sap_document_id  TEXT,
  created_at TEXT, updated_at TEXT,
  UNIQUE (thread_id, intent, action_type) -- C1: idempotency level 1
);

CREATE UNIQUE INDEX ux_actions_business_key
  ON actions(action_type, business_key) WHERE business_key IS NOT NULL; -- level 2

-- Every dispatch attempt, for audit and unknown-outcome reconciliation.
CREATE TABLE action_attempts (
  attempt_id     TEXT PRIMARY KEY,
  action_id      TEXT NOT NULL REFERENCES actions(action_id),
  attempt_no     INTEGER NOT NULL,
  idempotency_key TEXT NOT NULL,          -- (thread_id, action_id, attempt-invariant)
  request_at     TEXT, response_at TEXT,
  outcome        TEXT,                    -- success|failure|timeout|unknown (C1)
  error_code     TEXT, error_class TEXT,
  response_json  TEXT
);
```

---

## 6. LLM silver labeling

### 6.1 Output contract

Strict JSON, **single-label** (D1), with explicit multi-intent flagging:

```json
{
  "intent": "po_confirmation",
  "confidence": 0.91,
  "multi_intent": false,
  "other_intents_present": [],
  "rationale": "Supplier confirms receipt and acceptance of PO 4500."
}
```

Rules enforced in the prompt:
- Only intent ids from the frozen taxonomy.
- **Exactly one `intent`** — this is a single-label corpus (D1).
- Return `MISC` if nothing fits.
- **If the email genuinely contains more than one intent, set `multi_intent: true` and list them in `other_intents_present`.** Such emails are **excluded from training** (D13) and routed to `multi_intent_log` (§5.2.1) — the LLM must not silently pick one and discard the rest, because that is exactly the noise D1 exists to prevent.
- Few-shot seed sentences for the nearest candidate intents are retrieved and injected (§4.3).

### 6.2 Confidence via self-consistency — not self-report

**A model's verbalised "confidence: 0.9" is a generated token, not a probability**, and is systematically over-optimistic. In the consumer pilot only 5.4% of items were flagged low-confidence, almost certainly an underestimate.

Procedure:
- Run the labeler **k = 3** times at temperature 0.4 (escalate to k = 5 when the first 3 disagree).
- Confidence = **agreement rate** across runs (single-label makes this a simple vote, not a Jaccard computation).
- Final label = **majority vote**.

| Band | Condition | Routing |
|---|---|---|
| **HIGH** | 3/3 runs agree | stratified sample reviewed (§9.2) |
| **MEDIUM** | 2/3 agree | **full human review** |
| **LOW** | no majority, or `MISC` | **full human review** |
| **EXCLUDED** | any run flags `multi_intent` | removed from training; logged (§5.2.1) |

Where the endpoint exposes token logprobs, blend them in as a secondary signal. Prototype-classifier disagreement (§4.3) is a third.

### 6.3 Operational notes

- Concurrency ~8 workers; resumable via `email_id` tracking.
- Cost: 3× base calls (~30k calls for 10k emails). Budget accordingly.
- Every labeling run is stamped with `model_version` (LLM name + prompt hash) so results remain reproducible and comparable.
- **Track the multi-intent exclusion rate** — it is a required input to §12.8 sizing (§5.2.1).

---

## 7. Rules engine (decorrelated signal)

Neural models trained on the same text and the same labels make **largely the same mistakes**. Deterministic rules fail in *different* places, which is exactly what makes them valuable in an ensemble — and several are near-100% precision:

| Signal | Example | Use |
|---|---|---|
| PO number pattern | `\b45\d{8}\b` | strong evidence for PO-related intents |
| Invoice / credit-note reference | `INV[- ]?\d+`, `\bcredit note\b` | invoice intents |
| RFQ / tender keywords | `RFQ`, `RFP`, `tender`, `bid` | sourcing intents |
| Sender domain class | known supplier / internal / marketplace | prior over intents |
| `In-Reply-To` present | — | reply vs new request |
| Attachment MIME types | `application/pdf` + "invoice" | invoice submission |
| Bank-detail keywords | `IBAN`, `account number`, `remittance` | **high-risk** intent, fraud-sensitive |

Rules are **advisory by default**; only rules demonstrating ≥99% precision on the gold dev split may act as hard overrides. Bank-detail threads should always be surfaced regardless of model confidence, given fraud exposure.

---

## 8. Gold evaluation set

> **This is the single highest-value component of the plan.** Without it, every metric measures imitation of the LLM rather than correctness.

Evidence from the consumer pilot: the trained model reached **97.5% agreement with LLM labels** while scoring **89.1% on held-out data** — the first number describes how well it copied the teacher, not how often it was right.

### 8.1 Construction

| Property | Value |
|---|---|
| Size | **600 emails** |
| Selection | **Uniform random from real production-like traffic** — no confidence filtering, no stratification, and **never from the business-curated pool** (D3) |
| Annotation | **2 independent annotators**, blind to LLM output |
| **Cardinality** | **Multi-intent permitted here** — annotators record *every* intent present |
| Adjudication | Disagreements resolved by a third senior reviewer |
| Splits | `gold_dev` 300 (calibration + threshold tuning) · `gold_test` 300 (**frozen, reporting only**) |

> **The gold set is deliberately asymmetric with the training set.** Training is single-label and curated (D1, D13); gold is multi-intent and uncurated. This is intentional: the gold set must measure the model against **the distribution it will actually face**, not the one it was trained on. A gold set built from curated single-intent emails would simply re-confirm the training assumptions and hide the §2.2 gap entirely.

Because gold may carry several intents while the model emits one, evaluation uses **production-mode metrics** (§11.1): *is the predicted intent present in the gold intent set?*, plus intent-set recall via segmentation (§12.8).

The gold set also yields the **true multi-intent rate** in production traffic — the number that determines how much §12.8 machinery is actually justified.

### 8.2 Why the corrected corpus cannot substitute

Steps in §9 correct *low*-confidence items fully and *high*-confidence items only thinly. The result is non-uniformly clean — excellent training data, invalid as an evaluation set.

### 8.3 Inter-annotator agreement

Gold annotations may carry several intents, so agreement is measured over **sets**. Cohen's κ does not apply.

- **Jaccard similarity** (primary, interpretable)
- **Krippendorff's α with MASI distance** (set-aware, chance-corrected)
- Also report **primary-intent-only agreement**, which is the ceiling most directly comparable to the single-label model.

**IAA is the accuracy ceiling.** If two procurement analysts agree only 78% of the time, no model will reach 90%, and the finding itself is actionable: the disagreeing intent pairs are precisely the merge candidates from §4.2.

Target: Jaccard ≥ 0.75. Below 0.65 → **stop and revise the taxonomy** before training.

### 8.4 Rules

- Never trained on. Never fine-tuned on. Excluded from all sampling.
- `gold_test` is opened only for final reporting and promotion gates.
- Re-annotated when the taxonomy version changes.

---

## 9. Human review workflow

### 9.1 Budget

At ~30 s/email, **1,000 reviews ≈ 8 hours** of analyst time. Expected Phase 3 load:

| Task | Volume (10k corpus) | Effort |
|---|---|---|
| Gold set (×2 annotators) | 600 × 2 | ~10 h |
| LOW + MEDIUM band review | ~1,500–2,500 | ~12–20 h |
| HIGH band stratified sample | ~600 | ~5 h |
| **Business curation of single-intent examples** | ≥100 per intent | **team-dependent — size with business** |
| **Total (review only)** | | **~30 h** |

This must be resourced explicitly — it is the main non-compute cost of the programme. Note that D1/D13 shift effort *forward*: curation is additional up-front work, repaid by a much simpler labeling and review task (one label, not a set).

### 9.2 Sampling the HIGH band

Uniform 1-in-50 over ~8k high-confidence emails yields ~160 items ≈ 8 per intent across 20 intents — enough for a global error estimate, useless per class.

Instead:
- **Stratify per intent** — minimum 25 per intent, oversampling rare classes.
- **Prioritise model-disagreement cases** (LLM vs prototype vs rules). Disagreement-driven sampling surfaces errors **5–10× more efficiently** than random draws.
- Remainder allocated randomly, to retain an unbiased error-rate estimate for that band.

### 9.3 Purpose of each review stream

| Stream | Primary purpose |
|---|---|
| LOW / MEDIUM full review | **Improve training labels** where the teacher is weakest |
| HIGH stratified sample | **Estimate label error rate** per intent; detect systematic teacher bias |
| Gold set | **Measure model accuracy**; gate promotion |

### 9.4 Review interface contract

File-based for this build (CSV/JSONL export → import). The future UI must preserve this contract:

- Present **the email text**, LLM label + rationale, confidence band.
- Capture: corrected intent (**one**), **reason code**, free-text note.
- Support **"multiple intents present"** as an explicit outcome → excludes the email from training and logs it (§5.2.1). Reviewers must never be forced to choose one of two valid intents.
- Support **"needs a new intent"** as an explicit outcome — the taxonomy evolution feedback loop.
- Record annotator identity + timestamp (audit + IAA).
- Queue ordering by **informativeness** (active learning), not FIFO.
- Never overwrite LLM labels — corrections are inserted as `source='human'` rows alongside them.

---

## 10. Model training

> **Regime A.** One email in, one intent out. Nothing in this section is thread-aware; thread behaviour is built above the model in §12.

### 10.1 Base model

**ModernBERT-base** (`answerdotai/ModernBERT-base`)

- Native 8,192-token context. Single cleaned emails need far less — **512 tokens is sufficient** — which makes training and inference materially cheaper than the thread-level design would have been.
- Requires `transformers >= 4.48`.
- Encoder-only: fast, cheap, self-hosted inference; no data egress.

### 10.2 Classification head (D7)

**Softmax head with cross-entropy loss** — the direct consequence of D1. With a strictly single-intent corpus there is no multi-label signal to learn from, so a sigmoid/BCE head would be strictly worse: it would discard the mutual-exclusivity information the curation effort deliberately created.

| Approach | Status |
|---|---|
| **Softmax + cross-entropy** | **Chosen** — matches the single-label corpus; probabilities are comparative, which is what §12.8's margin check needs |
| Sigmoid + BCE (multi-label) | Rejected — no multi-intent training signal exists under D13 |
| Binary relevance (N models) | Rejected — N× cost, no shared representation |
| Label powerset | Rejected — combinations unlearnable, and moot under D1 |

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base",
    num_labels=N,
    problem_type="single_label_classification",   # → CrossEntropyLoss
)
# targets: integer class indices
```

> **A useful property of softmax here:** because the outputs compete for a fixed mass, a genuinely two-intent email tends to split its probability (~0.45/0.40) rather than confidently picking one. That **low margin is the detection signal** §12.8 relies on to recover multi-intent cases the model was never trained to emit.

Three details that make or break it:

| Concern | Handling |
|---|---|
| **Class imbalance** | Class-weighted cross-entropy, weight ∝ `1/√freq`, clipped at 10. (Replaces the multi-label `pos_weight`.) |
| **Thresholds** | Never a global 0.5. **Per-class**, tuned on `gold_dev` (§10.4). Saved as a model artefact, versioned with the checkpoint. |
| **Empty predictions** | If nothing clears its threshold → take argmax if it clears a floor (0.25), else **MISC**. This is the abstention hook. |
| **Runaway label counts** | N/A under softmax — the head emits exactly one intent. Multi-intent recovery is a §12.8 concern, not a model concern. |

### 10.3 Training configuration

| Parameter | Value | Note |
|---|---|---|
| `max_seq_length` | **512** | A single cleaned email fits comfortably; thread-level would have needed 1024–2048 |
| `per_device_train_batch_size` | 32 | Fits 8 GB VRAM at 512 tokens |
| `gradient_accumulation_steps` | 1 | Not needed at this sequence length |
| Learning rate | 3e-5, linear decay | |
| Warmup ratio | 0.06 | |
| Epochs | 4–6, early stopping | patience 2 on val macro-F1 |
| Precision | **bf16** | not fp16 — stability on Ada/Ampere |
| Split | 80 train / 10 val / 10 test, **stratified by intent** | Standard stratification is sufficient — single-label removes the need for iterative stratification |

Splitting must be **time-aware and sender-aware**: the test split should be drawn from a later period to detect temporal drift, and **near-duplicate templated emails from one sender must not straddle splits** — that leaks, and inflates test scores.

> D1/D2 make training materially cheaper and simpler: 512 tokens instead of 2048, standard stratification instead of iterative, a 4× larger batch, and no threshold matrix to tune. This is a real benefit of the decision, not just a constraint.

### 10.4 Calibration & threshold tuning

1. Fit **temperature scaling** on `gold_dev`. Raw softmax outputs are not calibrated probabilities.
2. Tune two threshold families:
   - **Abstention threshold** per intent — below it, the prediction routes to MISC/human. Default objective: maximise recall subject to precision ≥ target (§11.4).
   - **Margin threshold** (§12.8) — below it, the email is flagged multi-intent-suspected.
3. Persist `thresholds_json` in `model_registry`. Thresholds are part of the model, not a runtime config.

### 10.5 Prototype model

Nearest-centroid over sentence embeddings from seed sentences (§4.3), refreshed with human-labeled data once available. Roles: cold-start before enough labels exist; disagreement/drift signal thereafter. **Not** a co-equal decision maker — see §12.2.

### 10.6 Phase 2 — closing the train/serve gap

The first model trains on business-curated examples plus reviewed historical mail. That corpus is cleaner and more prototypical than production traffic (§2.2). Once the model ships, the corpus must be grown toward the real distribution — **without violating D1 or D13**:

1. Sample real production messages, weighted toward where the model is weakest: low-confidence predictions, low-margin cases, and human-overridden decisions.
2. Label them **single-intent** as usual; multi-intent ones are excluded and logged (§5.2.1).
3. Deliberately include **short replies** and **reply-position messages**, which curation under-supplies.
4. Retrain, gate, promote (§12.5).

Expected effect: little movement in curated-data accuracy, **material improvement on production traffic** — which is the only number that matters. Track the two separately; a widening divergence is the signal that the gap is reopening.

---

## 11. Evaluation

### 11.1 Metrics

Two metric families, because training is single-label but production reality is not (§8.1).

**A — Model metrics** (single-label, on `gold_test` primary intent):

| Metric | Role |
|---|---|
| **Macro-F1** | **Primary.** Treats rare intents as equally important. |
| Accuracy / micro-F1 | Volume-weighted view |
| Per-class precision / recall / F1 / support | Operational truth; where routing actually breaks |
| Confusion matrix | Identifies intent pairs the taxonomy should merge (§4.2) |
| ECE / reliability diagram | Calibration quality |

**B — Production-mode metrics** (evaluated against the full, possibly multi-intent gold set):

| Metric | Role |
|---|---|
| **Hit rate** — is the predicted intent *in* the gold intent set? | The realistic accuracy measure |
| **Intent-set recall** — fraction of gold intents recovered after segmentation (§12.8) | Measures how much multi-intent content is missed |
| **Multi-intent detection rate** — recall of the margin flag on truly multi-intent emails | Validates D14 |
| **Coverage @ precision ≥ target** | **The business KPI** (§11.4) |
| Abstention rate | Volume routed to humans |

> Family A can look excellent while family B is poor — that divergence *is* the §2.2 gap, quantified. **Both must be reported; family B governs go-live.**

### 11.2 Baselines

Every model is reported against:
1. Majority-class baseline.
2. Rules engine alone.
3. Prototype classifier alone.
4. LLM zero-shot (the teacher) on `gold_test` — **this reveals whether the student beat the teacher.**

### 11.3 Reporting

All evaluation is on **`gold_test`** (human truth, production-like distribution). Agreement-with-LLM is reported separately and explicitly labelled as *imitation fidelity, not accuracy*. Performance on curated-style data is reported separately again, and is **not** the go-live number.

### 11.4 Business-facing target

The number that matters to the sponsor:

> *"We auto-classify **X%** of inbound volume at **≥95%** precision; the remaining **(100−X)%** routes to an analyst."*

Per-class precision targets are set from `misroute_cost`:

| `misroute_cost` | Precision target |
|---|---|
| high (invoices, bank details, contracts) | ≥ 0.97 |
| medium | ≥ 0.95 |
| low (newsletters, FYI) | ≥ 0.90 |

Uniform thresholds are wrong under procurement's class imbalance and unequal error costs.

---

## 12. Production serving — recommended design (out of scope to build here)

> **Regime B.** Everything in this section sits **above** the model. The model itself remains what §10 built: one email in, one intent out. Threads, multi-intent sets, accumulation, revision, and SAP actions are all orchestration concerns — none of them require retraining or a different head.

### 12.1 Pipeline

```
inbound message (is_agent=0)
   │
   ├─► segmentation (§12.8) ─► 1..n text spans
   │        each span:
   │          ├─► rules engine   ─► high-precision rule hits
   │          ├─► ModernBERT     ─► ONE intent + calibrated prob + margin
   │          └─► prototype      ─► ONE intent + calibrated prob
   │                    │
   │                    ▼
   │          combine (§12.2) ─► abstain? (§12.3)
   │                    │
   ▼                    ▼
union of span intents ──► message intent set (may be >1 — §12.8)
                            │
                            ▼
              accumulate into THREAD intent set (§12.7)
                 ├─ new          → spawn action → SAP
                 ├─ conflicting  → supersede → compensate
                 └─ contradicted → invalidate → compensate + correct label
```

### 12.2 Ensembling

First, an honest caveat: **the two neural models are highly correlated** — same text, same labels, same failure modes. Ensembles pay off when errors are decorrelated. **Prove the ensemble beats ModernBERT alone on `gold_test`; if it does not, ship ModernBERT alone.** The rules engine is the genuinely decorrelated member.

If ensembling:
- Calibrate each model first (temperature scaling).
- Combine **per class** via weighted average — ModernBERT **0.7**, prototype **0.3**.
- Do **not** multiply raw softmax outputs: the result is unnormalised, uncalibrated, and lets one over-confident model veto the other with a near-zero value.
- High-precision rule hits may override.

### 12.3 Abstention policy — the core of the design

What the business needs is **selective prediction**, not a single blended score:

```
if models_disagree(bert, prototype):          → MISC   # disagreement beats any blended score
elif top_prob < abstention_threshold[intent]: → MISC
elif margin < margin_threshold:               → multi-intent suspected (§12.8)
else:                                         → auto-route
```

Disagreement between models is a **stronger uncertainty signal** than a low combined score, and should short-circuit the blend.

### 12.4 Thread mutability

Threads are not immutable. A thread classified today gains messages tomorrow and its intent may legitimately change (`po_confirmation` → `dispute`).

- Key everything on `thread_id`; store `last_message_at` and `classified_at`.
- Re-classification is triggered **per inbound message on arrival** (D11), not by a daily sweep.
- Retain full classification history for audit (`classifications`, append-only).

**"Classify once, never re-classify" was explicitly considered and rejected (D9).** It was attractive — it deletes revision events, compensating actions and stability gating — but it fails on two counts: it freezes a decision taken on deliberately incomplete evidence, and it cannot represent a thread whose business meaning legitimately changes after the fact. The cost of that simplification is borne entirely by the irreversible-action path, which is the one place the system can least afford to be wrong.

Full lifecycle model follows in §12.7.

### 12.5 Retraining & governance

> **Primary hazard: the self-confirmation loop.** Retraining on the model's own high-confidence predictions amplifies existing bias — the model becomes progressively more certain about what it already gets wrong, while the metrics look excellent.

**Hard rule: train only on human-reviewed labels plus the original reviewed corpus. Never pseudo-label unreviewed predictions.**

Weekly cycle:

1. Ingest the week's human corrections (~600–900 new labels expected).
2. Fine-tune incrementally from the current checkpoint.
3. Re-calibrate and re-tune thresholds on `gold_dev`.
4. **Promotion gate — all must pass:**
   - macro-F1 on `gold_test` ≥ current − 0.5pp, and
   - **no individual class precision drops below its target**, and
   - coverage at target precision ≥ current − 1pp.
   A model that lifts the average while degrading one business-critical class is **blocked**.
5. Shadow-run against live traffic before promotion.
6. Register in `model_registry`; retain one-command rollback.

### 12.6 Monitoring

| Signal | Alert condition |
|---|---|
| Predicted class-mix drift | PSI > 0.2 vs trailing 4-week baseline |
| Abstention rate | ±10pp week-over-week |
| Mean max probability | sustained decline (creeping OOD input) |
| New sender domains | volume spike from unseen domains |
| Human override rate | rising → model degradation |
| Actions in `dead_letter` | any sustained growth |
| Actions stuck in `dispatched` | age > SLA (D3) |
| Compensation events | any occurrence → review misclassification |
| Intents per thread | p99 breaching the thread-level cap (A10) |
| **Invalidation rate** (A12) | rising → **model degradation**, retraining trigger |
| Supersession rate (A3) | rising → business-process change, *not* a model signal |
| Correction-budget exhaustions (A16) | any sustained growth → unresolvable threads |

### 12.7 Thread lifecycle, intent accumulation & action orchestration

This section specifies the consequence of D9–D12: a thread is a **living entity** whose intent set accumulates over time, and each newly activated intent dispatches an action into SAP with its own retry lifecycle.

> **The defining property of this architecture:** with `on_arrival` classification, additive intents, and external side effects, **every classification is potentially an irreversible financial action.** Abstention (§12.3) and the human-approval tier therefore carry far more weight than in a pure classification system.

#### 12.7.1 Flow

```
inbound counterparty message
   │  (agent-authored / internal messages never trigger — B6)
   ▼
acquire per-thread lock ─────────────────────► (F1–F3)
   ▼
rebuild thread text ─► classify ─► append immutable `classifications` row
   ▼
reconcile predicted intents against `thread_intents`:
   ├─ already active                  → no-op (A1)
   ├─ new                             → insert state=active, spawn action
   ├─ conflicts with an active intent → supersede prior (A3), compensate if dispatched
   └─ previously active, now absent   → leave active (additive); retract only if
                                        its action is still `pending` (A4)
   ▼
release lock
```

#### 12.7.2 Intent lifecycle & supersession

```
detected ─► active ─► actioned
              │  ├──► superseded   (conflicting intent arrived — A3)
              │  ├──► invalidated  (new evidence proves it was never correct — A12)
              │  └──► retracted    (human correction / stale — E3)
```

Intents are **never hard-deleted** — the row persists with a terminal state, because an action may already have been taken on it.

`superseded` and `invalidated` are deliberately **separate terminal states**, not synonyms — see §12.7.8. Collapsing them corrupts both the training corpus and the model-quality metrics.

Supersession is declarative, via `intent_conflicts`:

| `intent` | `supersedes` | `mode` |
|---|---|---|
| `cancel_po` | `po_confirmation` | supersede |
| `dispute_invoice` | `invoice_approval` | supersede |
| `update_bank_details` | — | requires `vendor_exists` |
| `order_cancellation` | `delivery_confirmation` | mutually_exclusive |

Supersession triggers a cascade on the prior intent's action, by current state: `pending`→cancel, `dispatched`→compensate, `succeeded`→compensating action + human notification.

#### 12.7.3 Action risk tiers

Not every action needs the same gate. Most operational value is Tier 0, which needs no certainty about thread completeness at all.

| Tier | Examples | Gate |
|---|---|---|
| **0 — reversible / internal** | tag, route to queue, set priority, surface in UI | dispatch immediately |
| **1 — soft external** | auto-acknowledge, request missing info | reply budget + cooldown |
| **2 — system-of-record write** | create ticket, update PO status, post invoice | confidence ≥ threshold; dependency-gated |
| **3 — irreversible / financial** | payment release, **bank-detail change** | **human approval always**, regardless of model confidence |

**Tier 3 never auto-dispatches.** This is independent of model quality and is not subject to future relaxation on the basis of improved metrics.

#### 12.7.4 Dual idempotency

One key is not enough:

| Level | Key | Prevents |
|---|---|---|
| 1 — action | `(thread_id, intent, action_type)` → `action_id` | retry double-apply; unknown-outcome timeouts (C1) |
| 2 — business object | `(action_type, business_key)` e.g. invoice no. | the same invoice arriving in **two different threads** (C10), and duplicate posts after a thread merge (B2) |

Level 2 is the only defence against thread-assembly errors producing duplicate SAP postings.

#### 12.7.5 Action state machine

```
pending ─► blocked ─► ready ─► awaiting_approval ─► dispatched
                                                       │
                        ┌──────────────────────────────┼───────────────┐
                        ▼                              ▼               ▼
                    succeeded                       failed          (no reply)
                                                       │               │
                                            transient ─┤               ▼
                                                       ▼         timeout sweep
                                                   retrying ──► dead_letter
                                            terminal ─► dead_letter (human queue)

     any state ─► cancelled | compensating ─► compensated   (supersession / human override)
```

Rules:
- **Payload is frozen at creation.** A retry replays the original payload; it never re-derives from a fresh classification, or retry semantics become nondeterministic across model versions (D1).
- **Error classes are not interchangeable**: `transient` (SAP down) retries with exponential backoff + jitter + circuit breaker; `terminal` (PO does not exist, duplicate invoice, closed period) must **not** retry — retrying is futile and masks the real problem; `data_missing` (no extractable PO number) routes to human enrichment, not retry.
- **Status is per action, never per thread** — a 3-intent thread can be 2 succeeded / 1 dead-lettered, and there is no atomic transaction spanning them (C4).
- **Unknown outcomes are reconciled, not retried blindly** — a periodic sweep compares local state against SAP for `dispatched` actions past SLA.
- **Retries are never silently exhausted** — terminal path is always a human queue.

#### 12.7.6 Exceptional case catalogue

The cases this design must handle, by category. IDs are referenced throughout §5.8 and §12.7.

**A. Classification-level**

| # | Case | Handling |
|---|---|---|
| A1 | Same intent re-expressed later in the thread | Dedupe on `(thread_id, intent)` while active → no new action |
| A2 | Intent re-detected *after* its action completed (invoice resent) | Per-intent policy `repeatable` vs `once_per_thread`; non-repeatable → suppress + flag |
| A3 | Contradictory intent (`cancel_po` after `po_confirmation`) | `intent_conflicts` → supersede; cascade to the prior action |
| A4 | Previously-detected intent no longer predicted | Remains active (additive); retractable only while its action is `pending` |
| A5 | Misclassification, action already committed in SAP | Compensating action + mandatory human notification; cannot be auto-corrected |
| A6 | Borderline-confidence intent later contradicted | Handled as A3 if a conflicting intent appears; otherwise stands |
| A7 | Classification itself fails (model down, OOM, timeout) | Dead-letter the **message** and retry — a silent skip is a permanently missed action |
| A8 | Taxonomy version changes mid-thread | Mixed-version intents; stamp version per row; block action if the intent id was merged/split |
| A9 | Model version changes mid-thread | Benign; `model_version` recorded per intent for audit |
| A10 | **Thread-level intent explosion** | The per-classification cap of 3 does not bound accumulation over 20 messages → thread-level cap + alert |
| A11 | Intent present only in quoted/forwarded content | Highest-frequency false-positive source; depends entirely on quote-stripping (§5.3) |
| A12 | **New evidence invalidates an earlier intent** (retro-correction) | Distinct from A3 — the earlier decision was *never* right. Retract + compensate + **correct the training label** (§12.7.8) |
| A13 | Explicit retraction language (*"please ignore my last email"*, *"sent in error"*, *"disregard"*) | Strong retro-correction trigger; detected by rule, confirmed by re-scoring |
| A14 | Intent flapping across successive messages | Hysteresis (activate 0.60 / retract 0.25) + per-thread correction budget |
| A15 | A **newer model** disagrees with a frozen past decision | **Not a correction.** Never retro-invalidates; otherwise every promotion triggers mass compensation |
| A16 | Correction of a correction (flip-flop) | Correction budget exhausted → freeze intent set, route thread to human |
| A17 | Retro-correction not propagated to the training corpus | Known-wrong label would be trained on — must cascade to `email_labels` |

**B. Thread assembly**

| # | Case | Handling |
|---|---|---|
| B1 | Out-of-order delivery | Each message classifies on arrival; additive accumulation absorbs this naturally |
| B2 | **Thread merge** discovered late | Two threads with dispatched actions for one business event → only business-key idempotency prevents duplicate SAP posts |
| B3 | **Thread split** (subject-fallback over-merge) | Reassign intents/actions to the correct child; ambiguous → human |
| B4 | Message on an archived/closed thread | Reopen, or new thread with `continues_from` link |
| B5 | Same message delivered twice (UID change, folder move) | Dedupe on `Message-ID` at ingest, before classification |
| B6 | Agent's own reply re-ingested | `is_agent=1` → never triggers classification or action |
| B7 | Forwarded third-party thread | Contains a foreign conversation; can trigger unrelated intents |
| B8 | Thread exceeds token budget, middle dropped | Mitigated by per-message classification — the middle was classified on arrival. A genuine argument for this architecture |

**C. SAP integration**

| # | Case | Handling |
|---|---|---|
| C1 | **Timeout — did SAP commit?** | The central hazard. Idempotency key makes retry safe; never retry without one |
| C2 | SAP succeeds, local status write fails | Safe only with C1's key; reconciliation sweep catches it |
| C3 | Transient outage → retry storm | Exponential backoff + jitter + circuit breaker + max attempts |
| C4 | Partial success across a multi-intent thread | Per-action status; no cross-action transaction |
| C5 | Ordering dependency (`create_vendor` → `update_bank_details`) | `depends_on` gating; intents arrive in arbitrary order |
| C6 | Business-rule rejection (PO absent, duplicate, closed period) | **Terminal** — must not share a retry path with C3 |
| C7 | Required entity not extractable | Human enrichment queue, not retry |
| C8 | **Wrong entity extracted** (PO number from a quoted footer) | Validate against thread recency + SAP existence check before dispatch |
| C9 | Stale action (PO already cancelled) | Pre-execution freshness check |
| C10 | Same business object across two threads | Business-key idempotency (level 2) |
| C11 | Action succeeded on a misclassification | Audit + reversal workflow + human notification |

**D. Retry & state machine**

| # | Case | Handling |
|---|---|---|
| D1 | Retry after model retrained | Replay the **frozen** payload; never re-derive |
| D2 | Human override while a retry is in flight | Cancel-token; the human decision wins |
| D3 | Stuck in `dispatched`, no callback | Timeout sweeper → reconcile → resolve or dead-letter |
| D4 | Retries exhausted | Dead-letter → human queue; never silently dropped |
| D5 | Out-of-order callbacks | Keyed by `action_id`, not sequence |
| D6 | Duplicate callback | Idempotent status transitions |

**E. Human-in-the-loop**

| # | Case | Handling |
|---|---|---|
| E1 | Human corrects intents after actions fired | Cascade: cancel `pending`, compensate `dispatched`, flag `succeeded` |
| E2 | Human adds an intent manually | Spawns an action identically to a model-detected intent |
| E3 | Human removes an intent | Retraction path — the explicit exception to pure-additive |
| E4 | MISC thread classified by a human days later | Late dispatch; staleness check first (C9) |
| E5 | Human disagrees with an already-successful action | Compensation workflow + training signal |

**F. Concurrency & timing**

| # | Case | Handling |
|---|---|---|
| F1 | Rapid-fire messages under `on_arrival` | **Per-thread serialisation is mandatory** |
| F2 | Two workers classify one thread concurrently | Thread-level lock or optimistic concurrency; otherwise duplicate intents → duplicate actions |
| F3 | Message arrives mid-classification | Queue; re-run after lock release |

**G. Data & compliance**

| # | Case | Handling |
|---|---|---|
| G1 | Message deleted/recalled after action | `thread_text_snapshot` at classification time — audit must not depend on live mail |
| G2 | Retention policy vs audit snapshots | Retention applied to snapshots; legal hold for actioned threads |
| G3 | Bank-detail intents | Never auto-action regardless of confidence (Tier 3) |

#### 12.7.7 Resolving patterns

1. **Intent lifecycle** — terminal states, never hard deletes.
2. **Declarative `intent_conflicts`** — supersession is configuration, not code.
3. **Action state machine** with explicit compensation path.
4. **Dual idempotency** — action key *and* business-object key.
5. **Retryable vs terminal error taxonomy** — separate code paths.
6. **Frozen action payloads** — classification is revisable; a dispatched action's payload is not.
7. **Per-thread serialisation** — mandatory under `on_arrival`.
8. **Reconciliation sweep** — periodic SAP-vs-local comparison (C2, D3).
9. **Risk-tiered dispatch** — Tier 0 immediate, Tier 3 never automatic.

#### 12.7.8 Decision revision — supersession vs correction

A later decision may overwrite an earlier one for **three semantically different reasons**. They share no code path, because each has different consequences for compensation, training data, and metrics.

| | **Supersession** (A3) | **Invalidation / retro-correction** (A12) | **Human override** (E1/E3) |
|---|---|---|---|
| Trigger | A conflicting intent legitimately arrives | New evidence shows the earlier intent was never present | Reviewer disagrees |
| Was the earlier decision correct at the time? | **Yes** | **No** | Either |
| Terminal state | `superseded` | `invalidated` | `retracted` |
| Training corpus | **Keep** — a valid example | **Correct it** (A17) — otherwise a known-wrong label is trained on | **Correct it** — gold-quality signal |
| Metric | Business-process rate | **Model error rate** | Model error rate |
| Compensation urgency | Normal — the action was legitimate | **Elevated** — the action should never have fired | Per reviewer instruction |

> Conflating supersession with invalidation silently poisons the training corpus with labels already known to be wrong — the §12.5 self-confirmation hazard arriving through a different door.

**Trigger rule — evidence, not model version.**

> Retro-correction is triggered by **new evidence within the thread**, never by a **new model version** (A15). Production decisions are frozen per the `classifications` audit spine; if a model upgrade could retro-invalidate history, every promotion would cascade mass compensation into SAP. Offline re-scoring (§11) measures new models without touching frozen decisions.

**Detection.** On each re-classification, currently-active intents are re-scored — not just new ones. Because the model is single-label and per-message (§10.2), re-scoring an *earlier* intent means **re-running the model on the message that originally produced it**, now read in light of later messages (e.g. an explicit retraction). An active intent becomes an invalidation candidate when:

- re-scoring its originating message drops its probability below the **retraction threshold**, or
- an explicit retraction phrase is detected in a later message (A13), or
- the extracted entity is contradicted by a later message (C8).

**Hysteresis is mandatory (A14).** Activation and retraction thresholds must differ — e.g. activate at **0.60**, retract only below **0.25**. A single symmetric threshold makes intents flap across successive messages, and every flap is an SAP action or compensation.

**Auto-retraction is tier-gated.** A wrong retraction cancels a legitimate action, so autonomy narrows as risk rises:

| Tier | On invalidation |
|---|---|
| 0–1 | Auto-retract and auto-compensate |
| 2 | Auto-retract; compensation requires human confirmation |
| 3 | **Never auto-retract** — flag for human decision only |

**Correction budget (A16).** A thread may be corrected a bounded number of times (default 2). Beyond that the intent set freezes and the thread routes to a human — persistent flip-flopping indicates the model cannot resolve the thread and further automated churn only multiplies SAP traffic.

**Cascade on invalidation:**

1. `thread_intents.state → invalidated`, with `revision_id` and evidence reference.
2. Its action: `pending` → cancel · `dispatched` → compensate (tier-gated) · `succeeded` → compensating action + human notification.
3. **Correct the training label** — insert a `source='revision'` row into `email_labels` for the originating email, superseding the original, so the corpus never carries a known-wrong label (A17).
4. Emit a `decision_revisions` audit row.
5. Count against the thread's correction budget.

**Schema addition (§5.8):**

```sql
CREATE TABLE decision_revisions (
  revision_id       TEXT PRIMARY KEY,
  thread_id         TEXT NOT NULL REFERENCES threads(thread_id),
  intent            TEXT NOT NULL,
  revision_type     TEXT NOT NULL,   -- supersession | invalidation | human_override
  from_state        TEXT NOT NULL,
  to_state          TEXT NOT NULL,
  triggered_by_classification_id TEXT REFERENCES classifications(classification_id),
  triggered_by_message_id        TEXT REFERENCES messages(message_id),
  evidence          TEXT,            -- retraction phrase, prob. delta, entity contradiction
  prior_confidence  REAL,
  new_confidence    REAL,
  action_id         TEXT REFERENCES actions(action_id),
  action_outcome    TEXT,            -- cancelled | compensated | notified | none
  training_label_corrected INTEGER DEFAULT 0,   -- A17
  decided_by        TEXT,            -- model | human:<id>
  created_at        TEXT NOT NULL
);
```

**Metrics.** `invalidation_rate` is a **model-quality** signal and feeds the §12.6 monitors; `supersession_rate` is a **business-process** signal and must be reported separately. A rising invalidation rate is a retraining trigger; a rising supersession rate is not.

### 12.8 Multi-intent recovery — bridging Regime A and Regime B

The defining constraint of this architecture:

> **The model is single-label (D1). Production mail is not.** An email carrying two asks must still produce two actions, using a model that can only ever emit one intent.

Multi-intent emails were deliberately excluded from training (D13) to keep the learning signal clean. That decision is paid for **here**, in the orchestration layer — not in the model.

#### 12.8.1 Three recovery mechanisms

**1. Thread-level accumulation (free — the primary mechanism)**

Most multi-intent *threads* are not multi-intent *emails*. The asks usually arrive in separate messages: message 1 confirms the PO, message 3 requests the bank change. Per-message classification plus §12.7 accumulation already yields a multi-intent thread with **no extra machinery**.

This alone covers the majority of real cases and is the main reason D1/D2 are viable at all.

**2. Segmentation (for genuinely multi-ask single emails)**

Split the cleaned email into candidate spans and classify each independently, then take the union:

- Split on paragraph breaks, numbered/bulleted lists, and discourse markers (*"Also,"*, *"Separately,"*, *"One more thing,"*, *"Additionally,"*).
- Each span ≥ 20 characters is classified by the same single-label model.
- Union the span intents; discard spans predicted `MISC` or below the abstention threshold.
- **Cap at 3 intents per message** — more indicates segmentation noise → route to human.

Crucially this **requires no model change** — the model still does one-span-one-intent, exactly matching its training distribution. Spans of a single email actually resemble curated training examples *more* closely than the raw email does.

**3. Margin detection (the safety net — D14)**

Softmax outputs compete for fixed probability mass, so a genuine two-intent input splits it:

| Pattern | Interpretation |
|---|---|
| 0.92 / 0.04 | Confident single intent → auto-route |
| 0.45 / 0.40 | **Two intents competing** → segment, or route to human |
| 0.30 / 0.25 | Confused or out-of-taxonomy → MISC |

`margin = top_prob − runner_up_prob`. Below the tuned `margin_threshold` (§10.4), flag `multi_intent_suspected` and escalate to segmentation, then to a human if still unresolved.

> This is why the **softmax head is a genuine advantage over sigmoid here** (§10.2): competitive probabilities make multi-intent detectable *precisely because* the classes compete. Independent sigmoid outputs would not carry this signal.

#### 12.8.2 Escalation order

```
classify whole message
   │
   ├─ high prob, high margin  ─────────────► single intent, auto-route
   │
   ├─ low margin (D14)  ──► segment (§12.8.1)
   │        ├─ spans yield distinct confident intents ──► multi-intent, act on each
   │        ├─ spans agree ─────────────────────────────► single intent, auto-route
   │        └─ spans unclear ───────────────────────────► HUMAN
   │
   └─ low prob  ───────────────────────────────────────► MISC → HUMAN
```

#### 12.8.3 Residual risk — accepted, and measured

Some multi-intent emails will be missed: two asks in one paragraph, with one expressed briefly, may yield a confident single prediction and **silently lose the second intent — and therefore its action.**

This risk is **accepted by design**, but must be **quantified, not assumed**:

- The gold set (§8.1) permits multi-intent annotation specifically to measure it — **intent-set recall** (§11.1 family B) is that measurement.
- The `multi_intent_log` (§5.2.1) supplies a test set of known multi-intent emails to validate segmentation against.
- If measured intent-set recall is unacceptable, the escalation options are, in increasing cost: lower the margin threshold (more human review, no retraining); improve segmentation; or — only if the multi-intent rate proves genuinely high — revisit D1 and train a multi-label head on deliberately collected multi-intent data.

> **Sizing input required from the business:** the multi-intent rate. At ~3% of traffic, margin detection plus abstention is sufficient and segmentation is optional. At ~30%, segmentation is critical-path and D1 should be formally re-examined. This number comes out of §8 and §5.2.1 — **do not estimate it; measure it.**

---

## 13. Deliverables

**Regime A — built here:**

```
intentpipe/
  cleaning.py            HTML→text, quote/signature/disclaimer stripping, normalisation
                         ** shared verbatim with production — never forked (§5.4) **
  storage.py             SQLite schema + DAO (emails, email_labels, gold_annotations)
  formatting.py          model input construction (§5.5) — shared with production
  taxonomy.py            load/validate/version taxonomy; co-occurrence + separability checks
  connectors/
    base.py              RawMessage + MailConnector protocol
    imap_connector.py    IMAP implementation (captures threading headers for Regime B)
    (graph_connector.py) later

extract_emails.py        IMAP → SQLite; stratified sampling (incl. reply share), dedup
import_curated.py        ingest business-supplied single-intent examples
reconcile_taxonomy.py    embed + cluster + compare against business draft
label_emails.py          single-label LLM labeling, k-run self-consistency, bands,
                         multi-intent detection → multi_intent_log
gold_set.py              sample production-like mail, export, import, adjudicate, IAA
review_queue.py          export/import review batches (stratified + disagreement-driven)
train_classifier.py      ModernBERT single-label fine-tuning (softmax + CE)
calibrate.py             temperature scaling + abstention/margin threshold tuning
evaluate.py              gold-set benchmark: model metrics + production-mode metrics
segment_eval.py          validate §12.8 segmentation against multi_intent_log
infer.py                 inference testing harness (single email)

docs/
  email_spec.md          this document
  production_design.md   expanded §12 for the serving team
```

**Regime B — design only, built elsewhere:** thread assembly, intent accumulation and lifecycle, segmentation/margin orchestration, action dispatch + SAP integration, retry/compensation, reviewer UI.

> **Contract between the regimes:** production **must** import `cleaning.py` and `formatting.py` unchanged. Any divergence between training-time and inference-time text preparation silently degrades accuracy and is among the hardest ML defects to diagnose.

---

## 14. Phased plan

| Phase | Regime | Work | Exit criteria | Depends on |
|---|---|---|---|---|
| **0** | A | Business drafts intents + seed sentences + **curated single-intent examples** | Draft taxonomy, ≥10 seeds/intent, ≥100 examples/intent | Business team |
| **1** | A | Extraction + cleaning + storage (**emails, not threads**) | 5–10k emails in SQLite, dedup'd, ≥30% replies | Inbox credentials |
| **2** | A | Taxonomy reconciliation (clustering) | **Taxonomy v1 frozen**, business sign-off | Phases 0, 1 |
| **3** | A | LLM labeling + multi-intent exclusion | All emails labeled; **multi-intent rate measured** | Phase 2 |
| **4** | A | **Gold set** annotation (production-like, multi-intent allowed) | 600 emails, IAA ≥ 0.75, adjudicated | Phase 2, 2 annotators |
| **5** | A | Review queues (LOW/MED full, HIGH stratified) | Corrections imported | Phases 3, 4 |
| **6** | A | Training + calibration + threshold tuning | Model registered with thresholds | Phase 5 |
| **7** | A | Benchmark evaluation (**both metric families**) | Coverage @ precision vs baselines; intent-set recall reported | Phases 4, 6 |
| **8** | A | Inference testing | Spot-check harness validated on live mail | Phase 6 |
| **9** | A | **Phase-2 corpus augmentation** (§10.6) | Production-traffic accuracy improved; gap tracked | Phase 8 + live traffic |
| **→** | B | Production build (separate platform) | — | Phase 7 |

Phases 3 and 4 run in parallel. Phase 4 must **not** wait for Phase 5.

---

## 15. Risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Business taxonomy not text-separable | Wasted labeling budget; low ceiling | §4.2 reconciliation **before** labeling spend |
| R2 | Low IAA (analysts disagree) | Caps achievable accuracy | Measure early (Phase 4); merge confusable intents |
| R3 | Silver-label noise inherited from LLM | Model imitates teacher's errors | Self-consistency banding; full review of LOW/MED; gold set for truth |
| R4 | Self-confirmation retraining loop | Silent degradation, good-looking metrics | Human-labeled data only; frozen gold test; promotion gates |
| R5 | Class imbalance, rare intents | Rare classes never predicted | Class-weighted CE, per-intent abstention thresholds, macro-F1 primary, oversampled review |
| R6 | Thread assembly errors (Regime B) | Wrong unit of accumulation | Prefer connector conversation ids; constrain subject fallback; 30-day silence split |
| R7 | Temporal drift (new suppliers, systems) | Accuracy decays | Time-aware test split; drift monitoring; weekly retrain |
| R8 | Review capacity not resourced | Pipeline stalls at Phase 5 | ~30 h estimate agreed up front (§9.1) |
| R9 | Fraud-sensitive intents misrouted | Financial loss | Bank-detail rules always surface; precision target 0.97 |
| R10 | Taxonomy churn post-freeze | Labels invalidated | Versioned taxonomy + documented migration; `MISC` absorbs the unknown |
| R21 | **Train/serve distribution gap** (§2.2) | Model looks strong in eval, underperforms live | Gold set drawn from production-like mail (D3); production-mode metrics (§11.1B); Phase-2 augmentation (§10.6) |
| R22 | **Multi-intent emails silently lose an intent — and its action** | Missed PO/invoice/bank action | Margin detection + segmentation (§12.8); intent-set recall measured on gold; `multi_intent_log` as test set |
| R23 | Business curation skews optimistic | Inflated eval, weak live performance | Curation brief demands hard/boring/reply examples (§5.2); never evaluate on curated data |
| R24 | Cleaning code diverges between training and production | Silent accuracy loss, very hard to diagnose | `cleaning.py`/`formatting.py` shared verbatim (§13 contract) |
| R25 | Multi-intent rate much higher than assumed | §12.8 becomes critical-path; D1 may need revisiting | Measure in Phases 3–4 before committing production design (§12.8.3) |
| R11 | **Irreversible SAP action on a misclassification** | Financial loss, supplier impact | Risk-tiered dispatch; Tier 3 always human-approved; compensation workflow; entity validation before dispatch (C8) |
| R12 | Duplicate SAP posting from thread merge/split | Double payment or double PO | Business-object idempotency (level 2); merge/split reconciliation |
| R13 | Unknown-outcome SAP timeouts | Silent double-apply or silent no-op | Idempotency key on every attempt; reconciliation sweep; never blind-retry |
| R14 | Retry storms during SAP outage | Cascading load, alert fatigue | Backoff + jitter + circuit breaker + max attempts |
| R15 | Intent accumulation explosion on long threads | Action spam into SAP | Thread-level intent cap + alert (A10) |
| R16 | Concurrency race under `on_arrival` | Duplicate intents → duplicate actions | Per-thread serialisation (F1–F3) |
| R17 | Agent-authored mail re-ingested | Self-triggering loops; training contamination | `is_agent=1` exclusion at ingest, from both triggering and the training corpus |
| R18 | **Invalidation conflated with supersession** | Known-wrong labels trained on; model-quality metrics unreadable | Separate terminal states + separate metrics (§12.7.8) |
| R19 | Intent flapping across messages | Repeated SAP dispatch/compensation churn | Threshold hysteresis + correction budget (A14, A16) |
| R20 | Model upgrade retro-invalidating frozen decisions | Mass compensation cascade into SAP | Revision triggered by evidence only, never by model version (A15) |

---

## 16. Open items for sign-off

**Regime A — model building (blocking the build):**

1. **Intent list + seed sentences** — owner and delivery date for Phase 0.
2. **Curated single-intent examples** — ≥100 per intent; curation brief per §5.2 (hard/boring/reply examples, not just textbook ones).
3. **Annotator assignment** — two reviewers plus an adjudicator for the gold set.
4. **Shared inbox inventory** — names, credentials, and retention window available over IMAP.
5. **Per-intent `misroute_cost`** — needed to set precision targets (§11.4).
6. **Review capacity confirmation** — ~30 h of analyst time across Phases 4–5, plus curation effort.
7. **LLM endpoint + governance confirmation** — recorded as cleared (D4).

**Regime B — production (needed before the serving build, not before training):**

8. **Production platform owner** — recipient of the §12 design.
9. **Intent → action mapping + risk tier** per intent (§12.7.3).
10. **`intent_conflicts` matrix** — business-authored supersession rules (§12.7.2).
11. **SAP error-code taxonomy** — transient vs terminal vs data-missing (C6/C7).
12. **Business keys per action type** — invoice no., PO no., for level-2 idempotency (§12.7.4).
13. **Compensation procedures** — documented reversal path for each Tier 2/3 action (A5, C11).
14. **Approval routing for Tier 3** — who approves payment and bank-detail actions, and within what SLA.
15. **Activation / retraction thresholds per intent** — hysteresis band (§12.7.8); defaults 0.60 / 0.25.
16. **Correction budget** — automated revisions per thread before freezing to human (default 2).
17. **Retraction phrase list** — business-authored, incl. localisations (A13).

**Measured, not assumed — outputs of Phases 3–4 that gate the production design:**

18. **Multi-intent rate** in real traffic (§12.8.3) — determines whether segmentation is critical-path and whether D1 holds.
19. **IAA** (§8.3) — the accuracy ceiling; below 0.65 the taxonomy must be revised before training.
20. **Intent-set recall** (§11.1B) — how much multi-intent content the single-label design actually loses.
