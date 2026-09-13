"""Random real-mail inference test — pulls one email from before a cutoff
date straight from the mailbox and runs the fine-tuned ModernBERT classifier
on it, so you can eyeball real predictions rather than trusting metrics alone.

Reuses `extract_historical_emails.py`'s IMAP connection/fetch/body-cleaning
code (same folder, imported not copied) so the inference sees exactly the
same preprocessing the training data went through.

Usage (run with the project venv, which has torch/transformers installed):
    .venv/bin/python test_random_inference.py
    .venv/bin/python test_random_inference.py --before 01-Jan-2023 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from extract_historical_emails import (  # noqa: E402
    DEFAULT_FOLDER,
    _connect,
    _quote_mailbox,
    fetch_batch,
    load_mailbox_config,
)

DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "modernbert-intent-v1" / "model"
#: IMAP's BEFORE search key wants DD-Mon-YYYY; "01-Jan-2024" means everything
#: with an internal date strictly before Jan 1 2024.
DEFAULT_BEFORE = "01-Jan-2024"


def _build_text(subject: str, body: str) -> str:
    """Must match the training-time input format exactly (see
    `train_modernbert.py:_build_text`) — a model is only as good as the
    consistency between train-time and inference-time preprocessing."""
    return f"Subject: {subject}\n\n{body}"


def find_matching_uids(
    conn, folder: str, before_date: str | None, keyword: str | None, field: str
) -> list[int]:
    """Builds an IMAP SEARCH with whichever criteria were given.

    IMAP ANDs multiple search keys together automatically when passed in the
    same command, so BEFORE and SUBJECT/TEXT/FROM can be combined in one round
    trip rather than fetching everything and filtering client-side.
    """
    typ, data = conn.select(_quote_mailbox(folder), readonly=True)
    if typ != "OK":
        raise RuntimeError(f"Could not select folder {folder!r}: {data}")

    criteria: list[str] = []
    if before_date:
        criteria += ["BEFORE", before_date]
    if keyword:
        field_key = {"subject": "SUBJECT", "text": "TEXT", "from": "FROM"}[field]
        # Quote the keyword so multi-word phrases survive as one search term
        # rather than being read as several separate (ANDed) search keys.
        criteria += [field_key, f'"{keyword}"']
    if not criteria:
        criteria = ["ALL"]

    typ, data = conn.uid("SEARCH", None, *criteria)
    if typ != "OK":
        raise RuntimeError(f"UID SEARCH {criteria} failed on {folder!r}: {typ}")

    return [int(u) for u in data[0].split()]


def pick_random_email(
    folder: str, before_date: str | None, keyword: str | None, field: str, seed: int | None
) -> dict[str, Any]:
    cfg = load_mailbox_config()
    if cfg is None:
        raise RuntimeError("Mailbox not configured (email.imap_host / GYRFALCON_IMAP_PASSWORD).")

    conn = _connect(cfg)
    try:
        uids = find_matching_uids(conn, folder, before_date, keyword, field)
        if not uids:
            criteria_desc = f"before {before_date}" if before_date else ""
            if keyword:
                criteria_desc += f" {field}~{keyword!r}"
            raise RuntimeError(f"No messages found in {folder!r} matching:{criteria_desc or ' ALL'}")

        rng = random.Random(seed)
        uid = rng.choice(uids)

        records = fetch_batch(conn, folder, [uid])
        if not records:
            raise RuntimeError(f"Could not fetch UID {uid} in {folder!r}")
        return records[0]
    finally:
        try:
            conn.logout()
        except Exception:  # noqa: BLE001
            pass


def load_classifier(model_dir: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def classify(text: str, tokenizer, model, device, top_k: int = 3) -> list[tuple[str, float]]:
    import torch

    inputs = tokenizer(text, truncation=True, max_length=500, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    return [(model.config.id2label[i.item()], p.item()) for p, i in zip(top.values, top.indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="IMAP folder to sample from")
    parser.add_argument(
        "--before", default=DEFAULT_BEFORE,
        help="IMAP BEFORE date, e.g. 01-Jan-2024 (use --no-before to disable date filtering)",
    )
    parser.add_argument("--no-before", action="store_true", help="Don't filter by date at all")
    parser.add_argument(
        "--keyword", default=None,
        help="Only sample emails matching this keyword (see --field for where it's searched)",
    )
    parser.add_argument(
        "--field", choices=["subject", "text", "from"], default="text",
        help="Where --keyword is searched: subject line, full text (subject+body), or sender",
    )
    parser.add_argument("--seed", type=int, default=None, help="Fix the random pick for reproducibility")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--top-k", type=int, default=3, help="How many candidate intents to show")
    args = parser.parse_args()

    before_date = None if args.no_before else args.before

    criteria_desc = []
    if before_date:
        criteria_desc.append(f"before {before_date}")
    if args.keyword:
        criteria_desc.append(f"{args.field}~{args.keyword!r}")
    print(f"Sampling one email from {args.folder!r} ({', '.join(criteria_desc) or 'no filter'})...")

    record = pick_random_email(args.folder, before_date, args.keyword, args.field, args.seed)

    text = _build_text(record["subject"], record["body_text"])

    print("\n" + "=" * 70)
    print(f"Subject : {record['subject']}")
    print(f"From    : {record['from']}")
    print(f"Date    : {record['date']}")
    print(f"Gmail labels: {record['gmail_labels']}")
    print("-" * 70)
    print("Body:")
    print(record["body_text"][:2000] + ("..." if len(record["body_text"]) > 2000 else ""))
    print("=" * 70)

    print(f"\nLoading classifier from {args.model_dir}...")
    tokenizer, model, device = load_classifier(args.model_dir)
    predictions = classify(text, tokenizer, model, device, top_k=args.top_k)

    print(f"\nPredicted intent (device={device}):")
    for label, prob in predictions:
        marker = "-> " if (label, prob) == predictions[0] else "   "
        print(f"{marker}{label:40s} {prob:.1%}")


if __name__ == "__main__":
    main()
