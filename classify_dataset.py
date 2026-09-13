"""Batch-classifies an extracted email dataset with the fine-tuned ModernBERT
model and reports the resulting intent distribution.

Distinct from `label_with_llm.py` (which produces the LLM silver labels used
for *training*): this runs the *trained model itself* over the dataset, so
you can see what the model actually predicts at scale, compare it against the
LLM labels it was trained on, and spot systematic disagreements.

Usage:
    .venv/bin/python classify_dataset.py
    .venv/bin/python classify_dataset.py --data dataset/labeled.jsonl --compare-to-llm-labels
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_DATA = Path(__file__).parent / "dataset" / "historical_extract.jsonl"
DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "modernbert-intent-v1" / "model"
DEFAULT_OUTPUT = Path(__file__).parent / "dataset" / "model_predictions.jsonl"


def _build_text(subject: str, body: str) -> str:
    """Must match training-time input format exactly (see
    `train_modernbert.py:_build_text`)."""
    return f"Subject: {subject}\n\n{body}"


def load_records(data_path: Path) -> list[dict[str, Any]]:
    records = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_all(
    records: list[dict[str, Any]], model_dir: Path, batch_size: int = 32, max_length: int = 500
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Loaded model on device={device}")

    results = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        texts = [_build_text(r.get("subject") or "", r.get("body_text") or "") for r in batch]
        inputs = tokenizer(
            texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        confidences, pred_ids = torch.max(probs, dim=-1)

        for record, pred_id, confidence in zip(batch, pred_ids.tolist(), confidences.tolist()):
            results.append(
                {
                    "message_id": record["message_id"],
                    "subject": record.get("subject"),
                    "predicted_label": model.config.id2label[pred_id],
                    "predicted_confidence": confidence,
                    "llm_label": record.get("label"),
                }
            )

        done = min(i + batch_size, len(records))
        if done % (batch_size * 10) == 0 or done == len(records):
            print(f"Progress: {done}/{len(records)} classified")

    return results


def summarize(results: list[dict[str, Any]], compare_to_llm: bool) -> dict[str, Any]:
    pred_counts = Counter(r["predicted_label"] for r in results)
    summary: dict[str, Any] = {
        "total": len(results),
        "predicted_distribution": dict(pred_counts.most_common()),
    }

    if compare_to_llm:
        with_llm_label = [r for r in results if r.get("llm_label")]
        agree = sum(1 for r in with_llm_label if r["predicted_label"] == r["llm_label"])
        summary["agreement_with_llm_labels"] = {
            "compared": len(with_llm_label),
            "agree": agree,
            "agreement_rate": agree / len(with_llm_label) if with_llm_label else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--compare-to-llm-labels", action="store_true",
        help="Report agreement rate against the 'label' field already in the data (the LLM silver labels)",
    )
    args = parser.parse_args()

    records = load_records(args.data)
    print(f"Classifying {len(records)} emails from {args.data}")

    results = classify_all(records, args.model_dir, batch_size=args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(results, compare_to_llm=args.compare_to_llm_labels)
    print("\n=== Intent distribution (model predictions) ===")
    for label, count in summary["predicted_distribution"].items():
        pct = 100 * count / summary["total"]
        print(f"{label:40s} {count:5d}  ({pct:.1f}%)")
    print(f"\nTotal: {summary['total']}")

    if args.compare_to_llm_labels and summary.get("agreement_with_llm_labels"):
        a = summary["agreement_with_llm_labels"]
        print(f"\nAgreement with LLM labels: {a['agree']}/{a['compared']} ({a['agreement_rate']:.1%})")

    print(f"\nPer-email predictions written to {args.output}")


if __name__ == "__main__":
    main()
