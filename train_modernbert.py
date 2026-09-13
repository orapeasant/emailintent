"""Fine-tunes ModernBERT for single-label email intent classification.

Reads labeled records from a JSONL dataset (subject/body_text/label) and the
flexible taxonomy from `intents.json` (see `generate_intent_taxonomy.py` —
the label set is regenerated from the LLM, not hardcoded here), and produces
a fine-tuned ModernBERT-base checkpoint plus an evaluation report.

This script only trains — it does not label. Records with `"label": null`
(everything is null until the labeling step runs) are dropped before
training, so this is safe to run early to validate the pipeline on whatever
subset already has labels, and re-run as more labels accumulate.

Requires (installed in `.venv` alongside this file, not the gyrfalcon venv —
training deps are heavy and unrelated to the agent runtime):
    torch, transformers>=4.48 (ModernBERT support), datasets, scikit-learn, accelerate

Usage:
    .venv/bin/python train_modernbert.py \
        --data dataset/historical_extract.jsonl \
        --intents intents.json \
        --output-dir models/modernbert-intent-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("email.train_modernbert")

BASE_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_MAX_SEQ_LENGTH = 500

#: Below this many labeled examples for a class, the train/val/test stratified
#: split can't guarantee at least one example per split — surface it as a
#: warning rather than letting sklearn silently fail or crash mid-run.
MIN_EXAMPLES_PER_CLASS = 3


@dataclass
class SplitData:
    texts: list[str]
    labels: list[int]


def _build_text(record: dict[str, Any]) -> str:
    """Concatenates subject + body into one input string.

    Subject carries a disproportionate amount of intent signal for email
    (often the whole intent is legible from the subject alone) — keeping it
    prefixed and clearly separated lets the model use it without the body
    diluting/truncating it out of the window.
    """
    subject = (record.get("subject") or "").strip()
    body = (record.get("body_text") or "").strip()
    return f"Subject: {subject}\n\n{body}"


def load_labeled_dataset(data_path: Path, intents_path: Path) -> tuple[list[str], list[int], dict[int, str]]:
    intents = json.loads(intents_path.read_text(encoding="utf-8"))["intents"]
    label2id = {intent["id"]: i for i, intent in enumerate(intents)}
    id2label = {i: intent["id"] for i, intent in enumerate(intents)}

    texts: list[str] = []
    labels: list[int] = []
    dropped_unlabeled = 0
    dropped_unknown_label = 0

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record.get("label")
            if not label:
                dropped_unlabeled += 1
                continue
            if label not in label2id:
                logger.warning("Unknown label %r not in intents.json; dropping", label)
                dropped_unknown_label += 1
                continue
            texts.append(_build_text(record))
            labels.append(label2id[label])

    logger.info(
        "Loaded %d labeled examples (%d dropped: unlabeled, %d dropped: unknown label)",
        len(texts), dropped_unlabeled, dropped_unknown_label,
    )
    if not texts:
        raise RuntimeError(
            "No labeled examples found. Run the labeling step first — this script "
            "trains on records with a non-null 'label', it does not assign labels."
        )
    return texts, labels, id2label


def stratified_split(
    texts: list[str], labels: list[int], seed: int = 42
) -> tuple[SplitData, SplitData, SplitData]:
    """80/10/10 train/val/test, stratified so every split sees every class.

    sklearn's train_test_split(stratify=...) does this in one call for a
    binary split; two calls (train+rest, then val/test out of rest) gets the
    three-way split without a third-party splitter dependency.
    """
    from collections import Counter
    from sklearn.model_selection import train_test_split

    counts = Counter(labels)
    rare = {label: count for label, count in counts.items() if count < MIN_EXAMPLES_PER_CLASS}
    if rare:
        logger.warning(
            "Classes with <%d examples (stratified split may be unreliable for these): %s",
            MIN_EXAMPLES_PER_CLASS, rare,
        )

    train_texts, rest_texts, train_labels, rest_labels = train_test_split(
        texts, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        rest_texts, rest_labels, test_size=0.50, random_state=seed, stratify=rest_labels
    )
    return (
        SplitData(train_texts, train_labels),
        SplitData(val_texts, val_labels),
        SplitData(test_texts, test_labels),
    )


def build_datasets(
    train: SplitData, val: SplitData, test: SplitData, tokenizer: Any, max_seq_length: int
):
    from datasets import Dataset, DatasetDict

    def _tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length)

    raw = DatasetDict(
        {
            "train": Dataset.from_dict({"text": train.texts, "label": train.labels}),
            "validation": Dataset.from_dict({"text": val.texts, "label": val.labels}),
            "test": Dataset.from_dict({"text": test.texts, "label": test.labels}),
        }
    )
    return raw.map(_tokenize, batched=True, remove_columns=["text"])


def compute_metrics_fn(eval_pred) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        # Macro, not micro/weighted: an intent classifier is judged on how well
        # it does across *all* categories, not dominated by whichever class has
        # the most examples (procurement mail is expected to be imbalanced).
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def train(
    data_path: Path,
    intents_path: Path,
    output_dir: Path,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = 42,
    base_model: str = BASE_MODEL,
) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    random.seed(seed)
    torch.manual_seed(seed)

    texts, labels, id2label = load_labeled_dataset(data_path, intents_path)
    label2id = {v: k for k, v in id2label.items()}
    train_split, val_split, test_split = stratified_split(texts, labels, seed=seed)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", len(train_split.texts), len(val_split.texts), len(test_split.texts)
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    dataset = build_datasets(train_split, val_split, test_split, tokenizer, max_seq_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    use_cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s%s", use_cuda, f" ({torch.cuda.get_device_name(0)})" if use_cuda else "")

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        # bf16 over fp16 on newer GPUs (Ampere+/Ada like the 4060) avoids the
        # loss-scaling instability fp16 sometimes hits on transformer attention.
        bf16=use_cuda,
        logging_steps=25,
        report_to="none",
        seed=seed,
    )

    from transformers import DataCollatorWithPadding

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    val_metrics = trainer.evaluate(dataset["validation"])
    test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")

    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    report = {
        "base_model": base_model,
        "num_labels": len(id2label),
        "id2label": id2label,
        "train_size": len(train_split.texts),
        "val_size": len(val_split.texts),
        "test_size": len(test_split.texts),
        "max_seq_length": max_seq_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved model + report to %s", output_dir)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "dataset" / "labeled.jsonl")
    parser.add_argument("--intents", type=Path, default=Path(__file__).parent / "intents.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "models" / "modernbert-intent-v1")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    report = train(
        data_path=args.data,
        intents_path=args.intents,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        base_model=args.base_model,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "id2label"}, indent=2))


if __name__ == "__main__":
    main()
