"""LLM first-pass labeling — assigns an intent from `intents.json` to every
email in the extracted dataset, producing the labeled data the BERT training
script consumes.

This is the "silver labels" step (see design discussion): the LLM's output is
treated as noisy-but-useful supervision, not ground truth. Each record keeps
its `confidence` and `rationale` so a human-review pass can target exactly the
predictions worth checking, instead of everything or nothing.

Concurrent by design: 4500 sequential LLM calls would take ~1-2 hours at
1-2s/call; a thread pool (LLM calls are I/O-bound) gets this down to minutes
without needing a batch API. Resumable: message_ids already present in the
output are skipped, so a rate-limit or crash loses at most the in-flight batch.

Usage:
    uv run python label_with_llm.py --limit 50          # dry run
    uv run python label_with_llm.py                       # full dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path.home() / ".gyrfalcon" / "flows"))  # read-only import
from gyrfalcon.agent.auxiliary_client import call_llm  # noqa: E402

logger = logging.getLogger("email.label_with_llm")

DEFAULT_INPUT = Path(__file__).parent / "dataset" / "historical_extract.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "dataset" / "labeled.jsonl"
DEFAULT_INTENTS = Path(__file__).parent / "intents.json"

_TASK_NAME = "email_intent_labeling"

#: Below this, treat the label as noisy enough to route to human review rather
#: than trust outright — training can still use it, but it's flagged.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

#: Per-email text budget in the prompt. Generous relative to the taxonomy
#: prompt's per-sample budget because here it's one email per call, not
#: hundreds packed into one context.
MAX_BODY_CHARS = 1500

_write_lock = threading.Lock()


def _load_taxonomy(intents_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    data = json.loads(intents_path.read_text(encoding="utf-8"))
    intents = data["intents"]
    return intents, {intent["id"] for intent in intents}


def _build_system_prompt(intents: list[dict[str, Any]]) -> str:
    lines = [
        "You are labeling one email at a time with a single best-fit intent from "
        "the closed taxonomy below. Output STRICT JSON only, no markdown fences, "
        "no commentary, matching this shape:\n"
        '{"label": "<intent_id>", "confidence": 0.0-1.0, "rationale": "one short sentence"}\n\n'
        "Taxonomy:",
    ]
    for intent in intents:
        lines.append(f"- {intent['id']}: {intent['label']} — {intent['description']}")
    lines.append(
        "\nRules:\n"
        "- label MUST be exactly one of the intent_id values above, verbatim.\n"
        "- confidence reflects how unambiguous the fit is, not how important the email is.\n"
        "- If nothing fits well, use the catch-all/other intent rather than forcing a bad fit."
    )
    return "\n".join(lines)


def _extract_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return json.loads(text)


def _load_seen_message_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["message_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def label_one(
    record: dict[str, Any], system_prompt: str, valid_ids: set[str], model: Optional[str]
) -> dict[str, Any]:
    """Labels a single email. Never raises — failures degrade to a flagged
    'other' label so one bad response can't stall the whole batch."""
    subject = (record.get("subject") or "").strip()
    body = (record.get("body_text") or "").strip()[:MAX_BODY_CHARS]
    gmail_labels = record.get("gmail_labels") or []

    user_prompt = f"Subject: {subject}\nGmail labels: {gmail_labels}\nBody:\n{body}"

    try:
        response = call_llm(
            _TASK_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content or ""
        parsed = _extract_json(raw)
        label = parsed.get("label")
        confidence = float(parsed.get("confidence", 0.0))
        rationale = parsed.get("rationale", "")

        if label not in valid_ids:
            logger.warning(
                "Model returned unknown label %r for %s; falling back to 'other'",
                label, record.get("message_id"),
            )
            label = "other" if "other" in valid_ids else next(iter(valid_ids))
            confidence = 0.0
            rationale = f"[fallback: invalid label {parsed.get('label')!r}] {rationale}"
    except Exception as e:  # noqa: BLE001 - one failed call must not kill the run
        logger.warning("Labeling failed for %s: %s", record.get("message_id"), e)
        label = "other" if "other" in valid_ids else next(iter(valid_ids))
        confidence = 0.0
        rationale = f"[fallback: LLM call error: {e}]"

    return {
        **record,
        "label": label,
        "label_source": "llm",
        "label_confidence": confidence,
        "label_rationale": rationale,
    }


def run_labeling(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    intents_path: Path = DEFAULT_INTENTS,
    limit: Optional[int] = None,
    workers: int = 8,
    model: Optional[str] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    intents, valid_ids = _load_taxonomy(intents_path)
    system_prompt = _build_system_prompt(intents)

    already_seen = _load_seen_message_ids(output_path)
    logger.info("Resuming: %d already labeled at %s", len(already_seen), output_path)

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["message_id"] in already_seen:
                continue
            records.append(record)
    if limit:
        records = records[:limit]
    logger.info("Labeling %d emails with %d workers", len(records), workers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts: dict[str, int] = {}
    low_confidence_count = 0
    done = 0

    with output_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(label_one, record, system_prompt, valid_ids, model): record
            for record in records
        }
        for future in as_completed(futures):
            labeled = future.result()
            label_counts[labeled["label"]] = label_counts.get(labeled["label"], 0) + 1
            if labeled["label_confidence"] < confidence_threshold:
                low_confidence_count += 1

            with _write_lock:
                out.write(json.dumps(labeled, ensure_ascii=False) + "\n")
                out.flush()

            done += 1
            if done % 50 == 0 or done == len(records):
                logger.info("Progress: %d/%d labeled (%d low-confidence)", done, len(records), low_confidence_count)

    return {
        "status": "ok",
        "output_path": str(output_path),
        "newly_labeled": len(records),
        "total_labeled": len(already_seen) + len(records),
        "label_distribution": label_counts,
        "low_confidence_count": low_confidence_count,
        "confidence_threshold": confidence_threshold,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--intents", type=Path, default=DEFAULT_INTENTS)
    parser.add_argument("--limit", type=int, default=None, help="Only label the first N unlabeled emails")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=None, help="Override the model used for labeling")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    result = run_labeling(
        input_path=args.input,
        output_path=args.output,
        intents_path=args.intents,
        limit=args.limit,
        workers=args.workers,
        model=args.model,
        confidence_threshold=args.confidence_threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
