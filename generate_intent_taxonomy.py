"""Generates (or regenerates) the email-intent taxonomy from a sample of the
extracted dataset, using an LLM instead of a hand-maintained label list.

Why this exists: the taxonomy is a *moving target* right now — the current
`historical_extract.jsonl` is personal mail, but the eventual dataset is
procurement mail, so any list of intents hardcoded today would be wrong
tomorrow. Instead of editing labels by hand as the mailbox composition shifts,
this script re-derives them from whatever is actually in the dataset plus a
steering hint (e.g. "procurement"), and writes them to `intents.json` — the
single file every other step (labeling, training, eval) reads the label set
from. Re-run this any time the mail mix changes; nothing downstream needs to
change code, only re-read the file.

Usage:
    uv run python generate_intent_taxonomy.py \
        --input dataset/historical_extract.jsonl \
        --domain-hint "procurement (RFQs, POs, invoices, vendor comms)" \
        --num-intents 12
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.home() / ".gyrfalcon" / "flows"))  # read-only import, see extract_historical_emails.py

from gyrfalcon.agent.auxiliary_client import call_llm  # noqa: E402

logger = logging.getLogger("gyrfalcon.flow.email.generate_intent_taxonomy")

DEFAULT_OUTPUT = Path(__file__).parent / "intents.json"
#: Per-email text budget in the sample prompt — keeps a few hundred emails'
#: worth of signal inside a reasonable token count without truncating so hard
#: that intent becomes unrecognizable.
MAX_CHARS_PER_EMAIL = 300

_TASK_NAME = "email_intent_taxonomy"

_SYSTEM_PROMPT = """You are designing a label taxonomy for an email intent classifier.
You will be shown a sample of real emails (subject + truncated body) and a domain hint
describing what the eventual production mailbox will mostly contain. The sample may not
yet match that domain — the dataset is a work in progress — so use the domain hint to
anticipate categories that are not well represented (or absent) in the sample.

Output STRICT JSON only, no markdown fences, no commentary, matching this shape:
{
  "intents": [
    {
      "id": "snake_case_id",
      "label": "Human Readable Label",
      "description": "One or two sentences a labeler can apply consistently.",
      "example_subjects": ["short example subject line", "..."]
    }
  ]
}

Rules:
- 8-15 intents: a workable classifier granularity, not so many that classes overlap.
- Mutually exclusive as much as possible; each email should map to exactly one best-fit
  intent (single-label classification), so avoid near-duplicate categories.
- Always include a catch-all/"other" intent for anything that doesn't fit — a closed
  taxonomy without one forces bad forced-fits during labeling.
- Bias the taxonomy toward the domain hint even where the sample under-represents it;
  the sample is a snapshot, the domain hint is the target.
- Marketing/promotional/newsletter mail is NOT one bucket: split it by the underlying
  product or service category being promoted (e.g. "travel_rewards_promo",
  "saas_product_newsletter", "retail_offer", "financial_services_promo"), the same way
  you would split transactional mail by purpose. A single generic "marketing" or
  "promotional" intent throws away exactly the signal (what is this email actually
  about) that the classifier needs.
"""


def _load_sample(path: Path, sample_size: int, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        raise RuntimeError(f"No records found in {path}")

    random.Random(seed).shuffle(records)
    return records[:sample_size]


def _format_sample_for_prompt(records: list[dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(records, 1):
        subject = (r.get("subject") or "").strip()
        body = (r.get("body_text") or "").strip().replace("\n", " ")
        body = body[:MAX_CHARS_PER_EMAIL]
        gmail_labels = r.get("gmail_labels") or []
        lines.append(f"{i}. Subject: {subject}\n   Labels: {gmail_labels}\n   Body: {body}")
    return "\n".join(lines)


def _extract_json(raw_text: str) -> dict[str, Any]:
    """Strips markdown code fences if the model added them anyway, then parses."""
    text = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return json.loads(text)


def generate_taxonomy(
    input_path: Path,
    output_path: Path,
    domain_hint: str,
    sample_size: int = 150,
    num_intents_hint: int | None = None,
    seed: int = 42,
    model: str | None = None,
) -> dict[str, Any]:
    sample = _load_sample(input_path, sample_size, seed)
    sample_text = _format_sample_for_prompt(sample)

    user_prompt = (
        f"Domain hint (target production mailbox): {domain_hint}\n"
        + (f"Target roughly {num_intents_hint} intents.\n" if num_intents_hint else "")
        + f"\nSample of {len(sample)} emails from the current (in-progress) dataset:\n\n{sample_text}"
    )

    response = call_llm(
        _TASK_NAME,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.2,
        max_tokens=4000,
    )
    raw_text = response.choices[0].message.content or ""

    try:
        parsed = _extract_json(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON taxonomy: {e}\nRaw output:\n{raw_text}") from e

    intents = parsed.get("intents")
    if not isinstance(intents, list) or not intents:
        raise RuntimeError(f"Model output missing a non-empty 'intents' list: {parsed}")

    prior_version = 0
    if output_path.exists():
        try:
            prior_version = json.loads(output_path.read_text(encoding="utf-8")).get("version", 0)
        except (json.JSONDecodeError, OSError):
            pass

    taxonomy = {
        "version": prior_version + 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domain_hint": domain_hint,
        "source_file": str(input_path),
        "sample_size": len(sample),
        "intents": intents,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False), encoding="utf-8")
    return taxonomy


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "dataset" / "historical_extract.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--domain-hint",
        default="general personal/consumer email: transactional receipts and account notices, "
        "security/account alerts, newsletters and marketing (split by product/service type), "
        "job alerts and professional networking, event/webinar invitations, "
        "personal correspondence, and local/community content — no specific business domain "
        "should be assumed; let the intents reflect whatever is actually in the sample.",
    )
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--num-intents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None, help="Override the model used for taxonomy generation")
    args = parser.parse_args()

    taxonomy = generate_taxonomy(
        input_path=args.input,
        output_path=args.output,
        domain_hint=args.domain_hint,
        sample_size=args.sample_size,
        num_intents_hint=args.num_intents,
        seed=args.seed,
        model=args.model,
    )
    print(f"Wrote taxonomy v{taxonomy['version']} with {len(taxonomy['intents'])} intents to {args.output}")
    for intent in taxonomy["intents"]:
        print(f"  - {intent['id']}: {intent['label']}")


if __name__ == "__main__":
    main()
