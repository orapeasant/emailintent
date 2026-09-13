"""One-time historical extraction — pulls N past emails into a JSONL dataset.

Drop this file in `~/.gyrfalcon/flows/` alongside `mailbox_flow.py` and run it
directly:

    uv run python ~/.gyrfalcon/flows/extract_historical_emails.py --limit 5000

This is deliberately *not* the same code path as `poll_mailbox`: that flow is
an incremental, UID-cursor poller meant to run forever every 5 minutes. This
is a backfill — it walks backwards from the newest message, fetches full
bodies (not just headers), cleans them, and writes one JSON object per line to
a dataset file that the training pipeline reads from.

Output format is JSONL (see module docstring rationale in the design chat):
one record per line, so it streams into `datasets.load_dataset("json", ...)`
or pandas without loading 5000 emails into memory at once, and a crash mid-run
loses at most the batch in flight rather than a whole in-memory list.

Resumable by construction: message_ids already present in the output file are
skipped, so re-running after a crash/rate-limit only fetches what's missing.
"""

from __future__ import annotations

import argparse
import email
import html
import imaplib
import json
import logging
import os
import re
import sys
from email.header import decode_header, make_header
from email.message import Message
from pathlib import Path
from typing import Any, Iterable, Optional

#: This script lives in the working project folder, not ~/.gyrfalcon/flows —
#: only read from that flows directory (for the existing MailboxConfig /
#: load_mailbox_config helpers), don't write anything there.
_FLOWS_DIR = Path.home() / ".gyrfalcon" / "flows"
sys.path.insert(0, str(_FLOWS_DIR))
try:
    from mailbox_flow import MailboxConfig, load_mailbox_config  # noqa: E402
except ModuleNotFoundError:
    # This script also gets imported from the project's own .venv (for
    # inference, alongside torch/transformers) which intentionally does not
    # have the `gyrfalcon` package installed — mailbox_flow.py needs it just
    # to load config, not for anything IMAP-specific. Read the same two files
    # `load_mailbox_config` reads (~/.gyrfalcon/config.yaml, ~/.gyrfalcon/.env)
    # directly instead, so this module works from either venv.
    from dataclasses import dataclass as _dataclass

    @_dataclass(frozen=True)
    class MailboxConfig:  # type: ignore[no-redef]
        host: str
        port: int
        user: str
        password: str
        use_ssl: bool = True

        @property
        def account_key(self) -> str:
            return self.user if "@" in self.user else f"{self.user}@{self.host}"

    def load_mailbox_config() -> Optional["MailboxConfig"]:  # type: ignore[no-redef]
        import yaml

        gyrfalcon_home = Path.home() / ".gyrfalcon"
        config_path = gyrfalcon_home / "config.yaml"
        env_path = gyrfalcon_home / ".env"

        config: dict = {}
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        email_cfg = config.get("email", {})

        password = os.environ.get("GYRFALCON_IMAP_PASSWORD")
        if not password and env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GYRFALCON_IMAP_PASSWORD="):
                    password = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

        host = email_cfg.get("imap_host")
        user = email_cfg.get("imap_user")
        if not (host and user and password):
            return None
        return MailboxConfig(
            host=host,
            port=int(email_cfg.get("imap_port", 993)),
            user=user,
            password=password,
            use_ssl=bool(email_cfg.get("imap_ssl", True)),
        )

logger = logging.getLogger("gyrfalcon.flow.email.extract_historical")

#: Gmail's "All Mail" holds every message across every label (Inbox, Sent,
#: Archive, etc.) — the broadest pool to sample an intent-classification
#: dataset from. IMAP folder names differ by Gmail locale; override with
#: --folder if this doesn't match your account.
DEFAULT_FOLDER = "[Gmail]/All Mail"

#: One IMAP round trip covers this many UIDs. Gmail will happily hand back
#: thousands at once, but a single huge FETCH response is harder to recover
#: from mid-stream if the connection drops — smaller batches trade a few
#: extra round trips for restart granularity.
BATCH_SIZE = 100

#: Below this many characters after cleaning, a body carries no classification
#: signal (image-only newsletters, all-quoted forwards) — drop rather than
#: write a blank row.
MIN_BODY_LENGTH = 20

_QUOTE_MARKERS = re.compile(
    r"^\s*(>.*|On .{0,80} wrote:|-{2,}\s*Original Message\s*-{2,}.*|"
    r"From: .*|Sent from my .*)\s*$",
    re.IGNORECASE,
)
_SIGNATURE_MARKER = re.compile(r"^\s*--\s*$")


# ── Body cleaning ────────────────────────────────────────────────────────────

def _html_to_text(raw_html: str) -> str:
    """Best-effort HTML->text without a parser dependency.

    Good enough for email bodies: strip script/style blocks, turn block tags
    into line breaks, drop the rest, unescape entities. Not a substitute for
    a real renderer, but training data doesn't need one.
    """
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    # Table cells are the biggest source of layout noise in marketing HTML
    # (invisible preheader tables, spacer cells): give each a boundary so
    # adjacent cell text doesn't run together, and end each row on its own line.
    text = re.sub(r"(?i)</t[dh]>", " ", text)
    text = re.sub(r"(?i)</(p|div|tr|li|h[1-6])>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", "", text)
    return html.unescape(text)


#: Marketing HTML frequently leaves behind runs of bare separator characters
#: (table borders, spacer glyphs) once tags are stripped — "|  |  |  |" with
#: nothing between them. Two or more in a row carry no signal.
_SEPARATOR_RUN = re.compile(r"(?:[|•·▪]\s*){2,}")
_BLANK_LINES = re.compile(r"\n{3,}")
_REPEATED_SPACES = re.compile(r"[ \t]{2,}")
#: Zero-width joiners/non-joiners, BOM, combining marks, and other invisible
#: Unicode formatting chars are used by marketing senders to break up hidden
#: preheader text so email clients can't collapse it — visually invisible,
#: but they show up as garbage once decoded to plain text.
_INVISIBLE_CHARS = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u034f\u00ad]|[\u0300-\u036f]"
)
#: Some newsletter templates prefix the body with a bare preheader-width
#: number ("96", "150 characters of hidden preview text...") on its own line.
_LEADING_NUMERIC_LINE = re.compile(r"^\s*\d{1,4}\s*\n")


def _strip_invisible_chars(text: str) -> str:
    text = _INVISIBLE_CHARS.sub("", text)
    return text.replace("\xa0", " ")


def _collapse_whitespace_artifacts(text: str) -> str:
    """Removes layout noise left over after HTML stripping.

    Runs after quote/signature stripping so it cleans exactly the content
    that will end up in the dataset, not text that gets discarded anyway.
    """
    text = _strip_invisible_chars(text)
    text = _LEADING_NUMERIC_LINE.sub("", text)
    text = _SEPARATOR_RUN.sub(" ", text)
    lines = [_REPEATED_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _strip_quotes_and_signature(text: str) -> str:
    """Keeps only the new content: drops quoted history and the signature.

    Truncates at the first quote marker or signature delimiter and stops
    there — everything after is a prior message or a boilerplate footer,
    neither of which reflects the sender's intent for *this* email.
    """
    kept: list[str] = []
    for line in text.splitlines():
        if _QUOTE_MARKERS.match(line) or _SIGNATURE_MARKER.match(line):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def extract_body_text(msg: Message) -> str:
    """Pulls the most useful text out of a (possibly multipart) message.

    Prefers text/plain; falls back to text/html converted to text. Skips
    attachments entirely — bodies only, per the same "keep payload out of
    events/datasets" principle as the header-only poller.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition.lower():
                continue
            content_type = part.get_content_type()
            try:
                payload = part.get_payload(decode=True)
            except Exception:  # noqa: BLE001 - malformed MIME part
                continue
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                decoded = payload.decode(charset, errors="replace")
            except (LookupError, ValueError):
                decoded = payload.decode("utf-8", errors="replace")
            if content_type == "text/plain":
                plain_parts.append(decoded)
            elif content_type == "text/html":
                html_parts.append(decoded)
    else:
        try:
            payload = msg.get_payload(decode=True)
        except Exception:  # noqa: BLE001
            payload = None
        if payload is not None:
            charset = msg.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                html_parts.append(decoded)
            else:
                plain_parts.append(decoded)

    raw_text = "\n".join(plain_parts) if plain_parts else _html_to_text("\n".join(html_parts))
    stripped = _strip_quotes_and_signature(raw_text)
    return _collapse_whitespace_artifacts(stripped)


def _decode_header_value(msg: Message, name: str) -> str:
    value = msg.get(name)
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:  # noqa: BLE001 - malformed header is not worth failing on
        return value


# ── Gmail label parsing ──────────────────────────────────────────────────────

_GM_LABELS_RE = re.compile(rb"X-GM-LABELS \(([^)]*)\)")


def _parse_gm_labels(fetch_response_line: bytes) -> list[str]:
    """Extracts Gmail's own labels from the FETCH response line.

    These are free weak-supervision signal (CATEGORY_PROMOTIONS, IMPORTANT,
    user labels like "Receipts") — worth carrying through so the labeling
    step can use them as a prior instead of starting from nothing.
    """
    match = _GM_LABELS_RE.search(fetch_response_line)
    if not match:
        return []
    raw = match.group(1).decode("utf-8", errors="replace")
    # Labels are space-separated, quoted if they contain spaces, e.g.
    # X-GM-LABELS (\Inbox "Some Label" IMPORTANT)
    return [tok.strip('"') for tok in re.findall(r'"[^"]*"|\S+', raw)]


# ── Extraction ───────────────────────────────────────────────────────────────

def _connect(cfg: MailboxConfig) -> imaplib.IMAP4:
    opener = imaplib.IMAP4_SSL if cfg.use_ssl else imaplib.IMAP4
    conn = opener(cfg.host, cfg.port)
    conn.login(cfg.user, cfg.password)
    return conn


def _load_seen_message_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen: set[str] = set()
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


def _record_from_message(uid: int, folder: str, gm_labels: list[str], raw: bytes) -> dict[str, Any]:
    msg = email.message_from_bytes(raw)
    message_id = _decode_header_value(msg, "Message-ID").strip("<>") or f"{folder}#{uid}"
    return {
        "message_id": message_id,
        "thread_id": _decode_header_value(msg, "Thread-Index") or None,
        "gmail_labels": gm_labels,
        "from": _decode_header_value(msg, "From"),
        "to": _decode_header_value(msg, "To"),
        "subject": _decode_header_value(msg, "Subject"),
        "date": _decode_header_value(msg, "Date"),
        "body_text": extract_body_text(msg),
        "has_attachments": any(
            "attachment" in str(part.get("Content-Disposition", "")).lower()
            for part in (msg.walk() if msg.is_multipart() else [])
        ),
        "label": None,
        "label_source": None,
    }


def _quote_mailbox(folder: str) -> str:
    """imaplib does not auto-quote mailbox names with spaces/brackets, and
    Gmail's special folders (e.g. "[Gmail]/All Mail") have both."""
    if folder.startswith('"') and folder.endswith('"'):
        return folder
    return f'"{folder}"'


def iter_uids_newest_first(conn: imaplib.IMAP4, folder: str, limit: int) -> list[int]:
    typ, data = conn.select(_quote_mailbox(folder), readonly=True)
    if typ != "OK":
        raise RuntimeError(f"Could not select folder {folder!r}: {data}")

    typ, data = conn.uid("SEARCH", None, "ALL")
    if typ != "OK":
        raise RuntimeError(f"UID SEARCH failed on {folder!r}: {typ}")

    uids = [int(u) for u in data[0].split()]
    # Newest first: a backfill wants recent, relevant mail before it wants
    # mail from a decade ago, and this also means an interrupted run has
    # already captured the most useful slice.
    return sorted(uids, reverse=True)[:limit]


def fetch_batch(conn: imaplib.IMAP4, folder: str, uids: list[int]) -> list[dict[str, Any]]:
    """Fetches full bodies + Gmail labels for one batch of UIDs."""
    if not uids:
        return []
    uid_set = ",".join(str(u) for u in uids)
    typ, data = conn.uid("FETCH", uid_set, "(X-GM-LABELS BODY.PEEK[])")
    if typ != "OK" or not data:
        raise RuntimeError(f"FETCH failed for batch in {folder!r}: {typ}")

    records = []
    # imaplib returns alternating (metadata, literal) tuples per message,
    # plus bare closing-paren bytes we can ignore.
    pending_uid_line: Optional[bytes] = None
    for item in data:
        if isinstance(item, tuple):
            meta, raw = item
            uid_match = re.search(rb"UID (\d+)", meta)
            uid = int(uid_match.group(1)) if uid_match else None
            gm_labels = _parse_gm_labels(meta)
            if uid is None:
                logger.warning("Could not parse UID from FETCH response in %s", folder)
                continue
            records.append(_record_from_message(uid, folder, gm_labels, raw))
    return records


def extract_historical(
    folder: str = DEFAULT_FOLDER,
    limit: int = 5000,
    output_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    cfg = load_mailbox_config()
    if cfg is None:
        raise RuntimeError(
            "Mailbox not configured (email.imap_host / GYRFALCON_IMAP_PASSWORD). "
            "Set those up the same way poll_mailbox needs them."
        )

    output_path = output_path or (Path(__file__).parent / "dataset" / "historical_extract.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_seen = _load_seen_message_ids(output_path)
    logger.info("Resuming: %d messages already extracted at %s", len(already_seen), output_path)

    conn = _connect(cfg)
    written = 0
    skipped = 0
    empty_bodies = 0
    errors = 0
    try:
        uids = iter_uids_newest_first(conn, folder, limit)
        logger.info("Found %d candidate UIDs in %r (capped at limit=%d)", len(uids), folder, limit)

        with output_path.open("a", encoding="utf-8") as out:
            for i in range(0, len(uids), batch_size):
                batch = uids[i : i + batch_size]
                try:
                    records = fetch_batch(conn, folder, batch)
                except Exception as e:  # noqa: BLE001 - one bad batch must not kill the run
                    logger.warning("Batch %d-%d failed (%s); skipping", i, i + len(batch), e)
                    errors += len(batch)
                    continue

                for record in records:
                    if record["message_id"] in already_seen:
                        skipped += 1
                        continue
                    # Image-only / all-quoted emails clean down to nothing —
                    # keeping them would put blank rows in the training set
                    # instead of a usable example.
                    if len(record["body_text"]) < MIN_BODY_LENGTH:
                        empty_bodies += 1
                        already_seen.add(record["message_id"])
                        continue
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    already_seen.add(record["message_id"])
                    written += 1
                out.flush()
                logger.info(
                    "Progress: %d/%d UIDs processed (%d written, %d skipped, %d empty, %d errors)",
                    min(i + batch_size, len(uids)), len(uids), written, skipped, empty_bodies, errors,
                )
    finally:
        try:
            conn.logout()
        except Exception:  # noqa: BLE001
            pass

    return {
        "status": "ok",
        "folder": folder,
        "output_path": str(output_path),
        "written": written,
        "skipped_existing": skipped,
        "skipped_empty_body": empty_bodies,
        "errors": errors,
        "total_in_dataset": len(already_seen),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="IMAP folder to pull from")
    parser.add_argument("--limit", type=int, default=5000, help="Max messages to extract")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Messages per FETCH round trip")
    args = parser.parse_args()

    result = extract_historical(
        folder=args.folder, limit=args.limit, output_path=args.output, batch_size=args.batch_size
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
