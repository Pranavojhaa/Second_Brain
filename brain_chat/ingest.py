from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.core import AI_MODE, VAULT_DIR, ingest_notes


if __name__ == "__main__":
    try:
        stats = ingest_notes()
    except Exception as exc:
        print(exc)
        sys.exit(1)
    print(f"Ingested {VAULT_DIR} (AI mode: {AI_MODE}): " + ", ".join(f"{k}={v}" for k, v in stats.items()))
