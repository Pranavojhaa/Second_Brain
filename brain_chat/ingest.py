from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.core import ingest_notes


if __name__ == "__main__":
    try:
        did_ingest = ingest_notes()
    except Exception as exc:
        print(exc)
    else:
        if not did_ingest:
            print("Vectorstore is already up to date.")
