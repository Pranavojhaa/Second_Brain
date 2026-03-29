import sys
import os
import subprocess
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.chat_with_vault import start_chat
from brain_chat.core import has_vectorstore, ingest_notes


def start_streamlit() -> None:
    port = os.getenv("PORT", "8501")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "brain_chat/second_brain_gui.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            port,
            "--server.headless",
            "true",
        ],
        check=False,
    )


def maybe_auto_ingest() -> None:
    if os.getenv("AUTO_INGEST_ON_STARTUP", "").lower() != "true":
        return

    if has_vectorstore():
        return

    try:
        print("AUTO_INGEST_ON_STARTUP is enabled. Building vector store before launch...")
        ingest_notes()
    except Exception as exc:
        print(f"Automatic ingest skipped: {exc}")


if __name__ == "__main__":
    args = {arg.lower() for arg in sys.argv[1:]}

    if "ingest" in args:
        try:
            ingest_notes()
        except Exception as exc:
            print(exc)
            sys.exit(1)

    if "streamlit" in args:
        maybe_auto_ingest()
        start_streamlit()
    else:
        start_chat()
