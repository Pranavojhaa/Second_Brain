import sys
import subprocess
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.chat_with_vault import start_chat
from brain_chat.core import ingest_notes


def start_streamlit() -> None:
    subprocess.run(["streamlit", "run", "brain_chat/second_brain_gui.py"], check=False)


if __name__ == "__main__":
    args = {arg.lower() for arg in sys.argv[1:]}

    if "ingest" in args:
        try:
            ingest_notes()
        except Exception as exc:
            print(exc)
            sys.exit(1)

    if "streamlit" in args:
        start_streamlit()
    else:
        start_chat()
