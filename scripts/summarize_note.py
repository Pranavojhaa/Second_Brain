from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
VAULT_DIR = BASE_DIR / "vault"


def summarize_note(note_name: str) -> None:
    file_path = VAULT_DIR / note_name
    if not file_path.exists():
        raise FileNotFoundError(f"Note not found: {file_path}")

    content = file_path.read_text()
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes notes."},
            {"role": "user", "content": f"Summarize this note:\n\n{content}"},
        ],
    )

    summary = response.choices[0].message.content or ""
    new_content = f"## AI Summary\n{summary}\n\n---\n\n{content}"
    file_path.write_text(new_content)
    print(f"Summary added to {file_path.name}")


if __name__ == "__main__":
    target_note = sys.argv[1] if len(sys.argv) > 1 else "test.md"
    try:
        summarize_note(target_note)
    except Exception as exc:
        print(exc)
