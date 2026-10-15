from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.core import AI_MODE, VAULT_DIR


def summarize_note(note_name: str) -> None:
    if AI_MODE != "openai":
        raise RuntimeError("AI is off. Set SECOND_BRAIN_AI=openai in .env to send notes to OpenAI.")
    from openai import OpenAI

    file_path = VAULT_DIR / note_name
    if not file_path.exists():
        raise FileNotFoundError(f"Note not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
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
    # Write a temp file next to the note, then swap it in: a crash mid-write never truncates the note.
    fd, tmp_path = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(new_content)
        os.chmod(tmp_path, file_path.stat().st_mode)
        os.replace(tmp_path, file_path)
    except BaseException:
        os.unlink(tmp_path)
        raise
    print(f"Summary added to {file_path.name}")


if __name__ == "__main__":
    target_note = sys.argv[1] if len(sys.argv) > 1 else "test.md"
    try:
        summarize_note(target_note)
    except Exception as exc:
        print(exc)
