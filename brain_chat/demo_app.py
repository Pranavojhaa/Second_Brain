"""Public demo entry point: the normal UI over sample_vault/, offline, no API key, no personal notes."""
import runpy
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain_chat import core

# Pin the demo regardless of any .env: offline retrieval over the tracked sample notes only.
core.AI_MODE = "off"
core.VAULT_DIR = ROOT / "sample_vault"
core.CHROMA_DIR = Path(tempfile.gettempdir()) / "second_brain_demo"
st.cache_resource(core.ingest_notes, show_spinner=False)()

runpy.run_path(str(ROOT / "brain_chat" / "second_brain_gui.py"), run_name="__main__")

st.sidebar.markdown(
    "**Public demo.** Answers come from three sample notes in offline mode: keyword retrieval, "
    "no LLM. The full app runs locally against your own Obsidian vault. "
    "[Source and setup](https://github.com/Pranavojhaa/Second_Brain)"
)
st.sidebar.markdown("Try: *how do I update data with fastapi?* or *what is deep work?*")
