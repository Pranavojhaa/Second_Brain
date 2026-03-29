# Second Brain AI

Chat with your Obsidian-style markdown notes using OpenAI, LangChain, and Chroma.

## What It Does

- Loads markdown notes from `vault/`
- Creates embeddings with OpenAI
- Stores vectors locally with Chroma
- Lets you ask questions in:
  - CLI
  - Streamlit

## Project Structure

```text
brain_chat/
  chat_with_vault.py    # CLI chat entry point
  core.py               # Shared ingest + retrieval logic
  ingest.py             # Embeds notes into Chroma
  main.py               # Small launcher
  second_brain_gui.py   # Streamlit UI
scripts/
  summarize_note.py     # Optional note summarizer
vault/                  # Your markdown notes
chroma_db/              # Local vector database
```

## Setup

1. Clone the repo
   ```bash
   git clone https://github.com/YOUR_USERNAME/second-brain-ai
   cd second-brain-ai
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your OpenAI key
   ```bash
   cp .env.example .env
   ```

4. Put your markdown notes inside `vault/`

5. Build or refresh the vector store
   ```bash
   python brain_chat/ingest.py
   ```

## Run

CLI:

```bash
python brain_chat/chat_with_vault.py
```

Streamlit:

```bash
python -m brain_chat.main streamlit
```

After loading the local instance, open Obsidian and connect the local [`vault/`](/Users/pranavojha/ai-second-brain/vault) folder as your vault so the app and your notes stay in sync.

## Troubleshooting

- `OPENAI_API_KEY is missing`
  Add the key to `.env`

- `No vectorstore found`
  Run `python brain_chat/ingest.py` first

- Network-related OpenAI or `tiktoken` errors
  Make sure the machine has internet access when embedding or querying

## Deploy on Railway

This repo includes a [`Procfile`](/Users/pranavojha/ai-second-brain/Procfile) and [`railway.json`](/Users/pranavojha/ai-second-brain/railway.json) so Railway can start the Streamlit app with the correct host and port settings.

Recommended Railway setup:

1. Create a new Railway project from this repo
2. Add `OPENAI_API_KEY` as an environment variable
3. Mount a persistent volume and point these variables at it:
   ```text
   VAULT_DIR=/data/vault
   CHROMA_DIR=/data/chroma_db
   TRACKER_PATH=/data/embedded_files.json
   ```
4. Upload or sync your markdown notes into `VAULT_DIR`
5. Run one ingest job with:
   ```bash
   python brain_chat/ingest.py
   ```
   Or set:
   ```text
   AUTO_INGEST_ON_STARTUP=true
   ```
   if you want the app to build the vector store automatically when it first boots without one

Important:

- Railway filesystems are ephemeral unless you use a mounted volume
- If you deploy without a persistent volume, your uploaded notes and vector store will disappear on restart
- The default web start command is:
  ```bash
  python -m brain_chat.main streamlit
  ```

## Tech Stack

- OpenAI
- LangChain
- langchain-openai
- langchain-chroma
- Streamlit
- Obsidian-style markdown notes

## License

MIT
