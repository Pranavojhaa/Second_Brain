from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
VAULT_DIR = Path(os.getenv("VAULT_DIR", str(BASE_DIR / "vault"))).expanduser()
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))).expanduser()
TRACKER_PATH = Path(
    os.getenv("TRACKER_PATH", str(BASE_DIR / "brain_chat" / "embedded_files.json"))
).expanduser()
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def _read_embedded_files() -> set[str]:
    if not TRACKER_PATH.exists():
        return set()

    try:
        return set(json.loads(TRACKER_PATH.read_text()))
    except json.JSONDecodeError:
        return set()


def _write_embedded_files(file_paths: set[str]) -> None:
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(sorted(file_paths), indent=2))


def has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def create_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL)


def load_and_chunk_notes() -> list:
    already_embedded = _read_embedded_files()
    if not VAULT_DIR.exists():
        print(f"Vault directory does not exist yet: {VAULT_DIR}")
        return []

    loader = DirectoryLoader(
        str(VAULT_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        use_multithreading=True,
        show_progress=True,
    )

    all_docs = loader.load()
    new_docs = [doc for doc in all_docs if doc.metadata["source"] not in already_embedded]
    if not new_docs:
        print("No new notes found to embed.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = splitter.split_documents(new_docs)
    print(f"Found {len(new_docs)} new notes -> {len(split_docs)} chunks.")
    return split_docs


def embed_and_store(docs: list) -> bool:
    if not docs:
        return False

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=create_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.persist()

    embedded_files = {doc.metadata["source"] for doc in docs}
    _write_embedded_files(_read_embedded_files().union(embedded_files))
    print("Chunks embedded and saved to ChromaDB.")
    return True


def ingest_notes() -> bool:
    if not has_openai_api_key():
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env before ingesting notes.")

    docs = load_and_chunk_notes()
    return embed_and_store(docs)


def has_vectorstore() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


def load_vectorstore() -> Chroma:
    if not has_vectorstore():
        raise FileNotFoundError(
            f"No vectorstore found at {CHROMA_DIR}. Run `python brain_chat/ingest.py` first."
        )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=create_embeddings(),
    )


def build_qa_chain() -> ConversationalRetrievalChain:
    return ConversationalRetrievalChain.from_llm(
        llm=create_chat_model(),
        retriever=load_vectorstore().as_retriever(),
        return_source_documents=True,
    )
