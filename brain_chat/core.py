from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
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
MEMORY_TOP_K = 7
PROACTIVE_RECALL_DISTANCE = 0.45
CONTRADICTION_DISTANCE = 0.65
MAX_CHAT_HISTORY_TURNS = 4


@dataclass
class MemoryItem:
    text: str
    embedding: list[float]
    timestamp: str
    type: str
    score: float


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


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _infer_memory_type(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("goal", "plan to", "want to", "i need to", "i will")):
        return "goal"
    if any(keyword in lowered for keyword in ("idea", "explore", "maybe", "what if", "experiment")):
        return "idea"
    return "thought"


def _normalize_text(text: str, limit: int = 320) -> str:
    cleaned = " ".join(text.split())
    return cleaned[: limit - 3] + "..." if len(cleaned) > limit else cleaned


def _document_timestamp(source: str) -> str:
    try:
        return datetime.fromtimestamp(Path(source).stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return _utc_now_iso()


def _memory_metadata(text: str, source: str | None = None, timestamp: str | None = None) -> dict[str, Any]:
    return {
        "text": text,
        "timestamp": timestamp or _document_timestamp(source or ""),
        "type": _infer_memory_type(text),
        "source": source or "conversation",
    }


def _format_chat_history(chat_history: list[tuple[str, str]]) -> str:
    recent_turns = chat_history[-MAX_CHAT_HISTORY_TURNS:]
    if not recent_turns:
        return "No recent conversation history."

    lines = []
    for user_message, assistant_message in recent_turns:
        lines.append(f"User: {_normalize_text(user_message, 240)}")
        lines.append(f"Collaborator: {_normalize_text(assistant_message, 240)}")
    return "\n".join(lines)


def _format_memories(memories: list[MemoryItem]) -> str:
    if not memories:
        return "- No strongly relevant past thoughts were found."

    lines = []
    for memory in memories:
        lines.append(
            f"- [{memory.type} | {memory.timestamp}] {_normalize_text(memory.text)}"
        )
    return "\n".join(lines)


def _is_intent_mismatch(current_input: str, memory_text: str) -> bool:
    current = current_input.lower()
    memory = memory_text.lower()
    polarity_pairs = [
        ("should", "should not"),
        ("want", "don't want"),
        ("build", "avoid"),
        ("focus on", "stop"),
        ("prioritize", "deprioritize"),
        ("increase", "reduce"),
    ]

    has_shared_subject = bool(set(current.split()) & set(memory.split()))
    if not has_shared_subject:
        return False

    return any(
        (positive in current and negative in memory) or (negative in current and positive in memory)
        for positive, negative in polarity_pairs
    )


def _build_system_prompt() -> str:
    return (
        "You are a persistent AI thinking partner, not a generic assistant. "
        "Work like a long-term collaborator who remembers the user's evolving ideas.\n\n"
        "Always do the following:\n"
        "1. Connect the current input to relevant past thoughts when possible.\n"
        "2. Highlight contradictions, tension, or changed priorities when you notice them.\n"
        "3. Expand and refine ideas instead of giving generic summaries.\n"
        "4. Ask a focused follow-up question when the user is vague or underspecified.\n"
        "5. Avoid boilerplate assistant phrasing.\n"
        "6. Use the retrieved memories as working context before you answer.\n"
        "7. Sound like a thoughtful collaborator helping the user think better over time."
    )


def has_openai_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def create_chat_model() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL)


def load_and_chunk_notes() -> list[Document]:
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
    for doc in split_docs:
        doc.metadata.update(_memory_metadata(doc.page_content, source=doc.metadata.get("source")))

    print(f"Found {len(new_docs)} new notes -> {len(split_docs)} chunks.")
    return split_docs


def embed_and_store(docs: list[Document]) -> bool:
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

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=create_embeddings(),
    )
    _backfill_memory_schema(vectorstore)
    return vectorstore


def _backfill_memory_schema(vectorstore: Chroma) -> None:
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        return

    snapshot = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = snapshot.get("ids", [])
    documents = snapshot.get("documents", [])
    metadatas = snapshot.get("metadatas", [])
    embeddings = snapshot.get("embeddings", [])

    pending_ids: list[str] = []
    pending_documents: list[str] = []
    pending_metadatas: list[dict[str, Any]] = []
    pending_embeddings: list[list[float]] = []

    embedding_client = create_embeddings()
    for index, doc_id in enumerate(ids):
        document_text = documents[index] if index < len(documents) else ""
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        embedding = embeddings[index] if index < len(embeddings) else None

        changed = False
        if "text" not in metadata:
            metadata["text"] = document_text
            changed = True
        if "timestamp" not in metadata:
            metadata["timestamp"] = _utc_now_iso()
            changed = True
        if "type" not in metadata:
            metadata["type"] = _infer_memory_type(document_text)
            changed = True

        if embedding is None:
            embedding = embedding_client.embed_query(document_text)
            changed = True

        if changed:
            pending_ids.append(doc_id)
            pending_documents.append(document_text)
            pending_metadatas.append(metadata)
            pending_embeddings.append(embedding)

    if pending_ids:
        collection.update(
            ids=pending_ids,
            documents=pending_documents,
            metadatas=pending_metadatas,
            embeddings=pending_embeddings,
        )
        vectorstore.persist()


def _memory_from_document(
    document: Document, score: float, embedding: list[float], question: str
) -> MemoryItem | None:
    text = document.metadata.get("text") or document.page_content
    normalized_text = _normalize_text(text)
    if normalized_text.lower() == _normalize_text(question).lower():
        return None

    return MemoryItem(
        text=text,
        embedding=embedding,
        timestamp=document.metadata.get("timestamp", _utc_now_iso()),
        type=document.metadata.get("type", _infer_memory_type(text)),
        score=score,
    )


def _retrieve_memories(
    vectorstore: Chroma, embeddings: OpenAIEmbeddings, question: str, top_k: int = MEMORY_TOP_K
) -> list[MemoryItem]:
    query_embedding = embeddings.embed_query(question)
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        results = vectorstore.similarity_search_with_score(question, k=top_k)
        memories: list[MemoryItem] = []
        for document, score in results:
            memory = _memory_from_document(document, score, query_embedding, question)
            if memory is not None:
                memories.append(memory)
        return memories

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    memory_embeddings = results.get("embeddings", [[]])[0]

    memories: list[MemoryItem] = []
    for index, document_text in enumerate(documents):
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        document = Document(page_content=document_text, metadata=metadata)
        score = distances[index] if index < len(distances) else 1.0
        memory_embedding = memory_embeddings[index] if index < len(memory_embeddings) else query_embedding
        memory = _memory_from_document(document, score, memory_embedding, question)
        if memory is not None:
            memories.append(memory)

    return memories


def _proactive_recall(memories: list[MemoryItem]) -> str | None:
    if not memories:
        return None

    strongest = min(memories, key=lambda memory: memory.score)
    if strongest.score > PROACTIVE_RECALL_DISTANCE:
        return None

    return (
        "This connects to something you mentioned earlier: "
        f"{_normalize_text(strongest.text, 220)}"
    )


def _contradiction_note(question: str, memories: list[MemoryItem]) -> str | None:
    for memory in memories:
        if memory.score <= CONTRADICTION_DISTANCE and _is_intent_mismatch(question, memory.text):
            return (
                "Potential contradiction to address: "
                f"You previously said '{_normalize_text(memory.text, 180)}', which may point in a different direction."
            )
    return None


def _build_user_prompt(question: str, memories: list[MemoryItem], chat_history: list[tuple[str, str]]) -> str:
    contradiction_note = _contradiction_note(question, memories)
    sections = [
        "Relevant past thoughts:",
        _format_memories(memories),
        "",
        "Recent conversation:",
        _format_chat_history(chat_history),
        "",
        "Current input:",
        question,
        "",
        "Instruction:",
        "Think like a long-term collaborator. Connect ideas over time, surface tensions, and help refine the thinking.",
    ]
    if contradiction_note:
        sections.extend(["", contradiction_note])

    return "\n".join(sections)


def _store_memory(vectorstore: Chroma, text: str, memory_type: str, source: str) -> None:
    document = Document(
        page_content=text,
        metadata={
            "text": text,
            "timestamp": _utc_now_iso(),
            "type": memory_type,
            "source": source,
        },
    )
    vectorstore.add_documents([document], ids=[str(uuid4())])
    vectorstore.persist()


class ThinkingPartnerChain:
    def __init__(self, llm: ChatOpenAI, vectorstore: Chroma, embeddings: OpenAIEmbeddings):
        self.llm = llm
        self.vectorstore = vectorstore
        self.embeddings = embeddings

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        memories = _retrieve_memories(self.vectorstore, self.embeddings, question)
        prompt = _build_user_prompt(question, memories, chat_history)

        response = self.llm.invoke(
            [
                SystemMessage(content=_build_system_prompt()),
                HumanMessage(content=prompt),
            ]
        )
        base_answer = response.content if isinstance(response.content, str) else str(response.content)
        answer = base_answer

        proactive_recall = _proactive_recall(memories)
        if proactive_recall and proactive_recall not in answer:
            answer = f"{answer}\n\n{proactive_recall}"

        _store_memory(self.vectorstore, question, _infer_memory_type(question), "conversation:user")
        _store_memory(self.vectorstore, base_answer, "thought", "conversation:assistant")

        source_documents = [
            Document(
                page_content=memory.text,
                metadata={
                    "timestamp": memory.timestamp,
                    "type": memory.type,
                    "score": memory.score,
                },
            )
            for memory in memories
        ]
        return {"answer": answer, "source_documents": source_documents}


def build_qa_chain() -> ThinkingPartnerChain:
    embeddings = create_embeddings()
    return ThinkingPartnerChain(
        llm=create_chat_model(),
        vectorstore=load_vectorstore(),
        embeddings=embeddings,
    )
