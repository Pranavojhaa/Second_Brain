from __future__ import annotations

import hashlib
import math
import os
import re
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
# Your own notes live in vault/ (gitignored); a clean clone falls back to the sample vault.
_DEFAULT_VAULT = BASE_DIR / "vault" if (BASE_DIR / "vault").is_dir() else BASE_DIR / "sample_vault"
VAULT_DIR = Path(os.getenv("VAULT_DIR", str(_DEFAULT_VAULT))).expanduser()
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))).expanduser()
# Connectors are opt-in: "off" (default) never touches the network, "openai" enables OpenAI.
AI_MODE = os.getenv("SECOND_BRAIN_AI", "off").strip().lower()
AI_MODES = ("off", "openai")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
MEMORY_TOP_K = 7
PROACTIVE_RECALL_DISTANCE = 0.45
CONTRADICTION_DISTANCE = 0.65
MAX_CHAT_HISTORY_TURNS = 4

# Every stored chunk is tagged with "kind" so notes and chat memory never mix.
NOTE = "note"
CONVERSATION = "conversation"
REQUIRED_TAGS = ("source", "kind", "type", "timestamp")

POLARITY_PAIRS = [
    ("should", "should not"),
    ("want", "not want"),
    ("prioritize", "deprioritize"),
]
# Frozen: HashEmbeddings skips these words, so any change shifts the offline vectors of stored chunks.
_STOPWORDS = frozenset(
    "a about all also am an and any are as at be been but by can could did do does don't for from "
    "had has have he her him his how i i'm if in into is it its just me more my no not of on or our "
    "she so some than that the their them then there these they this to too up us was we were what "
    "when which who why will with would you your "
    "avoid build deprioritize focus increase prioritize reduce should stop want".split()
)
_MARKER = re.compile(r"\[(\d+)\]")


@dataclass
class MemoryItem:
    text: str
    timestamp: str
    type: str
    score: float
    source: str


def _content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9']+", text.lower()) if word not in _STOPWORDS}


class HashEmbeddings(Embeddings):
    """Offline embedding: hashed bag of words. Matches shared words, not meaning."""

    # ponytail: lexical overlap only; set SECOND_BRAIN_AI=openai for semantic retrieval.
    size = 256

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for word in _content_words(text):
            vector[zlib.crc32(word.encode()) % self.size] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _format_chat_history(chat_history: list[tuple[str, str]]) -> str:
    recent_turns = chat_history[-MAX_CHAT_HISTORY_TURNS:]
    if not recent_turns:
        return "No recent conversation history."

    lines = []
    for user_message, assistant_message in recent_turns:
        lines.append(f"User: {_normalize_text(_MARKER.sub('', user_message), 240)}")
        lines.append(f"Collaborator: {_normalize_text(_MARKER.sub('', assistant_message), 240)}")
    return "\n".join(lines)


def _format_notes(notes: list[MemoryItem]) -> str:
    if not notes:
        return "- No relevant notes were found."
    return "\n".join(
        f"[{index}] {note.source}: {_normalize_text(_MARKER.sub('', note.text), 600)}"
        for index, note in enumerate(notes, start=1)
    )


def _format_memories(memories: list[MemoryItem]) -> str:
    if not memories:
        return "- No relevant earlier conversation was found."
    return "\n".join(
        f"- [{memory.type} | {memory.timestamp}] {_normalize_text(_MARKER.sub('', memory.text))}"
        for memory in memories
    )


def _polarity(text: str, positive: str, negative: str) -> str | None:
    # Check the negative first: "should not" also contains "should".
    if re.search(rf"\b{re.escape(negative)}\b", text):
        return "negative"
    return "positive" if re.search(rf"\b{re.escape(positive)}\b", text) else None


def _is_intent_mismatch(current_input: str, memory_text: str) -> bool:
    # ponytail: keyword heuristic, not NLI; swap in an LLM check if false flags still matter.
    # ponytail: whole message, not per sentence; "I should X. I should not Y." can misfire on long turns.
    current, memory = (
        " ".join(
            text.lower()
            .replace("\u2019", "'")
            .replace("can't", "can not")
            .replace("won't", "will not")
            .replace("n't", " not")
            .split()
        )
        for text in (current_input, memory_text)
    )
    if len(_content_words(current) & _content_words(memory)) < 2:
        return False

    return any(
        {_polarity(current, positive, negative), _polarity(memory, positive, negative)}
        == {"positive", "negative"}
        for positive, negative in POLARITY_PAIRS
    )


def _build_system_prompt() -> str:
    return (
        "You are a persistent AI thinking partner, not a generic assistant. "
        "Work like a long-term collaborator who remembers the user's evolving ideas.\n\n"
        "Always do the following:\n"
        "1. Cite the source note for every claim about the user's notes, using its [n] marker.\n"
        "2. Never cite conversation memory as a source; it is context, not evidence.\n"
        "3. If the source notes do not cover the question, say so instead of guessing.\n"
        "4. Highlight contradictions, tension, or changed priorities when you notice them.\n"
        "5. Expand and refine ideas instead of giving generic summaries.\n"
        "6. Ask a focused follow-up question when the user is vague or underspecified.\n"
        "7. Avoid boilerplate assistant phrasing."
    )


def _check_ai_mode() -> None:
    if AI_MODE not in AI_MODES:
        raise RuntimeError(f"SECOND_BRAIN_AI must be one of {AI_MODES}, got '{AI_MODE}'.")
    if AI_MODE == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("SECOND_BRAIN_AI=openai but OPENAI_API_KEY is missing. Add it to your .env.")


def create_embeddings() -> Embeddings:
    _check_ai_mode()
    if AI_MODE == "off":
        return HashEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def create_chat_model() -> Any | None:
    _check_ai_mode()
    if AI_MODE == "off":
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=CHAT_MODEL)


def load_vectorstore() -> Chroma:
    from chromadb.config import Settings

    # One collection per mode: the two embedding spaces are not comparable.
    return Chroma(
        collection_name=f"second_brain_{AI_MODE}",
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DIR),
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
    )


def chunk_note(text: str, source: str, content_hash: str, timestamp: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    return [
        Document(
            page_content=chunk,
            metadata={
                "source": source,
                "kind": NOTE,
                "type": _infer_memory_type(chunk),
                "timestamp": timestamp,
                "content_hash": content_hash,
            },
        )
        for chunk in splitter.split_text(text)
    ]


def ingest_notes(vectorstore: Chroma | None = None, vault_dir: Path | None = None) -> dict[str, int]:
    """Sync the vault into the vectorstore: add new notes, replace edited ones, drop deleted ones."""
    vectorstore = vectorstore or load_vectorstore()
    vault_dir = Path(vault_dir or VAULT_DIR)
    if not vault_dir.is_dir():
        raise FileNotFoundError(f"Vault folder not found: {vault_dir}")

    # ponytail: reads every note chunk's metadata per run; fine for a personal vault.
    stored = vectorstore.get(where={"kind": NOTE}, include=["metadatas"])
    stored_hashes: dict[str, set[str]] = {}
    stored_ids: dict[str, list[str]] = {}
    for chunk_id, metadata in zip(stored["ids"], stored["metadatas"]):
        stored_hashes.setdefault(metadata["source"], set()).add(metadata.get("content_hash", ""))
        stored_ids.setdefault(metadata["source"], []).append(chunk_id)

    notes = {
        path.relative_to(vault_dir).as_posix(): path
        for path in sorted(vault_dir.rglob("*.md"))
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(vault_dir).parts)
    }
    if not notes and stored_ids:
        raise RuntimeError(
            f"No .md notes found in {vault_dir}; refusing to remove {len(stored_ids)} indexed notes. "
            f"Check VAULT_DIR. To clear the index on purpose, delete {CHROMA_DIR} "
            "(this also erases conversation memory)."
        )
    stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

    for source, chunk_ids in stored_ids.items():
        if source not in notes:
            vectorstore.delete(ids=chunk_ids)
            stats["removed"] += 1

    for source, path in notes.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Mixed hashes mean an earlier run stopped between add and delete: redo it.
        if stored_hashes.get(source) == {content_hash}:
            stats["unchanged"] += 1
            continue

        timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        chunks = chunk_note(text, source, content_hash, timestamp)
        if chunks:
            vectorstore.add_documents(chunks, ids=[str(uuid4()) for _ in chunks])
        # Add first, delete second: a failed embedding call never loses the old chunks,
        # and a crash before the delete is healed by the mixed-hash check on the next run.
        if source in stored_ids:
            vectorstore.delete(ids=stored_ids[source])
            stats["updated"] += 1
        elif chunks:
            stats["added"] += 1

    return stats


def _retrieve(vectorstore: Chroma, question: str, kind: str, top_k: int = MEMORY_TOP_K) -> list[MemoryItem]:
    results = vectorstore.similarity_search_with_score(question, k=top_k, filter={"kind": kind})
    normalized_question = _normalize_text(question).lower()
    return [
        MemoryItem(
            text=document.page_content,
            timestamp=document.metadata.get("timestamp", _utc_now_iso()),
            type=document.metadata.get("type", "thought"),
            score=score,
            source=document.metadata.get("source", kind),
        )
        for document, score in results
        if _normalize_text(document.page_content).lower() != normalized_question
    ]


def _proactive_recall(memories: list[MemoryItem]) -> str | None:
    # Only the user's own turns: assistant text must never come back as "you mentioned".
    memories = [memory for memory in memories if memory.source == "conversation:user"]
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
        if (
            memory.source == "conversation:user"
            and memory.score <= CONTRADICTION_DISTANCE
            and _is_intent_mismatch(question, memory.text)
        ):
            return (
                "Possible tension (keyword heuristic, unverified): "
                f"an earlier message of yours says '{_normalize_text(memory.text, 180)}'. "
                "Raise it as a question only if it is real; do not assert a contradiction."
            )
    return None


def _build_user_prompt(
    question: str,
    notes: list[MemoryItem],
    memories: list[MemoryItem],
    chat_history: list[tuple[str, str]],
) -> str:
    contradiction_note = _contradiction_note(question, memories)
    sections = [
        "Source notes (cite these as [n]):",
        _format_notes(notes),
        "",
        "Conversation memory (earlier chat turns; context only, not a citable source):",
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


def _offline_answer(question: str, notes: list[MemoryItem]) -> str:
    header = "AI is off (offline mode), so this is retrieval only, not a generated answer."
    if not notes:
        return f"{header}\nNo notes share a keyword with your question."
    excerpts = "\n".join(
        f"[{index}] {note.source}: {_normalize_text(_MARKER.sub('', note.text))}"
        for index, note in enumerate(notes, start=1)
    )
    return f"{header}\nClosest note excerpts:\n{excerpts}"


def _store_memory(vectorstore: Chroma, text: str, memory_type: str, source: str) -> None:
    document = Document(
        page_content=text,
        metadata={
            "source": source,
            "kind": CONVERSATION,
            "type": memory_type,
            "timestamp": _utc_now_iso(),
        },
    )
    vectorstore.add_documents([document], ids=[str(uuid4())])


class ThinkingPartnerChain:
    def __init__(self, llm: Any | None, vectorstore: Chroma):
        self.llm = llm
        self.vectorstore = vectorstore

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        notes = _retrieve(self.vectorstore, question, NOTE)
        memories = _retrieve(self.vectorstore, question, CONVERSATION)

        if self.llm is None:
            # Hash embeddings always return top-k, so keep only notes that share a real word.
            question_words = _content_words(question)
            notes = [note for note in notes if question_words & _content_words(note.text)]
            answer = _offline_answer(question, notes)
            cited = {int(n) for n in _MARKER.findall(answer)}
        else:
            response = self.llm.invoke(
                [
                    SystemMessage(content=_build_system_prompt()),
                    HumanMessage(content=_build_user_prompt(question, notes, memories, chat_history)),
                ]
            )
            base_answer = response.content if isinstance(response.content, str) else str(response.content)
            answer = base_answer
            cited = {int(n) for n in _MARKER.findall(base_answer)}
            for marker in sorted(cited - set(range(1, len(notes) + 1))):
                answer = (
                    f"{answer}\n\n(Marker [{marker}] does not match any retrieved note; "
                    "treat that claim as unsourced.)"
                )
            if notes and not cited & set(range(1, len(notes) + 1)):
                answer = f"{answer}\n\n(No note was cited in this answer, so treat it as unsourced.)"

            proactive_recall = _proactive_recall(memories)
            if proactive_recall and proactive_recall not in answer:
                answer = f"{answer}\n\n{proactive_recall}"
            _store_memory(self.vectorstore, base_answer, "thought", "conversation:assistant")

        _store_memory(self.vectorstore, question, _infer_memory_type(question), "conversation:user")

        return {
            "answer": answer,
            "sources": list(dict.fromkeys(note.source for note in notes)),
            # Numbered exactly as the notes were shown to the model (or in the offline answer).
            "citations": [
                {"n": n, "source": note.source, "excerpt": note.text, "cited": n in cited}
                for n, note in enumerate(notes, start=1)
            ],
            "source_documents": [
                Document(
                    page_content=note.text,
                    metadata={
                        "source": note.source,
                        "timestamp": note.timestamp,
                        "type": note.type,
                        "score": note.score,
                    },
                )
                for note in notes
            ],
        }


def cited_lines(result: dict[str, Any]) -> list[str]:
    """One display line per note the answer cites. A pointer only: the claim itself is not checked."""
    return [
        f"[{c['n']}] {c['source']} — {' '.join(c['excerpt'].split())[:160]}"
        for c in result["citations"]
        if c["cited"]
    ]


def build_qa_chain() -> ThinkingPartnerChain:
    vectorstore = load_vectorstore()
    if not vectorstore.get(where={"kind": NOTE}, limit=1)["ids"]:
        raise FileNotFoundError("No notes ingested yet. Run `python brain_chat/ingest.py` first.")
    return ThinkingPartnerChain(llm=create_chat_model(), vectorstore=vectorstore)
