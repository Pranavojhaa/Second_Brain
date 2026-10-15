"""Offline tests: fake embeddings, stub chat model, temp Chroma. No network, no API key."""
import re
import socket
import sys
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat import core


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every non-AF_UNIX connect fails (loopback TCP included), so a test can never reach the network."""
    for name in ("connect", "connect_ex"):
        real = getattr(socket.socket, name)

        def guarded(self, address, _real=real):
            if self.family != socket.AF_UNIX:
                raise RuntimeError("network disabled in tests")
            return _real(self, address)

        monkeypatch.setattr(socket.socket, name, guarded)


class StubChatModel:
    def __init__(self, reply="FastAPI uses decorators for routes [1]."):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return AIMessage(content=self.reply)


def make_store(tmp_path, embedding=None):
    return Chroma(
        collection_name="test",
        embedding_function=embedding or DeterministicFakeEmbedding(size=32),
        persist_directory=str(tmp_path / "chroma"),
    )


@pytest.fixture
def vault(tmp_path):
    folder = tmp_path / "vault"
    folder.mkdir()
    (folder / "fastapi.md").write_text("FastAPI uses decorators such as app.get for routes.")
    (folder / "deep work.md").write_text("Deep work means focusing without distraction.")
    return folder


def note_chunks(store):
    return store.get(where={"kind": core.NOTE})


def test_chunking_splits_and_tags():
    chunks = core.chunk_note("word " * 400, "long.md", "abc", "2026-01-01T00:00:00+00:00")
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 500 for chunk in chunks)
    for chunk in chunks:
        assert set(core.REQUIRED_TAGS) <= set(chunk.metadata)
        assert chunk.metadata["kind"] == core.NOTE and chunk.metadata["source"] == "long.md"


def test_ingest_is_incremental(tmp_path, vault):
    store = make_store(tmp_path)
    assert core.ingest_notes(store, vault) == {"added": 2, "updated": 0, "removed": 0, "unchanged": 0}
    assert core.ingest_notes(store, vault) == {"added": 0, "updated": 0, "removed": 0, "unchanged": 2}
    assert len(note_chunks(store)["ids"]) == 2


def test_edited_note_replaces_chunks(tmp_path, vault):
    """Regression pin: an edited multi-chunk note leaves only the new text under one hash."""
    (vault / "fastapi.md").write_text("FastAPI routes. " * 60)
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    assert len(store.get(where={"source": "fastapi.md"})["ids"]) > 1
    (vault / "fastapi.md").write_text("FastAPI now documented with app.post instead.")

    assert core.ingest_notes(store, vault)["updated"] == 1
    fastapi = store.get(where={"source": "fastapi.md"})
    assert fastapi["documents"] == ["FastAPI now documented with app.post instead."]
    assert len({m["content_hash"] for m in fastapi["metadatas"]}) == 1
    assert len(note_chunks(store)["ids"]) == 2


def test_interrupted_ingest_heals_on_rerun(tmp_path, vault, monkeypatch):
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    new_text = "FastAPI now documented with app.post instead."
    (vault / "fastapi.md").write_text(new_text)

    real_delete = store.delete

    def crash_once(*args, **kwargs):
        monkeypatch.setattr(store, "delete", real_delete)
        raise RuntimeError("simulated crash between add and delete")

    monkeypatch.setattr(store, "delete", crash_once)
    with pytest.raises(RuntimeError):
        core.ingest_notes(store, vault)

    core.ingest_notes(store, vault)
    fastapi = store.get(where={"source": "fastapi.md"})
    assert fastapi["documents"] == [new_text]
    assert len({m["content_hash"] for m in fastapi["metadatas"]}) == 1
    assert core.ingest_notes(store, vault)["unchanged"] == 2


def test_empty_vault_scan_refuses_to_wipe_store(tmp_path, vault):
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RuntimeError):
        core.ingest_notes(store, empty)
    assert len(note_chunks(store)["ids"]) == 2


def test_directory_named_like_note_is_skipped(tmp_path, vault):
    (vault / "x.md").mkdir()
    store = make_store(tmp_path)
    assert core.ingest_notes(store, vault) == {"added": 2, "updated": 0, "removed": 0, "unchanged": 0}


def test_deleted_note_removes_chunks(tmp_path, vault):
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    (vault / "fastapi.md").unlink()

    assert core.ingest_notes(store, vault)["removed"] == 1
    assert {m["source"] for m in note_chunks(store)["metadatas"]} == {"deep work.md"}


def test_answer_lists_sources_and_writes_memory(tmp_path, vault):
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    llm = StubChatModel()

    result = core.ThinkingPartnerChain(llm, store).invoke({"question": "How do routes work?"})

    assert sorted(result["sources"]) == ["deep work.md", "fastapi.md"]
    assert "[1] " in llm.prompts[0]
    memory = store.get(where={"kind": core.CONVERSATION})
    assert sorted(m["source"] for m in memory["metadatas"]) == ["conversation:assistant", "conversation:user"]
    assert all(set(core.REQUIRED_TAGS) <= set(m) for m in memory["metadatas"])


def test_repeat_question_does_not_surface_answer_as_note(tmp_path, vault):
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    llm = StubChatModel(reply="UNIQUE-EARLIER-ANSWER [1]")
    chain = core.ThinkingPartnerChain(llm, store)

    chain.invoke({"question": "How do routes work?"})
    second = chain.invoke({"question": "How do routes work?"})

    assert sorted(second["sources"]) == ["deep work.md", "fastapi.md"]
    assert all("UNIQUE-EARLIER-ANSWER" not in doc.page_content for doc in second["source_documents"])
    notes_section, memory_section = llm.prompts[1].split("Conversation memory")
    assert "UNIQUE-EARLIER-ANSWER" not in notes_section
    assert "UNIQUE-EARLIER-ANSWER" in memory_section


def test_uncited_answer_is_flagged(tmp_path, vault):
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    result = core.ThinkingPartnerChain(StubChatModel(reply="Trust me."), store).invoke({"question": "Routes?"})
    assert "unsourced" in result["answer"]


def test_contradiction_ignores_stopword_overlap():
    # Only "I", "should", "the", "to" are shared: no common subject, so no flag.
    assert not core._is_intent_mismatch(
        "I should go to the gym tomorrow", "I should not email the landlord"
    )
    # Same stance on the same subject is not a contradiction either.
    assert not core._is_intent_mismatch("I should not buy a car", "We should not buy that car")


def test_contradiction_flags_opposite_stance_on_same_subject():
    assert core._is_intent_mismatch("I should learn Rust", "I should not learn Rust this year")


def test_assistant_memory_is_never_attributed_to_user():
    ts = "2026-01-01T00:00:00+00:00"
    a = core.MemoryItem("I should not learn Rust this year", ts, "thought", 0.0, "conversation:assistant")
    u = core.MemoryItem(a.text, ts, "thought", 0.0, "conversation:user")

    assert core._proactive_recall([a]) is None
    assert core._build_user_prompt("I should learn Rust", [], [a], []).count(a.text) == 1
    # Positive control: the same text from the user is recalled and raised as a possible tension.
    assert core._build_user_prompt("I should learn Rust", [], [u], []).count(a.text) == 2


def test_citation_numbers_map_to_sources(tmp_path, vault):
    (vault / "long.md").write_text(" ".join(f"routing sentence number {i}." for i in range(60)))
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    llm = StubChatModel(reply="See [2].")

    result = core.ThinkingPartnerChain(llm, store).invoke({"question": "How does routing work?"})

    citations = result["citations"]
    notes_section = llm.prompts[0].split("Conversation memory")[0]
    lines = re.findall(r"^\[(\d+)\] (.+?): (.*)$", notes_section, flags=re.M)
    assert len(lines) == len(citations) > len(result["sources"])
    for n, source, text in lines:
        c = citations[int(n) - 1]
        assert c["n"] == int(n) and c["source"] == source
        assert core._normalize_text(c["excerpt"], 600) == text
    assert [c["cited"] for c in citations] == [c["n"] == 2 for c in citations]


def test_bracket_numbers_inside_notes_are_not_markers(tmp_path, vault):
    (vault / "fastapi.md").write_text("FastAPI routes, see [FastAPI docs][7] and section [2].")
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)

    class EchoModel(StubChatModel):
        """Quotes the fastapi.md line from the prompt, as a model citing it verbatim would."""

        def invoke(self, messages):
            self.prompts.append(messages[-1].content)
            n, text = re.search(r"^\[(\d+)\] fastapi\.md: (.*)$", self.prompts[-1], flags=re.M).groups()
            return AIMessage(content=f"Per [{n}]: {text}")

    result = core.ThinkingPartnerChain(EchoModel(), store).invoke({"question": "FastAPI routes?"})
    fastapi_n = next(c["n"] for c in result["citations"] if c["source"] == "fastapi.md")
    assert [c["n"] for c in result["citations"] if c["cited"]] == [fastapi_n]
    assert "does not match" not in result["answer"]


def test_out_of_range_citation_is_flagged(tmp_path, vault):
    store = make_store(tmp_path)
    core.ingest_notes(store, vault)
    result = core.ThinkingPartnerChain(StubChatModel(reply="Yes [9]."), store).invoke({"question": "Routes?"})
    assert "[9] does not match" in result["answer"]


@pytest.mark.parametrize(
    "current, earlier",
    [
        # Every row shares >= 2 content words, so the polarity rules (not the overlap check) decide.
        ("Focus on the novel draft", "Nonstop novel draft work wears me out"),
        ("Build the parser rewrite", "The parser rewrite is unavoidable"),
        ("Increase my savings rate", "Reduce my savings rate"),
        ("My shoulder needs the physio plan", "I should not skip the physio plan"),
        ("I can't wait, I want the car", "I can't say I don't want the car"),  # no "ca" leftover
        ("I should learn Rust", "I should not use Rust at work"),  # one shared word: overlap check
        ("I don\u2019t want a new car", "I don't want a new car"),
        ("I should not buy a car", "We should not buy that car"),  # regression pin
    ],
)
def test_contradiction_negatives(current, earlier):
    assert not core._is_intent_mismatch(current, earlier)


@pytest.mark.parametrize(
    "current, earlier",
    [
        ("I want to learn Rust", "I don\u2019t want to learn Rust"),
        ("I should learn Rust", "I should not learn Rust this year"),  # regression pin
        ("Prioritize the parser rewrite", "Deprioritize the parser rewrite"),  # regression pin
    ],
)
def test_contradiction_positives(current, earlier):
    assert core._is_intent_mismatch(current, earlier)


def test_network_is_blocked():
    """Regression pin for the autouse _no_network fixture."""
    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)


def test_offline_mode_end_to_end(tmp_path, vault, monkeypatch):
    monkeypatch.setattr(core, "AI_MODE", "off")
    assert core.create_chat_model() is None
    store = make_store(tmp_path, core.create_embeddings())
    core.ingest_notes(store, vault)

    result = core.ThinkingPartnerChain(None, store).invoke({"question": "what are fastapi routes?"})
    assert result["sources"] == ["fastapi.md"]
    assert "AI is off" in result["answer"] and "[1] fastapi.md" in result["answer"]
    # Citations follow the offline answer's numbering, i.e. after the keyword filter.
    lines = re.findall(r"^\[(\d+)\] (.+?): ", result["answer"], flags=re.M)
    assert len(lines) == len(result["citations"])
    for n, source in lines:
        assert (result["citations"][int(n) - 1]["n"], result["citations"][int(n) - 1]["source"]) == (int(n), source)

    miss = core.ThinkingPartnerChain(None, store).invoke({"question": "quantum chromodynamics"})
    assert miss["sources"] == []


def test_openai_mode_requires_key(monkeypatch):
    monkeypatch.setattr(core, "AI_MODE", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        core.create_embeddings()
