import streamlit as st
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.core import AI_MODE, build_qa_chain, cited_lines

st.set_page_config(page_title="Second Brain Chat", layout="centered")
st.title("Talk to Your Second Brain")
st.markdown("Ask questions based on your Obsidian notes.")
if AI_MODE == "off":
    st.info("AI is off: answers are note excerpts only. Set `SECOND_BRAIN_AI=openai` in `.env` for generated answers.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Entries from before a hot reload may have an older shape.
st.session_state.chat_history = [entry for entry in st.session_state.chat_history if len(entry) == 4]

try:
    qa_chain = build_qa_chain()
except Exception as exc:
    st.error(f"Unable to load the chat pipeline: {exc}")
    st.stop()


def show_answer(answer: str) -> None:
    # Offline answers are raw note text: never render it, or a note's ![](url) fetches a remote image.
    if AI_MODE == "off":
        st.text(answer)
    else:
        st.markdown(answer.replace("![", "\\!["))


def show_sources(sources: list, cited: list) -> None:
    st.caption("Retrieved notes: " + (", ".join(sources) or "none"))
    if cited:
        st.caption("Cited in answer (pointer only, not checked against the claim):")
        st.text("\n".join(cited))


for question, answer, sources, cited in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        show_answer(answer)
        show_sources(sources, cited)

user_query = st.chat_input("Ask your brain something...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke(
                    {
                        "question": user_query,
                        "chat_history": [(q, a) for q, a, _, _ in st.session_state.chat_history],
                    }
                )
            except Exception as exc:
                st.error(f"Chat request failed: {exc}")
            else:
                answer = result["answer"]
                cited = cited_lines(result) if AI_MODE != "off" else []
                show_answer(answer)
                show_sources(result["sources"], cited)
                st.session_state.chat_history.append((user_query, answer, result["sources"], cited))
