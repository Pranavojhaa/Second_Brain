import streamlit as st
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.core import build_qa_chain, has_openai_api_key, has_vectorstore

st.set_page_config(page_title="Second Brain Chat", layout="centered")
st.title("Talk to Your Second Brain")
st.markdown("Ask questions based on your Obsidian notes.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not has_openai_api_key():
    st.error("OPENAI_API_KEY is missing. Add it to your .env before starting the app.")
    st.stop()

if not has_vectorstore():
    st.error(
        "No vectorstore found yet. Run `python brain_chat/ingest.py` first, "
        "or set `AUTO_INGEST_ON_STARTUP=true` in Railway after mounting your vault."
    )
    st.stop()

try:
    qa_chain = build_qa_chain()
except Exception as exc:
    st.error(f"Unable to load the chat pipeline: {exc}")
    st.stop()

for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)

user_query = st.chat_input("Ask your brain something...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke(
                    {"question": user_query, "chat_history": st.session_state.chat_history}
                )
            except Exception as exc:
                st.error(f"Chat request failed: {exc}")
            else:
                answer = result["answer"]
                st.markdown(answer)
                st.session_state.chat_history.append((user_query, answer))
