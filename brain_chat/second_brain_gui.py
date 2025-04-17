import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# Load .env
load_dotenv()

# Initialize
st.set_page_config(page_title="Second Brain Chat 💬", layout="centered")
st.title("🧠 Talk to Your Second Brain")
st.markdown("Ask questions based on your Obsidian notes.")

# Set up session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Set up chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Chat input
user_query = st.chat_input("Ask your brain something...")

if user_query:
    with st.spinner("Thinking..."):
        result = qa_chain({"question": user_query, "chat_history": st.session_state.chat_history})
        answer = result["answer"]
        st.session_state.chat_history.append((user_query, answer))

# Display history
for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**🧠 Second Brain:** {a}")
    st.markdown("---")
