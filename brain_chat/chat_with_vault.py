import os
import json
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# Load API key
load_dotenv()

VAULT_DIR = "vault"
TRACKER_PATH = "brain_chat/embedded_files.json"

def load_and_chunk_notes():
    # Load previously embedded files
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, "r") as f:
            already_embedded = set(json.load(f))
    else:
        already_embedded = set()

    loader = DirectoryLoader(
        VAULT_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        use_multithreading=True,
        show_progress=True
    )

    all_docs = loader.load()
    new_docs = [doc for doc in all_docs if doc.metadata["source"] not in already_embedded]

    if not new_docs:
        print("🟢 No new notes found to embed.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = splitter.split_documents(new_docs)
    print(f"🧠 Found {len(new_docs)} new notes → {len(split_docs)} chunks.")
    return split_docs

def embed_and_store(docs):
    if not docs:
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    vectorstore.persist()

    # Update tracker
    embedded_files = list(set(doc.metadata["source"] for doc in docs))
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, "r") as f:
            existing = set(json.load(f))
    else:
        existing = set()

    updated = list(existing.union(set(embedded_files)))
    with open(TRACKER_PATH, "w") as f:
        json.dump(updated, f, indent=2)

    print("✅ Chunks embedded and saved to ChromaDB.")

def start_chat():
    if not os.path.exists("chroma_db"):
        print("⚠️ No vectorstore found. Run embedding first.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    chat_history = []
    print("\n🧠 Chat with your second brain! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting chat.")
            break

        result = qa_chain({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        print(f"\nSecond Brain: {answer}\n")
        chat_history.append((query, answer))

if __name__ == "__main__":
    docs = load_and_chunk_notes()
    embed_and_store(docs)
    start_chat()
