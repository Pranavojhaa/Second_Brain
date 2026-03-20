from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.core import build_qa_chain, has_openai_api_key


def start_chat() -> None:
    if not has_openai_api_key():
        print("OPENAI_API_KEY is missing. Add it to your .env before starting chat.")
        return

    try:
        qa_chain = build_qa_chain()
    except Exception as exc:
        print(exc)
        return

    chat_history = []
    print("\nChat with your second brain. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        try:
            result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        except Exception as exc:
            print(f"Chat request failed: {exc}")
            continue

        answer = result["answer"]
        print(f"\nSecond Brain: {answer}\n")
        chat_history.append((query, answer))


if __name__ == "__main__":
    start_chat()
