from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain_chat.core import AI_MODE, build_qa_chain, cited_lines


def start_chat() -> None:
    try:
        qa_chain = build_qa_chain()
    except Exception as exc:
        print(exc)
        return

    chat_history = []
    print("\nChat with your second brain. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break
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
        sources = ", ".join(result["sources"]) or "none"
        print(f"\nSecond Brain: {answer}\nRetrieved notes: {sources}")
        cited = cited_lines(result) if AI_MODE != "off" else []
        if cited:
            print("Cited in answer (pointer only, not checked against the claim):")
            print("\n".join(cited))
        print()
        chat_history.append((query, answer))


if __name__ == "__main__":
    start_chat()
