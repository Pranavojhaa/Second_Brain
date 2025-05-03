# 🧠 Second Brain

An AI-powered second brain using Obsidian, LangChain, Chroma, and OpenAI.  
Chat with your personal notes — right from the terminal or a web browser!

---

## 🚀 How to Run

This project supports **CLI mode** (terminal) and **Streamlit mode** (web app).

---

### ✅ CLI Mode (Terminal chat)

Embed new notes and chat from the terminal:
```bash
python brain_chat/main.py cli
```

---

### ✅ Streamlit Mode (Web GUI)

Chat with your notes in a browser interface:
```bash
python brain_chat/main.py streamlit
```

---

## 💾 Requirements

- Python 3.9+
- Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

- Create a `.env` file in the root folder:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

---

## 📦 Features

✅ Load and embed `.md` notes from your Obsidian vault  
✅ Track which notes have been embedded (no duplicates)  
✅ Chat in the terminal or Streamlit  
✅ Automatically updates only new files

---

## ⚠️ Known Warnings

- Deprecation warnings from LangChain → safe to ignore for now  
- `KeyboardInterrupt` on Ctrl+C → normal when stopping the app

---

## 🤝 Contributing

Pull requests and ideas are welcome! Feel free to fork and improve.

---

## 📄 License

MIT License
