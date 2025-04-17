
# 🧠 Second Brain AI

Chat with your personal knowledge vault using OpenAI, LangChain, and ChromaDB — all synced with your Obsidian notes.

Built by [@pranavojha](https://github.com/pranavojha), powered by ChatGPT 💡

---

## 🔍 What It Does

- ✅ Loads your markdown notes from your Obsidian vault
- 🧠 Embeds them using OpenAI’s `text-embedding-3-small`
- 💾 Stores them locally using ChromaDB
- 💬 Lets you chat with your notes via:
  - Terminal (CLI)
  - Streamlit web app

---

## 🚀 Demo

> _“What do my notes say about deep work?”_  
> _“Summarize my thoughts on startups.”_

![screenshot](screenshot.png) <!-- Optional: Add a screenshot here -->

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [Obsidian](https://obsidian.md/)

---

## ⚙️ How to Use

1. Clone the repo  
   ```bash
   git clone https://github.com/YOUR_USERNAME/second-brain-ai
   cd second-brain-ai
   ```

2. Create `.env`  
   ```
   OPENAI_API_KEY=your_openai_key
   ```

3. Install dependencies  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Add your `.md` notes inside the `vault/` folder

5. Run chatbot (CLI)  
   ```bash
   python brain_chat/chat_with_vault.py
   ```

6. Or launch the web app  
   ```bash
   streamlit run brain_chat/second_brain_gui.py
   ```

---

## 🙏 Credits

Built by [@pranavojha](https://github.com/pranavojha) with support from ChatGPT.  
LangChain is love. Markdown is power.

---

## 📄 License

MIT – free to use, learn, remix.
