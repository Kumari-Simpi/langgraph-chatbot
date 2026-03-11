# 🔬 Research Assistant Chatbot

An **agentic AI chatbot** built with **LangGraph**, **Groq LLM**, and deployed on **Streamlit Cloud**.

The chatbot intelligently routes your questions to the right tool — research papers, factual lookup, or current news — and maintains full conversation memory within a session.

---

## 🚀 Live Demo

👉 ## 🚀 Live Demo
👉 [Click here to try the app](https://langgraph-chatbot-hvruasrz5lujsynrpf8yet.streamlit.app/)

---

## 🧠 How It Works

This project uses a **LangGraph StateGraph** — an agentic loop where the LLM decides which tools to call:

```
User Input
    ↓
[tool_calling_llm]  ← Groq Qwen3-32b
    ↓ (tool call?)
   YES → [ToolNode] → back to LLM → Final Answer
   NO  → Final Answer
```

### Tools Available

| Tool | Source | Best For |
|------|--------|----------|
| 🧑‍🔬 `arxiv` | ArXiv API | Academic research papers |
| 📖 `wikipedia` | Wikipedia API | Factual background knowledge |
| 🌐 `tavily` | Tavily Search | Recent news & current events |

### Key Features
- **Agentic routing** — LLM decides which tool(s) to use per query
- **Multi-tool calls** — can call multiple tools in parallel (e.g., "latest AI news AND recent papers")
- **Conversation memory** — remembers context across the full session
- **Tool transparency** — UI shows which tools were used per response

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq API — `qwen/qwen3-32b` |
| Agentic Framework | LangGraph |
| Tools | LangChain Community |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
langgraph-chatbot/
├── app.py              ← Streamlit UI
├── chatbot.py          ← LangGraph graph logic
├── requirements.txt    ← Dependencies
├── runtime.txt         ← Python 3.11 pin
├── .env.example        ← Template for secrets
├── .gitignore
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/langgraph-chatbot.git
cd langgraph-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Get your free API keys:
- **Groq**: https://console.groq.com
- **Tavily**: https://app.tavily.com

### 5. Run locally
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** → connect your GitHub repo
4. Set **Main file**: `app.py`
5. Go to **Advanced Settings → Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_key_here"
TAVILY_API_KEY = "your_tavily_key_here"
```

6. Click **Deploy** 🚀

---

## 💬 Example Queries

- `What is the paper 1706.03762?` → uses Arxiv
- `What is quantum entanglement?` → uses Wikipedia
- `Latest AI news today?` → uses Tavily
- `Recent research on large language models and recent LLM news?` → uses both Arxiv + Tavily

---


