# 🔬 AI Research Agent

An agentic AI system that autonomously researches any topic by searching the web and ArXiv papers, then synthesizes a structured report.

## 🏗️ Architecture

```
User Query → Planner Agent → [Web Search + ArXiv Search] → Synthesizer → Report
```

This is a **multi-tool agentic system** built with LangChain that demonstrates:
- Autonomous planning and tool selection
- Parallel information gathering from multiple sources
- LLM-based synthesis and report generation

## ⚙️ Setup (5 minutes)

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/ai-research-agent
cd ai-research-agent
pip install -r requirements.txt
```

### 2. Add your API key
Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

> **Free alternative:** Use [Groq](https://groq.com) (free tier, very fast).  
> Replace the LLM in `agent.py`:
> ```python
> from langchain_groq import ChatGroq
> llm = ChatGroq(model="llama-3.1-70b-versatile", api_key="your-groq-key")
> ```

### 3. Run the agent (terminal)
```bash
python agent.py
```

### 4. Run the web UI
```bash
streamlit run app.py
```

## 📁 Project Structure

```
ai-research-agent/
├── agent.py          # Core agent logic (LangChain + tools)
├── app.py            # Streamlit web UI
├── requirements.txt  # Dependencies
└── README.md
```

## 🧠 How It Works

1. **Planner** — The LLM receives your topic and decides which searches to run
2. **Tool use** — Agent calls `web_search` (DuckDuckGo) and `arxiv_search` autonomously
3. **Reasoning loop** — Agent iterates (up to 8 steps) until it has enough information
4. **Synthesis** — Final LLM call merges all findings into a structured markdown report

## 🚀 Day 2 Upgrades (add these yourself!)

- [ ] Add a `url_reader` tool using `requests` + `BeautifulSoup` to scrape full articles
- [ ] Add memory so agent remembers previous research sessions
- [ ] Save reports to a SQLite database

## 🔧 Tech Stack

- **LangChain** — Agent framework
- **OpenAI GPT-4o-mini** — LLM backbone
- **DuckDuckGo Search** — Web search (free, no API key)
- **ArXiv API** — Academic paper search (free)
- **Streamlit** — Web UI
