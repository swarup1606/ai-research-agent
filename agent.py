from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent   # ignore the warning, this works
from ddgs import DDGS
import arxiv
import os

# 1. LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key="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"   # ← your groq key
)

# 2. Tools
@tool
def web_search(query: str) -> str:
    """Search the web for recent news and practical information."""
    try:
        results = DDGS().text(query, max_results=5)
        return "\n\n".join([f"Title: {r['title']}\n{r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {e}"

@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for real academic research papers."""
    try:
        search = arxiv.Search(query=query, max_results=5)
        papers = []
        for p in arxiv.Client().results(search):
            papers.append(
                f"Title: {p.title}\n"
                f"Authors: {', '.join(a.name for a in p.authors[:3])}\n"
                f"ArXiv ID: {p.entry_id}\n"
                f"Summary: {p.summary[:400]}"
            )
        return "\n\n---\n\n".join(papers)
    except Exception as e:
        return f"ArXiv search failed: {e}"

# 3. Agent
tools = [web_search, arxiv_search]
agent = create_react_agent(llm, tools)

# 4. Run
def run_research(topic: str):
    print(f"\n🔍 Researching: {topic}\n{'='*50}")
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"""You MUST use the web_search and arxiv_search tools.
Do NOT make up papers or information. Only use real results from the tools.

Step 1: Call web_search tool with topic: {topic}
Step 2: Call arxiv_search tool with topic: {topic}
Step 3: Write a report ONLY from the real tool results with these sections:
  - 📌 Summary
  - 🔬 Key Research Papers (real ArXiv IDs only)
  - 🌐 Recent Developments
  - 💡 Key Insights
  - 🚀 Future Directions"""
        }]
    })
    return result["messages"][-1].content

if __name__ == "__main__":
    topic = "Retrieval Augmented Generation RAG latest advances 2024"
    report = run_research(topic)
    print("\n" + "="*50)
    print("📄 FINAL REPORT")
    print("="*50)
    print(report)