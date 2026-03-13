# ============================================================
#  AI Research Agent — Day 2 (llama-3.1-8b-instant version)
#  Multi-agent: Planner → Researcher → Synthesizer
# ============================================================

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from ddgs import DDGS
import arxiv
import requests
from bs4 import BeautifulSoup
import time

# ── 1. Config ─────────────────────────────────────────────────
GROQ_KEY = "sk-your-key-here"   # ← paste your Groq key here

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_KEY
)

llm_smart = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    api_key=GROQ_KEY
)

# ── 2. Tools ──────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for recent news and practical information."""
    try:
        results = DDGS().text(query, max_results=2)
        return "\n\n".join([
            f"Title: {r['title']}\n{r['body'][:150]}"
            for r in results
        ])
    except Exception as e:
        return f"Search failed: {e}"

@tool
def arxiv_search(query: str) -> str:
    """Search ArXiv for real academic research papers."""
    try:
        search = arxiv.Search(query=query, max_results=2)
        papers = []
        for p in arxiv.Client().results(search):
            papers.append(
                f"Title: {p.title}\n"
                f"ArXiv ID: {p.entry_id}\n"
                f"Published: {p.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {p.summary[:200]}"
            )
        return "\n\n---\n\n".join(papers)
    except Exception as e:
        return f"ArXiv search failed: {e}"

@tool
def read_url(url: str) -> str:
    """Read and extract the main text content from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        return "\n".join(lines[:30])
    except Exception as e:
        return f"Could not read URL: {e}"

# ── 3. Shared State ───────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    search_queries: list[str]
    raw_research: Annotated[list[str], operator.add]
    final_report: str

# ── 4. Agent 1: Planner ───────────────────────────────────────
def planner_agent(state: ResearchState) -> dict:
    print("\n🧠 PLANNER: Generating search queries...")

    prompt = f"""Generate 2 short search queries for this topic: {state['topic']}

Return ONLY a Python list. Example: ["query one", "query two"]"""

    response = llm_smart.invoke(prompt)
    content = response.content.strip()

    try:
        import ast
        start = content.find("[")
        end = content.rfind("]") + 1
        queries = ast.literal_eval(content[start:end])
        print(f"   Queries: {queries}")
    except Exception:
        queries = [state['topic'], f"{state['topic']} research 2024"]
        print(f"   Using fallback queries")

    return {"search_queries": queries}

# ── 5. Agent 2: Researcher ────────────────────────────────────
researcher_tools = [web_search, arxiv_search, read_url]
researcher = create_react_agent(llm, researcher_tools)

def researcher_agent(state: ResearchState) -> dict:
    print("\n🔍 RESEARCHER: Searching sources...")

    all_results = []

    for i, query in enumerate(state['search_queries'][:2]):
        print(f"   [{i+1}/2] Searching: {query}")

        result = researcher.invoke({
            "messages": [{
                "role": "user",
                "content": f"""Search for: "{query}"
Use web_search and arxiv_search tools. Return key findings only."""
            }]
        })

        raw = result["messages"][-1].content
        all_results.append(f"=== Query: {query} ===\n{raw[:1000]}")
        print(f"   ✓ Done")
        time.sleep(3)

    return {"raw_research": all_results}

# ── 6. Agent 3: Synthesizer ───────────────────────────────────
def synthesizer_agent(state: ResearchState) -> dict:
    print("\n✍️  SYNTHESIZER: Writing report...")

    all_research = "\n\n".join(state['raw_research'])

    prompt = f"""Write a research report on: {state['topic']}

Based on this research:
{all_research[:3000]}

Write a structured report with these sections:
- 📌 Summary (3-4 sentences)
- 🔬 Key Research Papers (with real ArXiv IDs)
- 🌐 Recent Developments
- 💡 Key Insights (top 3-4 points)
- 🚀 Future Directions

Only use information from the research above."""

    response = llm_smart.invoke(prompt)
    print("   ✓ Report ready!")
    return {"final_report": response.content}

# ── 7. Build Pipeline ─────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(ResearchState)

    graph.add_node("planner",     planner_agent)
    graph.add_node("researcher",  researcher_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner",     "researcher")
    graph.add_edge("researcher",  "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()

pipeline = build_pipeline()

# ── 8. Run ────────────────────────────────────────────────────
def run_research(topic: str) -> str:
    print(f"\n{'='*55}")
    print(f"  🚀 Multi-Agent Research System")
    print(f"  Topic: {topic}")
    print(f"{'='*55}")

    result = pipeline.invoke({
        "topic": topic,
        "search_queries": [],
        "raw_research": [],
        "final_report": ""
    })

    return result["final_report"]


if __name__ == "__main__":
    topic = "Retrieval Augmented Generation RAG latest advances 2024"
    report = run_research(topic)

    print("\n" + "="*55)
    print("📄  FINAL RESEARCH REPORT")
    print("="*55)
    print(report)

    filename = f"report_{topic[:30].replace(' ', '_')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {topic}\n\n{report}")
    print(f"\n💾 Saved to: {filename}")
