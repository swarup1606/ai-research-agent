# ============================================================
#  AI Research Agent — Streamlit UI
#  Run: streamlit run app.py
# ============================================================

import streamlit as st
import time
from agent_v2 import run_research

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ How It Works")
    st.markdown("""
    Three specialized AI agents collaborate:

    **🧠 Planner Agent**
    Breaks your topic into targeted search queries

    **🔍 Researcher Agent**
    Searches Web + ArXiv papers + reads URLs

    **✍️ Synthesizer Agent**
    Merges everything into a structured report

    ---
    **🛠️ Tech Stack**
    - 🔗 LangGraph (multi-agent)
    - 🦜 LangChain
    - ⚡ Groq LLM (llama-3.1-8b)
    - 🔎 DuckDuckGo Search
    - 📚 ArXiv API
    - 🌐 BeautifulSoup
    - 🎈 Streamlit
    """)
    st.divider()
    if "report_count" not in st.session_state:
        st.session_state.report_count = 0
    st.metric("Reports Generated", st.session_state.report_count)

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔬 Multi-Agent AI Research System</p>',
            unsafe_allow_html=True)
st.caption("Powered by LangGraph · LangChain · Groq · DuckDuckGo · ArXiv")

st.markdown("""
Three AI agents work together to:
1. **Plan** — break your topic into targeted search queries
2. **Research** — search the web, ArXiv papers, and read URLs
3. **Synthesize** — produce a clean structured report
""")

# ── Agent Pipeline Visual ─────────────────────────────────────
st.divider()
col1, col2, col3, col4, col5 = st.columns([2, 0.4, 2, 0.4, 2])
with col1:
    st.info("🧠 **Planner**\nGenerates search queries")
with col2:
    st.markdown("<h3 style='text-align:center;margin-top:14px'>→</h3>",
                unsafe_allow_html=True)
with col3:
    st.info("🔍 **Researcher**\nSearches Web + ArXiv")
with col4:
    st.markdown("<h3 style='text-align:center;margin-top:14px'>→</h3>",
                unsafe_allow_html=True)
with col5:
    st.info("✍️ **Synthesizer**\nWrites final report")
st.divider()

# ── Example Topics ────────────────────────────────────────────
with st.expander("💡 Example topics to try"):
    examples = [
        "Multi-agent AI systems and frameworks",
        "Transformer architecture improvements 2024",
        "AI agents for healthcare applications",
        "LLM fine-tuning techniques — LoRA and QLoRA",
        "Vision language models — latest advances",
        "Graph neural networks applications",
    ]
    ex_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with ex_cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["selected_topic"] = ex

# ── Input ─────────────────────────────────────────────────────
topic = st.text_input(
    "🎯 Enter your research topic:",
    value=st.session_state.get("selected_topic", ""),
    placeholder="e.g. Retrieval Augmented Generation for LLMs"
)

run_btn = st.button("🚀 Start Research", type="primary")

# ── Run Pipeline ──────────────────────────────────────────────
if run_btn and topic:
    st.divider()
    st.markdown(f"### 🔎 Researching: *{topic}*")

    # Live status indicators
    status_cols = st.columns(3)
    with status_cols[0]:
        p_status = st.empty()
        p_status.warning("🧠 Planner: Running...")
    with status_cols[1]:
        r_status = st.empty()
        r_status.info("🔍 Researcher: Waiting...")
    with status_cols[2]:
        s_status = st.empty()
        s_status.info("✍️ Synthesizer: Waiting...")

    progress = st.progress(0, text="Starting agents...")
    log_box  = st.empty()

    try:
        log_box.code("🧠 Planner agent: generating search queries...")
        progress.progress(15, text="Planner running...")
        time.sleep(1)

        p_status.success("🧠 Planner: ✓ Done")
        r_status.warning("🔍 Researcher: Searching...")
        progress.progress(35, text="Researcher searching web + ArXiv...")
        log_box.code("🔍 Researcher agent: searching web and ArXiv papers...")

        # ── Run the actual 3-agent pipeline ──────────────────
        report = run_research(topic)

        r_status.success("🔍 Researcher: ✓ Done")
        s_status.warning("✍️ Synthesizer: Writing...")
        progress.progress(85, text="Synthesizer writing report...")
        log_box.code("✍️ Synthesizer agent: writing structured report...")
        time.sleep(1)

        s_status.success("✍️ Synthesizer: ✓ Done")
        progress.progress(100, text="Complete!")
        log_box.empty()

        st.session_state.report_count += 1
        st.success("✅ Research complete!")

        # ── Display Report ────────────────────────────────────
        st.divider()
        st.markdown("## 📄 Research Report")
        st.markdown(report)
        st.divider()

        # ── Download Buttons ──────────────────────────────────
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="⬇️ Download as Markdown",
                data=f"# Research Report: {topic}\n\n{report}",
                file_name=f"research_{topic[:30].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with dl2:
            st.download_button(
                label="⬇️ Download as Text",
                data=f"Research Report: {topic}\n\n{report}",
                file_name=f"research_{topic[:30].replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("💡 Check your GROQ_API_KEY in agent_v2.py or .streamlit/secrets.toml")

elif run_btn and not topic:
    st.warning("⚠️ Please enter a research topic first.")