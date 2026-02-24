import streamlit as st
import tempfile, os
from dotenv import load_dotenv
from core_rag import build_rag_chain
from config import config
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv()

st.set_page_config(
    page_title=config.app_title,
    page_icon=config.app_icon,
    layout="wide"
)

# ───────────────────────── UI STYLE ───────────────────────── #

st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f172a,#020617);
}

.main-header {
    font-size:2.5rem;
    font-weight:800;
    background: linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6);
    -webkit-background-clip:text;
    color:transparent;
    text-align:center;
}

.sub-header {
    color:#c7d2fe;
    text-align:center;
    margin-bottom:2rem;
}

.stat-box {
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:18px;
    padding:1.2rem;
    text-align:center;
    backdrop-filter: blur(12px);
    box-shadow:0 0 25px rgba(56,189,248,0.15);
    transition: 0.3s;
}
.stat-box:hover {
    transform:scale(1.05);
}

.stChatMessage.user {
    background:#1e293b;
    border-radius:16px;
    padding:12px;
}

.stChatMessage.assistant {
    background:#020617;
    border-radius:16px;
    padding:12px;
    border:1px solid #312e81;
}

.sidebar .block-container {
    background: linear-gradient(180deg,#020617,#0f172a);
}

button[kind="primary"] {
    background: linear-gradient(90deg,#38bdf8,#818cf8);
    border-radius:12px;
}

.footer {
    text-align:center;
    color:#94a3b8;
    font-size:0.8rem;
    margin-top:3rem;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────── HEADER ───────────────────────── #

st.markdown(f"<p class='main-header'>{config.app_title}</p>", unsafe_allow_html=True)
st.markdown(f"<p class='sub-header'>{config.app_description}</p>", unsafe_allow_html=True)

# ───────────────────────── SESSION STATE ───────────────────────── #

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ───────────────────────── SIDEBAR ───────────────────────── #

with st.sidebar:
    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        f"Choose a PDF (max {config.max_file_size_mb}MB)",
        type="pdf"
    )

    if uploaded_file:
        file_mb = uploaded_file.size / (1024 * 1024)
        if file_mb > config.max_file_size_mb:
            st.error(f"File too large! Max size is {config.max_file_size_mb}MB.")
        elif st.button("⚡ Process PDF", type="primary", use_container_width=True):
            with st.spinner("Indexing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name
                try:
                    chain, count = build_rag_chain(tmp_path)
                    st.session_state.chain = chain
                    st.session_state.chunk_count = count
                    st.session_state.chat_history = []
                    st.session_state.display_history = []
                    st.success(f"✅ Ready! {count} chunks indexed.")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    os.unlink(tmp_path)

    st.markdown("---")

    if st.session_state.chain:
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.chunk_count)
        col2.metric("Messages", len(st.session_state.display_history))

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.display_history = []
            st.rerun()

    st.markdown("---")
    st.markdown(f"""
    **⚙️ Model Info**  
    🤖 `{config.llm_model}`  
    🌡️ Temp: `{config.llm_temperature}`  
    🔍 Top-K: `{config.retriever_k}`
    """)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.8rem; color:#94a3b8'>
    Built by <b>{config.author_name}</b><br>
    <a href='{config.github_url}' target='_blank'>GitHub</a> · 
    <a href='{config.linkedin_url}' target='_blank'>LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────── MAIN AREA ───────────────────────── #

if not st.session_state.chain:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-box">⚡<br><b>LLaMA 3.1 8B</b><br><small>via Groq</small></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-box">🔍<br><b>RAG Pipeline</b><br><small>LangChain</small></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-box">💰<br><b>100% Free</b><br><small>No cost to run</small></div>', unsafe_allow_html=True)

    st.info("👈 Upload a PDF from the sidebar to start chatting!")

else:
    # Display history
    for msg in st.session_state.display_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📖 View Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i} · Page {src.metadata.get('page', '?') + 1}**")
                        st.caption(src.page_content[:400] + "...")
                        st.divider()

    # Chat input
    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.display_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                sources = result.get("context", [])

            st.write(answer)

            if sources:
                with st.expander("📖 View Sources"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i} · Page {src.metadata.get('page', '?') + 1}**")
                        st.caption(src.page_content[:400] + "...")
                        st.divider()

        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])

        st.session_state.display_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

# ───────────────────────── FOOTER ───────────────────────── #

st.markdown(
    f"<div class='footer'>AI Study Assistant · Built by {config.author_name} · Powered by LangChain, Groq & Streamlit</div>",
    unsafe_allow_html=True
)