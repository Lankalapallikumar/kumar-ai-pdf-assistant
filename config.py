# config.py
from dataclasses import dataclass

@dataclass
class AppConfig:
    # ── App Info ─────────────────────────
    app_title: str = "🤖 Kumar AI Study Assistant"
    app_icon: str = "🤖"
    app_description: str = "Chat with any PDF using Conversational RAG + LLaMA 3 — built for smart learning."
    layout: str = "wide"

    # ── Model Settings ───────────────────
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.25
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG Config ───────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100
    retriever_k: int = 2   # <-- TOP 2 chunks only

    # ── UI Limits ────────────────────────
    max_file_size_mb: int = 10

    # ── Author Info ──────────────────────
    author_name: str = "Lankalapalli Kumar"
    github_url: str = "https://github.com/Lankalapallikumar"
    linkedin_url: str = "https://www.linkedin.com/in/kumar-lankalapalli-datascience-ml"

config = AppConfig()