---
title: Kumar AI PDF Chat
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🤖 Kumar AI Study Assistant

> Conversational AI that lets you chat with any PDF using Retrieval-Augmented Generation (RAG) + LLaMA 3.1.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Overview

Kumar AI Study Assistant allows users to upload any PDF (textbooks, notes, research papers) and interact with it through a conversational AI interface.

Instead of manually searching documents, the system retrieves the most relevant sections using vector search and generates accurate, context-aware answers using LLaMA 3.1.

This project demonstrates a complete **Conversational RAG pipeline**.

---

## 🧠 How It Works
PDF Upload
↓
Text Splitting (Chunking)
↓
Embedding Generation
↓
FAISS Vector Store
↓
User Question
↓
Top-K Semantic Retrieval
↓
LLaMA 3.1 (via Groq)
↓
Answer + Source References

---

## ✨ Key Features

- 📄 Upload any PDF
- 💬 Continuous conversational chat
- 🔍 Semantic Top-K retrieval
- 📚 Source citation preview
- ⚡ Fast inference using Groq
- 🎨 Modern Streamlit UI
- 🐳 Docker-ready
- ☁️ Hugging Face Spaces compatible
- 🧠 Chat memory support

---

## ⚙️ Configuration

All major settings are configurable in `config.py`:

| Parameter | Description |
|----------|-------------|
| `llm_model` | LLaMA model version |
| `llm_temperature` | Controls creativity |
| `chunk_size` | PDF text chunk size |
| `chunk_overlap` | Overlap between chunks |
| `retriever_k` | Number of retrieved chunks |
| `max_file_size_mb` | Upload size limit |

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG orchestration |
| FAISS | Vector similarity search |
| HuggingFace Embeddings | Text embeddings |
| Groq API | LLaMA 3.1 inference |
| Streamlit | Web UI |
| Docker | Containerized deployment |

---

## ▶️ Run Locally (Docker)

```bash
docker-compose up --build
