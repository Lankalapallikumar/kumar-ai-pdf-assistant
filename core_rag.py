# core_rag.py

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# LangChain classic imports
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_rag_chain(file_path: str):
    """
    Loads a PDF, creates a FAISS vector store,
    and returns a conversational RAG chain + chunk count.
    """

    # 1️⃣ Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2️⃣ Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # 3️⃣ Create embeddings + FAISS vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Top-K retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4️⃣ Initialize Groq LLaMA (reads GROQ_API_KEY automatically)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.25
    )

    # 5️⃣ History-aware retriever prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "reformulate the question into a standalone question. "
        "Do NOT answer the question."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # 6️⃣ Question answering prompt
    qa_system_prompt = (
        "You are a helpful study assistant. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, say you don't know.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 7️⃣ Final RAG chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    return rag_chain, len(chunks)