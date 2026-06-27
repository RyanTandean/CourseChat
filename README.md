# CourseChat | RAG Study Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions grounded in your own course notes, with source highlighting so you can see exactly where every answer comes from.

Built with LangChain, ChromaDB, HuggingFace, Groq, and Streamlit.

This repository is currently set up as a local demo rather than a deployed service. The focus is on fast iteration, grounded answers, and clear source tracing.

---

<img width="1901" height="880" alt="image" src="https://github.com/user-attachments/assets/ef3642b1-6b1b-44ac-9e16-397fa446e472" />

---

## Rationale

The primary driver behind this mini-project was an experimental deep-dive to understand the mechanics of LLMs, orchestration frameworks (LangChain), and Retrieval-Augmented Generation (RAG).

While models like GPT, Claude, and Gemini possess immense general knowledge, this breadth becomes a liability in structured academic environments. University courses sometimes introduce complex subjects by deliberately scoping down concepts and utilizing specific terminology that might not be reflective of terminology used in the real world.

When a student asks a general LLM for help with a specialized course concept, the model often pulls in external methodologies, advanced formulations, or conflicting definitions. Instead of helping, this extra "out-of-scope" data introduces noise, breaks the instructor's intended learning progression, and ultimately accelerates student confusion.

This experiment consists of a chatbot that enforces an absolute boundary on information retrieval by grounding the LLM strictly within the official course notes. This ensures that:

- The model only explains concepts using the specific definitions and boundaries established by the course curriculum
- Explanations are consistent with what students see on exams and assignments
- Hallucinations are minimized by restricting the context window strictly to lecture materials

---

## Features

- **PDF ingestion** — upload one or more PDFs per course (lecture notes, textbook chapters, problem sets)
- **Markdown-aware extraction** — uses `pymupdf4llm` to preserve tables, headers, and math notation as structured markdown rather than raw text
- **Semantic chunking** — splits documents into overlapping chunks to preserve context across page boundaries
- **Local embeddings** — `all-MiniLM-L6-v2` via HuggingFace, runs entirely on CPU with no API cost
- **ChromaDB vector store** — named collections per course, persists across sessions
- **Tool-based retrieval** — agent decides when to call the retrieval tool, avoiding retrieval on greetings and non-course questions
- **Persistent conversation history** — per-course chat history saved to disk as JSON, survives browser close and app restart
- **Source highlighting** — every answer links back to the exact source chunk, with the original PDF page rendered inline so you can verify the context without leaving the app
- **Multi-course support** — isolated ChromaDB collections per course, switch between courses like switching tabs

---

## Architecture

```
PDF upload (via Streamlit UI)
│
▼
pymupdf4llm (markdown extraction, page-by-page)
│
▼
RecursiveCharacterTextSplitter (chunk_size=800, overlap=100)
│
▼
HuggingFace all-MiniLM-L6-v2 (local embeddings, free)
│
▼
ChromaDB (named collection per course, persisted to ./chroma_db)
│
▼
User query
│
▼
LangChain create_agent + retrieve_context tool
│
├── agent decides to call retrieve_context
│   └── similarity search (k=6) → top chunks + metadata
│
└── Groq llama-3.3-70b-versatile (answer generation)
    │
    ▼
Streamlit chat UI + collapsible source panel (renders PDF page inline)
```

---

## Project Structure

```
coursenotes-rag/
├── app.py                  # Streamlit entrypoint, UI, session state
├── rag/
│   ├── ingest.py           # PDF loading, chunking, embedding, ChromaDB upsert
│   ├── retriever.py        # Agent setup, retrieve_context tool, build_chain()
│   └── history.py          # Per-course conversation persistence (JSON)
├── data/
│   ├── notes/              # Drop PDFs here for CLI ingestion (gitignored)
│   └── conversations/      # Per-course chat history JSON files (gitignored)
├── chroma_db/              # Persisted vector store (gitignored)
├── .streamlit/
│   └── config.toml         # Disables file watcher noise
├── requirements.txt        # Runtime dependencies
├── .env.example
└── README.md
```

---

## Stack

| Component      | Library                                       | Why                                                      |
| -------------- | --------------------------------------------- | -------------------------------------------------------- |
| PDF extraction | `pymupdf4llm`                                 | Markdown output preserves tables, headers, math notation |
| Text splitting | `langchain` `RecursiveCharacterTextSplitter`  | Overlap preserves context at chunk boundaries            |
| Embeddings     | `all-MiniLM-L6-v2` (HuggingFace)             | Free, local, no API key, good retrieval quality          |
| Vector store   | `chromadb`                                    | Local, persistent, named collections per course          |
| LLM            | `llama-3.3-70b-versatile` (Groq)             | Free tier, fast inference, strong instruction following  |
| Agent          | `langchain` `create_react_agent`              | Tool-based retrieval with conditional calling            |
| UI             | `streamlit`                                   | Chat primitives built in, fast to iterate                |

---

## Learning Highlights

**Retrieval-Augmented Generation**
RAG outperforms pure LLM generation for domain-specific knowledge by constraining the model to retrieved context, reducing hallucination and making answers verifiable. Key engineering decisions: chunk size (larger = more context per chunk, noisier retrieval), overlap (more overlap = better cross-boundary coherence), and k (more chunks = more coverage, higher token cost).
Sliding window conversation history caps context to the last N turns to prevent token limit degradation over long sessions, trading full recall for stable inference cost.

**Tool-based vs middleware retrieval**
Two approaches to RAG with LangChain agents: always-retrieve middleware (intercepts every LLM call and injects context regardless of query type) vs tool-based retrieval (agent decides when to invoke the retrieval tool based on the query). Middleware is simpler to implement and guarantees context is always present, but wastes tokens on greetings, clarifications, and follow-ups that don't need retrieval. Tool-based is more natural, since the agent reasons about whether the question warrants a vector store lookup, which reduces unnecessary retrieval calls and keeps the context window focused on relevant chunks. The tradeoff is that tool-based retrieval adds an extra reasoning step and can occasionally miss retrieval on ambiguous queries.

**Vector databases and embeddings**
Embeddings transform unstructured text into dense vector representations where semantic similarity translates into geometric proximity in a high-dimensional space. During runtime, user queries are embedded using the same `all-MiniLM-L6-v2` model and a cosine similarity nearest-neighbor search is performed against the ChromaDB store. This highlighted how critical clean text ingestion and metadata filtering are, since poorly formatted chunks degrade retrieval quality regardless of model capability.

---

## Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/RyanTandean/CourseChat.git
cd coursenotes-rag
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. CLI Demo

Terminal-based demo for exercising the model flow without the browser UI:

```bash
python test_cli.py
```

### 5. Evaluation

Measures retrieval recall by checking whether top retrieved chunks contain expected keywords for each test question:

```bash
python eval.py
```

---

## Roadmap

- Hybrid search (BM25 + dense retrieval with reciprocal rank fusion)
- Cross-encoder reranking for improved answer quality
- Cloud deployment with persistent storage (Railway + Supabase)
