# CourseChat | RAG Study Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions grounded in your own course notes, with source highlighting so you can see exactly where every answer comes from.

Built with LangChain, Supabase pgvector, HuggingFace, Groq, and Streamlit.

**[Live demo →](https://coursechat-ryant.streamlit.app/)**

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
- **Vision enrichment** — Groq vision model describes embedded images and equations that text extraction skips, keeping math content searchable
- **Semantic chunking** — splits documents into overlapping chunks to preserve context across page boundaries
- **Local embeddings** — `all-MiniLM-L6-v2` via HuggingFace, runs entirely on CPU with no API cost
- **Supabase pgvector** — named collections per course, persisted in the cloud across sessions and deployments
- **Tool-based retrieval** — agent decides when to call the retrieval tool, avoiding retrieval on greetings and non-course questions
- **Sliding window history** — caps context sent to the agent at the last 10 message pairs to prevent token limit degradation over long sessions
- **Persistent conversation history** — per-course chat history saved to disk, survives browser close and app restart
- **Source highlighting** — every answer links back to the exact source chunk, with the original PDF page rendered inline
- **Multi-course support** — isolated pgvector collections per course; add, switch, and delete courses from the sidebar
- **Per-file management** — view and delete individual indexed files without removing the whole course

---

## Architecture

```
PDF upload (via Streamlit UI)
│
▼
pymupdf4llm (markdown extraction, page-by-page)
│
▼
Groq llama-4-scout vision model (describes embedded images/equations)
│
▼
RecursiveCharacterTextSplitter (chunk_size=800, overlap=100)
│
▼
HuggingFace all-MiniLM-L6-v2 (local embeddings, free)
│
▼
Supabase pgvector (named collection per course, cloud-persisted)
    └── Supabase Storage (original PDFs, for page image rendering)
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
│   ├── ingest.py           # PDF loading, vision enrichment, chunking, pgvector upsert
│   ├── retriever.py        # Agent setup, retrieve_context tool, build_chain()
│   ├── history.py          # Per-course conversation persistence + Supabase queries
│   └── config.py           # Shared constants
├── data/
│   ├── notes/              # Local PDF cache (gitignored, sourced from Supabase Storage)
│   └── conversations/      # Per-course chat history JSON files (gitignored)
├── .streamlit/
│   └── config.toml         # Disables file watcher noise
├── requirements.txt        # Pinned runtime dependencies
├── .env.example
└── README.md
```

---

## Stack

| Component      | Library                                      | Why                                                      |
| -------------- | -------------------------------------------- | -------------------------------------------------------- |
| PDF extraction | `pymupdf4llm`                                | Markdown output preserves tables, headers, math notation |
| Vision         | `llama-4-scout` (Groq)                       | Describes embedded images and equations in PDFs          |
| Text splitting | `langchain` `RecursiveCharacterTextSplitter` | Overlap preserves context at chunk boundaries            |
| Embeddings     | `all-MiniLM-L6-v2` (HuggingFace)            | Free, local, no API key, good retrieval quality          |
| Vector store   | Supabase pgvector (`langchain-postgres`)     | Cloud-persisted, named collections per course            |
| PDF storage    | Supabase Storage                             | Survives container restarts on Streamlit Cloud           |
| LLM            | `llama-3.3-70b-versatile` (Groq)            | Free tier, fast inference, strong instruction following  |
| Agent          | `langchain` `create_agent`                   | Tool-based retrieval with conditional calling            |
| UI             | `streamlit`                                  | Chat primitives built in, fast to iterate                |

---

## Learning Highlights

**Retrieval-Augmented Generation**
RAG outperforms pure LLM generation for domain-specific knowledge by constraining the model to retrieved context, reducing hallucination and making answers verifiable. Key engineering decisions: chunk size (larger = more context per chunk, noisier retrieval), overlap (more overlap = better cross-boundary coherence), and k (more chunks = more coverage, higher token cost).
Sliding window conversation history caps context to the last 10 message pairs to prevent token limit degradation over long sessions, trading full recall for stable inference cost.

**Tool-based vs middleware retrieval**
Two approaches to RAG with LangChain agents: always-retrieve middleware (intercepts every LLM call and injects context regardless of query type) vs tool-based retrieval (agent decides when to invoke the retrieval tool based on the query). Middleware is simpler but wastes tokens on greetings and follow-ups that don't need retrieval. Tool-based is more natural — the agent reasons about whether the question warrants a vector store lookup. The tradeoff is an extra reasoning step and occasional missed retrieval on ambiguous queries.

**Vector databases and embeddings**
Embeddings transform unstructured text into dense vector representations where semantic similarity translates into geometric proximity in a high-dimensional space. During runtime, user queries are embedded using the same `all-MiniLM-L6-v2` model and a cosine similarity nearest-neighbor search is performed against Supabase pgvector. Migrating from local ChromaDB to pgvector highlighted the difference between SQLAlchemy-based connection strings (`postgresql+psycopg://`) and raw psycopg connection strings (`postgresql://`), and how connection pooling (port 6543) is required on managed cloud databases that restrict direct connections.

**Vision enrichment for math content**
Two separate vision-related features exist in the pipeline. At ingest time, `pymupdf4llm` skips embedded images and replaces them with placeholders — for math-heavy PDFs this means equations in images are invisible to the retriever. A vision model pass (`llama-4-scout`) transcribes each image into LaTeX at ingest time, replacing placeholders with searchable text so equations end up in the vector store. At query time, a separate feature fetches the original PDF from Supabase Storage and renders the specific source page as an image inline in the sources panel, giving the user visual context alongside the text excerpt.

---

## Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/RyanTandean/CourseChat.git
cd coursenotes-rag
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with the following:

```
GROQ_API_KEY=your_groq_api_key
SUPABASE_CONNECTION_STRING=postgresql+psycopg://postgres.xxxx:password@aws-0-region.pooler.supabase.com:6543/postgres
SUPABASE_DB_URL=postgresql://postgres.xxxx:password@aws-0-region.pooler.supabase.com:6543/postgres
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
HF_TOKEN=your_huggingface_token
```

Supabase setup: enable the `vector` extension in your project's SQL editor (`create extension if not exists vector;`), create a private storage bucket named `course-pdfs`. LangChain creates the embedding tables automatically on first ingest.

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
