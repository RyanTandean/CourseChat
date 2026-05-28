# 📚 CourseChat — RAG Study Assistant for Course Notes

A retrieval-augmented generation (RAG) chatbot that answers questions grounded in your own course notes, with source highlighting so you can see exactly where every answer comes from.

Built with LangChain, ChromaDB, OpenAI, and Streamlit.

---

## Why I built this

Generic LLMs hallucinate course-specific content. This tool forces the model to answer only from material you've uploaded — your lecture slides, textbook excerpts, and notes — and shows you the exact passage it drew from for every response. Inspired by the same architecture used in scientific research assistants that help domain experts query internal datasets alongside public literature.

---

## Demo

> *screenshot / GIF here*

---

## Features

- **PDF ingestion** — upload one or more PDFs (lecture notes, textbook chapters, problem sets)
- **Semantic chunking** — splits documents into overlapping chunks to preserve context across page boundaries
- **ChromaDB vector store** — embeds and indexes chunks locally, persists across sessions
- **Conversational retrieval** — maintains chat history so follow-up questions work correctly
- **Source highlighting** — every answer shows the exact source chunk(s) it was retrieved from, with document name and page number
- **Multi-document support** — upload notes from multiple courses, filter by course at query time
- **Streamlit UI** — single-page chat interface, no frontend setup required

---

## Architecture

```
PDF upload
    │
    ▼
PyMuPDF (text extraction)
    │
    ▼
RecursiveCharacterTextSplitter (chunk_size=800, overlap=100)
    │
    ▼
OpenAI text-embedding-3-small (embeddings)
    │
    ▼
ChromaDB (local vector store, persisted to ./chroma_db)
    │
    ▼
User query ──► ConversationalRetrievalChain (LangChain)
                    │
                    ├── top-k retrieved chunks (k=4)
                    │       └── returned as source documents
                    │
                    └── gpt-4o-mini (answer generation)
                            │
                            ▼
                    Streamlit chat UI
                    + source highlight panel
```

---

## Project structure

```
coursenotes-rag/
├── app.py                  # Streamlit entrypoint
├── rag/
│   ├── ingest.py           # PDF loading, chunking, embedding, ChromaDB upsert
│   ├── retriever.py        # ChromaDB query, ConversationalRetrievalChain setup
│   └── highlight.py        # Source chunk formatting and display logic
├── data/
│   └── notes/              # Drop PDFs here (gitignored)
├── chroma_db/              # Persisted vector store (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Build plan

### Phase 1 — Ingestion pipeline (Day 1 morning)
- [ ] Set up project structure and virtual environment
- [ ] Install dependencies: `langchain`, `langchain-openai`, `chromadb`, `pymupdf`, `streamlit`, `python-dotenv`
- [ ] Write `ingest.py`:
  - Load PDF with `PyMuPDFLoader`
  - Split with `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)`
  - Embed with `OpenAIEmbeddings(model="text-embedding-3-small")`
  - Store in `Chroma` with metadata: `{source: filename, page: n}`
  - Persist to `./chroma_db`
- [ ] Test: ingest a set of MATH 239 notes, verify chunks look sane

### Phase 2 — Retrieval + LLM chain (Day 1 afternoon)
- [ ] Write `retriever.py`:
  - Load persisted Chroma store
  - Set up `ConversationalRetrievalChain` with `gpt-4o-mini`
  - Return both answer and `source_documents`
  - Add `ConversationBufferMemory` for chat history
- [ ] Test: query from CLI, verify answers are grounded in notes and not hallucinated
- [ ] Verify source documents contain correct page/filename metadata

### Phase 3 — Source highlighting (Day 1 evening)
- [ ] Write `highlight.py`:
  - Format retrieved chunks for display: filename, page number, excerpt (first 300 chars)
  - Deduplicate chunks from same page
- [ ] Design the UI pattern: answer first, then collapsible "Sources" expander showing each chunk
- [ ] Test with a question that spans multiple source chunks

### Phase 4 — Streamlit UI (Day 2 morning)
- [ ] Write `app.py`:
  - Sidebar: PDF uploader, "Ingest" button, course filter dropdown
  - Main panel: `st.chat_message` conversation history
  - Below each assistant message: `st.expander("Sources")` with highlighted chunks
  - Session state for chat history and memory
- [ ] Handle edge cases: no documents ingested yet, query returns no relevant chunks
- [ ] Add a confidence/relevance indicator (retrieval score from Chroma) next to each source

### Phase 5 — Polish + README (Day 2 afternoon)
- [ ] Record a demo GIF (use a real course — combinatorics or linear algebra notes work well)
- [ ] Write `.env.example` with `OPENAI_API_KEY=your_key_here`
- [ ] Add `.gitignore`: `chroma_db/`, `data/notes/`, `.env`
- [ ] Clean up README, add architecture diagram, fill in demo section
- [ ] Push to GitHub

---

## Stack

| Component | Library | Why |
|---|---|---|
| PDF loading | `pymupdf` (`fitz`) | Fast, handles complex layouts, returns page numbers |
| Text splitting | `langchain` `RecursiveCharacterTextSplitter` | Overlap preserves context at chunk boundaries |
| Embeddings | `text-embedding-3-small` (OpenAI) | Cheap ($0.02/1M tokens), high quality |
| Vector store | `chromadb` | Local, persistent, no setup, returns similarity scores |
| LLM | `gpt-4o-mini` | Fast, cheap, sufficient for retrieval-grounded answers |
| Chain | `ConversationalRetrievalChain` (LangChain) | Built-in memory + source document passthrough |
| UI | `streamlit` | Single file, chat primitives built in, fast to iterate |

---

## Learning highlights

This project covers the core AI engineering skills used in production RAG systems:

**Retrieval-Augmented Generation**
Understanding why RAG outperforms pure LLM generation for domain-specific knowledge: the model is constrained to retrieved context, reducing hallucination and making answers verifiable. The tradeoff between chunk size (larger = more context per chunk, noisier retrieval) and overlap (more overlap = better cross-boundary coherence, more storage) is a real engineering decision.

**Vector databases and embeddings**
Embeddings are dense vector representations of text where semantic similarity maps to geometric proximity. ChromaDB stores these vectors and retrieves the top-k nearest neighbours to a query embedding using cosine similarity. The choice of embedding model affects retrieval quality independently of the LLM — a weak embedding model will retrieve irrelevant chunks even if the LLM is strong.

**Conversational memory**
`ConversationBufferMemory` maintains a running history of the conversation and injects it into each new query. Without this, "what did you mean by that?" has no referent. The tradeoff: longer history = more tokens per call = higher cost and latency.

**Source attribution as explainability**
Showing source chunks is the RAG equivalent of SHAP/LIME for ML models — it makes the system's reasoning transparent and auditable. In a scientific context (the motivation for this project), a researcher needs to know *why* the model gave an answer, not just what the answer is. This is also a trust and safety mechanism: if the source chunk doesn't support the answer, the chain has hallucinated despite the retrieval constraint.

**Chunking strategy**
`RecursiveCharacterTextSplitter` splits on paragraph → sentence → word boundaries in order, preserving semantic units. The `chunk_overlap` parameter ensures that a concept split across two chunks is represented in both, so retrieval doesn't miss it depending on where the query lands.

---

## Potential extensions

- **Reranking** — use a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM`) to rerank retrieved chunks before passing to LLM, improving answer quality
- **Hybrid search** — combine dense (embedding) retrieval with sparse (BM25 keyword) retrieval for better coverage on exact-match queries like equation names
- **Course filter** — tag chunks by course at ingest time, filter Chroma queries by metadata so "explain the binomial theorem" only searches MATH 239 notes
- **Eval harness** — generate a small set of question/answer pairs from known notes, measure retrieval recall and answer faithfulness automatically
- **Docker** — containerize with a `Dockerfile` for reproducible deployment (directly relevant to the AC posting's Docker nice-to-have)

---

## Setup

```bash
git clone https://github.com/yourusername/coursenotes-rag
cd coursenotes-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add your OPENAI_API_KEY to .env
streamlit run app.py
```

---

## Requirements

```
langchain
langchain-openai
langchain-community
chromadb
pymupdf
streamlit
python-dotenv
openai
```

---

*Built as a portfolio project exploring RAG architecture, vector retrieval, and explainability in AI systems.*
