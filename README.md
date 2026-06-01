# CourseChat — RAG Study Assistant

A retrieval-augmented generation (RAG) chatbot that answers questions grounded in your own course notes, with source highlighting so you can see exactly where every answer comes from.

Built with LangChain, ChromaDB, HuggingFace, Groq, and Streamlit.

---

## Demo

<img width="1901" height="880" alt="image" src="https://github.com/user-attachments/assets/ef3642b1-6b1b-44ac-9e16-397fa446e472" />


---

## Rationale
The primary driver behind this mini-project was an experimental deep-dive to understand the mechanics of LLMs, orchestration frameworks (LangChain), and Retrieval-Augmented Generation (RAG).

While models like GPT, Claude, and Gemini possess immense general knowledge, this breadth becomes a liability in structured academic environments. 
University courses sometimes introduce complex subjects by deliberately scoping down concepts and utilizing specific terminology that might not be reflective of terminilogy used in the real world.

When a student asks a general LLM for help with a specialized course concept, the model often pulls in external methodologies, advanced formulations, or conflicting definitions. 
Instead of helping, this extra "out-of-scope" data introduces noise, breaks the instructor's intended learning progression, and ultimately accelerates student confusion.

As such, this experiment consists of a chatbot that enforces an absolute boundary on information retrieval by grounding the LLM strictly within the official course notes.
This ensures that:

- The model only explains concepts using the specific definitions and boundaries established by the course curriculum.
- Explanations are consistent with what students see on exams and assignments
- Hallucinations are minimized, by restricting the context window strictly to lecture materials


---

## Features

- **PDF ingestion** — upload one or more PDFs per course (lecture notes, textbook chapters, problem sets)
- **Markdown-aware extraction** — uses `pymupdf4llm` to preserve tables, headers, and math notation as structured markdown rather than raw text
- **Semantic chunking** — splits documents into overlapping chunks to preserve context across page boundaries
- **Local embeddings** — `all-MiniLM-L6-v2` via HuggingFace, runs entirely on CPU with no API cost
- **ChromaDB vector store** — named collections per course, persists across sessions
- **Tool-based retrieval** — agent decides when to call the retrieval tool, avoiding retrieval on greetings and non-course questions
- **Persistent conversation history** — per-course chat history saved to disk as JSON, survives browser close and app restart
- **Source highlighting** — every answer shows the source file, page number, section, and excerpt it was retrieved from
- **Multi-course support** — isolated ChromaDB collections per course, switch between courses like switching tabs

---

## Architecture
