############# NO LONGER NEEDED #######
# Was created because there was no way to get the source documents: 
#   Docs were used to build the system prompt, then discarded
#   This file was then going to be used to rerun the same similarity search (redundant, cheap anyways but still)
#   and return the sources alongside the answer in the UI.

# LangChain actually has a class-based middleware approach (AgentMiddleware) that stores
# retrieved docs directly in agent state under state["context"], so they're accessible
# after the call without re-querying ChromaDB.





from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Source highlighting works by re-running the same similarity search
# that the middleware ran, and returning the source Documents so the UI
# can display where each answer came from.
# We can't get them directly from the agent response since dynamic_prompt
# middleware doesn't expose the retrieved docs — so we query ChromaDB again (twice per call).
# This is a second identical query, cheap since embeddings are local.

def get_sources(query: str, db: Chroma, k: int = 4) -> list[dict]:
    # run the same similarity search as the middleware
    # k=4 means we get the same 4 chunks that were injected into the prompt as context, so the sources we show match the retrieved context the model saw when answering.
    retrieved_docs = db.similarity_search(query, k=k)

    sources = []
    seen = set()  # deduplicate chunks from the same page
    
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")
        
        # use (source, page) as a unique key — don't show the same page twice
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        
        sources.append({
            "source": source,
            "page": page,
            "section": section,
            # first 300 chars of the chunk as a preview excerpt
            "excerpt": doc.page_content[:300].strip()
        })
    
    return sources