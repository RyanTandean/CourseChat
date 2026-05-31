

import pymupdf4llm 
# PyMuPDF is a python lib for data extraction of PDF and this is a wrapper that outputs markdown, 
# so when chunks get passed to the LLM, it has structual context, like markdown tables instead of just raw text
from langchain_core.documents import Document
# Document is the Langchain's std container for text + metadata 
# every chunk that goes into ChromaDB is a Document with two fields:
#   - page_content: the actual text
#   - metadata: a dict of arbitrary info (source filename, page number, etc.)
# metadata is what lets us later say "this answer came from page 7 of module4CS241.pdf"
# https://reference.langchain.com/python/langchain-core/documents/base/Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# splits long Documents into smaller chunks for retrieval, as original PDFs can be very long and exceed token limits.
# "Recursive" since it tries to split on paragraphs first, then sentences, then words, to preserve context as much as possible.
# we pass chunk_size=800 (max chars per chunk) and chunk_overlap=100
# (consecutive chunks share 100 chars so concepts at boundaries appear in both)

from langchain_chroma import Chroma
# LangChain's wrapper around ChromaDB.
# ChromaDB is a vector database where it stores text chunks alongside their embedding vectors
# and lets us query "which stored vectors are closest to this query vector?"
# using cosine similarity. persists to disk so we don't re-ingest on every run.

# Cosine similarity: a measure of similarity between two vectors that calculates the cosine of the angle between them.
# Indifferent to magnitude, focuses on direction

from langchain_huggingface import HuggingFaceEmbeddings
# loads a local sentence embedding model (all-MiniLM-L6-v2, 384 dimensions).
# an embedding model converts text into a dense vector of numbers where
# semantically similar text ends up geometrically close in vector space.
# "what is a CFG" and "define context-free grammar" will produce similar vectors
# even though they share no words. this is what enables semantic search.
# we use a local HuggingFace model here instead of OpenAI embeddings so we need no API key
# and pay nothing for embeddings only the LLM calls cost money.

from dotenv import load_dotenv
import os

load_dotenv()
# authenticates with HuggingFace Hub to get higher rate limits when downloading the embedding model.
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

CHROMA_PATH = "./chroma_db"
NOTES_PATH = "./data/notes"

# load the local embedding model. on first run this downloads ~90MB from HuggingFace.
# subsequent runs load from cache instantly.
# all-MiniLM-L6-v2 is a good balance of speed and quality for semantic search —
# small enough to run fast on CPU, accurate enough for retrieval tasks.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_pdf(filepath: str, collection_name: str = "course_notes"):
    # pymupdf4llm extracts the PDF page by page and returns a list of dicts.
    # page_chunks=True means one dict per page instead of one big string.
    # each dict has: 
    #       'text' (markdown string), 
    #       'metadata' (page number, filename, etc.),
    #       'toc_items' (table of contents entries on this page), 
    #       'page_boxes' (layout info)
    pages = pymupdf4llm.to_markdown(filepath, page_chunks=True)
    
    # wrap each page dict into a LangChain Document object.
    # we only keep 'text' and two metadata fields we actually need for source highlighting:
    # source (filename) and page (page number).
    # + toc_items, which is the sections hierarchy of the PDF, so we can say "this answer came from the Section 2.3"
    # each page now has a list of entries of [level, title, page number]

    # Don't need page boxes
    # everything else from pymupdf4llm's metadata dict is discarded.
    documents = [
        Document(
            page_content=page["text"],
            metadata={
                "source": os.path.basename(filepath),
                "page": page["metadata"]["page_number"],
                "section": page["toc_items"][0][1] if page["toc_items"] else ""
            }
        )
        for page in pages
    ]
    # split each page Document into smaller chunks.
    # 800 chars is roughly a paragraph which is small enough for precise retrieval,
    # large enough to contain a complete thought.
    # 100 char overlap means a concept split across two chunks appears in both,
    # so retrieval doesn't miss it depending on where the query vector lands.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    # splits a page Document into multiple chunk Documents, each with the same metadata as the parent page.
    # NOTE: if you run ingest.py twice on the same PDF, ChromaDB will store duplicate chunks. It has no deduplication by default.
    # This might lead to repeated answers, so the fix is to delete chroma_db/ before re-ingesting, 
    # which you should do any time you change the chunking parameters anyway since old and new chunks would be inconsistent.
    chunks = splitter.split_documents(documents)

    # chunks is now a list of Documents, each with page_content <= 800 chars
    # and metadata inherited from the parent Document (source + page number)

    # embed all chunks and store them in ChromaDB.
    # from_documents() does three things:
    #   1. calls embeddings.embed_documents() on each chunk's page_content
    #   2. stores the resulting vectors alongside the text and metadata in ChromaDB
    #   3. persists everything to disk at CHROMA_PATH
    # after this, the chroma_db/ folder contains the full searchable index.
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=collection_name # new, organizes vectors in ChromaDB into named collections
    )

    print(f"Ingested {len(chunks)} chunks from {filepath}")
    return db

if __name__ == "__main__":
    # scan data/notes/ for PDFs and ingest each one.
    # running this multiple times will add duplicate chunks to ChromaDB —
    # for now that's fine, but a production system would check if a file
    # was already ingested before re-processing it.
    for filename in os.listdir(NOTES_PATH):
        if filename.endswith(".pdf"):
            ingest_pdf(os.path.join(NOTES_PATH, filename))