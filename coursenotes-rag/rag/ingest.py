import pymupdf4llm
# PyMuPDF is a python lib for data extraction of PDF and this is a wrapper that outputs markdown,
# so when chunks get passed to the LLM, it has structural context, like markdown tables instead of just raw text

import fitz
# fitz is the actual PyMuPDF library (pymupdf4llm wraps it).
# we use it directly here to extract embedded images from PDF pages —
# something pymupdf4llm deliberately skips, replacing them with placeholder text instead.
# fitz gives us low-level access: open a page, list its images, extract raw bytes.

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
# and pay nothing for embeddings — only the LLM calls cost money.

from groq import Groq
# same Groq client used in retriever.py for the LLM.
# here we use it for vision: the llama-4-scout model accepts base64-encoded images
# alongside text prompts, letting us describe math equations extracted from PDFs.

from dotenv import load_dotenv
import os
import re
import base64
import time

load_dotenv()
# authenticates with HuggingFace Hub to get higher rate limits when downloading the embedding model.
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN", "")

CHROMA_PATH = "./chroma_db"
NOTES_PATH  = "./data/notes"

# ── Page image storage ────────────────────────────────────────────────────────
#
# At query time, the sources panel can render the actual PDF page the retrieved
# chunk came from. To do this we need the original PDF available on disk.
#
# Option A (current): save the uploaded PDF to data/notes/ at ingest time and
# render the specific page with fitz at query time. Simple, works locally.
#
# Option B (production): render all pages to PNGs at ingest time and store them.
# Avoids fitz dependency at query time but uses more disk space.
#
# For deployment either option requires object storage (S3, GCS, R2) since
# ephemeral filesystems (Render, Streamlit Cloud) wipe local files on restart.
# Migration path: swap open(filepath) for s3.get_object(Key=filepath) and
# store the S3 key in ChromaDB metadata instead of the local path.
PDF_STORAGE_PATH = "./data/notes"

# ── Vision model config ───────────────────────────────────────────────────────
#
# meta-llama/llama-4-scout-17b-16e-instruct is Groq's current vision model.
# It accepts messages with image_url content blocks containing base64-encoded images.
# We use it to transcribe math equations that pymupdf4llm drops as placeholders.
#
# VISION_SLEEP_SECONDS: delay between consecutive vision API calls.
# Groq's free tier allows ~30 requests/min on vision models (~2s between calls).
# Increase this if you hit rate limit errors during ingestion of image-heavy PDFs.
VISION_MODEL         = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_SLEEP_SECONDS = 2

# placeholder pattern that pymupdf4llm inserts whenever it skips an embedded image.
# format: ==> picture [W x H] intentionally omitted <==
# we use this to find how many images were dropped per page and where to splice in
# the vision model's descriptions.
PLACEHOLDER_PATTERN = r"==> picture \[\d+ x \d+\] intentionally omitted <=="

# load the local embedding model. on first run this downloads ~90MB from HuggingFace.
# subsequent runs load from cache instantly.
# all-MiniLM-L6-v2 is a good balance of speed and quality for semantic search —
# small enough to run fast on CPU, accurate enough for retrieval tasks.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# single Groq client instance reused across all vision calls in this process.
# picks up GROQ_API_KEY from the .env file automatically.
groq_client = Groq()


# ── Vision helper ─────────────────────────────────────────────────────────────

def describe_page_images(page_text: str, fitz_page: fitz.Page, fitz_doc: fitz.Document) -> str:
    """Replace pymupdf4llm image placeholders with vision model descriptions.

    pymupdf4llm skips embedded images and writes a placeholder line:
        ==> picture [W x H] intentionally omitted <==

    This function:
        1. Counts how many placeholders are on this page
        2. Extracts the corresponding embedded images from the PDF page using fitz
        3. Sends each image to the Groq vision model with a math-focused prompt
        4. Replaces each placeholder with the model's LaTeX transcription or description
        5. Returns the enriched page text

    If a vision call fails (rate limit, bad image, network error), the original
    placeholder is left in place rather than crashing the whole ingest — silent
    degradation is better than losing the entire document.

    Args:
        page_text:  the markdown string for this page from pymupdf4llm
        fitz_page:  the corresponding fitz.Page object for raw image access
        fitz_doc:   the open fitz.Document, needed to extract image bytes by xref

    Returns:
        page_text with placeholders replaced (or unchanged if no images / all calls failed)
    """
    # count placeholders — tells us how many images we need to process on this page
    placeholders = re.findall(PLACEHOLDER_PATTERN, page_text)
    if not placeholders:
        # no images on this page, nothing to do
        return page_text

    # get_images(full=True) returns a list of tuples for every image on the page.
    # each tuple is: (xref, smask, width, height, bpc, colorspace, ...)
    # xref is the PDF object reference number — we use it to extract raw bytes.
    # full=True includes indirect image references (images referenced via XObject)
    image_list = fitz_page.get_images(full=True)

    if not image_list:
        # pymupdf4llm found images (hence placeholders) but fitz found none —
        # this can happen with certain PDF encodings. leave placeholders as-is.
        print(f"    [vision] page has {len(placeholders)} placeholder(s) but fitz found no images — skipping")
        return page_text

    # we match images to placeholders by index (first image → first placeholder, etc.)
    # if counts differ, we process up to min(images, placeholders) and leave the rest
    num_to_process = min(len(image_list), len(placeholders))
    if len(image_list) != len(placeholders):
        print(f"    [vision] placeholder count ({len(placeholders)}) != image count ({len(image_list)}) — processing {num_to_process}")

    enriched_text = page_text

    for i in range(num_to_process):
        xref = image_list[i][0]  # PDF object reference for this image

        try:
            # extract_image() returns a dict with:
            #   "image": raw bytes of the image
            #   "ext":   file extension string ("png", "jpeg", "jp2", etc.)
            #   "width", "height", "colorspace": image properties
            base_image  = fitz_doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext   = base_image["ext"]

            # Groq's vision API requires RGB or RGBA images.
            # PDFs sometimes embed CMYK images (common in print-oriented documents).
            # fitz can detect this via the colorspace name and convert to RGB pixmap.
            # n > 4 means CMYK or other non-RGB colorspace.
            pix = fitz.Pixmap(fitz_doc, xref)
            if pix.n > 4:
                # convert CMYK → RGB so the vision model can process it
                pix = fitz.Pixmap(fitz.csRGB, pix)
                image_bytes = pix.tobytes("png")
                image_ext   = "png"

            # base64-encode the image bytes for the Groq API.
            # the API expects a data URI: "data:<mime>;base64,<encoded>"
            image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
            mime_type = f"image/{image_ext}"

            print(f"    [vision] calling vision model for image {i+1}/{num_to_process} ({image_ext}, {len(image_bytes)//1024}KB)...")

            # vision API call — same Groq client as the LLM but with an image content block.
            # the prompt is tuned for academic math content:
            #   - asks for LaTeX using \(...\) / \[...\] delimiters so normalise_latex()
            #     in app.py will render them correctly (same delimiter convention as the LLM answers)
            #   - falls back to plain text description for diagrams/graphs
            #   - "return only" prevents the model from adding preamble like "Sure, here is..."
            response = groq_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "This image is from a university mathematics course. "
                                "If it contains a mathematical expression, equation, or formula, "
                                "transcribe it exactly as LaTeX. "
                                "Use \\(...\\) for inline expressions and \\[...\\] for display equations. "
                                "If it is a diagram, graph, or table, describe it concisely in plain text. "
                                "Return only the LaTeX or description — no preamble, no explanation."
                            )
                        }
                    ]
                }],
                temperature=0  # deterministic output for consistent indexing
            )

            description = response.choices[0].message.content.strip()
            print(f"    [vision] got: {description[:80]}{'...' if len(description) > 80 else ''}")

            # replace the FIRST remaining placeholder in the text with this description.
            # re.sub with count=1 replaces only the leftmost match, so processing
            # left-to-right correctly maps image[0]→placeholder[0], image[1]→placeholder[1].
            enriched_text = re.sub(PLACEHOLDER_PATTERN, description, enriched_text, count=1)

        except Exception as e:
            # any failure (rate limit, bad image bytes, network error) is caught here.
            # we print a warning and leave the placeholder for this image unchanged.
            # the loop continues to process remaining images on this page.
            print(f"    [vision] WARNING: failed to describe image {i+1}: {e}")
            # skip this placeholder by advancing past it — replace with a labelled fallback
            # so it's clear in the index that a vision call was attempted but failed
            enriched_text = re.sub(
                PLACEHOLDER_PATTERN,
                "[image: vision description unavailable]",
                enriched_text,
                count=1
            )

        # sleep between vision calls to stay within Groq's free tier rate limit.
        # only sleep if there are more images to process after this one.
        if i < num_to_process - 1:
            time.sleep(VISION_SLEEP_SECONDS)

    return enriched_text


# ── Main ingestion function ───────────────────────────────────────────────────

def ingest_pdf(filepath: str, collection_name: str = "course_notes", original_filename: str = None, use_vision: bool = True) -> tuple[object, bool, str | None]:
    """Extract, enrich, chunk, embed and store a PDF in ChromaDB.

    Pipeline:
        1. Save the PDF to data/notes/ for later page image rendering
        2. pymupdf4llm extracts text as markdown, page by page
        3. (if use_vision=True) fitz + Groq vision fills in image placeholders
           with LaTeX transcriptions or plain-text descriptions
        4. LangChain RecursiveCharacterTextSplitter chunks each page
        5. HuggingFace all-MiniLM-L6-v2 embeds each chunk
        6. ChromaDB stores vectors + text + metadata

    Args:
        filepath:         path to the PDF file on disk (temp file from upload)
        collection_name:  ChromaDB collection to store chunks in (one per course)
        original_filename: display name stored in metadata (shown in sources panel)
        use_vision:       if True, call the Groq vision model to describe embedded
                          images. Adds ~2s per image. Set False for non-math PDFs.

    Returns:
        (db, was_skipped, saved_pdf_path):
            db             — the Chroma collection object
            was_skipped    — True if the file was already indexed (dedup check)
            saved_pdf_path — path where the PDF was saved for page rendering,
                             or None if the file was skipped (already existed)
    """
    # pymupdf4llm extracts the PDF page by page and returns a list of dicts.
    # each dict has:
    #       'text' (markdown string),
    #       'metadata' (page number, filename, etc.),
    #       'toc_items' (table of contents entries on this page),
    #       'page_boxes' (layout info)
    pages = pymupdf4llm.to_markdown(filepath, page_chunks=True)

    # the filename we'll store in metadata and use as the deduplication key
    source_name = original_filename or os.path.basename(filepath)

    # ── Save PDF for page rendering ───────────────────────────────────────────
    #
    # The sources panel in app.py renders the actual PDF page that a retrieved
    # chunk came from, using fitz at query time. This requires the PDF to be
    # available on disk — so we copy it to a stable location before deleting
    # the temp file that app.py created for the upload.
    #
    # Stored at: data/notes/<source_name>
    # app.py uses source metadata (stored in every chunk) to find this file.
    #
    # Deployment note: on ephemeral filesystems (Render, Streamlit Cloud) this
    # file won't survive a restart. Production fix: upload to S3/GCS instead and
    # store the object key in ChromaDB metadata.
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
    saved_pdf_path = os.path.join(PDF_STORAGE_PATH, source_name)
    if not os.path.exists(saved_pdf_path):
        # only copy if not already there — avoids overwriting on re-ingest attempts
        import shutil
        shutil.copy2(filepath, saved_pdf_path)

    # ── Deduplication check ──────────────────────────────────────────────────
    #
    # ChromaDB has no built-in deduplication — calling from_documents() twice
    # on the same file silently doubles every chunk in the collection. This causes:
    #   - duplicate source entries in the UI sources panel
    #   - retrieval returning the same chunk twice, wasting the k=6 budget
    #   - inflated similarity scores since duplicates vote for each other
    #
    # Fix: before ingesting, load the existing collection and query for any chunk
    # whose metadata 'source' field matches this filename. If any exist, this file
    # has already been indexed — skip it and return the existing collection instead.
    #
    # Why metadata filter instead of hashing the file content?
    # A content hash would be more robust (catches renamed duplicates) but requires
    # storing the hash at ingest time and querying by it. The source filename is
    # already stored in metadata and is sufficient for the common case: a user
    # accidentally clicking "Upload & Index" twice on the same file.
    try:
        existing_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        existing = existing_db.get(where={"source": source_name}, limit=1)
        if existing["ids"]:
            print(f"Skipping {source_name} — already indexed in '{collection_name}'")
            return existing_db, True, saved_pdf_path  # True = was skipped (already existed)
    except Exception:
        # collection doesn't exist yet (first ingest for this course) — that's fine,
        # from_documents() below will create it. Any other error falls through
        # to a fresh ingest rather than silently failing.
        pass

    # ── Vision enrichment ────────────────────────────────────────────────────
    #
    # Open the PDF with fitz so we can extract raw image bytes page by page.
    # We keep this open for the duration of the enrichment loop, then close it.
    # fitz page numbers are 0-indexed; pymupdf4llm page numbers are 1-indexed.
    #
    # use_vision=False skips this entirely — useful for non-math PDFs (no equations
    # to extract), or when you need a fast ingest and don't want the ~2s-per-image
    # delay from Groq's rate limit throttle.
    if use_vision:
        print(f"Opening {source_name} for vision enrichment...")
        fitz_doc = fitz.open(filepath)

        enriched_pages = []
        for page in pages:
            page_number_1indexed = page["metadata"]["page_number"]
            page_text = page["text"]

            # check quickly if this page even has placeholders before touching fitz
            if re.search(PLACEHOLDER_PATTERN, page_text):
                fitz_page = fitz_doc[page_number_1indexed - 1]  # fitz is 0-indexed
                print(f"  page {page_number_1indexed}: found image placeholder(s), calling vision model...")
                page_text = describe_page_images(page_text, fitz_page, fitz_doc)

            enriched_pages.append({**page, "text": page_text})

        fitz_doc.close()
    else:
        # skip vision — use the raw pymupdf4llm output as-is.
        # image placeholders (==> picture ... intentionally omitted <==) will remain
        # in the chunks, which means equations in images won't be searchable.
        # acceptable tradeoff for non-math content or fast re-ingests.
        print(f"Vision enrichment disabled for {source_name} — skipping image extraction.")
        enriched_pages = pages

    # ── Build LangChain Documents ────────────────────────────────────────────
    #
    # wrap each enriched page into a Document object.
    # source_name is used for dedup (above) and for source highlighting in the UI.
    # section comes from the PDF's table of contents — lets us say
    # "this answer came from Section 2.3" in the sources panel.
    documents = [
        Document(
            page_content=page["text"],
            metadata={
                "source":  source_name,
                "page":    page["metadata"]["page_number"],
                "section": page["toc_items"][0][1] if page["toc_items"] else ""
            }
        )
        for page in enriched_pages
    ]

    # ── Chunking ─────────────────────────────────────────────────────────────
    #
    # split each page Document into smaller chunks.
    # 800 chars is roughly a paragraph — small enough for precise retrieval,
    # large enough to contain a complete thought.
    # 100 char overlap means a concept split across two chunks appears in both,
    # so retrieval doesn't miss it depending on where the query vector lands.
    #
    # NOTE: if you ever change chunking parameters (chunk_size, chunk_overlap),
    # delete chroma_db/ and re-ingest everything — mixing chunk sizes in the same
    # collection produces inconsistent retrieval.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # ── Embed and store ───────────────────────────────────────────────────────
    #
    # from_documents() does three things:
    #   1. calls embeddings.embed_documents() on each chunk's page_content
    #   2. stores the resulting vectors alongside the text and metadata in ChromaDB
    #   3. persists everything to disk at CHROMA_PATH
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=collection_name
    )

    print(f"Ingested {len(chunks)} chunks from {source_name} into '{collection_name}'")
    return db, False, saved_pdf_path  # False = freshly ingested (not skipped)


if __name__ == "__main__":
    # scan data/notes/ for PDFs and ingest each one.
    for filename in os.listdir(NOTES_PATH):
        if filename.endswith(".pdf"):
            ingest_pdf(os.path.join(NOTES_PATH, filename))
