# eval.py — Retrieval Recall Evaluation Harness
#
# ── What this measures ────────────────────────────────────────────────────────
#
# RAG quality has two independent failure modes:
#
#   1. Retrieval failure — the right chunk was never returned by the similarity
#      search, so the LLM never had the information it needed. No matter how good
#      the LLM is, it can't answer from context it didn't receive.
#
#   2. Generation failure — the right chunk was retrieved but the LLM still gave
#      a wrong or incomplete answer.
#
# This script measures (1): RETRIEVAL RECALL.
# For each test question, it runs the same similarity search that the agent uses
# (ChromaDB, k=6, all-MiniLM-L6-v2 embeddings) and checks whether any of the
# returned chunks contains the expected keywords.
#
# Why retrieval recall specifically?
#   - It's the most important metric for RAG — retrieval quality is the bottleneck
#   - It requires no LLM API calls (local embeddings only, free and fast)
#   - It produces a deterministic, reproducible number you can report
#   - It catches chunking and embedding quality problems directly
#
# ── How to add test cases ─────────────────────────────────────────────────────
#
# Each test case is a dict with:
#   "question"  : the query string, written the way a student would ask it
#   "keywords"  : a list of strings that MUST appear (case-insensitive) in at
#                 least one retrieved chunk for the test to pass
#   "course"    : the ChromaDB collection name to search in (your course name)
#
# Keyword choice matters:
#   - Use specific terms from the notes, not generic words
#   - Multiple keywords in one test all must appear in the SAME chunk
#   - If a concept spans multiple chunks, split it into multiple test cases
#   - Aim for 10-20 tests covering different sections of the material
#
# ── Running ───────────────────────────────────────────────────────────────────
#
#   cd coursenotes-rag
#   python eval.py
#
# No API keys needed — uses local HuggingFace embeddings only.

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN", "")

CHROMA_PATH = "./chroma_db"

# ── same embedding model as ingest.py and retriever.py ───────────────────────
# MUST be identical — if the model differs, the query vectors live in a different
# vector space than the stored chunk vectors and cosine similarity breaks down.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── test cases ────────────────────────────────────────────────────────────────
# Written from Enumeration-1.pdf (the course notes currently indexed).
# Each "keywords" list contains terms that should appear verbatim in a retrieved
# chunk if the similarity search is working correctly for this question.
#
# To test a different course: change "course" to your collection name and
# replace the test cases with questions from that material.
TEST_CASES = [
    {
        "question": "What is a composition?",
        "keywords": ["composition", "positive integers"],
        "course": "CS241",
    },
    {
        "question": "What is the Product Lemma?",
        # the Product Lemma definition is largely in images in this PDF,
        # but surrounding text references it by name alongside these terms
        "keywords": ["Product Lemma"],
        "course": "CS241",
    },
    {
        "question": "What is the String Lemma?",
        "keywords": ["String Lemma"],
        "course": "CS241",
    },
    {
        "question": "What is the weight function used for in generating series?",
        # notes use "weight function" as a phrase, tested against generating series context
        "keywords": ["weight function"],
        "course": "CS241",
    },
    {
        "question": "What is a generating series?",
        # notes introduce generating series in the context of counting objects in a set
        "keywords": ["generating series", "count"],
        "course": "CS241",
    },
    {
        "question": "What does the Sum Lemma say?",
        "keywords": ["Sum Lemma", "disjoint"],
        "course": "CS241",
    },
    {
        "question": "What is a binary string?",
        "keywords": ["binary", "string"],
        "course": "CS241",
    },
    {
        "question": "How do you count k-element subsets?",
        # pymupdf4llm renders italic k as _k_ in markdown, so the chunk text
        # contains "_k_ -element subset" not "k-element subset"
        "keywords": ["element subset"],
        "course": "CS241",
    },
]

# ── evaluation logic ──────────────────────────────────────────────────────────

def chunk_contains_all_keywords(chunk_text: str, keywords: list[str]) -> bool:
    """Return True if the chunk contains ALL keywords (case-insensitive).

    Requiring all keywords to appear in the same chunk is stricter than
    checking across all chunks — it ensures the retrieved context is actually
    about the right concept rather than matching each keyword in a different,
    unrelated chunk.
    """
    text_lower = chunk_text.lower()
    return all(kw.lower() in text_lower for kw in keywords)


def evaluate(test_cases: list[dict], k: int = 6) -> None:
    """Run all test cases and print a retrieval recall report.

    Args:
        test_cases: list of test case dicts (see module docstring for schema)
        k: number of chunks to retrieve per query — should match the k used
           in retriever.py so the eval reflects real production retrieval
    """
    # group tests by course so we only load each ChromaDB collection once
    # loading a collection is cheap but avoids redundant disk reads
    courses = set(tc["course"] for tc in test_cases)
    dbs = {}
    for course in courses:
        try:
            dbs[course] = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings,
                collection_name=course
            )
        except Exception as e:
            print(f"ERROR: could not load collection '{course}': {e}")
            print("Make sure you have ingested notes for this course before running eval.")
            return

    passed = 0
    failed = 0
    results = []

    for tc in test_cases:
        question = tc["question"]
        keywords = tc["keywords"]
        course   = tc["course"]
        db       = dbs[course]

        # run the same similarity search the agent uses at query time
        # k=6 matches retriever.py — eval should reflect production conditions
        retrieved_docs = db.similarity_search(question, k=k)

        # a test PASSES if at least one of the k retrieved chunks contains
        # all the expected keywords — we don't require the best chunk to be
        # ranked first, just that it's somewhere in the top k
        hit = any(
            chunk_contains_all_keywords(doc.page_content, keywords)
            for doc in retrieved_docs
        )

        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1
        else:
            failed += 1

        results.append((status, question, keywords))

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RETRIEVAL RECALL EVALUATION")
    print("=" * 60)

    for status, question, keywords in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"\n  {icon} [{status}] {question}")
        if status == "FAIL":
            # print the keywords that weren't found so you know what to investigate
            print(f"        expected keywords: {keywords}")

    total = passed + failed
    recall = passed / total if total > 0 else 0.0

    print("\n" + "-" * 60)
    print(f"  Retrieval Recall:  {passed}/{total}  ({recall:.0%})")
    print("-" * 60)

    # ── interpretation guide ─────────────────────────────────────────────────
    # 100%  — retrieval is working well for this test set
    #  80%+ — acceptable for most use cases
    #  60%  — retrieval is struggling; consider:
    #            - smaller chunk_size (currently 800) for more precise matching
    #            - larger k (currently 6) to cast a wider net
    #            - better test case keywords (too generic = false negatives)
    # below 60% — systematic retrieval problem; check that the correct collection
    #             is being queried and that ingest ran without errors
    print()
    if recall == 1.0:
        print("  Retrieval is working well for this test set.")
    elif recall >= 0.8:
        print("  Retrieval is acceptable. Review FAILs above for gaps.")
    elif recall >= 0.6:
        print("  Retrieval is marginal. Consider tuning chunk_size or k.")
    else:
        print("  Retrieval is poor. Check ingestion and collection names.")
    print()


if __name__ == "__main__":
    evaluate(TEST_CASES)
