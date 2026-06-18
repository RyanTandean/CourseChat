# history.py
# 
# Why this exists:
# create_agent has no built-in memory — every agent.stream() call is stateless.
# Without this, the agent forgets everything between messages:
#   - "explain that differently" has no context of what "that" is referring to, so it fails
#   - "give me an example of what you just described" fails
#   - follow-up questions are impossible
#
# This module solves it by maintaining conversation history manually:
#   - messages are stored as a list of {role, content} dicts per course
#   - persisted to disk as JSON so history survives browser close and app restart
#   - loaded back on startup and passed into every agent.stream() call
#   - each course has its own isolated history file

import json
import os
import chromadb
from rag.config import CHROMA_PATH, CONVERSATIONS_PATH

def _course_path(course_name: str) -> str:
    # sanitize course name to a safe filename
    # e.g. "CS 241" -> "CS_241.json"
    safe_name = course_name.strip().replace(" ", "_")
    return os.path.join(CONVERSATIONS_PATH, f"{safe_name}.json")

def load_history(course_name: str) -> list[dict]:
    # load conversation history for a course from disk
    # returns empty list if no history exists yet (first time opening a course)
    path = _course_path(course_name)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[load_history] could not parse {path}: {e}")
        return []

def save_history(course_name: str, messages: list[dict]) -> None:
    # persist conversation history to disk after every message
    # creates the conversations directory if it doesn't exist yet
    os.makedirs(CONVERSATIONS_PATH, exist_ok=True)
    path = _course_path(course_name)
    try:
        with open(path, "w") as f:
            json.dump(messages, f, indent=2)
    except OSError as e:
        print(f"[save_history] could not write {path}: {e}")

def clear_history(course_name: str) -> None:
    # delete the history file for a course — used by the "clear conversation" button
    # resets the context window so the agent starts fresh
    path = _course_path(course_name)
    if os.path.exists(path):
        os.remove(path)

def list_files(course_name: str) -> list[str]:
    """Return the distinct filenames indexed in a course's ChromaDB collection.

    Each chunk stored at ingest time has a 'source' metadata field set to the
    original PDF filename. We query the collection for all chunks, pull the
    'source' field from each, and deduplicate — giving us the list of PDFs
    currently indexed for this course.

    Returns an empty list if the collection doesn't exist or has no chunks.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(course_name.replace(" ", "_"))
        # get() with no filter returns all chunks; include=["metadatas"] skips
        # fetching embeddings and documents — we only need the metadata dicts
        result = collection.get(include=["metadatas"])
        sources = {m["source"] for m in result["metadatas"] if "source" in m}
        return sorted(sources)
    except Exception:
        return []


def delete_file(course_name: str, filename: str) -> None:
    """Delete all chunks belonging to a single PDF from a course's collection.

    Uses ChromaDB's delete() with a 'where' metadata filter on the 'source'
    field — only chunks from this file are removed, leaving all other files
    in the collection untouched.

    Also removes the saved PDF from data/notes/ so the sources panel doesn't
    try to render pages from a file that's no longer indexed.
    """
    CHROMA_PATH    = "./chroma_db"
    NOTES_PATH     = "./data/notes"

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(course_name.replace(" ", "_"))
        # delete all chunks whose 'source' metadata matches this filename
        collection.delete(where={"source": filename})
    except Exception as e:
        print(f"[delete_file] ChromaDB delete failed for {filename}: {e}")

    # remove the saved PDF so page rendering doesn't find a stale file
    pdf_path = os.path.join(NOTES_PATH, filename)
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception as e:
            print(f"[delete_file] could not remove PDF {pdf_path}: {e}")


def list_courses() -> list[str]:
    # ── Why ChromaDB is the source of truth, not the conversations folder ────
    #
    # Previously this function scanned data/conversations/ for JSON files.
    # That caused two silent bugs:
    #
    #   1. Ingest a course, don't chat yet → no JSON file created → course never
    #      appears in the sidebar even though vectors are fully indexed in ChromaDB
    #
    #   2. Delete or lose the JSON file (e.g. clearing the conversations folder)
    #      → course vanishes from the sidebar, but all its vectors are still in
    #      ChromaDB taking up space — orphaned and inaccessible
    #
    # The fix: use ChromaDB's collection list as the authoritative source.
    # A course exists if and only if it has a ChromaDB collection. The JSON history
    # file is optional — if it's missing we just start with an empty conversation.
    #
    # chromadb.PersistentClient loads the same on-disk database that ingest.py
    # writes to. client.list_collections() returns all named collections.
    # Each collection name was set to the course name at ingest time (with spaces
    # replaced by underscores to satisfy ChromaDB's naming rules — we reverse that
    # here to get the display name back).
    #
    # If ChromaDB doesn't exist yet (fresh install, no courses ingested) we return
    # an empty list rather than crashing.

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        # list_collections() returns a list of Collection objects
        # each has a .name attribute — the collection name set at ingest time
        collections = client.list_collections()
        # reverse the underscore substitution applied at ingest time so the
        # display name in the sidebar matches what the user originally typed
        return [col.name.replace("_", " ") for col in collections]
    except Exception:
        # ChromaDB directory doesn't exist yet or is unreadable — no courses yet
        return []