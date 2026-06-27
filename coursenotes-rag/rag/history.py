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
import psycopg
from dotenv import load_dotenv
from rag.config import CONVERSATIONS_PATH

load_dotenv()
CONNECTION_STRING = os.getenv("SUPABASE_DB_URL", "")

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
    """Return the distinct filenames indexed in a course's collection.

    Queries langchain_pg_embedding for distinct cmetadata->>'source' values
    belonging to the given collection. cmetadata is a jsonb column so we use
    the ->> operator to extract the 'source' key as text.
    """
    try:
        conn = psycopg.connect(CONNECTION_STRING)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT e.cmetadata->>'source'
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s
            """,
            (course_name,)
        )
        sources = [row[0] for row in cur.fetchall() if row[0]]
        cur.close()
        conn.close()
        return sorted(sources)
    except Exception as e:
        print(f"[list_files] query failed: {e}")
        return []


def delete_file(course_name: str, filename: str) -> None:
    """Delete all chunks belonging to a single PDF from a course's collection.

    Deletes rows from langchain_pg_embedding where the collection matches
    and cmetadata->>'source' matches the filename. Leaves all other files
    in the collection untouched.

    Also removes the saved PDF from data/notes/ so the sources panel doesn't
    try to render pages from a file that's no longer indexed.
    """
    NOTES_PATH = "./data/notes"

    try:
        conn = psycopg.connect(CONNECTION_STRING)
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM langchain_pg_embedding e
            USING langchain_pg_collection c
            WHERE e.collection_id = c.uuid
              AND c.name = %s
              AND e.cmetadata->>'source' = %s
            """,
            (course_name, filename)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[delete_file] SQL delete failed for {filename}: {e}")

    # remove the saved PDF so page rendering doesn't find a stale file
    pdf_path = os.path.join(NOTES_PATH, filename)
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception as e:
            print(f"[delete_file] could not remove PDF {pdf_path}: {e}")


def list_courses() -> list[str]:
    # langchain_pg_collection is the source of truth — one row per course.
    # A course exists if and only if it has a collection in Supabase.
    # The JSON history file is optional — missing just means empty conversation.
    try:
        conn = psycopg.connect(CONNECTION_STRING)
        cur = conn.cursor()
        cur.execute("SELECT name FROM langchain_pg_collection")
        courses = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return courses
    except Exception as e:
        print(f"[list_courses] query failed: {e}")
        return []
