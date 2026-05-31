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

CONVERSATIONS_PATH = "./data/conversations"

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
    with open(path, "r") as f:
        return json.load(f)

def save_history(course_name: str, messages: list[dict]) -> None:
    # persist conversation history to disk after every message
    # creates the conversations directory if it doesn't exist yet
    os.makedirs(CONVERSATIONS_PATH, exist_ok=True)
    path = _course_path(course_name)
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)

def clear_history(course_name: str) -> None:
    # delete the history file for a course — used by the "clear conversation" button
    # resets the context window so the agent starts fresh
    path = _course_path(course_name)
    if os.path.exists(path):
        os.remove(path)

def list_courses() -> list[str]:
    # scan the conversations directory to find all courses with saved history
    # used to populate the sidebar on startup
    # returns course names with underscores converted back to spaces
    if not os.path.exists(CONVERSATIONS_PATH):
        return []
    return [
        f.replace(".json", "").replace("_", " ")
        for f in os.listdir(CONVERSATIONS_PATH)
        if f.endswith(".json")
    ]