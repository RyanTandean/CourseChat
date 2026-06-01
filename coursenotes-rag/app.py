import streamlit as st
import os
from rag.retriever import build_chain
from rag.ingest import ingest_pdf
from rag.history import load_history, save_history, clear_history, list_courses
import chromadb
import tempfile
import re

def strip_markdown(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{2,}', ' ', text)
    return text.strip()

st.set_page_config(
    page_title="CourseChat",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2d3148; }
    [data-testid="stChatMessage"] { background-color: #1a1d27; border-radius: 8px; margin-bottom: 8px; }
    [data-testid="stExpander"] { background-color: #12151f; border: 1px solid #2d3148; border-radius: 6px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* welcome screen */
    .welcome-container {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; height: 65vh; gap: 12px;
    }
    .welcome-title { font-size: 1.8rem; font-weight: 600; color: #e0e0e0; margin: 0; }
    .welcome-sub { color: #666; font-size: 0.95rem; margin: 0; }
    .suggestion-row { display: flex; gap: 10px; margin-top: 8px; flex-wrap: wrap; justify-content: center; }
    .suggestion-chip {
        background: #1a1d27; border: 1px solid #2d3148; border-radius: 20px;
        padding: 8px 16px; color: #aaa; font-size: 0.85rem; cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

CHROMA_PATH = "./chroma_db"

# ── session state ───────────────────────────────────────────────────────────

if "active_course" not in st.session_state:
    st.session_state.active_course = None
if "agents" not in st.session_state:
    st.session_state.agents = {}
if "courses" not in st.session_state:
    st.session_state.courses = list_courses()

# ── helpers ─────────────────────────────────────────────────────────────────

def get_agent(course_name: str):
    if course_name not in st.session_state.agents:
        agent, db = build_chain(collection_name=course_name)
        st.session_state.agents[course_name] = (agent, db)
    return st.session_state.agents[course_name]

def switch_course(course_name: str):
    st.session_state.active_course = course_name

def delete_course(course_name: str):
    clear_history(course_name)
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(course_name.replace(" ", "_"))
    except Exception:
        pass
    if course_name in st.session_state.agents:
        del st.session_state.agents[course_name]
    st.session_state.courses = [c for c in st.session_state.courses if c != course_name]
    if st.session_state.active_course == course_name:
        st.session_state.active_course = None

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## CourseChat")
    st.divider()

    st.markdown("**New Course**")
    new_course = st.text_input("Course name", placeholder="e.g. CS241", label_visibility="collapsed")
    uploaded_files = st.file_uploader(
        "Upload notes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("Upload & Index", use_container_width=True, type="primary"):
        if not new_course:
            st.error("Enter a course name first.")
        elif not uploaded_files:
            st.error("Upload at least one PDF.")
        else:
            with st.spinner(f"Indexing {len(uploaded_files)} file(s)..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    ingest_pdf(tmp_path, collection_name=new_course, original_filename=uploaded_file.name)
                    os.unlink(tmp_path)

            if new_course not in st.session_state.courses:
                st.session_state.courses.append(new_course)
                save_history(new_course, [])

            st.success(f"Ready — {new_course} indexed.")
            switch_course(new_course)
            st.rerun()

    st.divider()

    st.markdown("**Courses**")
    if not st.session_state.courses:
        st.caption("No courses yet.")
    else:
        for course in st.session_state.courses:
            col1, col2 = st.columns([4, 1])
            with col1:
                is_active = st.session_state.active_course == course
                if st.button(
                    course,
                    key=f"course_{course}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    switch_course(course)
                    st.rerun()
            with col2:
                if st.button("✕", key=f"delete_{course}", help=f"Delete {course}"):
                    delete_course(course)
                    st.rerun()

    if st.session_state.active_course:
        st.divider()
        with st.expander("Settings"):
            if st.button("Clear conversation", use_container_width=True):
                clear_history(st.session_state.active_course)
                st.rerun()

# ── main panel ───────────────────────────────────────────────────────────────

if st.session_state.active_course is None:
    st.markdown("""
    <div class='welcome-container'>
        <p class='welcome-title'>What do you want to study?</p>
        <p class='welcome-sub'>Select a course from the sidebar or add a new one to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    course = st.session_state.active_course
    history = load_history(course)

    # empty conversation state — show centered prompt like Claude
    if not history:
        st.markdown(f"""
        <div class='welcome-container'>
            <p class='welcome-title'>Ask me anything about {course}</p>
            <p class='welcome-sub'>Your questions are grounded in your uploaded notes.</p>
            <div class='suggestion-row'>
                <span class='suggestion-chip'>Summarize key concepts</span>
                <span class='suggestion-chip'>Explain a definition</span>
                <span class='suggestion-chip'>Walk me through an example</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # render existing messages
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['source']}** — page {src['page']} — *{src['section']}*")
                        if src.get("excerpt"):
                            st.caption(strip_markdown(src["excerpt"]))
                        st.divider()

    # chat input
    if prompt := st.chat_input(f"Ask about {course}..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        agent_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ] + [{"role": "user", "content": prompt}]

        agent, db = get_agent(course)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            final_state = None

            for step in agent.stream(
                {"messages": agent_messages},
                stream_mode="values"
            ):
                final_state = step
                last_msg = step["messages"][-1]
                if last_msg.type == "ai" and not last_msg.tool_calls:
                    full_response = last_msg.content
                    response_placeholder.markdown(full_response)

            sources = []
            if final_state:
                for msg in final_state["messages"]:
                    if msg.type == "tool" and hasattr(msg, "artifact") and msg.artifact:
                        seen = set()
                        for doc in msg.artifact:
                            key = (doc.metadata.get("source"), doc.metadata.get("page"))
                            if key not in seen:
                                seen.add(key)
                                sources.append({
                                    "source": doc.metadata.get("source", "unknown"),
                                    "page": doc.metadata.get("page", "?"),
                                    "section": doc.metadata.get("section", ""),
                                    "excerpt": doc.page_content[:300].strip()
                                })
                        break

            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"**{src['source']}** — page {src['page']} — *{src['section']}*")
                        st.caption(strip_markdown(src["excerpt"]))
                        st.divider()

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": full_response, "sources": sources})
        save_history(course, history)