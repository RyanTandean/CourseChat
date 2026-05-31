import streamlit as st
import os
from rag.retriever import build_chain
from rag.ingest import ingest_pdf
from rag.history import load_history, save_history, clear_history, list_courses
import chromadb
import tempfile

# page config must be first streamlit call
st.set_page_config(
    page_title="CourseChat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# dark theme CSS
st.markdown("""
<style>
    /* main background */
    .stApp { background-color: #0f1117; }
    
    /* sidebar */
    [data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2d3148; }
    
    /* chat messages */
    [data-testid="stChatMessage"] { background-color: #1a1d27; border-radius: 8px; margin-bottom: 8px; }
    
    /* source expander */
    [data-testid="stExpander"] { background-color: #12151f; border: 1px solid #2d3148; border-radius: 6px; }
    
    /* course buttons in sidebar */
    .course-btn { 
        width: 100%; text-align: left; padding: 8px 12px;
        background: #1e2235; border: 1px solid #2d3148;
        border-radius: 6px; color: #e0e0e0; cursor: pointer;
        margin-bottom: 4px;
    }
    .course-btn.active { background: #2d3a6e; border-color: #4a5fd4; }
    
    /* hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

CHROMA_PATH = "./chroma_db"

# ── session state init ──────────────────────────────────────────────────────

if "active_course" not in st.session_state:
    st.session_state.active_course = None

if "agents" not in st.session_state:
    # cache agent instances per course so we don't rebuild on every rerun
    st.session_state.agents = {}

if "courses" not in st.session_state:
    # load existing courses from saved history files on startup
    st.session_state.courses = list_courses()

# ── helper functions ────────────────────────────────────────────────────────

def get_agent(course_name: str):
    # return cached agent or build a new one for this course
    if course_name not in st.session_state.agents:
        agent, db = build_chain(collection_name=course_name)
        st.session_state.agents[course_name] = (agent, db)
    return st.session_state.agents[course_name]

def switch_course(course_name: str):
    st.session_state.active_course = course_name

def delete_course(course_name: str):
    # clear history file
    clear_history(course_name)
    # delete ChromaDB collection
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(course_name.replace(" ", "_"))
    except Exception:
        pass
    # remove from session state
    if course_name in st.session_state.agents:
        del st.session_state.agents[course_name]
    st.session_state.courses = [c for c in st.session_state.courses if c != course_name]
    if st.session_state.active_course == course_name:
        st.session_state.active_course = None

# ── sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📚 CourseChat")
    st.divider()

    # new course section
    st.markdown("**Add Course**")
    new_course = st.text_input("Course name", placeholder="e.g. CS241", label_visibility="collapsed")
    uploaded_files = st.file_uploader(
        "Upload notes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("Ingest Notes", use_container_width=True, type="primary"):
        if not new_course:
            st.error("Enter a course name first.")
        elif not uploaded_files:
            st.error("Upload at least one PDF.")
        else:
            with st.spinner(f"Ingesting {len(uploaded_files)} file(s)..."):
                for uploaded_file in uploaded_files:
                    # save uploaded file to a temp path so ingest_pdf can read it
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    ingest_pdf(tmp_path, collection_name=new_course)
                    os.unlink(tmp_path)  # clean up temp file

            # add to courses list if not already there
            if new_course not in st.session_state.courses:
                st.session_state.courses.append(new_course)
                # save an empty history so list_courses() picks it up on next startup
                save_history(new_course, [])

            st.success(f"Ingested into {new_course}!")
            switch_course(new_course)
            st.rerun()

    st.divider()

    # course list
    st.markdown("**Your Courses**")
    if not st.session_state.courses:
        st.caption("No courses yet — add one above.")
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
                if st.button("🗑", key=f"delete_{course}", help=f"Delete {course}"):
                    delete_course(course)
                    st.rerun()

    # danger zone
    if st.session_state.active_course:
        st.divider()
        with st.expander("⚠️ Danger Zone"):
            if st.button("Clear Conversation", use_container_width=True):
                clear_history(st.session_state.active_course)
                st.rerun()

# ── main panel ───────────────────────────────────────────────────────────────

if st.session_state.active_course is None:
    # empty state
    st.markdown("""
    <div style='display:flex; flex-direction:column; align-items:center; justify-content:center; height:60vh; color:#555;'>
        <h1 style='font-size:3rem;'>📚</h1>
        <h3>Select or create a course to start studying</h3>
        <p>Upload your lecture notes and ask questions grounded in your material.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    course = st.session_state.active_course
    st.markdown(f"### {course}")
    st.divider()

    # load history for this course
    history = load_history(course)

    # render existing messages
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # render sources if present
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['source']}** — page {src['page']} — *{src['section']}*")
                        if src.get("excerpt"):
                            st.caption(src["excerpt"])
                        st.divider()

    # chat input
    if prompt := st.chat_input(f"Ask about {course}..."):
        # show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # build message list to pass to agent (full history + new message)
        agent_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ] + [{"role": "user", "content": prompt}]

        # get or build agent for this course
        agent, db = get_agent(course)

        # stream response
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
                    # stream token by token if possible, otherwise show full response
                    full_response = last_msg.content
                    response_placeholder.markdown(full_response)

            # extract sources from tool message artifact
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

            # show sources
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"**{src['source']}** — page {src['page']} — *{src['section']}*")
                        st.caption(src["excerpt"])
                        st.divider()

        # save updated history
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": full_response, "sources": sources})
        save_history(course, history)