# STREAMLIT FRONT END
# Most important thing to understand: 
# the script app.py reruns from top to bottom on every single user interaction,
# unlike React. To maintain variables, we have a st.session_state

import streamlit as st
import os
from rag.retriever import build_chain
from rag.ingest import ingest_pdf
from rag.history import load_history, save_history, clear_history, list_courses, list_files, delete_file
import chromadb
import tempfile
import re
import fitz  # PyMuPDF — used to render PDF pages as images for the sources panel

# strip_markdown is currently unused — was previously used to flatten excerpt text
# before passing to st.caption(). Replaced with st.markdown() so pymupdf4llm's
# extracted formatting (bullets, bold, headers) renders correctly in the sources panel.
# Kept here in case plain-text excerpts are needed elsewhere later.
# def strip_markdown(text: str) -> str:
#     text = re.sub(r'\*+', '', text)
#     text = re.sub(r'#+\s', '', text)
#     text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
#     text = re.sub(r'`+', '', text)
#     text = re.sub(r'<[^>]+>', '', text)
#     text = re.sub(r'\n{2,}', ' ', text)
#     return text.strip()

############### LaTeX delimiter normalisation ####################
#
# Why this exists:
# Streamlit's st.markdown has BUILT-IN KaTeX rendering
# (KaTeX) = JS lib that converts LaTeX math notation into visual HTML/CSS math symbols in a browser
# It recognises two delimiter styles:
#   $...$   — inline math   (e.g.  $x^2 + y^2 = r^2$)
#   $$...$$  — block/display math  (e.g.  $$\int_0^\infty e^{-x} dx = 1$$)
#
# The problem: LLMs trained on LaTeX documents often produce a DIFFERENT
# delimiter style — the native LaTeX convention:
#   \(...\)  — inline math
#   \[...\]  — display/block math
#
# Streamlit does NOT recognise \(...\) or \[...\], so they render as raw text
# with backslashes and parentheses instead of rendered equations.
#
# The fix is a preprocessing step: convert \(...\) → $...$ and \[...\] → $$...$$
# before passing the string to st.markdown. 
# This happens entirely in Python so no JavaScript injection, no iframes, no hacks.
#
# Regex breakdown for INLINE: re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
#   \\\(     — matches a literal \( in the string  (two backslashes: one to escape
#              the regex engine, one for the Python string literal)
#   (.+?)    — capture group: the actual math content, non-greedy so it stops
#              at the FIRST \) rather than consuming everything up to the last one
#   \\\)     — matches a literal \)
#   r'$\1$'  — replacement: wraps the captured group in Streamlit's $...$ delimiters
#   re.DOTALL — makes . match newlines too, so multi-line inline expressions work
#
# Regex breakdown for BLOCK: re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
#   Same idea but with \[...\] → $$...$$
#   Block math is typically multi-line (e.g. a matrix), so re.DOTALL is critical.
#
# Order matters: process block delimiters BEFORE inline. If you did inline first
# and the text contained something like \[\frac{1}{2}\], the \( inside \frac
# would not actually be a delimiter, but doing block first avoids any ambiguity.
#
# Reference: https://docs.streamlit.io/develop/api-reference/text/st.latex
# (st.markdown uses the same underlying KaTeX renderer as st.latex)
#
# Why not just prompt the LLM to use custom delimiters to match streamlit's format?
# Most likely, models have seen vast amounts of documents using default LaTeX rendering,
# It might be more prone to making more mistakes if we forced it to use a abnormal format
# Takeaway: If you can handle it deterministically in code, don't rely on the model to do it consistently.
#           Models are good at reasoning and language, not rigidly following formatting and strict conditions.

def normalise_latex(text: str) -> str:
    """Convert LLM-style LaTeX delimiters to Streamlit-compatible delimiters.

    LLMs trained on academic text commonly output:
        \\[...\\]  for display/block math
        \\(...\\)  for inline math

    Streamlit's st.markdown KaTeX renderer expects:
        $$...$$   for display/block math
        $...$     for inline math

    This function converts between the two styles so math renders
    correctly without any JavaScript or custom components.
    """
    # Step 1: block math — \[...\] → $$...$$
    # Must run before inline so that any \( sequences inside a block
    # expression are not accidentally matched as inline delimiters.
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

    # Step 2: inline math — \(...\) → $...$
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)

    return text

# ── Page image rendering ─────────────────────────────────────────────────────
#
# The sources panel shows the actual PDF page a retrieved chunk came from.
# This gives the user visual context — they can see the original layout,
# diagrams, and any equations that weren't captured as text.
#
# How it works:
#   1. Every chunk in ChromaDB has metadata: {source: filename, page: n}
#   2. At ingest time, the PDF is saved to data/notes/<filename>
#   3. At query time, we open that PDF with fitz, render the specific page
#      to a pixmap (an in-memory image), convert to PNG bytes, and pass to
#      st.image() which displays it inline in the sources expander.
#
# fitz.Matrix(zoom, zoom): a transformation matrix that scales the render.
#   zoom=2 means 2x the native PDF resolution (72 DPI → 144 DPI).
#   Higher zoom = sharper image but larger bytes. 2x is a good balance.
#
# Deployment note: this requires the PDF to be on the local filesystem.
# For production deployment on ephemeral hosts (Render, Streamlit Cloud),
# replace the fitz.open(pdf_path) with an S3 fetch and store the object
# key in chunk metadata instead of the local filename.

PDF_STORAGE_PATH = "./data/notes"

def render_pdf_page(source_filename: str, page_number: int) -> bytes | None:
    """Render a specific PDF page to PNG bytes for display with st.image().

    Args:
        source_filename: the PDF filename as stored in chunk metadata (e.g. "Enumeration-1.pdf")
        page_number:     1-indexed page number from chunk metadata

    Returns:
        PNG image bytes, or None if the PDF file doesn't exist or rendering fails.
        Callers should check for None and skip st.image() gracefully.
    """
    pdf_path = os.path.join(PDF_STORAGE_PATH, source_filename)

    if not os.path.exists(pdf_path):
        # PDF not found — either the file was deleted, or this is an older course
        # indexed before page rendering was added. Fail silently.
        return None

    try:
        doc  = fitz.open(pdf_path)
        # fitz uses 0-indexed pages; chunk metadata uses 1-indexed
        page = doc[page_number - 1]

        # render at 1.5x resolution — crisp enough for reading, smaller than 2x
        # Matrix(1.5, 1.5) scales both x and y by 1.5
        zoom   = 1.5
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix)

        # tobytes("png") converts the pixmap to PNG bytes in memory —
        # no temp file needed, st.image() accepts raw bytes directly
        image_bytes = pixmap.tobytes("png")
        doc.close()
        return image_bytes

    except Exception as e:
        # page out of range, corrupted PDF, etc. — fail silently
        print(f"[page render] failed to render {source_filename} page {page_number}: {e}")
        return None


##################### Page Config + CSS injection ############################
# Runs once every rerun, sets the overall styling
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

######### session state ################
# 
# "if not in" pattern:
# On the first run, these components get set and 
# "if not in" checks if they already exist on each rerun, which is how state
# survices across interactions
# 

# which course is selected
if "active_course" not in st.session_state:
    st.session_state.active_course = None
# cached agent instances
if "agents" not in st.session_state:
    st.session_state.agents = {}
# list of courses, for course sidebar
if "courses" not in st.session_state:
    st.session_state.courses = list_courses()

################# helpers #######################
def get_agent(course_name: str):
    """Return the cached (agent, db) pair for a course, building it if necessary.

    build_chain() is expensive — it loads the ChromaDB index from disk, initializes
    the HuggingFace embedding model, and creates the Groq LLM connection. Streamlit
    reruns app.py from top to bottom on every interaction, so without this cache,
    build_chain() would be called on every keypress and message.

    st.session_state.agents acts as a per-session cache: the first call for a given
    course pays the build cost; every subsequent call returns the already-built objects
    instantly. The cache lives for the duration of the browser session and is cleared
    on page refresh.

    Returns None if build_chain() fails (corrupted ChromaDB, embedding model
    download failure, missing API key, etc.) — callers must check for None.
    """
    if course_name not in st.session_state.agents:
        try:
            agent, db = build_chain(collection_name=course_name)
            st.session_state.agents[course_name] = (agent, db)
        except Exception as e:
            # build_chain can fail if:
            #   - ChromaDB files are corrupted or locked
            #   - HuggingFace embedding model fails to download (no internet, cache issue)
            #   - GROQ_API_KEY is missing from .env
            # return None so the caller can show a user-facing error instead of crashing
            st.error(f"Failed to load agent for {course_name}: {e}")
            return None
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

############## sidebar #############

# streamlit object that renders a collapsible panel on the left side of the page
# 'with' keyword from python; everything indented in the block belongs to this container
# Basically simplifies resource management with setup and cleanup
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

    # vision enrichment toggle — shown next to the uploader so the user can
    # decide before clicking Upload & Index.
    # on: Groq vision model describes embedded images (equations, diagrams) — slower
    # off: image placeholders stay in the index — faster, fine for non-math PDFs
    use_vision = st.checkbox(
        "Vision enrichment (math PDFs)",
        value=True,
        help="Uses AI vision to extract equations from images. Adds ~2s per image. Turn off for non-math PDFs."
    )

    if st.button("Upload & Index", use_container_width=True, type="primary"):
        if not new_course:
            st.error("Enter a course name first.")
        elif not uploaded_files:
            st.error("Upload at least one PDF.")
        else:
            with st.spinner(f"Indexing {len(uploaded_files)} file(s)..."):
                # track which files succeeded, were skipped, and which failed
                failed = []
                skipped = []
                for uploaded_file in uploaded_files:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        _, was_skipped, _ = ingest_pdf(tmp_path, collection_name=new_course, original_filename=uploaded_file.name, use_vision=use_vision)
                        os.unlink(tmp_path)
                        if was_skipped:
                            skipped.append(uploaded_file.name)
                    except Exception as e:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        failed.append((uploaded_file.name, str(e)))

            if failed:
                for filename, error in failed:
                    st.error(f"Failed to index {filename}: {error}")
            else:
                if skipped:
                    for filename in skipped:
                        st.info(f"{filename} was already indexed — skipped.")
                if new_course not in st.session_state.courses:
                    st.session_state.courses.append(new_course)
                    save_history(new_course, [])
                if len(skipped) < len(uploaded_files):
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

        # ── Indexed files ─────────────────────────────────────────────────────
        # Show every PDF currently indexed for the active course.
        # Each file has an ✕ button that deletes only that file's chunks from
        # ChromaDB and removes the saved PDF — leaving other files intact.
        # The cached agent is evicted so the next query rebuilds against the
        # updated collection rather than serving stale vectors.
        with st.expander("Indexed files"):
            files = list_files(st.session_state.active_course)
            if not files:
                st.caption("No files indexed yet.")
            else:
                for fname in files:
                    col_f, col_x = st.columns([5, 1])
                    with col_f:
                        st.caption(fname)
                    with col_x:
                        if st.button("✕", key=f"delfile_{fname}", help=f"Remove {fname}"):
                            delete_file(st.session_state.active_course, fname)
                            # evict cached agent — its db reference now points to
                            # a collection that no longer has this file's vectors
                            if st.session_state.active_course in st.session_state.agents:
                                del st.session_state.agents[st.session_state.active_course]
                            st.rerun()

        with st.expander("Settings"):
            if st.button("Clear conversation", use_container_width=True):
                clear_history(st.session_state.active_course)
                st.rerun()

####################### main panel ######################
# Huge if else, if no course selected, show welcome screen, otherwise show chat
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

    # render every past message on every rereun, for a conversation to "persist"
    for msg in history:
        with st.chat_message(msg["role"]):
            # Normalise LaTeX delimiters before rendering so that math from
            # previous conversations displays correctly on reload.
            # User messages are passed through too — harmless if they contain
            # no math, and correct if they typed a \(...\) expression.
            st.markdown(normalise_latex(msg["content"]))
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['source']}** — page {src['page']} — *{src['section']}*")
                        if src.get("excerpt"):
                            # render the excerpt as markdown so pymupdf4llm's extracted
                            # formatting (bullet points, bold, headers) displays correctly
                            st.markdown(src["excerpt"])
                        # render the actual PDF page as an image so the user can see
                        # the original layout, diagrams, and any visual context.
                        # render_pdf_page returns None if the PDF isn't available —
                        # silently skips the image rather than showing an error.
                        page_img = render_pdf_page(src["source"], src["page"])
                        if page_img:
                            with st.expander(f"Page {src['page']} image"):
                                st.image(page_img, width="stretch")
                        st.divider()

    # chat input
    # st.chat_input will be None most of the reruns that occur (clicking buttons that don't submit a user message)
    # Otherwise, this entire block runs
    # Walrus operator ':=' checks if st.chat_input is truthy, then assigns it to prompt
    if prompt := st.chat_input(f"Ask about {course}..."):

        # technically renders user messages as markdown if possible before sending to LLM
        # I think it's fine as it is right now, but worth noting if theres problems on this side,
        # Possible edge cases: "it costs $5 and $10", remember st.markdown uses $ as delimiters
        with st.chat_message("user"):
            st.markdown(prompt)

        # Recall that agent is stateless, has no memory of previous calls so we need to
        # reconstruct the full convo every time by saving history and sending one huge message back to
        # the agent.
        #
        # Sliding window: cap the history sent to the agent at the most recent
        # HISTORY_WINDOW message pairs (user + assistant = 2 messages per pair).
        # This prevents the context window from growing unboundedly over long sessions
        # and keeps token usage stable. The full history is still saved to disk and
        # rendered in the UI — only the agent's input is trimmed.
        #
        # Why 10 pairs (20 messages)?
        # Study sessions are usually focused on one topic at a time. Referencing
        # context from more than 10 exchanges back is rare, and the tradeoff
        # (losing old context vs. hitting token limits / degraded answers) favours
        # the window. Increase HISTORY_WINDOW if your sessions are longer.
        HISTORY_WINDOW = 10  # message pairs to keep (1 pair = 1 user + 1 assistant)
        windowed_history = history[-(HISTORY_WINDOW * 2):]

        agent_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in windowed_history
        ] + [{"role": "user", "content": prompt}]

        agent, db = get_agent(course)

        # get_agent returns None if build_chain failed — show error and bail out
        # rather than crashing with an AttributeError on agent.stream()
        if agent is None:
            st.error("Could not load the agent. Check your GROQ_API_KEY and that ChromaDB is intact.")
        else:
            with st.chat_message("assistant"):
                # st.empty() reserves a single fixed slot in the UI at this position.
                # Calling .markdown() on it overwrites the slot in place rather than
                # appending a new element below — like erasing a whiteboard and rewriting it.
                # Without this, each iteration of the streaming loop below would add a new
                # message bubble, leaving multiple partial copies of the response on screen.
                # With it, the user sees one message that updates as new content arrives.
                #
                # NOTE: this app does NOT do true token-by-token streaming. agent.stream()
                # yields full state snapshots at each agent step (human message → tool call
                # → final answer), not individual tokens. The placeholder is only written
                # once — on the final step when the complete response is ready. The st.empty()
                # pattern is kept here as scaffolding for if true streaming is added later.
                response_placeholder = st.empty()
                full_response = ""
                final_state = None
                # Remember agent.stream() generates a full snapshot of full agent state and the associated steps,
                # 1. Human message, 2. Agent call retrieve_context tool, add to state, 3. Agent produces final answer
                try:
                    for step in agent.stream(
                        {"messages": agent_messages},
                        stream_mode="values"
                    ):
                        final_state = step
                        last_msg = step["messages"][-1]

                        # filter intermediate steps, optionally could show it calling the tool but it isn't needed
                        if last_msg.type == "ai" and not last_msg.tool_calls:
                            full_response = last_msg.content
                            # Normalise delimiters on every streaming chunk so the math
                            # renders correctly as the response arrives, not just at the end.
                            # We normalise the full accumulated string each time rather than
                            # the delta chunk, because a delimiter like \( might arrive in
                            # one chunk and \) in the next — the regex needs the complete pair.
                            response_placeholder.markdown(normalise_latex(full_response))
                except Exception as e:
                    # agent.stream() can fail if:
                    #   - Groq API is down or rate-limited
                    #   - network connection dropped mid-stream
                    #   - the LLM returned a malformed response
                    # show a friendly error in the placeholder slot rather than a raw traceback
                    response_placeholder.error(f"Something went wrong while generating a response: {e}")
                    full_response = ""
                    final_state = None

                sources = []
                if final_state:
                    # need to dig out the source documents out of the agent's full message history
                    # recall we embedded the source documents as part of messages
                    for msg in final_state["messages"]:
                        # source = tool result, artifact field exists, and is not empty
                        # back in retriever.py, tool was decorated with
                        #   1. content: serialized test LLM sees as context
                        #   2. artifact: the actual document Objects which has the meta data for source display
                        if msg.type == "tool" and hasattr(msg, "artifact") and msg.artifact:
                            # Deduplication
                            # Multiple chunks can come from the same page so 'seen' as a set
                            # keeps track of sources such that duplicates are skipped
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
                            break # stops after finding the first tool message, don't scan further

                # Rendering sources
                if sources:
                    with st.expander("Sources"): # collapsible "Sources" panel
                        for src in sources:
                            st.markdown(f"**{src['source']}** — page {src['page']} — *{src['section']}*")
                            # render the excerpt as markdown so pymupdf4llm's extracted
                            # formatting (bullet points, bold, headers) displays correctly
                            st.markdown(src["excerpt"])
                            # render the actual PDF page as an image — gives visual context
                            # alongside the text excerpt. silently skipped if PDF unavailable.
                            page_img = render_pdf_page(src["source"], src["page"])
                            if page_img:
                                with st.expander(f"Page {src['page']} image"):
                                    st.image(page_img, width="stretch")
                            st.divider() # big horizontal line

            # Append history
            history.append({"role": "user", "content": prompt})
            # Store the normalised version so that on reload, st.markdown renders
            # the already-converted $...$ delimiters rather than re-running the
            # conversion on \(...\) strings that would have been doubly converted.
            history.append({"role": "assistant", "content": normalise_latex(full_response), "sources": sources})
            save_history(course, history) # saving convo to disk as a JSON file
            # TODO: right now everything is being saved locally, should consider deploying, using the hosted options
            # Might be easier than I think since streamlit, ChromaDB, etc surely has options for this