from langchain_chroma import Chroma
# for loading the persisted index from disk, not creating a new one

from langchain_huggingface import HuggingFaceEmbeddings
# same embedding model and ChromaDB setup as ingest.py, we need the same embeddings object here to query the same vector space.

from langchain_groq import ChatGroq
# Groq is free, runs on their hardware, for now atleast
# looks for the api key automatically

from langchain.agents import create_agent
# creates a simple agent loop: model + optional tools + optional middleware

from langchain.tools import tool

from dotenv import load_dotenv



load_dotenv()

CHROMA_PATH = "./chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# must match ingest.py — same model means same vector space

def build_chain(collection_name: str = "course_notes"):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,  
        collection_name=collection_name
    ) # load the persisted ChromaDB index from disk, using the same embeddings object to ensure we query the same vector space

    # the agent calls this tool when it decides it needs context to answer
    # the docstring is important the agent reads it to decide when to call the tool
    # decorator registers this function as a tool the agent can call
    # response_format="content_and_artifact" tells LangChain the tool returns a tuple of two things
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant course notes to help answer a question about course material."""
        retrieved_docs = db.similarity_search(query, k=6)
        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}, "
            f"Page: {doc.metadata.get('page', '?')}, "
            f"Section: {doc.metadata.get('section', '')}\n"
            f"{doc.page_content}"
            for doc in retrieved_docs
        )
        # return both the string for the LLM and the raw docs for source highlighting
        return serialized, retrieved_docs

    # temp = 0 means deterministic output
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # system prompt tells the agent when to use the tool and when not to
    prompt = (
        "You are a helpful study assistant with access to a course notes retrieval tool. "
        "You can retrieve relevant course notes at any time using the retrieve_context tool. "
        "Always use the retrieve_context tool before answering any technical, conceptual, "
        "or course-related question, even if you think you already know the answer. "
        "The tool ensures your answers are grounded in the student's specific course notes "
        "rather than general knowledge, which may differ from what their professor teaches. "
        "Only respond without using the tool for greetings, clarifications, or questions "
        "that are clearly not about course material. "
        "Never guess — if the retrieved notes don't contain the answer, say so clearly."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=prompt)
    return agent, db