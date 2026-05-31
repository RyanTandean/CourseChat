from rag.retriever import build_chain

# Calls build_chain from rag.retriever to initializes ChromaDB, the embedding model, the Groq LLM,
# the retrieve_context tool, and wires them into a create_agent agent.
# Recall: Agent for answering, Db for source highlighting
agent, db = build_chain()

print("CourseChat CLI Test — type 'quit' to exit\n")

while True:
    question = input("You: ") # read from std in, the user's question
    if question.lower() == "quit":
        break
        
    # agent.stream() runs the agent step by step, yielding the full state after each step.
    # unlike agent.invoke() which waits for the complete response, stream() lets you
    # see intermediate steps as they happen — useful for debugging and for streaming
    # output to the UI later.
    #
    # stream_mode="values" means each yielded snapshot is the FULL messages list at that
    # point in time, not just the delta. so step["messages"][-1] always gives the
    # most recent message regardless of which step we're on.
    #
    # for a dynamic_prompt agent with no tools, the steps are:
    #   step 1: human message added → step["messages"][-1] is the human message
    #   step 2: LLM generates answer → step["messages"][-1] is the AI response
    #
    # if we were using tools it would be 3 steps:
    #   step 1: human message
    #   step 2: tool call + result
    #   step 3: final AI answer
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):  
        # pretty_print() is a LangChain helper that formats messages cleanly in terminal
        # step["messages"] contains the entire history, but we only want to print the most recent message at each step, which is [-1]
        step["messages"][-1].pretty_print()
    
    print()