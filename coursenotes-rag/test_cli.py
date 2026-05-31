from rag.retriever import build_chain

# Calls build_chain from rag.retriever to initializes ChromaDB, the embedding model, the Groq LLM,
# the retrieve_context tool, and wires them into a create_agent agent.
# Recall: Agent for answering, db incase we need ChromaDB later
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
    # with tool-based retrieval, steps are:
    #   step 1: human message
    #   step 2: AI decides to call retrieve_context tool, tool runs, result added
    #   step 3: final AI answer grounded in retrieved context
    # for greetings/non-course questions: only 2 steps (no tool call)
    final_state = None
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):  
        # pretty_print() is a LangChain helper that formats messages cleanly in terminal
        # step["messages"] contains the entire history, but we only want to print the most recent message at each step, which is [-1]
        final_state = step # keeps getting overwritten each step, but at the end of the loop it will be the final state after the full response is generated
        # step["messages"][-1].pretty_print() # prints literally every step
        msg = step["messages"][-1]
        # print(msg)
        # print()
        # print(type(msg))
        if msg.type == "ai": # only print assistant messages, not tool calls or human messages
            msg.pretty_print()
    
     # final_state is the full message history after the complete response is generated
     # final_state is the last snapshot from the stream — the most complete state.
    
    if final_state:
        for msg in final_state["messages"]:
            if msg.type == "tool" and hasattr(msg, "artifact") and msg.artifact:
                print("\nSources:")
                seen = set()
                for doc in msg.artifact:
                    key = (doc.metadata.get('source'), doc.metadata.get('page'))
                    if key not in seen:
                        seen.add(key)
                        print(f"  {doc.metadata.get('source')} — page {doc.metadata.get('page')} — {doc.metadata.get('section')}")
                break
    
    print()