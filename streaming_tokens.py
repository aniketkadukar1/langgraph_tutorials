from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")


class State(MessagesState):
    summary: str

def call_model(state: State):

    # Get summary
    summary = state.get("summary", "")

    if summary:
        system_message = SystemMessage(f"Summary of conversation earlier: {summary}")
        return {'messages': llm.invoke([system_message] + state['messages'])}

    return {'messages': llm.invoke(state['messages'])}

def summerize_conversation(state: State):
    # Get summary
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is a summary of conversation to date: {summary}"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summery of the conversation above:"

    messages = state['messages'] + [HumanMessage(summary_message)]

    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State):
    """ Returns the next node to execute"""

    messages = len(state['messages'])

    if messages > 6:
        return "summerize_conversation"
    
    return END

# Create a graph
builder = StateGraph(State)

builder.add_node("call_model", call_model)
builder.add_node("summerize_conversation", summerize_conversation)

builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue)
builder.add_edge("summerize_conversation", END)


memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

print(graph.get_graph().draw_ascii())


config = {"configurable": {"thread_id": "2"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Lance")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    for m in event['messages']:
        m.pretty_print()
    print("---"*25)




config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="Tell me about the 49ers NFL team")
for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
    print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")