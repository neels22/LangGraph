from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

memory = MemorySaver()

search_tool = TavilySearchResults(max_results=5)

tools = [search_tool]

llm_with_tools = llm.bind_tools(tools=tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]



def model(state: State):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]

    }

def tools_router(state: State):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    else:
        return END
    

graph = StateGraph(State)

graph.add_node("model", model)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("model")

graph.add_conditional_edges(
    "model",
    tools_router,
    {
        "tools": "tools",
        END: END
    }
)
graph.add_edge("tools", "model")

app = graph.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

config = {
    "configurable": {
        "thread_id": "1"
    }
}

events = app.stream(
    {"messages": [HumanMessage(content="What is the weather in Tokyo?")]},
    config=config,
    stream_mode="values"
)

for event in events:
    print(event["messages"][-1].pretty_print())


snapshots = app.get_state(config=config)
print(snapshots.next)

events = app.stream(
    None,
    config=config,
    stream_mode="values"
)

for event in events:
    print(event["messages"][-1].pretty_print())

