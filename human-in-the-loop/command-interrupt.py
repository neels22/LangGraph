from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

memory = MemorySaver()

class State(TypedDict):
    value: str



def node_a(state: State):
    print("node_a")
    return Command(
        goto="node_b",
        update={"value": state["value"] + "a"},
    )

def node_b(state: State):
    print("node_b")

    human_input = interrupt("do you want to go to c or d ")
    print("human review:", human_input)

    if human_input.lower() == "c":
        return Command(
            goto="node_c",
            update={"value": state["value"] + "b"},
        )
    else:
        return Command(
            goto="node_d",
            update={"value": state["value"] + "b"},
        )

def node_c(state: State):
    print("node_c")
    return Command(
        goto=END,
        update={"value": state["value"] + "c"},
    )

def node_d(state: State):
    print("node_d")
    return Command(
        goto=END,
        update={"value": state["value"] + "d"},
    )


graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)

graph.set_entry_point("node_a")

app = graph.compile(checkpointer=memory)
config = {
    "configurable": {
        "thread_id": "1"
    }
}

initial_state = {
    "value": ""
}

# First run - this will pause at the interrupt in node_b
result = app.invoke(initial_state, config=config)
print("First result:", result)

# Check the current state
state = app.get_state(config=config)
print("Current state:", state.values)
print("Next nodes:", state.next)

# Resume with human input "d" to go to node_d
secondresult = app.invoke(Command(resume="d"),config=config)

print("Second result:", secondresult)