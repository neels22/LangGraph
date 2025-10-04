

from typing import TypedDict
from langgraph.graph import END, StateGraph

class simplestate(TypedDict):
    count:int 



def increment(state: simplestate) -> simplestate:
    return {
        "count":state["count"] + 1
    }

def should_continue(state: simplestate) -> bool:
    if state["count"] <5:
        return "continue"
    return "stop"


graph = StateGraph(simplestate)


graph.add_node("increment", increment)
graph.add_conditional_edges(
    "increment",
    should_continue,
    {
        "continue": "increment",
        "stop": END
    }
)

graph.set_entry_point("increment")

app = graph.compile()

# print(app.get_graph().draw_mermaid())
# app.get_graph().print_ascii()


state = {
    "count": 0
}

result = app.invoke(state)

print(result)