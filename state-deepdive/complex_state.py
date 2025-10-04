

from typing import TypedDict, List,Annotated
from langgraph.graph import END, StateGraph
import operator

class simplestate(TypedDict):
    count:int 
    sum:Annotated[int, operator.add]
    history:Annotated[List[int], operator.concat]



def increment(state: simplestate) -> simplestate:

    new_count = state["count"] + 1
    return {
        "count":new_count,
        "sum":new_count,
        "history":[new_count],

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
    "count": 0,
    "sum": 0,
    "history": []
}

result = app.invoke(state)

print(result)