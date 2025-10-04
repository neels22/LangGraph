
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

# Constants
MAX_ITERATIONS = 1

# Create the graph
graph = MessageGraph()

def event_loop(state: List[BaseMessage]) -> str:
    """Determine the next step in the graph based on the current state."""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    
    if num_iterations > MAX_ITERATIONS:
        return END
    
    return "execute_tools"




# add the nodes 
graph.add_node("first_responder", first_responder_chain)
graph.add_node("revisor", revisor_chain)
graph.add_node("execute_tools", execute_tools)


# add the edges 
graph.add_edge("first_responder", "execute_tools")
graph.add_edge("execute_tools", "revisor")


graph.add_conditional_edges(
    "revisor",
    event_loop,
    {
        "execute_tools": "execute_tools",
        END: END
    }
)

graph.set_entry_point("first_responder")

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

# Test the graph
from langchain_core.messages import HumanMessage

response = app.invoke("Write about how small business can leverage AI to grow")

print("Final response:")
print(response[-1].tool_calls[0]["args"]["answer"])

print(response)







