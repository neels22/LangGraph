from dotenv import load_dotenv
from langchain_core.agents import AgentFinish, AgentAction
from langgraph.graph import END, StateGraph

from nodes import reason_node, act_node
from react_state import AgentState

load_dotenv()

REASON_NODE = "reason_node"
ACT_NODE = "act_node"


def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT_NODE


graph = StateGraph(AgentState)

graph.add_node(REASON_NODE, reason_node)
graph.add_node(ACT_NODE, act_node)

graph.add_conditional_edges(
    REASON_NODE,
    should_continue,
    {
        ACT_NODE: ACT_NODE,
        END: END
    }
)

graph.add_edge(ACT_NODE, REASON_NODE)
graph.set_entry_point(REASON_NODE)

app = graph.compile()

result = app.invoke({
    "input": "how many days ago was the last spacex launch?",
    "agent_outcome": None,
    "intermediate_steps": []
    })

print(result)

print(result["agent_outcome"].return_values["output"], "final result")