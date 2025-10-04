from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain  

load_dotenv()


graph = MessageGraph()


REFLECT_NODE = "reflect"
GENERATE_NODE = "generate"


def generate_node(state): # think of state as list of messages happended in past 
    result = generation_chain.invoke({
        "messages": state,
    })
    
    return [result]


def reflect_node(state):
    response = reflection_chain.invoke({
        "messages": state,
    })

    return [HumanMessage(content=response.content)] # trick the systems 




graph.add_node(GENERATE_NODE, generate_node)
graph.add_node(REFLECT_NODE, reflect_node)

graph.set_entry_point(GENERATE_NODE)


def should_continue(state):
    if len(state) > 3:  # After 4+ messages (user + generate + reflect + generate), end
        return END

    return REFLECT_NODE

#you need to explicitly define the path map that shows what each return value from should_continue maps to.  
graph.add_conditional_edges(
    GENERATE_NODE, 
    should_continue,
    {
        REFLECT_NODE: REFLECT_NODE,
        END: END
    }
)

graph.add_edge(REFLECT_NODE, GENERATE_NODE)

app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

# # Actually run the graph with an initial user request
if __name__ == "__main__":
    initial_message = HumanMessage(content="Write a tweet about the future of AI in software development")
    
    print("\n" + "="*50)
    print("Starting Reflection Agent")
    print("="*50 + "\n")
    
    for i, state in enumerate(app.stream([initial_message])):
        print(f"\n--- Step {i+1} ---")
        print(state)
        print("\n")

