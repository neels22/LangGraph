from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages, StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class State(TypedDict):
    messages: Annotated[list, add_messages]



GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post(state: State):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

def get_review_decision(state: State):
    post_content = state["messages"][-1].content
    print("current linkedin post")
    print(post_content)
    print("\n")

    decision = input("Enter 'yes' or 'no': ")
    
    if decision.lower() == "yes":
        return POST
    else:
        return COLLECT_FEEDBACK


def post(state: State):
    final_post = state["messages"][-1].content
    print("final linkedin post")
    print(final_post)
    print("post has been published")



def collect_feedback(state: State):
    feedback = input("Enter feedback: ")
    return {
        "messages": [HumanMessage(content=feedback)]
    }


graph = StateGraph(State)

graph.add_node(GENERATE_POST, generate_post)
graph.add_node(GET_REVIEW_DECISION, get_review_decision)
graph.add_node(POST, post)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)

graph.set_entry_point(GENERATE_POST)

graph.add_conditional_edges(
    GENERATE_POST,
    get_review_decision,
    {
        POST: POST,
        COLLECT_FEEDBACK: COLLECT_FEEDBACK
    }
)

graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)
graph.add_edge(POST, END)

app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="Write a blog post about the future of AI in software development")]
})

print(result["messages"][-1].content)