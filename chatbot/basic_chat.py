

from typing import List, TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from dotenv import load_dotenv
load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: BasicChatState):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot",END)



app = graph.compile()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit","bye"]:
        break
    response = app.invoke({"messages": [HumanMessage(content=user_input)]})
    print("Bot: ", response["messages"][-1].content)

