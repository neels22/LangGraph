

from typing import List, TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()
search_tool = TavilySearchResults(max_results=5)
tools = [search_tool]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools=tools)





class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]





def chatbot(state: BasicChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response]
    }


def tools_router(state: BasicChatState):
    last_message = state["messages"][-1]

    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tools_node"
    else:
        return END


tools_node= ToolNode(tools=tools)









graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.add_node("tools_node", tools_node)

graph.set_entry_point("chatbot")

graph.add_conditional_edges(
    "chatbot",
    tools_router,
    {
        "tools_node": "tools_node",
        END: END
    }
)
graph.add_edge("tools_node", "chatbot")



app = graph.compile()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit","bye"]:
        break
    response = app.invoke({"messages": [HumanMessage(content=user_input)]})
    print("Bot: ", response["messages"][-1].content)

