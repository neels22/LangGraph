import json
from typing import List
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_community.tools import TavilySearchResults

load_dotenv()


search_tool = TavilySearchResults(max_results=5)

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]

    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []
    
    # Process the AnswerQuestion or ReviseAnswer tool calls to extract search queries
    tool_messages = []
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])
            
            # Execute each search query using the tavily tool
            query_results = {}
            for query in search_queries:
                result = search_tool.invoke(query)
                query_results[query] = result # each search term has 5 results returned by tavily tool
            
            # Create a tool message with the results
            tool_messages.append(  # we need to provide tool call id 
                ToolMessage(
                    content=json.dumps(query_results), 
                    tool_call_id=call_id
                )
            )
    
    return tool_messages


