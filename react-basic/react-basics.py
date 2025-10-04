from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType, tool


from langchain_community.tools import TavilySearchResults
from datetime import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = llm.invoke("give me a tweet about the weather?")

search_tool = TavilySearchResults(search_depth="basic")


@tool
def get_date_time():
    """ return the current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = initialize_agent(
    tools=[search_tool, get_date_time],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print(agent.invoke({"input": "when was spacex last launch and how many days ago was that from today?"}))
