
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
from dotenv import load_dotenv


from langchain_core.messages import HumanMessage , AIMessage

from langchain.agents import tool , create_react_agent
from langchain import hub
from langchain_community.tools import TavilySearchResults

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

search_tool = TavilySearchResults(search_depth="basic")

@tool
def get_date_time():
    """ return the current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools=[search_tool, get_date_time]

# Use the standard ReAct prompt from hub
prompt = hub.pull("hwchase17/react")

react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)






