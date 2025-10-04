from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class Country(BaseModel):
    """Information about a country"""
    name: str = Field(description="name of the country")
    language: str = Field(description="language of the country")
    capital: str = Field(description="Capital of the country")

structured_llm = llm.with_structured_output(Country) # this is considered as a tool and we are forcing llm to only use this tool as a output type 

print(structured_llm.invoke("What is the capital of France?"))





from typing_extensions import Annotated, TypedDict
from typing import Optional

# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""
    setup: Annotated[str, ..., "The setup of the joke"]
    # Alternatively, we could have specified setup as:
    # setup: str
    # setup: Annotated[str, ...]
    # setup: Annotated[str, "foo"]
    # no default, no description
    # no default, no description
    # default, no description
    punchline: Annotated[str, ...,"The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_llm = llm.with_structured_output(Joke)
print(structured_llm.invoke("Tell me a joke about cats"))   



########################################################
json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke"
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke"
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10"
        }
    },
    "required": ["setup", "punchline"]
}

structured_llm = llm.with_structured_output(json_schema)
print(structured_llm.invoke("Tell me a joke about cats"))