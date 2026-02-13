from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import os

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

messages = [
    SystemMessage(content= "You are a helful AI assitant."),
    HumanMessage(content="WHAT are VAEs")]


response = llm.invoke(messages)
print(response.content)