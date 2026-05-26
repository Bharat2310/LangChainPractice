from langchain_core.tools import tool, InjectedToolArg
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import requests
import json
 
from typing import Annotated
load_dotenv()

@tool
def fetch_conversion_rate(base:str, target:str)->float:
    """this function fetches the conversion rate between base currency and target currency """
    url = f"https://v6.exchangerate-api.com/v6/e6ae17357eeb136c61225a9a/pair/{base}/{target}"
    response = requests.get(url)
    return response.json()["conversion_rate"]

@tool
def convert(base: float, rate:float) -> float:
    """given a base currency and conversion rate, this function return the target currency value"""
    return base*rate


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

llm_tools = model.bind_tools([fetch_conversion_rate, convert])

query = HumanMessage("get the conversion rate between USD and INR and then convert 10 USD in INR")
messages = [query]


while True:
    ai_message = llm_tools.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        print("\nFinal Answer:", ai_message.content)
        break

    for tool_call in ai_message.tool_calls:
        print(f"Model requested tool: {tool_call['name']}")
        
        if tool_call['name'] == 'fetch_conversion_rate':
            tool_message = fetch_conversion_rate.invoke(tool_call)
            messages.append(tool_message)

        elif tool_call['name'] == 'convert':
            tool_message = convert.invoke(tool_call)
            messages.append(tool_message)
