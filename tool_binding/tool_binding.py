from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """given a and b this tool returns the multiplication of a and b"""
    return a * b

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

query = HumanMessage("multiply 4 and 9000")
messages = [query] 

llm_with_tools = model.bind_tools([multiply])

result = llm_with_tools.invoke(messages)
messages.append(result)

tool_call = result.tool_calls[0]

tool_result = multiply.invoke(tool_call)

messages.append(tool_result)

final_result = llm_with_tools.invoke(messages)

print(final_result.content)