from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import os
from transformers import logging
logging.set_verbosity_error()
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage 

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2b-it",
    
    task="text-generation",
    model_kwargs={"token": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                   'device_map' : "auto"},
    pipeline_kwargs=dict(temperature=1)
)
model = ChatHuggingFace(llm = llm)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI", result.content)

print(chat_history)