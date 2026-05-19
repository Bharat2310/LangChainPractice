from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from transformers import logging
logging.set_verbosity_error()

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2b-it",
    
    task="text-generation",
    model_kwargs={"token": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                   'device_map' : "auto"},
    pipeline_kwargs=dict(temperature=1)
)
model = ChatHuggingFace(llm = llm)

messages = [
    HumanMessage(content="You are a helpful gardener.\n\nTell me about different soils")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)