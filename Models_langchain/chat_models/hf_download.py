import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-2b-it",
    task="text-generation",
    device=0,
    model_kwargs={"token": os.getenv("HUGGINGFACEHUB_API_TOKEN")},
    pipeline_kwargs=dict(
    temperature=0.5, 
    max_new_tokens=100, 
    return_full_text=False  
)
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("till what date are you trained?")

print(result.content)