from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

prompt = PromptTemplate(
    template = "generate 5 super interesting facts about {topic} ",
    input_variables=["topic"]
)
parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite")
# model = ChatHuggingFace(llm = llm)

chain = prompt | model | parser 

result = chain.invoke({"topic": "sci-fy"})
print(result)

chain.get_graph().print_ascii()