from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

parser = JsonOutputParser()
model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template = "give me the name, age and city of fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables = {'format_instruction': parser.get_format_instructions() }
)

chain = template1 | model | parser
result = chain.invoke({})

print(result)