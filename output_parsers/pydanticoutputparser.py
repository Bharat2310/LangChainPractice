from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, EmailStr

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)
class Person(BaseModel):
    name : str = Field(description="Name of a person")
    age : int = Field(description="age of a person", gt = 18, lt= 20)
    email : EmailStr = Field(description="Email of that person ending with @gmail.com")
    city : str = Field(description="city of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template= "generate the name, age, city and email of a female living in {place} \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)
chain = template | model  | parser
final_result = chain.invoke({"place":"pakistan"})

print(final_result)