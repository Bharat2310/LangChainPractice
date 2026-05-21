from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

class Facts(BaseModel):
    fact1 : str = Field(description="insert fact 1 \n")
    fact2 : str = Field(description="insert fact 2 \n")
    fact3 : str = Field(description="insert fact 3 \n")
    fact4 : str = Field(description="insert fact 4 \n")
    fact5 : str = Field(description="insert fact 5 \n")

parser2 = PydanticOutputParser(pydantic_object=Facts)


model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite", max_output_tokens = 100)

prompt1 = PromptTemplate(
    template = "generate a short report on {topic} ",
    input_variables=["topic"]
    )
prompt2 = PromptTemplate(
    template = "wrtie a 5 lines short summary on the given text \n {text} \n {formatting_instructions}",
    input_variables=["text"],
    partial_variables={"formatting_instructions" : parser2.get_format_instructions()}

)
parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser2

result = chain.invoke({"topic": "python as a programming language"})
print(result)
