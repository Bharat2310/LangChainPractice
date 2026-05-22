from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite", temperature = 0.3)

prompt1 = PromptTemplate(
    template= "write a short joke on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "explain the following joke \n {text}",
    input_variables=["text"]
)


parser = StrOutputParser()

chain  = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({"topic": "Coding"}))