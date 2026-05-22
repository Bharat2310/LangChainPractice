from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = "create a tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="create a linkedIn post on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
parallel_chain  = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedIn": RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({"topic": "India"})
print(result["tweet"])
print("")
print(result["linkedIn"])