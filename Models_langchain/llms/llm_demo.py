from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result  = llm.invoke("what is capital of india?")

print(result.content)

# not to be used in new projects as chatbots are used nowadays

