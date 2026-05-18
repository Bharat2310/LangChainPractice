from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-2", output_dimensionality=2)

docs = ["delhi is the capital of india",
        "london = france", 
        "usa = americe"]

result = embedding_model.embed_documents("delhi is the capital of France")
print(str(result))