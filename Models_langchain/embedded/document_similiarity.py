from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding_model = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-2", output_dimensionality=200)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about rohit sharma"

doc_embed = embedding_model.embed_documents(documents)
query_embed = embedding_model.embed_query(query)

score = cosine_similarity([query_embed], doc_embed)[0]
index, score = sorted(list(enumerate(score)), key = lambda x:x[1])[-1]
print(documents[index])
