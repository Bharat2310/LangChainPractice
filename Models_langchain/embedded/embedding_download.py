from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"

model_kwargs = {'device': 'cpu'} 

encode_kwargs = {'normalize_embeddings': True} 

print(f"Loading {model_name}...")
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

documents = [
    "Hugging Face hosts millions of open-source machine learning models.",
    "Local embeddings run securely without needing an internet connection.",
    "Using a dedicated GPU drastically reduces vector generation time."
]
doc_vectors = embeddings.embed_documents(documents)

print(f"Successfully embedded {len(doc_vectors)} documents.")
print(f"Each vector has {len(doc_vectors[0])} dimensions.\n")
print(str(doc_vectors))

