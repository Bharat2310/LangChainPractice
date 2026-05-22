from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("glorot10a.pdf")

docs = loader.load()

print(len(docs))