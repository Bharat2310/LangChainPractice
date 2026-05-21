from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite")

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro",
    task = "text-generation"
)
model2 = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template="generate detailed notes on the provided text\n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template = "generate 5 questions and answers based the on given text\n {text}",
    input_variables=["text"]

)
prompt3 = PromptTemplate(
    template="merge the following given notes and questions \n notes -> {notes} \n questions -> {questions}",
    input_variables=["notes", "questions"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "questions": prompt2 | model2 | parser}
)

text  = """Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64."""

merged_chain = prompt3 | model1 | parser

chain = parallel_chain | merged_chain

result = chain.invoke(text)

print(result)

chain.get_graph().print_ascii()