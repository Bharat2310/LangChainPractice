from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Email(BaseModel):
    email_type: Literal["complain", "refund", "general_query"] = Field(
        description="categorise the email into complain, refund or general query email"
    )

parser1 = PydanticOutputParser(pydantic_object=Email)

parser2 = StrOutputParser()

prompt1 = PromptTemplate(
    template="categorise the email into complain, refund or general query email\n Email:\n {text} \n {formatting_instructions}",
    input_variables=['text'],
    partial_variables={"formatting_instructions": parser1.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="act as a customer care agent at any company and answer this customer query in short and concise manner \n {query}",
    input_variables=["query"]
)

categorising_chain = prompt1 | model | parser1

def complain(x):
    return "Stay tuned, customer care will be contacting you."

def refund(x):
    return "Your refund will be provided in 1-2 business days."

branch_chain = RunnableBranch(
    ((lambda x: x["category"].email_type == "complain"), RunnableLambda(complain)),
    
    ((lambda x: x["category"].email_type == "refund"), RunnableLambda(refund)),
    
    RunnableLambda(lambda x: {"query": x["text"]}) | prompt2 | model | parser2
)

final_chain = RunnablePassthrough.assign(category=categorising_chain) | branch_chain

result = final_chain.invoke({
    "text": """Subject/Reason for Contact: General Inquiry – Product Specifications
Customer Name: Sarah Chen

Order Number: N/A (Pre-purchase)

Message/Review:

"Hi there! I am really interested in purchasing the [Product Name], but I had a quick question before I place my order. The description online mentions it is 'water-resistant,' but doesn't specify the exact rating. Can this handle heavy rain, or is it just meant for light splashes? Also, if I am right between sizes, do you recommend sizing up or down for the best fit? Looking forward to your response so I can complete my checkout. Thanks!"""
})

print(result)