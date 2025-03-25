from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini",openai_api_key=openai_api_key )

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal["Positive","Negative"] = Field(description="Give the sentiment of the feedback")
    

parser2 = PydanticOutputParser(pydantic_object=Feedback)

template = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=["feedback"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

chain = template | model | parser2

template2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=["feedback"],
)

template3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=["feedback"],
)


chain_branch = RunnableBranch(
    (lambda x : x.sentiment=="Positive",template2 | model | parser),
    (lambda x : x.sentiment=="Negative",template3 | model | parser),
    RunnableLambda(lambda x :"could not find sentiment")
)


final_chain = chain | chain_branch

result = final_chain.invoke({"feedback" : "This is a bad phone"})

print(result)