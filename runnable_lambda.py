from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini",openai_api_key=openai_api_key)


parser = StrOutputParser()


prompt1 = PromptTemplate(
    template="explain the detail about {topic}",
    input_variables=["topic"]
)

chain_seq = RunnableSequence(prompt1,model, parser )


def count(text):
    return len(text.split())
    

chain = RunnableParallel({
    "detail": RunnablePassthrough(),
    "count" : RunnableLambda(count)
})


chain_final = RunnableSequence(chain_seq,chain)

result = chain_final.invoke({"topic" : "This is a bad phone"})

print(result)
