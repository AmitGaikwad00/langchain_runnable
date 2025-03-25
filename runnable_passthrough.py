from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel,RunnablePassthrough,RunnableSequence

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini",openai_api_key=openai_api_key)

prompt1 = PromptTemplate(
    template="write a joke on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

joke_seq = RunnableSequence(prompt1,model, parser)


prompt2 = PromptTemplate(
    template="explain the summarize about {text}",
    input_variables=["text"]
)

chain_parallel = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "summarize" : RunnableSequence(prompt2,model, parser )
})

chain = RunnableSequence(joke_seq, chain_parallel)

result = chain.invoke({"topic" : "blackhole"})

print(result)