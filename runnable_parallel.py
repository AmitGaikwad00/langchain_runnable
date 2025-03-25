from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini",openai_api_key=openai_api_key)

prompt1 = PromptTemplate(
    template="explain the tweet about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="explain the linkedin about {topic}",
    input_variables=["text"]
)


parser = StrOutputParser()

chain_Parallel = RunnableParallel({
    "tweet" : RunnableSequence(prompt1,model, parser),
    "linkedin" : RunnableSequence(prompt2,model, parser)
}
)

result = chain_Parallel.invoke({"topic" : "blackhole"})

print(result)