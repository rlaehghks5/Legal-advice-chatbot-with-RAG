import chainlit as cl
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma

from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# from pinecone import Pinecone, ServerlessSpec
import os

# Pinecone 설정

# pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = 'vector-db'
# index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)
database = PineconeVectorStore.from_existing_index(index_name, embeddings)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'])


prompt = PromptTemplate(template="""문장을 바탕으로 질문에 답하세요.

문장: 
{document}

질문: {query}
""", input_variables=["document", "query"])

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="준비되었습니다! 메시지를 입력하세요!").send()

@cl.on_message
async def on_message(input_message):
    input_message = input_message.content
    documents = database.similarity_search(input_message, k=3) #← input_message로 변경

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ---------------------------
    {document.page_content}
    """
        break
    result = chat([
        HumanMessage(content=prompt.format(document=documents_string,
                                           query=input_message)) #← input_message로 변경
    ])
    await cl.Message(content=result.content).send() #← 챗봇의 답변을 보냄