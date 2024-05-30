import chainlit as cl
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import itertools
from datasets import load_dataset

# !pip install datasets
from datasets import load_dataset
data_st_plus = load_dataset("lbox/lbox_open", "statute_classification_plus")
train_data = data_st_plus['train']

# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# from pinecone import Pinecone, ServerlessSpec
import os
os.environ['PINECONE_API_KEY'] = ''


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# HuggingFace Model ID
model_id = "Alphacode-AI/AlphaMist7B-slr-v4-slow2"

# HuggingFacePipeline 객체 생성
llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    device=1,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    # device_map="auto",
    task="text-generation", # 텍스트 생성
    pipeline_kwargs={"temperature": 0.1, "max_length": 8192},
    model_kwargs={"torch_dtype": torch.float16}  # Applying fp16

)

# 템플릿
template = """다음은 대한민국 법원에서 내려진 실제 판결 사례들 중 내 상황과 유사한 사례를 가져온 것들이야. 다음 세 개의 판결 사례들을 기반으로 내가 처한 상황에서 적용될 수 있는 법령 조항들을 모두 알려줘.

--- 사례 1
## 범죄 사실
{doc1}

## 법령의 적용
{law1}

--- 사례 2
## 범죄 사실
{doc2}

## 법령의 적용
{law2}

---
다음은 내 상황이야.
{query}

이러한 상황에서 단계별로 어떤 법령 조항들이 적용될 수 있을까? 앞서 주어진 사례들을 기반으로 답변해줘.

---
답변 :

"""

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

embedding_model_path = 'jhgan/ko-sroberta-multitask'
embeddings = SentenceTransformer(embedding_model_path)

# pinecone DB index 가져오기
index_name = "vector-db"
index = pc.Index(index_name)
prompt = PromptTemplate.from_template(template)     # 프롬프트 템플릿 생성

# ### 예시 쿼리 
query = """내 친구가 나한테 캐리어를 맡겼는데, 그 친구가 내가 여행떄문에 집을 비운 동안 그 캐리어가 필요하다고 돌려달라는거야.
난 당연히 지금 여행중이라 캐리어를 건네주기 어렵다고 했지.
근데 친구는 화를 내면서 내 집 도어락 비밀번호를 안다고 하면서, 자기가 우리 집 문을 열고 캐리어를 가져갔어.
그리고 의심하건데 아마 우리 집에 현금이 100만원 정도 있었는데, 이것도 그떄 같이 가져간 것 같거든.
이런 경우에는 내 친구를 어떤 죄목으로 고소할 수 있을까??"""



@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="준비되었습니다! 메시지를 입력하세요!").send()


@cl.on_message
async def on_message(input_message):
    input_message = input_message.content
    query_embedding = model.encode(input_message).tolist()
    documents = index.query(vector=query_embedding, top_k=3) #← input_message로 변경

    retrieval_docs = []

    for match in documents['matches']:
        doc_id = (match['id'])
        print(doc_id)
        temp = index.fetch(ids = [doc_id])
        case_id = temp['vectors'][doc_id]['metadata']['case_id']
        print(case_id)
        print()
        retrieval_docs.append(case_id)
    
    docs = []
    laws = []

    for case_id in retrieval_docs:
        docs.append(train_data[int(case_id)]['facts'])
        laws.append(train_data[int(case_id)]['statutes'])

    doc1 = docs[0]
    doc2 = docs[1]
    doc3 = docs[2]

    law1 = laws[0]
    law2 = laws[1]
    law3 = laws[2]
    
    chain = prompt | llm    # 체인 구성

    result = chain.invoke({"query": input_message, 
                    "doc1": doc1, "doc2": doc2, 
                    "law1": law1, "law2": law2,})
    
    result = result.split('---')[-1]
    
    
    await cl.Message(content=result).send() #← 챗봇의 답변을 보냄
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    