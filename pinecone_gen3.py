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
    device=0,               # -1: CPU(default), 0번 부터는 CUDA 디바이스 번호 지정시 GPU 사용하여 추론
    # device_map="auto",
    task="text-generation", # 텍스트 생성
    pipeline_kwargs={"temperature": 0.1, "max_length": 8192},
    model_kwargs={"torch_dtype": torch.float16}  # Applying fp16

)

# 템플릿
template = """다음은 대한민국 법원에서 내려진 실제 판결 사례들 중 내 상황과 유사한 사례를 가져온 것들이야. 다음 두개의 판결 사례들을 기반으로 내가 처한 상황에서 적용될 수 있는 법령 조항들을 모두 알려줘.

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

이러한 상황에서 단계별로 어떤 법령 조항들을 적용할 수 있을까? 앞서 주어진 사례들을 기반으로 답변해줘.
"""

model = SentenceTransformer('jhgan/ko-sroberta-multitask')
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )


# pinecone DB index 가져오기
index_name = "vector-db"
index = pc.Index(index_name)


### 예시 쿼리 수행
# query = """내 친구가 나한테 캐리어를 맡겼는데, 그 친구가 내가 여행떄문에 집을 비운 동안 그 캐리어가 필요하다고 돌려달라는거야.
# 난 당연히 지금 여행중이라 캐리어를 건네주기 어렵다고 했지.
# 근데 친구는 화를 내면서 내 집 도어락 비밀번호를 안다고 하면서, 자기가 우리 집 문을 열고 캐리어를 가져갔어.
# 그리고 의심하건데 아마 우리 집에 현금이 100만원 정도 있었는데, 이것도 그떄 같이 가져간 것 같거든.
# 이런 경우에는 내 친구를 어떤 죄목으로 고소할 수 있을까??"""

query = """내가 친구에게 돈을 빌렸는데, 6개월동안 돈이 없어서 안갚았는데, 친구가 나를 고소한다고 해 이런 경우 어떤 처벌을 받을 수 있어?"""

query_embedding = model.encode(query).tolist()
results = index.query(vector=query_embedding, top_k=3)

retrieval_docs = []

for match in results['matches']:
    doc_id = (match['id'])
    print(doc_id)
    temp = index.fetch(ids = [doc_id])
    case_id = temp['vectors'][doc_id]['metadata']['case_id']
    print(case_id)
    print()
    retrieval_docs.append(case_id)

prompt = PromptTemplate.from_template(template)     # 프롬프트 템플릿 생성
chain = prompt | llm    # 체인 구성


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

print(chain.invoke({"query": query, 
                    "doc1": doc1, "doc2": doc2, 
                    "law1": law1, "law2": law2,}))

# print(chain.invoke({"query": query, 
#                     "doc1": doc1, "doc2": doc2, "doc3": doc3, 
#                     "law1": law1, "law2": law2, "law3": law3 }))





# 문서 가져와서 실행하는거 demo
doc1 = """2023고단1493 판결 [야간주거침입절도, 야간주거침입절도미수, 절도, 절도미수]
## 범죄 사실
피고인은 2023. 3.경 일정한 직업 없이 지내던 중 생활비가 부족해지자 시정되지 않은 길가에 주차된 자동차 또는 빈집에 들어가 물건을 훔치기로 마음먹었다.

피고인은 2023. 3. 초순경 광주 동구 B ○층에 있는 피해자 C의 집 앞에 이르러, 피해자가 장바구니에 담아 현관문 앞에 놓아둔 시가 합계 3만 원 상당의 두부, 토마토, 고춧가루 등 식료품을 들고 가 피해자의 재물을 절취한 것을 비롯하여 그때부터 2023. 3. 하순경까지 사이에 별지 범죄일람표 기재와 같은 방법으로 총 13회에 걸쳐 야간에 타인의 주거에 침입하여 물건을 절취하거나 타인의 자동차 문을 열고 들어가 합계 5,633,000원 상당의 물건을 절취하고, 총 5회에 걸쳐 야간에 타인의 주거에 침입하여 물건을 절취하려 하였으나 범행이 발각되거나, 타인의 자동차 안을 물색하였으나 훔칠 물건을 발견하지 못하여 그 뜻을 이루지 못하는 등 미수에 그쳤다.


## 법령의 적용
각 형법 제329조(절도의 점, 징역형 선택), 각 형법 제330조(야간주거침입절도의 점), 각 형법 제342조, 제329조(절도미수의 점, 징역형 선택), 각 형법 제342조, 제330조(야간주거침입절도미수의 점)
"""

doc2 = """2023고단684 판결 [야간주거침입절도, 야간주거침입절도미수]

## 범죄 사실
1. 야간건조물침입절도

가. 2023. 2. 4.경 범행

피고인은 2023. 2. 4. 23:36 전북 전주시 완산구 B에 있는 피해자 C가 운영하는 'D'에 이르러 시정되지 않은 좌측 셔터문을 열고 들어가 그곳에 보관되어 있던 피해자 소유의 현금 450,000원1) 및 시가 100,000원 상당의 차와 약재를 꺼내어 가지고 갔다.

이로써 피고인은 야간에 타인이 관리하는 건조물에 침입하여 피해자의 재물을 절취하였다.

나. 2023. 2. 10. 23:57경 범행

피고인은 2023. 2. 10. 23:57경 제가항 기재 장소에 이르러 시정되어 있지 않은 좌측 셔터문을 열고 들어가 그곳에 보관되어 있던 피해자 소유의 시가 30,000원 상당의 튀밥, 시가 70,000원 상당의 약초, 시가 10,000원 상당의 보리쌀을 꺼내어 가지고 갔다. 이로써 피고인은 야간에 타인이 관리하는 건조물에 침입하여 피해자의 재물을 절취하였다.

다. 2023. 2. 24. 22:36경 범행

피고인은 2023. 2. 24. 22:36경 제가항 기재 장소에 이르러 시정되어 있지 않은 좌측 셔터문을 열고 들어가 그곳에 보관되어 있던 피해자 소유의 시가 미상의 쌀 1봉지를 꺼내어 가지고 갔다.

이로써 피고인은 야간에 타인이 관리하는 건조물에 침입하여 피해자의 재물을 절취하였다.

2. 야간건조물침입절도미수

피고인은 2023. 3. 2. 22:40경 제가항 기재 장소에 이르러 시정되어 있지 않은 좌측셔터문을 열고 들어가 절취할 물건을 찾던 중 신고를 받고 출동한 경찰관에게 현행범인으로 체포되었다.

이로써 피고인은 야간에 건조물에 침입하여 피해자의 재물을 절취하려다가 그 뜻을 이루지 못하고 미수에 그쳤다.


## 법령의 적용
각 형법 제330조(야간건조물침입절도의 점), 형법 제342조, 제330조(야간건조물침입절도미수의 점)
"""


## TODO

# index_name = 'vector-db'
# # index = pc.Index(index_name)

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-ada-002"
# )
# database = PineconeVectorStore.from_existing_index(index_name, embeddings)

# chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'])


# prompt = PromptTemplate(template="""문장을 바탕으로 질문에 답하세요.

# 문장: 
# {document}

# 질문: {query}
# """, input_variables=["document", "query"])

# @cl.on_chat_start
# async def on_chat_start():
#     await cl.Message(content="준비되었습니다! 메시지를 입력하세요!").send()

# @cl.on_message
# async def on_message(input_message):
#     input_message = input_message.content
#     documents = database.similarity_search(input_message, k=3) #← input_message로 변경

#     documents_string = ""

#     for document in documents:
#         documents_string += f"""
#     ---------------------------
#     {document.page_content}
#     """
#         break
#     result = chat([
#         HumanMessage(content=prompt.format(document=documents_string,
#                                            query=input_message)) #← input_message로 변경
#     ])
#     await cl.Message(content=result.content).send() #← 챗봇의 답변을 보냄
