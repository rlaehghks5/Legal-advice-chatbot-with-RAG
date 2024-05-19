# from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_pinecone import PineconeVectorStore
# import chainlit as cl

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# 환경 변수 설정

# 텍스트 파일이 있는 폴더 경로
folder_path = "./document"

# 폴더 내 모든 텍스트 파일 로드 및 문서 분할
all_docs = []
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        all_docs.extend(docs)

# 임베딩 생성
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Pinecone 벡터 스토어에 업로드
index_name = "vector-db"
docsearch = PineconeVectorStore.from_documents(all_docs, embeddings, index_name=index_name)

# 예시 쿼리 수행
query = "종빈이형이 내 친군데 비트코인 투자하라고 해서 사이트 회원가입을 하라고 해서 가입했어. 그리고 캐쉬를 충전했어, 그런데 종빈이형이 잠적했어. 이런경우 어떻게 처벌돼?"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)