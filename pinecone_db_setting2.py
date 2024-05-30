import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# 환경 변수 설정
folder_path = "/home/compu/KDH/langchain/Legal-advice-chatbot-with-RAG/document"
# os.environ['PINECONE_API_KEY'] = ''

pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
# 폴더 내 모든 텍스트 파일 로드 및 문서 분할
all_docs = []
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        loader = TextLoader(file_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        case_id = filename.split('_')[-1].split('.')[0]
        
        with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
        for doc in docs:
            doc.metadata = {"case_id": case_id, "text": text_content}
        all_docs.extend(docs)
    # break
# SentenceTransformer 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 문서 임베딩
def embed_documents(docs):
    embeddings = [model.encode(doc.page_content) for doc in docs]
    return embeddings

embeddings = embed_documents(all_docs)

# # Pinecone 벡터 스토어에 업로드
index_name = "vector-db"
# # if index_name not in pinecone.list_indexes():
# #     pinecone.create_index(index_name, dimension=embeddings[0].shape[0])

index = pc.Index(index_name)

# 문서와 임베딩을 인덱스에 추가
vectors = [ {"id" : str(i), "values": emb, "metadata": doc.metadata} for i, (doc, emb) in enumerate(zip(all_docs, embeddings))]
index.upsert(vectors)

### 예시 쿼리 수행
query = "저는 이번에 자가격리 법을 위반했어용"
query_embedding = model.encode(query).tolist()
results = index.query(vector=query_embedding, top_k=3)

for match in results['matches']:
    doc_id = (match['id'])
    print(doc_id)
    temp = index.fetch(ids = [doc_id])
    
    print('case_id:', temp['vectors'][doc_id]['metadata']['case_id'])
    print('text:', temp['vectors'][doc_id]['metadata']['text'])
    print()
