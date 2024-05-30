import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator

# Sentence Transformer 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# data.json에서 데이터 읽기
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 나만의 학습 데이터 구축
train_examples = []
for item in data:
    train_examples.append(InputExample(texts=[item['anchor'], item['positive'], item['negative']], label=1)

train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

# 손실 함수 정의
train_loss = losses.TripletLoss(model=model)

# 모델 미세 조정
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)}

# 모델 저장
model.save("tuned_models/fine_tuned_model")
