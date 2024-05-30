import openai
import os
import json
from datasets import load_dataset
import time

# OpenAI API 키 설정
openai.api_key = ''

# 데이터 로드
data_st_plus = load_dataset("lbox/lbox_open", "statute_classification_plus")

def generate_positive_negative_examples(anchor_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 법률 텍스트 데이터를 생성하는 데 도움이 되는 조수입니다."},

            {"role": "user", "content": f"""
### 범죄 사실 텍스트 :
질병관리청장, 시·도지사 또는 시장·군수·구청장은 감염병을 예방하기 위하여 감염병의심자를 적당한 장소에 일정한 기간 입원 또는 격리시키는 조치를 하거나 그에 필요한 일부 조치를 하여야 하고, 위와 같은 조치를 받은 사람은 이를 위반하여서는 아니 된다. 피고인은 감염병의심자에 해당하여 2021. 4. 12. 원주시장으로부터 ‘2021. 4. 12.부터 같은 달 22. 12:00경까지, 원주시 B아파트 C호 자택에서 격리조치 하라’는 내용의 격리통지를 받았음에도, 2021. 4. 14. 11:00경부터 같은 날 16:27경까지 위 격리장소에서 이탈하여 원주시 무실동에 있는 PC방을 방문하는 등 격리 조치를 위반하였다.

### 유사한 텍스트 :
2021년 4월 12일, 원주시장으로부터 받은 격리통지에 따라 원주시 B아파트 C호 자택에서 격리 조치를 받아야 했던 피고인은, 2021년 4월 14일 오전 11시부터 같은 날 오후 4시 27분까지 이를 위반하고 PC방을 방문했습니다. 이는 감염병 예방을 위한 법적 조치에 대한 중대한 위반이며, 해당 행위에 대해 적절한 법적 조치가 필요합니다.

### 다른 텍스트 :
피고인 G은 A에 있는 회사 B의 사무실에서 직원들과 논의 중이었는데, 그곳을 지나가던 C 등 다수의 직원들이 듣고 있음에도 불구하고, 피해자에게 "쟤는 정말 바보 같은 녀석이야, 무슨 생각으로 그런 소리를 하는 거야? 진짜로 귀찮아."라고 큰소리로 말하여 피해자를 모욕하였다.

### 범죄 사실 텍스트 :
피고인은 (차량번호 1 생략) 쏘나타 택시의 운전업무에 종사하는 사람이다. 피고인은 2020. 12. 15. 18:30경 위 택시를 운전하여 서울 강북구 도봉로 87에 있는 교차로를 삼양사거리 방면에서 B초교 방면으로 진행하게 되었다. 그곳은 신호기에 의하여 교통정리가 행하여지는 교차로이므로, 이러한 경우 운전업무에 종사하는 사람에게는 교통 신호에 따라 안전하게 운전하여야 할 업무상 주의의무가 있었다. 그럼에도 불구하고 피고인은 이를 게을리 한 채, 차량 신호등이 황색 등화로 바뀌었음에도 정지하지 아니하고 신호를 위반하여 교차로에 진입한 과실로, 마침 차량 진행신호에 따라 교차로에 진입하는 피해자 C(남, 29세)이 운전하는 (차량번호 2 생략) 오토바이를 피고인 택시의 우측 앞 범퍼 부분으로 들이받았다. 결국 피고인은 위와 같은 업무상 과실로 피해자에게 약 6주간의 치료가 필요한 비장열상 등 상해를 입게 하였다.

### 유사한 텍스트 :
피고인 A씨는 아이오닉 전기차의 운전사로서 업무를 수행하던 중, 2023년 7월 10일 오후 3시 경 서울 강남구 테헤란로에서 신호등이 빨간불로 바뀌자 정지해야 할 의무를 게을리 하고 횡단보도를 건너던 보행자에게 충격을 주어 중상을 입혔다. 결국 피해자는 사고로 인해 병원에 입원하여 치료를 받아야 했다.

### 다른 텍스트 :
지난 달, 지역 은행에서 고객들을 대상으로 한 이메일 사기가 발생했습니다. 가짜 은행 이메일을 받은 고객들은 링크를 클릭하면서 개인 정보를 입력했고, 이로써 사기꾼들이 고객의 계정을 침해했습니다. 경찰이 수사 중이며, 은행은 고객들에게 보안 주의를 당부하고 보호 조치를 강화했습니다.

### 범죄 사실 텍스트 :
누구든지 불특정인을 상대로 금품이나 그 밖의 재산상의 이익을 수수하거나 수수하기로 약속하고 성교행위를 하는 것을 알선하여서는 아니 된다. 그럼에도 불구하고 피고인은 2020. 12. 20.경부터 2021. 1. 18.경까지 사이에 고양시 일산동구 B건물 C호를 임차하고, 성매매 여성으로 D 등을 고용하고, 인터넷 성매매 광고 사이트에 ‘E’이라는 상호로 위 업소를 광고하여 광고를 보고 찾아온 불특정 다수의 남성 손님들로부터 성매매 시간 등에 따라 성매매 대금으로 17만 원에서 34만 원을 받고 위 오피스텔로 안내하여 안에서 대기하고 있던 D 등의 성매매 여성과 성교행위를 하도록 하였다. 이로써 피고인은 영업으로 성매매를 알선하였다.

### 유사한 텍스트 :
2023년 7월 10일, 피고인은 서울시 강남구의 한 아파트를 임차하여 성매매 업소를 운영했다. 그는 인터넷 성매매 사이트에 광고를 올려 'F'라는 가명으로 홍보를 하였으며, 이에 응답한 남성 손님들로부터 20만 원에서 50만 원의 대금을 받았다. 피고인은 이 대금을 받은 후, 손님들을 해당 아파트로 안내하여 성매매를 중개하였다. 이에 대해 경찰은 피고인을 체포하고 영업으로 성매매를 알선한 혐의로 기소했다.

### 다른 텍스트 :
2022년 9월 5일, 피고인은 서울 중구의 한 은행에서 강도로 은행원을 위협하고 5천만 원의 현금을 강탈하였다. 그는 은행 내 CCTV를 뚫고 방탄 조끼를 착용하여 범행을 진행했으며, 은행원과 고객들을 위협하며 자신의 요구를 이행하도록 했다. 경찰은 피고인을 추적하여 체포하였으며, 강도로 혐의를 적발하여 기소할 예정이다.

위 에시를 참고하여 범죄 사실에 대한 텍스트에 대해서 주제가 유사한 텍스트, 주제가 다른 텍스트를 생성하세요.

구체적으로, 먼저 범죄 사실에 대한 텍스트를 생성하고 그에 맞게 주제가 유사한 텍스트, 주제가 다른 텍스트를 각각 생성하여 아래 [] 에 들어갈 텍스트를 생성하세요. 
반드시 아래의 형태를 만족하게 생성하세요.

### 범죄 사실 텍스트 :
[]
### 유사한 텍스트 :
[]
### 다른 텍스트 :
[]
### 범죄 사실 텍스트 :
[]
### 유사한 텍스트 :
[]
### 다른 텍스트 :
[]
### 범죄 사실 텍스트 :
[]
### 유사한 텍스트 :
[]
### 다른 텍스트 :
[]
"""},
        ]
    )

    # similar_text = response.choices[0].message["content"].strip()

    return response.choices[0].message["content"]

# 트레이닝 예제 생성 및 저장
train_examples = []
# t = 9
# for example in data_st_plus['train']:
#     if t >= 10: break
#     anchor = example['facts']
#     text = generate_positive_negative_examples(anchor)
#     # train_examples.append({'anchor': anchor, 'positive': positive, 'negative': negative})
#     train_examples.append({'text': text})

#     t+=1

print('############ gen')
for i in range(1000):
    try:
        text = generate_positive_negative_examples('')
        train_examples.append({'text': text})
        print(f'{i}th text: ', text)
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(30)
        continue

# 예제를 파일로 저장
with open('train_examples.json', 'w', encoding='utf-8') as f:
    json.dump(train_examples, f)
    
import json

# JSON 파일 읽어오기
with open('train_examples.json', 'r', encoding='utf-8') as f:
    train_examples = json.load(f)
    
result_list = []

print('############ split')

for idx in range(len(train_examples)):

    try:
        text_data = train_examples[idx]['text']
        split_texts = text_data.split("### 범죄 사실 텍스트 :")[1:]

        for text in split_texts:
            # 각 섹션에서 "### 유사한 텍스트 :", "### 다른 텍스트 :"를 기준으로 세 부분으로 나눕니다.
            parts = text.split("### 유사한 텍스트 :")
            anchor = parts[0].strip()
            
            parts = parts[1].split("### 다른 텍스트 :")
            positive = parts[0].strip()
            negative = parts[1].strip()
            
            # 결과 리스트에 딕셔너리 형태로 추가합니다.
            result_list.append({"anchor": anchor, "positive": positive, "negative": negative})
    except IndexError:
                # 예외가 발생하면 오류 메시지 출력 후 다음 인덱스로 넘어감
                print("IndexError: 인덱스가 범위를 초과했습니다.")
                continue
            
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(result_list, f)
    # print(result_list)

with open('data.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

print(res)




