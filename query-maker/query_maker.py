import os
from typing import *

import pandas as pd
from datasets import Dataset
from openai import OpenAI


class GPTQueryMaker():
    def __init__(self, prompt, model="gpt-3.5-turbo"):
        self.prompt = prompt
        self.model = model
        
        
    def generate_simple_query(self, crime_facts: str) -> str:
        """
        주어진 범죄사실을 일반인이 이해할 수 있는 자연어 쿼리로 변환합니다.
        
        Args:
        crime_facts (str): 복잡한 범죄사실을 설명하는 텍스트. 
        문자열 포매팅의 형태로 prompt에 
        
        Returns:
        str: 간단한 자연어 쿼리
        """
        
        formatted_prompt = self.prompt.format(crime_facts=crime_facts)
        
        
        client = OpenAI(api_key="your-api-key")
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system", "content": formatted_prompt
                }
            ],
            model=self.model,
        )

        
        return response.choices[0].message.content
    
    
    def generate_querys_from_dataframe(self, df: pd.DataFrame, new_col_name: str="자연어 쿼리", from_col_name: str="범죄 사실") -> pd.DataFrame:
        df[new_col_name] = df[from_col_name].apply(self.generate_querys_from_dataframe)
        
        return df
    

    def generate_querys_from_datasets(self, dataset: Dataset, new_col_name: str="자연어 쿼리", from_col_name: str="범죄 사실") -> Dataset:
        def generate_query(row):
            row[new_col_name] = self.generate_simple_query(row[from_col_name])
            return row
        
        dataset = dataset.map(generate_query)
        
        return dataset
    
    
    
if __name__ == "__main__":
    import os
    os.environ["OPEN_AI_API_KEY"] = "your-api-key"
    
    from datasets import load_from_disk, DatasetDict

    
    
    ds = load_from_disk("../dataset/")['test']
    ds1 = ds.select(range(500))
    ds2 = ds.select(range(500, 1000))
    ds3 = ds.select(range(1000, 1500))
    ds4 = ds.select(range(1500, 2137))
    
    prompt = """당신은 복잡한 범죄 사실들을 마치 일반인이 얘기하듯 자연스러운 문장으로 풀어쓰는 한국어 챗봇입니다.
    제가 범죄사실을 말하면, 당신은 실제 변호사에게 물어보듯 해당 범죄사실을 자연스러운 문장으로 풀어쓰세요. 
    이때 다음과 같은 규칙들을 반드시 준수해야합니다.
    1. 당신이 범죄 사실을 풀어쓸 때 담당할 역할을 피의자, 피해자, 혹은 그 주변인물 중 무작위로 정하세요.
    2. 다음의 범죄 사실에을 법률 전문가가 아닌 일반인이 물어보듯 자연스럽게 다시 풀어쓰세요.
    3. 이때, 500자 이내로 서술하세요. 그리고 매번 범죄사실이 주어질 때 마다 다른 사람이 질문하듯 감정, 역할, 말투, 법률 지식 수준 등을 바꿔주세요.
    4. 마지막에는 실제로 당신이 담당한 역할을 맡은 사람(피의자, 피해자, 혹은 그 주변인물)이 변호사에게 해당 범죄의 경우 어떤 법령에 의해서 처벌을 받는지 법률상담을 요청하는 다양한 질문들을 덧붙여주세요. 
    5. 1인칭으로 대답하세요.

    범죄 사실은 다음과 같습니다:
    {crime_facts}"""

    query_maker = GPTQueryMaker(prompt=prompt)
    
    print("query making start!")
    for ds_part in [ds1, ds2, ds3, ds4]:
        ds_part_with_query = query_maker.generate_querys_from_datasets(ds_part, new_col_name='query', from_col_name="facts")
        ds_part_with_query.save_to_disk(f'./{ds_part}/')
        print(f"========== {ds_part} query generated ==========")
