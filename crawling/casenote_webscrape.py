import json
import time
from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
import re
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup

# start = time.time()
# today = datetime.now()

# css 찾을때 까지 10초대기
def time_wait(num, code):
    try:
        wait = WebDriverWait(driver, num).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, code)))
    except:
        print(code, '태그를 찾지 못하였습니다.')
        driver.quit()
    return wait
driver = webdriver.Chrome()

# (1) 고단 ------------------------------------------------
driver.get('https://casenote.kr/search/?q=&partial=0&exclusion=%EB%B3%91%ED%95%A9&sort=0&period=0&court=0,2&case=1&nu=%EA%B3%A0%EB%8B%A8')
time.sleep(1) 

# 클릭 가능할 때까지 대기
wait = WebDriverWait(driver, 10)   

element = wait.until(EC.element_to_be_clickable((By.XPATH , '/html/body/div[4]/div[1]/div[3]/div[1]/a')))
element.click()
time.sleep(3) 

# css를 찾을때 까지 10초 대기
# time_wait(10, 'div.cn-case-contents > div.cn-case-left.fs3 > div.cn-case-title > h1')
#cn-case > div.cn-case-contents
# 제목 출력
# 페이지 소스를 가져옴
html = driver.page_source

# 페이지 소스 출력 (디버깅 목적)
# print(html)


# 'cn-case-title' 클래스를 가진 <div> 태그 안의 <h1> 태그 찾기
# title= driver.find_element(By.CSS_SELECTOR ,'#cn-case > div.cn-case-contents > div.cn-case-left.fs3 > div.cn-case-title > h1')
# title= driver.find_element(By.XPATH ,'/html/head/title')
download_button = driver.find_element(By.XPATH, '//*[@id="cn-case"]/div[1]/div[1]/div[2]/div[4]/button')
download_button.click()

# print(title)



driver.quit()
