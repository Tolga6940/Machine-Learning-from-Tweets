import snscrape.modules.twitter as sntwitter
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as ec
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
import pandas as pd
chromeOptions = Options()
chromeOptions.add_argument("--start-maximized")
Path="C:\common\chromedriver.exe"
driver = webdriver.Chrome(Path, options=chromeOptions)
tweeter_url = "https://twitter.com/login"
driver.get(tweeter_url)
wait = WebDriverWait(driver, 10)
username_input = wait.until(ec.visibility_of_element_located((By.NAME, "text")))
username_input.send_keys('vizeodevi')
password_input = wait.until(ec.visibility_of_element_located((By.NAME, "password")))
password_input.send_keys('vizeödevi12345')
login_button = wait.until(ec.visibility_of_element_located((By.XPATH, "//div[@data-testid='LoginForm_Login_Button']")))
login_button.click()
search_input = wait.until(ec.visibility_of_element_located((By.XPATH, "//div/input[@data-testid='SearchBox_Search_Input']")))
search_input.clear()
search_input.send_keys("Elden Ring" + Keys.ENTER)
search_input="Elden Ring"
maxtweets = 20000
ekle = []
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_input + ' since:2022-02-25 until:2022-04-09 lang:"en" ').get_items()):
    if (i > maxtweets):
        break
    print(str(i),"tane tweet",tweet.content)
    data=tweet.content
    ekle.append(data)
tweets_df2 = pd.DataFrame(ekle, columns=['Text'])
with open('C:\\Users\\tolga\\OneDrive\\Masaüstü\\tweet.csv', 'w', encoding="utf-8")as f:
    writer=csv.writer(f)
    writer.writerows(tweets_df2['Text'])
f.close()
def clean(location):
    letters = re.sub("@[A-Za-z0-9]+","",location)
    letters = re.sub(r"(?:\@|http?\://|https?\://|www)\S+","", letters)
    letters = re.sub(',', '', letters)
    letters = re.sub('//', '', letters)
    letters=" ".join(letters.split())
    letters=letters.replace("#","").replace("_","")
    return letters
tweets_df2['Text'] = tweets_df2['Text'].apply(clean)
with open('C:\\Users\\tolga\\OneDrive\\Masaüstü\\tweetler1.csv', 'w', encoding="utf-8")as f:
    for x in tweets_df2['Text']:
        f.write(x)
        f.write('\n')
    f.close()