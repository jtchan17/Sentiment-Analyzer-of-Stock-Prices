import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from newspaper import Article
from newspaper import Config
from wordcloud import WordCloud, STOPWORDS
from gnews import GNews

nltk.download('punkt')
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

# save the company name in a variable
# company_name = input("Please provide the name of the Company or a Ticker: ")
#As long as the company name is valid, not empty...
# if company_name != '':
print('Searching for and analyzing company_name, Please be patient, it might take a while...')

# #Extract News with Google News
companies = ['AAPL', 'AMZN', 'TSLA', 'MSFT', 'META']
google_news = GNews()
google_news.max_results = 1000 # number of responses across a keyword

#use date range 
google_news.start_date = (2023, 10, 1) # Search from 1st Jan 2021
google_news.end_date = (2024, 1, 1) # Search until 31st December 2023

for i in companies:
    news_by_keyword = google_news.get_news(i)
    df = pd.DataFrame(news_by_keyword)
    df.to_csv(f'{i}_2023Q4_Financial_News.csv', index=True)
    print(df)