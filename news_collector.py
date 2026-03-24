import requests
import time
from bs4 import BeautifulSoup

class NewsCollector:
    def __init__(self):
        self.sources = [self.get_newsapi_news, self.get_alpha_vantage_news, self.web_scrape_news]

    def collect_news(self):
        for source in self.sources:
            try:
                return source()
            except Exception as e:
                print(f"Error retrieving news from {source.__name__}: {e}")
                time.sleep(1)  # wait before retrying
        print("All sources failed to retrieve news.")
        return None

    def get_newsapi_news(self):
        url = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_NEWSAPI_KEY'
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['articles']

    def get_alpha_vantage_news(self):
        url = 'https://www.alphavantage.co/query?function=NEWS&apikey=YOUR_ALPHA_VANTAGE_KEY'
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['feed']

    def web_scrape_news(self):
        url = 'https://example.com/news'
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')
        return [{'title': article.h2.text, 'link': article.a['href']} for article in articles]

if __name__ == '__main__':
    collector = NewsCollector()
    news_data = collector.collect_news()
    if news_data:
        for news in news_data:
            print(news)