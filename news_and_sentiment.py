import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

def collect_news(url):
    """
    Collect news articles from the given URL.
    :param url: URL of the news site to scrape.
    :return: List of news articles as strings.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('h2')  # Example tag, change according to actual HTML structure
    return [article.get_text() for article in articles]

def clean_article(article):
    """
    Clean the article text.
    :param article: Raw article text.
    :return: Cleaned article text.
    """
    return article.strip()

def analyze_sentiment(article):
    """
    Analyze the sentiment of a given article.
    :param article: Cleaned article text.
    :return: Sentiment polarity and subjectivity.
    """
    analysis = TextBlob(article)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def main():
    url = 'https://example.com/news'  # Change to a valid news website
    news_articles = collect_news(url)
    
    for article in news_articles:
        cleaned_article = clean_article(article)
        polarity, subjectivity = analyze_sentiment(cleaned_article)
        print(f"Article: {cleaned_article}\nSentiment Polarity: {polarity}, Subjectivity: {subjectivity}\n")

if __name__ == "__main__":
    main()