import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def clean_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text()

def extract_news_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return ""
        return clean_text(response.text)
    except:
        return ""

def fetch_google_news_rss(query, max_articles=5):
    feed_url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    articles = []

    for entry in feed.entries[:max_articles]:
        published = datetime(*entry.published_parsed[:6])
        if published >= datetime.now() - timedelta(days=7):
            content = extract_news_content(entry.link)
            if content:
                articles.append(content)
    return articles

def analyze_sentiment(contents):
    scores = []
    for text in contents:
        score = analyzer.polarity_scores(text)["compound"]
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

def get_sentiment_score_from_news(query):
    contents = fetch_google_news_rss(query)
    return analyze_sentiment(contents)
