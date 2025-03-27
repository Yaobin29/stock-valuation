import feedparser
from newspaper import Article
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time

analyzer = SentimentIntensityAnalyzer()

def fetch_google_news_rss(query):
    base_url = "https://news.google.com/rss/search?q="
    now = datetime.utcnow()
    one_week_ago = now - timedelta(days=7)
    url = f"{base_url}{query}+after:{one_week_ago.strftime('%Y-%m-%d')}&hl=en-US&gl=US&ceid=US:en"
    return feedparser.parse(url)

def fetch_news_sentiment(query):
    feed = fetch_google_news_rss(query)
    scores = []

    for entry in feed.entries[:10]:  # 只分析前10篇
        url = entry.link
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            if len(text) > 100:
                sentiment = analyzer.polarity_scores(text)
                scores.append(sentiment["compound"])
                time.sleep(0.5)  # 避免被封IP
        except Exception as e:
            continue

    if scores:
        return sum(scores) / len(scores)
    else:
        return 0.0
