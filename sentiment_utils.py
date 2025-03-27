import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

analyzer = SentimentIntensityAnalyzer()

def fetch_google_news_rss(query, max_articles=5):
    url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:max_articles]]

def fetch_news_sentiment(query):
    titles = fetch_google_news_rss(query)
    scores = []

    for title in titles:
        score = analyzer.polarity_scores(title)["compound"]
        scores.append(score)

    if not scores:
        return 0.0  # 默认中性情绪
    return sum(scores) / len(scores)
