import nltk
nltk.download('punkt')

from newspaper import Article
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import feedparser

analyzer = SentimentIntensityAnalyzer()

def fetch_google_news_rss(query):
    query_encoded = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query_encoded}+when:7d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return feed.entries[:5]  # 取前 5 条新闻

def fetch_news_sentiment(query):
    entries = fetch_google_news_rss(query)
    if not entries:
        return "中性"

    scores = []

    for entry in entries:
        try:
            article = Article(entry.link)
            article.download()
            article.parse()
            text = article.text
            if not text.strip():
                continue

            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)
        except Exception as e:
            print(f"❌ Failed to process article: {e}")
            continue

    if not scores:
        return "中性"

    avg_score = sum(scores) / len(scores)

    if avg_score > 0.2:
        return "正面"
    elif avg_score < -0.2:
        return "负面"
    else:
        return "中性"
