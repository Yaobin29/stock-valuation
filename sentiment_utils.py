import feedparser
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import urllib.parse

analyzer = SentimentIntensityAnalyzer()

def fetch_google_news_rss(query):
    if not query.strip():
        return []

    query_encoded = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/rss/search?q={query_encoded}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return feed.entries[:10]

def extract_article_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except:
        return ""

def analyze_sentiment(text):
    if not text.strip():
        return 0.0
    score = analyzer.polarity_scores(text)["compound"]
    return score

def fetch_news_sentiment(query):
    articles = fetch_google_news_rss(query)
    scores = []
    cutoff_date = datetime.utcnow() - timedelta(days=7)

    for entry in articles:
        try:
            pub_date = datetime(*entry.published_parsed[:6])
            if pub_date < cutoff_date:
                continue
            url = entry.link
            content = extract_article_text(url)
            score = analyze_sentiment(content)
            scores.append(score)
        except:
            continue

    return sum(scores) / len(scores) if scores else 0.0
