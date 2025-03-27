import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

analyzer = SentimentIntensityAnalyzer()

def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    return ' '.join(soup.stripped_strings)

def fetch_google_news_rss(query, max_articles=5):
    """
    从 Google News RSS 抓取最近一周的新闻条目。
    """
    url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries[:max_articles]:
        published = datetime(*entry.published_parsed[:6])
        if (datetime.now() - published).days > 7:
            continue
        articles.append(entry.link)

    return articles

def extract_article_text(url):
    """
    从新闻网页中提取正文内容。
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return ""
        return clean_html(response.text)
    except:
        return ""

def fetch_news_sentiment(query):
    """
    主函数：根据公司英文名抓取新闻并计算平均情绪分数。
    """
    articles = fetch_google_news_rss(query)
    scores = []

    for url in articles:
        text = extract_article_text(url)
        if text and len(text) > 300:  # 太短的不分析
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

    if not scores:
        return 0.0  # 默认中性

    return np.mean(scores)
