import feedparser
import trafilatura
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def fetch_news_sentiment(keyword: str, max_articles: int = 5) -> float:
    """
    使用 Google News RSS 搜索最近一周新闻，抓取标题与正文进行加权情绪分析
    :param keyword: 英文关键词（如 'Apple'）
    :param max_articles: 最大处理新闻数量
    :return: 平均 compound 情绪得分（正面为正值，负面为负值）
    """
    try:
        url = f"https://news.google.com/rss/search?q={keyword}+when:7d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        entries = feed.entries[:max_articles]

        scores = []
        for entry in entries:
            title = entry.title
            article_url = entry.link
            downloaded = trafilatura.fetch_url(article_url)
            body_score = 0.0

            if downloaded:
                extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                if extracted and len(extracted) > 200:
                    body_score = analyzer.polarity_scores(extracted)["compound"]

            title_score = analyzer.polarity_scores(title)["compound"]
            combined_score = 0.2 * title_score + 0.8 * body_score if body_score else title_score
            scores.append(combined_score)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    except Exception as e:
        print(f"❌ Error in fetch_news_sentiment: {e}")
        return 0.0
