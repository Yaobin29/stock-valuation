import feedparser
import trafilatura
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def fetch_news_sentiment(keyword: str) -> float:
    """
    使用 Google News RSS 搜索最近一周新闻，抓取正文进行情绪分析
    :param keyword: 英文关键词（如 'Apple'）
    :return: 平均 compound 情绪得分（正面为正值，负面为负值）
    """
    try:
        # 构建 RSS URL（最近一周，英文新闻）
        url = f"https://news.google.com/rss/search?q={keyword}+when:7d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        entries = feed.entries[:5]  # 最多取 5 篇新闻

        texts = []
        for entry in entries:
            article_url = entry.link
            downloaded = trafilatura.fetch_url(article_url)
            if downloaded:
                extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                if extracted and len(extracted) > 200:  # 排除超短无效正文
                    texts.append(extracted)

        if not texts:
            return 0.0  # 无有效内容，视为中性

        # 情绪分析
        scores = [analyzer.polarity_scores(text)["compound"] for text in texts]
        avg_score = sum(scores) / len(scores)
        return avg_score

    except Exception as e:
        print(f"❌ Error in fetch_news_sentiment: {e}")
        return 0.0
