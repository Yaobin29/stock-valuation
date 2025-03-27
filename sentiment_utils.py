import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def fetch_news_sentiment_rss(keyword: str, max_articles: int = 5) -> float:
    """
    使用 RSS 获取新闻标题与摘要，进行情绪分析（不访问网页）
    :param keyword: 英文关键词
    :param max_articles: 最多分析文章数
    :return: 平均 compound 情绪得分
    """
    try:
        url = f"https://news.google.com/rss/search?q={keyword}+when:7d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        entries = feed.entries[:max_articles]

        scores = []
        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            combined_text = title + ". " + summary
            if len(combined_text.strip()) > 30:  # 过滤太短的
                score = analyzer.polarity_scores(combined_text)["compound"]
                scores.append(score)

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    except Exception as e:
        print(f"❌ Error in RSS sentiment: {e}")
        return 0.0
