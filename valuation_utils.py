import numpy as np
import yfinance as yf
import joblib
from sentiment_utils import fetch_news_sentiment_rss

# 加载模型
model = joblib.load("valuation_model.pkl")

def safe_get(info, key):
    return info.get(key, np.nan)

def evaluate_stock(row, stock_map):
    try:
        stock = yf.Ticker(row["code"])
        info = stock.info
        code = row["code"]
        name = row["name_cn"]
        industry = row["industry"]

        current_price = safe_get(info, "currentPrice")
        if np.isnan(current_price):
            return None

        pe = safe_get(info, "trailingPE")
        pb = safe_get(info, "priceToBook")
        roe = safe_get(info, "returnOnEquity")
        eps = safe_get(info, "trailingEps")
        rev_growth = safe_get(info, "revenueGrowth")
        margin = safe_get(info, "grossMargins")
        fcf = safe_get(info, "freeCashflow")
        mktcap = safe_get(info, "marketCap")

        sentiment = fetch_news_sentiment_rss(code)
        sentiment_judge = "正面" if sentiment > 0.1 else "负面" if sentiment < -0.1 else "中性"
        sentiment_score = {"负面": 1, "中性": 0.5, "正面": 0}[sentiment_judge]

        # 技术面预测
        features = {
            "trailingPE": pe, "priceToBook": pb, "returnOnEquity": roe,
            "trailingEps": eps, "revenueGrowth": rev_growth, "grossMargins": margin,
            "marketCap": mktcap, "freeCashflow": fcf, "sentiment": sentiment
        }

        if any(np.isnan(v) for v in features.values()):
            return None  # 缺失严重则跳过

        pred_price = model.predict([list(features.values())])[0]
        tech_judge = "低估" if current_price < pred_price else "高估"
        tech_score = {"低估": 0, "高估": 1}[tech_judge]

        # 模型判断（情绪+技术）
        model_score = 0.6 * tech_score + 0.4 * sentiment_score
        model_judge = "低估" if model_score < 0.5 else "高估" if model_score > 0.5 else "合理"

        # 行业判断逻辑（PE > PB > ROE）
        peer_codes = stock_map[stock_map["industry"] == industry]["code"]
        peers = [yf.Ticker(t).info for t in peer_codes if t != code]
        peer_pe = np.nanmean([p.get("trailingPE", np.nan) for p in peers])
        peer_pb = np.nanmean([p.get("priceToBook", np.nan) for p in peers])
        peer_roe = np.nanmean([p.get("returnOnEquity", np.nan) for p in peers])

        if not np.isnan(pe) and not np.isnan(peer_pe):
            industry_judge = "低估" if pe < peer_pe else "高估"
        elif not np.isnan(pb) and not np.isnan(peer_pb):
            industry_judge = "低估" if pb < peer_pb else "高估"
        elif not np.isnan(roe) and not np.isnan(peer_roe):
            industry_judge = "低估" if roe > peer_roe else "高估"
        else:
            industry_judge = "合理"

        industry_score = {"低估": 0, "高估": 1, "合理": 0.5}[industry_judge]
        final_score = 0.5 * model_score + 0.5 * industry_score
        final_judge = "低估" if final_score < 0.5 else "高估" if final_score > 0.5 else "合理"

        return {
            "代码": code,
            "公司": name,
            "当前价格": f"${current_price:.2f}",
            "预测价格": f"${pred_price:.2f}",
            "最终判断": final_judge
        }

    except Exception as e:
        print(f"跳过 {row['code']}：{e}")
        return None
