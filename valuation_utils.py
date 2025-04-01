
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from sentiment_utils import fetch_news_sentiment_rss

model = joblib.load("valuation_model.pkl")

def get_metrics(info):
    return {
        "trailingPE": info.get("trailingPE", np.nan),
        "priceToBook": info.get("priceToBook", np.nan),
        "returnOnEquity": info.get("returnOnEquity", np.nan),
        "trailingEps": info.get("trailingEps", np.nan),
        "revenueGrowth": info.get("revenueGrowth", np.nan),
        "grossMargins": info.get("grossMargins", np.nan),
        "marketCap": info.get("marketCap", np.nan),
        "freeCashflow": info.get("freeCashflow", np.nan),
    }

def evaluate_stock(row, stock_map):
    try:
        code = row["code"]
        industry = row["industry"]
        stock = yf.Ticker(code)
        info = stock.info
        current_price = info.get("currentPrice", np.nan)
        if np.isnan(current_price):
            return None

        sentiment = fetch_news_sentiment_rss(code)
        sentiment_judge = "正面" if sentiment > 0.1 else "负面" if sentiment < -0.1 else "中性"

        metrics = get_metrics(info)
        if any(pd.isna(v) for v in metrics.values()):
            return None

        features = pd.DataFrame([{
            **metrics,
            "sentiment": sentiment
        }])
        pred_price = model.predict(features)[0]
        tech_judge = "低估" if current_price < pred_price else "高估"

        if sentiment_judge == "负面":
            model_judge = "高估"
        elif sentiment_judge == "正面":
            model_judge = "低估"
        else:
            model_judge = "合理"

        # 行业判断（仅用 PE）
        peer_codes = stock_map[stock_map["industry"] == industry]["code"]
        peer_pes = []
        for p in peer_codes:
            try:
                peer_pe = yf.Ticker(p).info.get("trailingPE", np.nan)
                if not np.isnan(peer_pe):
                    peer_pes.append(peer_pe)
            except:
                continue
        avg_pe = np.nanmean(peer_pes)
        industry_judge = "低估" if metrics["trailingPE"] < avg_pe else "高估" if not np.isnan(avg_pe) else "合理"

        score_map = {"低估": 0, "合理": 0.5, "高估": 1}
        model_score = score_map.get(model_judge, 0.5)
        industry_score = score_map.get(industry_judge, 0.5)
        final_score = 0.5 * model_score + 0.5 * industry_score
        final_judge = "低估" if final_score < 0.5 else "高估" if final_score > 0.5 else "合理"

        return {
            "股票代码": code,
            "公司名称": row["name_cn"],
            "当前价格": f"${current_price:.2f}",
            "预测价格": f"${pred_price:.2f}",
            "最终判断": final_judge
        }
    except Exception as e:
        print(f"❌ 错误处理 {row['code']}：{e}")
        return None
