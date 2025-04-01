import pandas as pd
import numpy as np
import yfinance as yf
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
        "currentPrice": info.get("currentPrice", np.nan)
    }

def tag(val, avg, high_good=True):
    if np.isnan(val) or np.isnan(avg):
        return 0.5
    return 1 if (val > avg if high_good else val < avg) else 0

def evaluate_stock(row, stock_map):
    try:
        stock = yf.Ticker(row["code"])
        info = stock.info
        metrics = get_metrics(info)
        if any(pd.isna(v) for v in metrics.values()):
            return None
        current_price = metrics["currentPrice"]

        # 行业均值
        industry = row["industry"]
        peers = stock_map[stock_map["industry"] == industry]["code"].tolist()
        pe_list, pb_list, roe_list = [], [], []
        for p in peers:
            try:
                p_info = yf.Ticker(p).info
                pe_list.append(p_info.get("trailingPE", np.nan))
                pb_list.append(p_info.get("priceToBook", np.nan))
                roe_list.append(p_info.get("returnOnEquity", np.nan))
            except:
                continue
        avg_pe, avg_pb, avg_roe = np.nanmean(pe_list), np.nanmean(pb_list), np.nanmean(roe_list)

        score_pe = tag(metrics["trailingPE"], avg_pe, False)
        score_pb = tag(metrics["priceToBook"], avg_pb, False)
        score_roe = tag(metrics["returnOnEquity"], avg_roe, True)
        industry_score = (score_pe + score_pb + score_roe) / 3
        industry_judge = "低估" if industry_score >= 0.6 else "高估"
        industry_judge = "合理" if industry_score == 0.5 else industry_judge

        # 情绪分析
        sentiment = fetch_news_sentiment_rss(row["code"])
        if sentiment > 0.1:
            sentiment_judge = "正面"
        elif sentiment < -0.1:
            sentiment_judge = "负面"
        else:
            sentiment_judge = "中性"

        # 技术面预测
        features = {
            "trailingPE": metrics["trailingPE"],
            "priceToBook": metrics["priceToBook"],
            "returnOnEquity": metrics["returnOnEquity"],
            "trailingEps": metrics["trailingEps"],
            "revenueGrowth": metrics["revenueGrowth"],
            "grossMargins": metrics["grossMargins"],
            "marketCap": metrics["marketCap"],
            "freeCashflow": metrics["freeCashflow"],
            "sentiment": sentiment
        }
        pred_price = model.predict(pd.DataFrame([features]))[0]
        tech_judge = "低估" if current_price < pred_price else "高估"

        # 模型判断
        if sentiment_judge == "负面":
            model_judge = "高估"
        elif sentiment_judge == "正面":
            model_judge = "低估"
        else:
            model_judge = "合理"

        score_map = {"低估": 0, "合理": 0.5, "高估": 1}
        final_score = 0.5 * score_map[model_judge] + 0.5 * score_map[industry_judge]
        if final_score < 0.5:
            final_judge = "低估"
        elif final_score > 0.5:
            final_judge = "高估"
        else:
            final_judge = "合理"

        return {
            "股票代码": row["code"],
            "公司名称": row["name_cn"],
            "当前价格": f"${current_price:.2f}",
            "预测价格": f"${pred_price:.2f}",
            "行业判断": industry_judge,
            "技术判断": tech_judge,
            "情绪判断": sentiment_judge,
            "模型判断": model_judge,
            "最终判断": final_judge
        }
    except Exception as e:
        print(f"跳过 {row['code']}: {e}")
        return None
