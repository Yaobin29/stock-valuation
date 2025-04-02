import numpy as np
import yfinance as yf
import pandas as pd
from sentiment_utils import fetch_news_sentiment_rss
import joblib

model = joblib.load("valuation_model.pkl")

def get_financial_metrics(info):
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

def evaluate_stock(row, stock_map):
    try:
        code = row["code"]
        industry = row["industry"]
        info = yf.Ticker(code).info
        metrics = get_financial_metrics(info)
        current_price = metrics["currentPrice"]

        if current_price is None or pd.isna(current_price):
            return None

        # 行业均值（优先顺序 PE > PB > ROE）
        industry_codes = stock_map[stock_map["industry"] == industry]["code"]
        industry_data = []
        for ic in industry_codes:
            try:
                i_info = yf.Ticker(ic).info
                industry_data.append(get_financial_metrics(i_info))
            except:
                continue
        industry_df = pd.DataFrame(industry_data)

        industry_judge = "合理"
        if not pd.isna(metrics["trailingPE"]):
            avg_pe = np.nanmean(industry_df["trailingPE"])
            if not pd.isna(avg_pe):
                industry_judge = "低估" if metrics["trailingPE"] < avg_pe else "高估"
        elif not pd.isna(metrics["priceToBook"]):
            avg_pb = np.nanmean(industry_df["priceToBook"])
            if not pd.isna(avg_pb):
                industry_judge = "低估" if metrics["priceToBook"] < avg_pb else "高估"
        elif not pd.isna(metrics["returnOnEquity"]):
            avg_roe = np.nanmean(industry_df["returnOnEquity"])
            if not pd.isna(avg_roe):
                industry_judge = "高估" if metrics["returnOnEquity"] < avg_roe else "低估"

        # 情绪面
        sentiment = fetch_news_sentiment_rss(code)
        if sentiment > 0.1:
            sentiment_judge = "正面"
        elif sentiment < -0.1:
            sentiment_judge = "负面"
        else:
            sentiment_judge = "中性"

        # 技术面估值
        input_features = {
            key: metrics[key] for key in [
                "trailingPE", "priceToBook", "returnOnEquity", "trailingEps",
                "revenueGrowth", "grossMargins", "marketCap", "freeCashflow"
            ]
        }
        input_features["sentiment"] = sentiment
        if any(pd.isna(v) for v in input_features.values()):
            return None

        pred_price = model.predict(pd.DataFrame([input_features]))[0]
        tech_judge = "低估" if current_price < pred_price else "高估"

        # 模型判断
        if sentiment_judge == "负面":
            model_judge = "高估"
        elif sentiment_judge == "正面":
            model_judge = "低估"
        else:
            model_judge = "合理"

        score_map = {"低估": 0, "合理": 0.5, "高估": 1}
        model_score = score_map[model_judge]
        industry_score = score_map[industry_judge]
        final_score = 0.5 * model_score + 0.5 * industry_score

        if final_score < 0.5:
            final_judge = "低估"
        elif final_score > 0.5:
            final_judge = "高估"
        else:
            final_judge = "合理"

        return {
            "公司名称": row["name_cn"],
            "股票代码": code,
            "当前价格": f"${current_price:.2f}",
            "预测价格": f"${pred_price:.2f}",
            "最终判断": final_judge
        }
    except Exception as e:
        print(f"跳过 {row['code']}: {e}")
        return None
