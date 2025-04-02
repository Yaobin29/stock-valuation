import numpy as np
import pandas as pd
import yfinance as yf
from sentiment_utils import fetch_news_sentiment_rss

def get_metric(info, name):
    return info.get(name, np.nan)

def evaluate_stock(row, stock_map, model):
    try:
        code = row["code"]
        industry = row["industry"]
        name_cn = row["name_cn"]

        stock = yf.Ticker(code)
        info = stock.info

        pe = get_metric(info, "trailingPE")
        pb = get_metric(info, "priceToBook")
        roe = get_metric(info, "returnOnEquity")
        eps = get_metric(info, "trailingEps")
        revenue_growth = get_metric(info, "revenueGrowth")
        gross_margin = get_metric(info, "grossMargins")
        free_cashflow = get_metric(info, "freeCashflow")
        market_cap = get_metric(info, "marketCap")
        current_price = get_metric(info, "currentPrice")

        # 行业平均比较
        industry_pe, industry_pb, industry_roe = [], [], []
        industry_stocks = stock_map[stock_map["industry"] == industry]["code"].tolist()
        for ticker in industry_stocks:
            try:
                data = yf.Ticker(ticker).info
                industry_pe.append(data.get("trailingPE", np.nan))
                industry_pb.append(data.get("priceToBook", np.nan))
                industry_roe.append(data.get("returnOnEquity", np.nan))
            except:
                continue
        avg_pe = np.nanmean(industry_pe)
        avg_pb = np.nanmean(industry_pb)
        avg_roe = np.nanmean(industry_roe)

        def tag(val, avg, high_good=True):
            if np.isnan(val) or np.isnan(avg):
                return 0.5
            return 1 if (val > avg if high_good else val < avg) else 0

        score_pe = tag(pe, avg_pe, False)
        score_pb = tag(pb, avg_pb, False)
        score_roe = tag(roe, avg_roe, True)
        industry_score = (score_pe + score_pb + score_roe) / 3
        industry_judge = "低估" if industry_score >= 0.6 else "高估"
        if industry_score == 0.5:
            industry_judge = "合理"

        # 模型判断
        sentiment = fetch_news_sentiment_rss(code)
        if sentiment > 0.1:
            sentiment_judge = "正面"
        elif sentiment < -0.1:
            sentiment_judge = "负面"
        else:
            sentiment_judge = "中性"

        features = pd.DataFrame([{
            "trailingPE": pe,
            "priceToBook": pb,
            "returnOnEquity": roe,
            "trailingEps": eps,
            "revenueGrowth": revenue_growth,
            "grossMargins": gross_margin,
            "marketCap": market_cap,
            "freeCashflow": free_cashflow,
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

        score_map = {"低估": 0, "合理": 0.5, "高估": 1}
        model_score = score_map.get(model_judge, 0.5)
        industry_score = score_map.get(industry_judge, 0.5)
        final_score = 0.5 * model_score + 0.5 * industry_score

        if final_score < 0.5:
            final_judge = "低估"
        elif final_score > 0.5:
            final_judge = "高估"
        else:
            final_judge = "合理"

        return {
            "股票代码": code,
            "公司名称": name_cn,
            "当前价格": f"${current_price:.2f}",
            "预测价格": f"${pred_price:.2f}",
            "行业判断": industry_judge,
            "情绪判断": sentiment_judge,
            "模型判断": model_judge,
            "最终判断": final_judge
        }
    except Exception as e:
        print(f"❌ 评估失败 {row['code']}: {e}")
        return None
