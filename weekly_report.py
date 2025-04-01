import os
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sentiment_utils import fetch_news_sentiment_rss

# 邮箱配置（通过 GitHub Secret 注入）
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAILS = [
    "wuyaobin89@gmail.com"
    
]

# 加载数据与模型
stock_map = pd.read_csv("stock_map.csv")
model = joblib.load("valuation_model.pkl")

# 财务数据提取
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

# 单只股票估值判断
def evaluate(row):
    try:
        stock = yf.Ticker(row["code"])
        info = stock.info
        current_price = info.get("currentPrice")
        if current_price is None:
            return None

        sentiment = fetch_news_sentiment_rss(row["code"])
        sentiment_judge = (
            "正面" if sentiment > 0.1 else
            "负面" if sentiment < -0.1 else
            "中性"
        )

        metrics = get_metrics(info)
        if any(pd.isna(v) for v in metrics.values()):
            return None

        features = metrics.copy()
        features["sentiment"] = sentiment
        pred_price = model.predict(pd.DataFrame([features]))[0]
        tech_judge = "低估" if current_price < pred_price else "高估"

        # 模型判断（基于情绪）
        if sentiment_judge == "负面":
            model_judge = "高估"
        elif sentiment_judge == "正面":
            model_judge = "低估"
        else:
            model_judge = "合理"

        # 行业判断：同行业 PE/PB/ROE 均值
        industry_codes = stock_map[stock_map["industry"] == row["industry"]]["code"].tolist()
        pe_list, pb_list, roe_list = [], [], []

        for c in industry_codes:
            try:
                data = yf.Ticker(c).info
                pe_list.append(data.get("trailingPE", np.nan))
                pb_list.append(data.get("priceToBook", np.nan))
                roe_list.append(data.get("returnOnEquity", np.nan))
            except:
                continue

        avg_pe = np.nanmean(pe_list)
        avg_pb = np.nanmean(pb_list)
        avg_roe = np.nanmean(roe_list)

        def tag(val, avg, high_good=True):
            if np.isnan(val) or np.isnan(avg):
                return 0.5
            return 1 if (val > avg if high_good else val < avg) else 0

        score_pe = tag(metrics["trailingPE"], avg_pe, False)
        score_pb = tag(metrics["priceToBook"], avg_pb, False)
        score_roe = tag(metrics["returnOnEquity"], avg_roe, True)
        industry_score = (score_pe + score_pb + score_roe) / 3
        industry_judge = (
            "低估" if industry_score >= 0.6 else
            "高估" if industry_score < 0.4 else
            "合理"
        )

        # 最终估值判断
        score_map = {"低估": 0, "合理": 0.5, "高估": 1}
        model_score = score_map.get(model_judge, 0.5)
        industry_score = score_map.get(industry_judge, 0.5)
        final_score = 0.5 * model_score + 0.5 * industry_score
        final_judge = (
            "低估" if final_score < 0.5 else
            "高估" if final_score > 0.5 else
            "合理"
        )

        return {
            "股票代码": row["code"],
            "公司名称": row["name_cn"],
            "当前价格": f"${current_price:.2f}",
            "预测价格": f"${pred_price:.2f}",
            "最终判断": final_judge
        }
    except Exception as e:
        print(f"❌ 跳过 {row['code']}：{e}")
        return None

# 批量扫描所有股票
results = []
for _, row in stock_map.iterrows():
    res = evaluate(row)
    if res and res["最终判断"] == "低估":
        results.append(res)

# 构建邮件 HTML 内容
html = "<h3>📊 每周低估股票清单</h3>"
if results:
    df = pd.DataFrame(results)
    html += df.to_html(index=False, escape=False)
else:
    html += "<p>本周暂无符合条件的低估股票。</p>"

# 发送邮件
msg = MIMEMultipart("alternative")
msg["Subject"] = "📉 每周低估股票提醒"
msg["From"] = SENDER_EMAIL
msg["To"] = ", ".join(RECEIVER_EMAILS)
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

print("✅ 邮件发送成功！")
