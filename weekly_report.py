import os
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sentiment_utils import fetch_news_sentiment_rss

# 邮箱配置（需通过 GitHub Secret 注入）
SENDER_EMAIL = os.getenv("ybwu29@gmail.com")
APP_PASSWORD = os.getenv("zsqxqoairqeeiakh")
RECEIVER_EMAILS = [
    "wuyaobin89@gmail.com",
    "wangling0607@gmail.com",
    "jakingtang1993@gmail.com"
]

# 加载数据与模型
stock_map = pd.read_csv("stock_map.csv")
model = joblib.load("valuation_model.pkl")

# 工具函数：财务数据提取
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

# 综合估值判断逻辑
def evaluate(row):
    try:
        stock = yf.Ticker(row["code"])
        info = stock.info
        current_price = info.get("currentPrice", None)
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

        if sentiment_judge == "负面":
            model_judge = "高估"
        elif sentiment_judge == "正面":
            model_judge = "低估"
        else:
            model_judge = "合理"

        peers = stock_map[stock_map["industry"] == row["industry"]]["code"].tolist()
        peer_pes = []
        for p in peers:
            try:
                peer_pe = yf.Ticker(p).info.get("trailingPE", np.nan)
                if not pd.isna(peer_pe): peer_pes.append(peer_pe)
            except:
                continue
        avg_pe = np.nanmean(peer_pes)
        industry_judge = (
            "低估" if metrics["trailingPE"] < avg_pe else "高估"
            if not pd.isna(avg_pe) and not pd.isna(metrics["trailingPE"]) else "合理"
        )

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
        print(f"跳过 {row['code']}：{e}")
        return None

# 执行全股票扫描
results = []
for _, row in stock_map.iterrows():
    res = evaluate(row)
    if res and res["最终判断"] == "低估":
        results.append(res)

# 构建邮件内容
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
