import os
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sentiment_utils import fetch_news_sentiment_rss

# 💌 邮箱配置
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAILS = [
    "wuyaobin89@gmail.com",
    "wangling0607@gmail.com",
    "jakingtang1993@gmail.com"
]

# 🔍 读取模型与股票映射表
stock_map = pd.read_csv("stock_map.csv")
model = joblib.load("valuation_model.pkl")

# 🧾 提取财务指标
def get_metrics(info):
    return {
        "trailingPE": info.get("trailingPE", np.nan),
        "priceToBook": info.get("priceToBook", np.nan),
        "returnOnEquity": info.get("returnOnEquity", np.nan),
        "trailingEps": info.get("trailingEps", np.nan),
        "revenueGrowth": info.get("revenueGrowth", np.nan),
        "grossMargins": info.get("grossMargins", np.nan),
        "marketCap": info.get("marketCap", np.nan),
        "freeCashflow": info.get("freeCashflow", np.nan)
    }

# ✅ 与 app.py 一致的综合判断逻辑
def evaluate(row):
    try:
        code = row["code"]
        stock = yf.Ticker(code)
        info = stock.info
        current_price = info.get("currentPrice", None)
        if current_price is None:
            return None

        # 获取情绪得分与判断
        sentiment = fetch_news_sentiment_rss(code)
        if sentiment > 0.1:
            sentiment_judge = "正面"
        elif sentiment < -0.1:
            sentiment_judge = "负面"
        else:
            sentiment_judge = "中性"

        # 财务指标获取
        metrics = get_metrics(info)
        if any(pd.isna(v) for v in metrics.values()):
            return None
        metrics["sentiment"] = sentiment

        # 技术面预测价
        pred_price = model.predict(pd.DataFrame([metrics]))[0]
        tech_judge = "低估" if current_price < pred_price else "高估"

        # 模型判断（基于情绪）
        if sentiment_judge == "负面":
            model_judge = "高估"
        elif sentiment_judge == "正面":
            model_judge = "低估"
        else:
            model_judge = "合理"

        # 行业均值（PE, PB, ROE）
        peers = stock_map[stock_map["industry"] == row["industry"]]["code"].tolist()
        pe_list, pb_list, roe_list = [], [], []
        for p in peers:
            try:
                peer_info = yf.Ticker(p).info
                pe_list.append(peer_info.get("trailingPE", np.nan))
                pb_list.append(peer_info.get("priceToBook", np.nan))
                roe_list.append(peer_info.get("returnOnEquity", np.nan))
            except:
                continue

        def tag(val, avg, high_good=True):
            if np.isnan(val) or np.isnan(avg):
                return 0.5
            return 1 if (val > avg if high_good else val < avg) else 0

        score_pe = tag(metrics["trailingPE"], np.nanmean(pe_list), False)
        score_pb = tag(metrics["priceToBook"], np.nanmean(pb_list), False)
        score_roe = tag(metrics["returnOnEquity"], np.nanmean(roe_list), True)
        industry_score = (score_pe + score_pb + score_roe) / 3
        if industry_score >= 0.6:
            industry_judge = "低估"
        elif industry_score < 0.4:
            industry_judge = "高估"
        else:
            industry_judge = "合理"

        # 最终估值判断（模型 × 行业）
        score_map = {"低估": 0, "合理": 0.5, "高估": 1}
        model_score = score_map.get(model_judge, 0.5)
        industry_score_val = score_map.get(industry_judge, 0.5)
        final_score = 0.5 * model_score + 0.5 * industry_score_val
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
            "技术判断": tech_judge,
            "情绪判断": sentiment_judge,
            "模型判断": model_judge,
            "行业判断": industry_judge,
            "最终判断": final_judge
        }

    except Exception as e:
        print(f"跳过 {row['code']}：{e}")
        return None

# 🔄 遍历所有股票，收集最终判断为“低估”的
results = []
for _, row in stock_map.iterrows():
    r = evaluate(row)
    if r and r["最终判断"] == "低估":
        results.append(r)

# 📧 构建邮件 HTML 内容
html = "<h3>📉 每周低估股票提醒</h3>"
if results:
    df = pd.DataFrame(results)
    html += df[["公司名称", "股票代码", "当前价格", "预测价格", "模型判断", "行业判断", "最终判断"]].to_html(index=False, escape=False)
else:
    html += "<p>本周暂无符合条件的低估股票。</p>"

# 📬 发送邮件
msg = MIMEMultipart("alternative")
msg["Subject"] = "📉 每周低估股票提醒"
msg["From"] = SENDER_EMAIL
msg["To"] = ", ".join(RECEIVER_EMAILS)
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

print("✅ 邮件发送成功！")
