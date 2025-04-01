import os
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sentiment_utils import fetch_news_sentiment_rss
from valuation_utils import evaluate_stock

# 邮箱配置（来自 GitHub Secrets）
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAILS = [
    "wuyaobin89@gmail.com",
    "wangling0607@gmail.com",
    "jakingtang1993@gmail.com"
]

# 加载股票列表与模型
stock_map = pd.read_csv("stock_map.csv")
model = joblib.load("valuation_model.pkl")

# 三类结果容器
low_list, fair_list, high_list = [], [], []

# 遍历所有股票
for _, row in stock_map.iterrows():
    result = evaluate_stock(row, stock_map, model)
    if result:
        judge = result["最终判断"]
        if judge == "低估":
            low_list.append(result)
        elif judge == "合理":
            fair_list.append(result)
        elif judge == "高估":
            high_list.append(result)

# 构建 HTML 邮件内容
html = "<h2>📊 每周估值判断报告</h2>"

def df_to_html(title, data, emoji):
    if not data:
        return f"<h3>{emoji} {title}</h3><p>无符合条件的股票。</p>"
    df = pd.DataFrame(data)
    return f"<h3>{emoji} {title}</h3>" + df.to_html(index=False, escape=False)

html += df_to_html("📉 低估股票", low_list, "🟩")
html += df_to_html("⚖️ 合理股票", fair_list, "🟨")
html += df_to_html("📈 高估股票", high_list, "🟥")

# 发送邮件
msg = MIMEMultipart("alternative")
msg["Subject"] = "📬 每周股票估值判断报告"
msg["From"] = SENDER_EMAIL
msg["To"] = ", ".join(RECEIVER_EMAILS)
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

print("✅ 每周估值报告邮件已发送！")
