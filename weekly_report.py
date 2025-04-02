import os
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sentiment_utils import fetch_news_sentiment_rss
from valuation_utils import evaluate_stock  # ✅ 使用与 app.py 一致的逻辑

# 邮件配置（GitHub Secrets 中设置）
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAILS = [
    "wuyaobin89@gmail.com",
   
]

# 加载数据
stock_map = pd.read_csv("stock_map.csv")
model = joblib.load("valuation_model.pkl")

# 收集分类结果
low_list, fair_list, high_list = [], [], []

for _, row in stock_map.iterrows():
    result = evaluate_stock(row, stock_map, model)
    if result:
        entry = {
            "股票代码": row["code"],
            "公司名称": row["name_cn"],
            "当前价格": result["当前价格"],
            "预测价格": result["预测价格"]
        }
        if result["最终判断"] == "低估":
            low_list.append(entry)
        elif result["最终判断"] == "合理":
            fair_list.append(entry)
        elif result["最终判断"] == "高估":
            high_list.append(entry)

# 组装 HTML
html = "<h2>📊 每周估值分类报告</h2>"

def section(title, data, color):
    if data:
        df = pd.DataFrame(data)
        return f"<h3 style='color:{color}'>{title}</h3>" + df.to_html(index=False, escape=False)
    else:
        return f"<h3 style='color:{color}'>{title}</h3><p>暂无数据</p>"

html += section("🟩 低估股票", low_list, "green")
html += section("🟨 合理股票", fair_list, "orange")
html += section("🟥 高估股票", high_list, "red")

# 发送邮件
msg = MIMEMultipart("alternative")
msg["Subject"] = "📬 每周股票估值分类报告"
msg["From"] = SENDER_EMAIL
msg["To"] = ", ".join(RECEIVER_EMAILS)
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

print("✅ 报告邮件发送成功！")
