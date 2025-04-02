import os
import pandas as pd
from valuation_utils import evaluate_stock
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

# Secrets
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAILS = ["wuyaobin89@gmail.com", "wangling0607@gmail.com", "jakingtang1993@gmail.com"]

# 股票列表
stock_map = pd.read_csv("stock_map.csv")

# 分类列表
low_list, fair_list, high_list = [], [], []

# 分析每只股票
for _, row in stock_map.iterrows():
    result = evaluate_stock(row, stock_map)
    if not result: continue
    judge = result["最终判断"]
    if judge == "低估":
        low_list.append(result)
    elif judge == "高估":
        high_list.append(result)
    else:
        fair_list.append(result)

# 构建HTML邮件
def to_html(title, data, color):
    html = f"<h3 style='color:{color}'>📌 {title}</h3>"
    if data:
        df = pd.DataFrame(data)
        html += df.to_html(index=False, escape=False)
    else:
        html += "<p>暂无数据</p>"
    return html

html = "<h2 style='color:purple'>📊 每周估值分类报告</h2>"
html += to_html("🟩 低估股票", low_list, "green")
html += to_html("🟨 合理股票", fair_list, "#e69500")
html += to_html("🟥 高估股票", high_list, "red")

# 发送邮件
msg = MIMEMultipart("alternative")
msg["Subject"] = "📬 每周股票估值分类报告"
msg["From"] = SENDER_EMAIL
msg["To"] = ", ".join(RECEIVER_EMAILS)
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

print("✅ 邮件发送成功")
