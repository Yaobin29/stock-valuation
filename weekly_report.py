import os
import pandas as pd
from valuation_utils import evaluate_stock
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

# 邮件配置
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAILS = [
    "wuyaobin89@gmail.com",
]

# 读取股票数据
stock_map = pd.read_csv("stock_map.csv")

# 分析结果分类收集
low, fair, high = [], [], []

for _, row in stock_map.iterrows():
    result = evaluate_stock(row, stock_map)
    if result:
        judge = result["最终判断"]
        if judge == "低估":
            low.append(result)
        elif judge == "合理":
            fair.append(result)
        elif judge == "高估":
            high.append(result)

# 构建 HTML 报告
html = "<h2>📊 每周估值分类报告</h2>"

def render_section(title, color, data):
    html = f"<h3 style='color:{color}'>🟩 {title}</h3>"
    if data:
        df = pd.DataFrame(data)
        html += df.to_html(index=False, escape=False)
    else:
        html += "<p>暂无数据</p>"
    return html

html += render_section("低估股票", "green", low)
html += render_section("合理股票", "orange", fair)
html += render_section("高估股票", "red", high)

# 发送邮件
msg = MIMEMultipart("alternative")
msg["Subject"] = "📬 每周股票估值分类报告"
msg["From"] = SENDER_EMAIL
msg["To"] = ", ".join(RECEIVER_EMAILS)
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

print("✅ 邮件发送成功！")
