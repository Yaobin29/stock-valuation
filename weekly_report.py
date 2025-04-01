
import os
import pandas as pd
import smtplib
import joblib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from valuation_utils import evaluate_stock

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAILS = [
    "wuyaobin89@gmail.com",
]

stock_map = pd.read_csv("stock_map.csv")

# 执行估值分析
results = []
for _, row in stock_map.iterrows():
    res = evaluate_stock(row, stock_map)
    if res and res["最终判断"] == "低估":
        results.append(res)

# 构建 HTML 邮件内容
html = "<h3>📊 每周低估股票清单</h3>"
if results:
    df = pd.DataFrame(results)
    html += df.to_html(index=False, escape=False)
else:
    html += "<p>本周暂无符合条件的低估股票。</p>"

# 构建邮件并发送
msg = MIMEMultipart("alternative")
msg["Subject"] = "📉 每周低估股票提醒"
msg["From"] = SENDER_EMAIL
msg["To"] = ", ".join(RECEIVER_EMAILS)
msg.attach(MIMEText(html, "html"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

print("✅ 每周报告邮件发送成功！")
