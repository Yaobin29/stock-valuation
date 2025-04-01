import os
import pandas as pd
import smtplib
import joblib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from valuation_utils import evaluate_stock  # 🔁 调用统一估值逻辑
import warnings
warnings.filterwarnings("ignore")

# 邮箱配置（从 GitHub Secrets 注入）
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAILS = [
    "wuyaobin89@gmail.com",
   
]

# 加载股票列表
stock_map = pd.read_csv("stock_map.csv")

# 扫描所有股票，评估并筛选“最终判断为低估”
low_estimates = []
for _, row in stock_map.iterrows():
    result = evaluate_stock(row, stock_map)
    if result and result["最终判断"] == "低估":
        low_estimates.append(result)

# 构建邮件内容
html = "<h2>📉 每周低估股票清单</h2>"
if low_estimates:
    df = pd.DataFrame(low_estimates)
    html += df.to_html(index=False, escape=False)
else:
    html += "<p>本周暂无符合条件的低估股票。</p>"

# 邮件组装
msg = MIMEMultipart("alternative")
msg["Subject"] = "📉 每周低估股票提醒"
msg["From"] = SENDER_EMAIL
msg["To"] = ", ".join(RECEIVER_EMAILS)
msg.attach(MIMEText(html, "html"))

# 发送邮件
with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())

print("✅ 邮件发送成功！")
