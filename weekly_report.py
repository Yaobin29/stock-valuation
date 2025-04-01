import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sentiment_utils import fetch_news_sentiment_rss
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 邮件配置（⚠️ 后续改为 GitHub Secret）
SENDER_EMAIL = os.getenv("ybwu29@gmail.com")
APP_PASSWORD = os.getenv("xhfq ycxv eeuk jyga")
RECEIVER_EMAIL = "wuyaobin89@gmail.com"
RECEIVER_EMAIL = "wangling0607@gmail.com"

# 加载模型与股票映射
model = joblib.load("valuation_model.pkl")
stock_map = pd.read_csv("stock_map.csv")

# 最终结果列表
results = []

for _, row in stock_map.iterrows():
    code = row["code"]
    name_cn = row["name_cn"]
    name_en = row["name_en"]
    industry = row["industry"]

    try:
        stock = yf.Ticker(code)
        info = stock.info

        # 获取财务指标
        def get_metric(name): return info.get(name, np.nan)
        pe = get_metric("trailingPE")
        pb = get_metric("priceToBook")
        roe = get_metric("returnOnEquity")
        eps = get_metric("trailingEps")
        revenue_growth = get_metric("revenueGrowth")
        gross_margin = get_metric("grossMargins")
        free_cashflow = get_metric("freeCashflow")
        market_cap = get_metric("marketCap")
        current_price = get_metric("currentPrice")

        # 行业判断（PE+PB+ROE）
        industry_pe, industry_pb, industry_roe = [], [], []
        industry_stocks = stock_map[stock_map["industry"] == industry]["code"].tolist()
        for ticker in industry_stocks:
            try:
                ind_info = yf.Ticker(ticker).info
                industry_pe.append(ind_info.get("trailingPE", np.nan))
                industry_pb.append(ind_info.get("priceToBook", np.nan))
                industry_roe.append(ind_info.get("returnOnEquity", np.nan))
            except:
                continue
        avg_pe = np.nanmean(industry_pe)
        avg_pb = np.nanmean(industry_pb)
        avg_roe = np.nanmean(industry_roe)

        def tag(val, avg, high_good=True):
            if np.isnan(val) or np.isnan(avg): return 0.5
            return 1 if (val > avg if high_good else val < avg) else 0

        score_pe = tag(pe, avg_pe, False)
        score_pb = tag(pb, avg_pb, False)
        score_roe = tag(roe, avg_roe, True)
        industry_score = (score_pe + score_pb + score_roe) / 3
        if industry_score >= 0.6:
            industry_judge = "低估"
        elif industry_score <= 0.3:
            industry_judge = "高估"
        else:
            industry_judge = "合理"

        # 情绪判断
        sentiment = fetch_news_sentiment_rss(name_en)
        if sentiment > 0.1:
            sentiment_judge = "正面"
        elif sentiment < -0.1:
            sentiment_judge = "负面"
        else:
            sentiment_judge = "中性"

        # 技术面预测
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

        # 模型判断（仅基于情绪面）
        if sentiment_judge == "负面":
            model_judge = "高估"
        elif sentiment_judge == "正面":
            model_judge = "低估"
        else:
            model_judge = "合理"

        # 综合判断（模型×行业）
        score_map = {"低估": 0, "合理": 0.5, "高估": 1}
        final_score = 0.5 * score_map[model_judge] + 0.5 * score_map[industry_judge]
        if final_score < 0.5:
            final_judge = "低估"
        elif final_score > 0.5:
            final_judge = "高估"
        else:
            final_judge = "合理"

        if final_judge == "低估":
            results.append({
                "公司": name_cn,
                "代码": code,
                "当前价": f"${current_price:.2f}",
                "预测价": f"${pred_price:.2f}",
                "技术面": tech_judge,
                "情绪面": sentiment_judge,
                "行业": industry_judge,
                "最终判断": final_judge
            })

    except Exception as e:
        print(f"跳过 {code}: {e}")
        continue

# 转为表格
report_df = pd.DataFrame(results)

# 构造邮件内容
msg = MIMEMultipart()
msg["From"] = SENDER_EMAIL
msg["To"] = RECEIVER_EMAIL
msg["Subject"] = "📬 本周低估股票报告（估值平台 1.1）"
if report_df.empty:
    html = "<p>本周无“低估”判断的股票。</p>"
else:
    html = report_df.to_html(index=False)

msg.attach(MIMEText(f"<h3>以下为您关注的股票中综合判断为“低估”的列表：</h3>{html}", "html"))

# 发送邮件
try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
    print("✅ 邮件已发送！")
except Exception as e:
    print(f"❌ 邮件发送失败: {e}")
