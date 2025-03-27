import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
from sentiment_utils import fetch_news_sentiment

st.set_page_config(page_title="📈 中英文股票估值分析平台", layout="wide")

# 加载股票映射
stock_map = pd.read_csv("stock_map.csv")
stock_map["display"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"

# 标题和搜索
st.title("📈 中英文股票估值分析平台")
query = st.text_input("请输入公司名称或股票代码（支持中英文，如 苹果、NVDA、0700.HK）", "")
matched = stock_map[stock_map["display"].str.contains(query, case=False, na=False)] if query else stock_map
selected = st.selectbox("请选择股票：", matched["display"].tolist())

# 获取选中公司信息
row = stock_map[stock_map["display"] == selected].iloc[0]
code, industry, name_en = row["code"], row["industry"], row["name_en"]
stock = yf.Ticker(code)
info = stock.info

# 提取财务数据
def get_metric(name):
    return info.get(name, np.nan)

pe = get_metric("trailingPE")
pb = get_metric("priceToBook")
roe = get_metric("returnOnEquity")
eps = get_metric("trailingEps")
revenue_growth = get_metric("revenueGrowth")
gross_margin = get_metric("grossMargins")
free_cashflow = get_metric("freeCashflow")
current_price = get_metric("currentPrice")
market_cap = get_metric("marketCap")

# 股票基本信息展示
st.markdown(f"### 📌 股票：{row['name_cn']} ({code})")
st.markdown("---")

# 股票关键指标
st.markdown("### 📊 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "-")
col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "-")
col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "-")

# 行业平均指标计算
industry_pe, industry_pb, industry_roe = [], [], []
industry_stocks = stock_map[stock_map["industry"] == industry]["code"].tolist()
for ticker in industry_stocks:
    try:
        data = yf.Ticker(ticker).info
        industry_pe.append(data.get("trailingPE", np.nan))
        industry_pb.append(data.get("priceToBook", np.nan))
        industry_roe.append(data.get("returnOnEquity", np.nan))
    except:
        continue

avg_pe, avg_pb, avg_roe = np.nanmean(industry_pe), np.nanmean(industry_pb), np.nanmean(industry_roe)

# 行业平均指标展示
st.markdown(f"### 🏭 {industry} 行业平均指标")
col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{avg_pe:.2f}" if not np.isnan(avg_pe) else "-")
col5.metric("行业平均PB", f"{avg_pb:.2f}" if not np.isnan(avg_pb) else "-")
col6.metric("行业平均ROE", f"{avg_roe*100:.2f}%" if not np.isnan(avg_roe) else "-")

# 行业估值判断
def tag(val, avg, high_good=True):
    if np.isnan(val) or np.isnan(avg):
        return 0.5
    threshold = 0.2
    if abs(val - avg) / avg <= threshold:
        return 0.5
    return 1 if (val > avg if high_good else val < avg) else 0

score_pe = tag(pe, avg_pe, high_good=False)
score_pb = tag(pb, avg_pb, high_good=False)
score_roe = tag(roe, avg_roe, high_good=True)
industry_score = (score_pe + score_pb + score_roe) / 3

industry_judge = "合理"
if industry_score >= 0.7:
    industry_judge = "低估"
elif industry_score <= 0.3:
    industry_judge = "高估"

st.markdown(f"### 🧠 行业对比判断：{industry_judge}")
st.markdown("---")

# 获取情绪得分
sentiment = fetch_news_sentiment(name_en)
if sentiment > 0.1:
    sentiment_judge = "正面"
elif sentiment < -0.1:
    sentiment_judge = "负面"
else:
    sentiment_judge = "中性"

st.markdown(f"### 💬 市场情绪判断：{sentiment_judge}")
st.markdown("---")

# 预测价格（技术面分析）
try:
    model = joblib.load("valuation_model.pkl")
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
    diff_ratio = (pred_price - current_price) / current_price

    tech_judge = "合理"
    if diff_ratio >= 0.1:
        tech_judge = "低估"
    elif diff_ratio <= -0.1:
        tech_judge = "高估"
except:
    pred_price = None
    tech_judge = "-"

# 估值结果展示
st.markdown("### 💲 估值结果")
col7, col8, col9 = st.columns(3)
col7.metric("📉 当前价格", f"${current_price:.2f}" if current_price else "-")
col8.metric("📈 预测价格", f"${pred_price:.2f}" if pred_price else "N/A")
col9.metric("⚙️ 技术面分析判断", tech_judge)

# 综合模型判断（技术60% + 情绪40%）
sentiment_score = {"负面": 1, "中性": 0.5, "正面": 0}[sentiment_judge]
tech_score = {"高估": 1, "合理": 0.5, "低估": 0}[tech_judge]
model_score = tech_score * 0.6 + sentiment_score * 0.4

model_judge = "合理"
if model_score >= 0.7:
    model_judge = "高估"
elif model_score <= 0.3:
    model_judge = "低估"

# 最终综合判断（模型50% + 行业50%）
industry_final_score = {"高估": 1, "合理": 0.5, "低估": 0}[industry_judge]
final_score = model_score * 0.5 + industry_final_score * 0.5
final_judge = "合理"
if final_score >= 0.7:
    final_judge = "高估"
elif final_score <= 0.3:
    final_judge = "低估"

st.markdown("---")
st.markdown(f"### 🧮 综合估值判断：{final_judge}")

# 股票走势图
st.markdown("### 📈 股票近6个月价格走势")
try:
    hist = stock.history(period="6mo", interval="1d")
    price_data = hist["Close"].dropna()
    price_df = pd.DataFrame({"日期": price_data.index, "收盘价": price_data.values}).set_index("日期")
    st.line_chart(price_df)
except:
    st.warning("⚠️ 无法获取历史价格数据，可能无日度数据或接口异常。")
