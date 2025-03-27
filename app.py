import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
from sentiment_utils import fetch_news_sentiment_rss
from datetime import datetime, timedelta

st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")
stock_map = pd.read_csv("stock_map.csv")
stock_map["display"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"

# 搜索部分
st.title("📈 中英文股票估值分析平台")
query = st.text_input("请输入公司名称或股票代码（支持中英文，如 苹果、NVDA、0700.HK）", "")
matched = stock_map[stock_map["display"].str.contains(query, case=False, na=False)] if query else stock_map
selected = st.selectbox("请选择股票：", matched["display"].tolist())

row = stock_map[stock_map["display"] == selected].iloc[0]
code = row["code"]
industry = row["industry"]
stock = yf.Ticker(code)
info = stock.info

# 获取财务指标
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

st.markdown(f"### 📌 股票：{row['name_cn']} ({code})")
st.markdown("### 📊 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "-")
col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "-")
col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "-")

# 行业平均
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

avg_pe = np.nanmean(industry_pe)
avg_pb = np.nanmean(industry_pb)
avg_roe = np.nanmean(industry_roe)

st.markdown(f"### 🏭 {industry}行业平均指标")
col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{avg_pe:.2f}" if not np.isnan(avg_pe) else "-")
col5.metric("行业平均PB", f"{avg_pb:.2f}" if not np.isnan(avg_pb) else "-")
col6.metric("行业平均ROE", f"{avg_roe*100:.2f}%" if not np.isnan(avg_roe) else "-")

# 行业判断
def tag(val, avg, high_good=True):
    if np.isnan(val) or np.isnan(avg):
        return 0.5
    return 1 if (val > avg if high_good else val < avg) else 0

score_pe = tag(pe, avg_pe, high_good=False)
score_pb = tag(pb, avg_pb, high_good=False)
score_roe = tag(roe, avg_roe, high_good=True)
industry_score = (score_pe + score_pb + score_roe) / 3
industry_judge = "低估" if industry_score >= 0.6 else "高估"
industry_judge = "合理" if industry_score == 0.5 else industry_judge
st.markdown(f"### 🧠 行业对比判断：{industry_judge}")

# 获取情绪指标
sentiment = fetch_news_sentiment_rss(code)

# 加载模型并预测
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
    tech_judge = "低估" if current_price < pred_price else "高估"
except:
    pred_price = None
    tech_judge = "-"

# 估值结果展示
st.markdown("### 💲 估值结果")
col7, col8, col9 = st.columns(3)
col7.metric("📉 当前价格", f"${current_price:.2f}" if current_price else "-")
col8.metric("📈 预测价格", f"${pred_price:.2f}" if pred_price else "N/A")
col9.metric("🧠 技术面分析判断", tech_judge)

# 情绪面分析判断
if sentiment > 0.1:
    sentiment_judge = "正面"
elif sentiment < -0.1:
    sentiment_judge = "负面"
else:
    sentiment_judge = "中性"
st.markdown(f"### 💬 情绪面分析判断：{sentiment_judge}")

# 模型估值判断（技术60% + 情绪40%）
if sentiment_judge == "负面":
    model_judge = "高估"
elif sentiment_judge == "正面":
    model_judge = "低估"
else:
    model_judge = "合理"

st.divider()
st.markdown("### 📊 模型内部估值判断（基于技术 + 情绪）")
color_map = {"高估": "red", "合理": "orange", "低估": "green"}
st.markdown(f"**<span style='color:{color_map[model_judge]}; font-size: 20px;'>模型判断：{model_judge}</span>**", unsafe_allow_html=True)

# 最终估值判断（模型 × 行业）
judge_score_map = {"低估": 0, "合理": 0.5, "高估": 1}
model_score = judge_score_map.get(model_judge, 0.5)
industry_score_final = judge_score_map.get(industry_judge, 0.5)
final_score = 0.5 * model_score + 0.5 * industry_score_final

if final_score < 0.5:
    final_judge = "低估"
elif final_score > 0.5:
    final_judge = "高估"
else:
    final_judge = "合理"

st.divider()
st.markdown("### 🧮 最终综合估值判断（模型 × 行业）")
st.markdown(f"**<span style='color:{color_map[final_judge]}; font-size: 24px;'>最终判断：{final_judge}</span>**", unsafe_allow_html=True)

# 股票价格走势
st.markdown("### 📈 股票近6个月价格走势")
try:
    hist = stock.history(period="6mo", interval="1d")
    if hist.empty or "Close" not in hist.columns:
        raise ValueError("无有效价格数据")
    price_data = hist["Close"].dropna()
    price_df = pd.DataFrame({"日期": price_data.index, "收盘价": price_data.values}).set_index("日期")
    st.line_chart(price_df)
except Exception as e:
    st.warning("⚠️ 无法获取历史价格数据。可能该股票无日度数据或接口异常。")
