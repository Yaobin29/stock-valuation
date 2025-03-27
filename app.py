import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sentiment_utils import fetch_news_sentiment

st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")
st.markdown("<h1>📈 中英文股票估值分析平台</h1>", unsafe_allow_html=True)

# 读取股票映射
stock_map = pd.read_csv("stock_map.csv")
stock_map["display"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"

query = st.text_input("请输入公司名称或股票代码（支持中英文，如 苹果、NVDA、0700.HK）", "")
matched = stock_map[stock_map["display"].str.contains(query, case=False, na=False)] if query else stock_map
selected = st.selectbox("请选择股票：", matched["display"].tolist())

row = stock_map[stock_map["display"] == selected].iloc[0]
code = row["code"]
industry = row["industry"]
name_cn = row["name_cn"]
name_en = row["name_en"]

st.markdown(f"### 📌 股票：<strong>{name_cn} ({code})</strong>", unsafe_allow_html=True)

# 获取财务指标
stock = yf.Ticker(code)
info = stock.info
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

st.markdown("---")
st.markdown("### 📊 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "-")
col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "-")
col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "-")

# 行业平均
industry_stocks = stock_map[stock_map["industry"] == industry]["code"].tolist()
industry_pe, industry_pb, industry_roe = [], [], []
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

st.markdown("### 🏭 {}行业平均指标".format(industry))
col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{avg_pe:.2f}" if not np.isnan(avg_pe) else "-")
col5.metric("行业平均PB", f"{avg_pb:.2f}" if not np.isnan(avg_pb) else "-")
col6.metric("行业平均ROE", f"{avg_roe*100:.2f}%" if not np.isnan(avg_roe) else "-")

# 行业估值判断
def tag(val, avg, high_good=True):
    if np.isnan(val) or np.isnan(avg):
        return 0.5
    return 1 if (val > avg if high_good else val < avg) else 0

score_pe = tag(pe, avg_pe, high_good=False)
score_pb = tag(pb, avg_pb, high_good=False)
score_roe = tag(roe, avg_roe, high_good=True)
industry_score = (score_pe + score_pb + score_roe) / 3
industry_judge = "低估" if industry_score >= 0.6 else "高估"
st.markdown(f"### 🧠 行业对比判断：<span style='color:#444;'>{industry_judge}</span>", unsafe_allow_html=True)

# 情绪面分析
sentiment_score = fetch_news_sentiment(name_en)
if sentiment_score > 0.2:
    sentiment_judge = "正面"
elif sentiment_score < -0.2:
    sentiment_judge = "负面"
else:
    sentiment_judge = "中性"

st.markdown("---")
st.markdown(f"### 💬 情绪面分析判断：<strong>{sentiment_judge}</strong>", unsafe_allow_html=True)

# 模型预测
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
        "sentiment": sentiment_score
    }])

    pred_price = model.predict(features)[0]
    pct_diff = (current_price - pred_price) / pred_price

    if pct_diff > 0.2:
        tech_judge = "高估"
    elif pct_diff < -0.2:
        tech_judge = "低估"
    else:
        tech_judge = "合理"

except:
    pred_price = None
    tech_judge = "-"

st.markdown("### 💲 估值结果")
col7, col8, col9 = st.columns(3)
col7.metric("📉 当前价格", f"${current_price:.2f}" if current_price else "-")
col8.metric("📈 预测价格", f"${pred_price:.2f}" if pred_price else "N/A")
col9.metric("🧠 技术面分析判断", tech_judge)

# 综合模型判断
tech_score = {"低估": 0, "合理": 0.5, "高估": 1}.get(tech_judge, 0.5)
sent_score = {"负面": 1, "中性": 0.5, "正面": 0}.get(sentiment_judge, 0.5)
model_score = tech_score * 0.6 + sent_score * 0.4

if model_score > 0.6:
    model_judge = "高估"
elif model_score < 0.4:
    model_judge = "低估"
else:
    model_judge = "合理"

# 综合估值判断（模型+行业）
industry_score_final = 0 if industry_judge == "低估" else (1 if industry_judge == "高估" else 0.5)
final_score = model_score * 0.5 + industry_score_final * 0.5
if final_score > 0.6:
    final_judge = "高估"
elif final_score < 0.4:
    final_judge = "低估"
else:
    final_judge = "合理"

# 显示最终估值判断结果
st.markdown("---")
color = "red" if final_judge == "高估" else "green" if final_judge == "低估" else "orange"
st.markdown(
    f"<div style='border:2px solid {color}; padding: 10px; border-radius: 8px;'>"
    f"<h4 style='color:{color};'>🧮 综合估值判断（技术60% + 情绪40%） × 模型50% + 行业50% ：{final_judge}</h4></div>",
    unsafe_allow_html=True
)

# 股票价格图
st.markdown("### 📉 股票近6个月价格走势")
try:
    hist = stock.history(period="6mo", interval="1d")
    if hist.empty or "Close" not in hist.columns:
        raise ValueError("无有效价格数据")
    price_data = hist["Close"].dropna()
    price_df = pd.DataFrame({"日期": price_data.index, "收盘价": price_data.values}).set_index("日期")
    st.line_chart(price_df)
except Exception as e:
    st.warning("⚠️ 无法获取历史价格数据。可能该股票无日度数据或接口异常。")
