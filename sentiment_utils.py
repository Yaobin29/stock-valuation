import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentiment_utils import fetch_news_sentiment

st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")
stock_map = pd.read_csv("stock_map.csv")
stock_map["display"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"

# 用户输入
st.title("📈 中英文股票估值分析平台")
query = st.text_input("请输入公司名称或股票代码（支持中英文，如 苹果、NVDA、0700.HK）", "")
matched = stock_map[stock_map["display"].str.contains(query, case=False, na=False)] if query else stock_map
selected = st.selectbox("请选择股票：", matched["display"].tolist())

row = stock_map[stock_map["display"] == selected].iloc[0]
code = row["code"]
industry = row["industry"]
name_en = row["name_en"]
stock = yf.Ticker(code)
info = stock.info

# 财务指标
def get_metric(name):
    return info.get(name, np.nan)

pe = get_metric("trailingPE")
pb = get_metric("priceToBook")
roe = get_metric("returnOnEquity")
eps = get_metric("trailingEps")
revenue_growth = get_metric("revenueGrowth")
gross_margin = get_metric("grossMargins")
free_cashflow = get_metric("freeCashflow")
market_cap = get_metric("marketCap")
current_price = get_metric("currentPrice")

st.markdown(f"### 📌 股票：**{row['name_cn']} ({code})**")
st.markdown("---")
st.markdown("### 📊 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "-")
col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "-")
col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "-")

# 行业均值
industry_stocks = stock_map[stock_map["industry"] == industry]["code"].tolist()
pe_list, pb_list, roe_list = [], [], []
for t in industry_stocks:
    try:
        i = yf.Ticker(t).info
        pe_list.append(i.get("trailingPE", np.nan))
        pb_list.append(i.get("priceToBook", np.nan))
        roe_list.append(i.get("returnOnEquity", np.nan))
    except:
        continue

avg_pe, avg_pb, avg_roe = np.nanmean(pe_list), np.nanmean(pb_list), np.nanmean(roe_list)

st.markdown(f"### 🏭 {industry}行业平均指标")
col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{avg_pe:.2f}" if not np.isnan(avg_pe) else "-")
col5.metric("行业平均PB", f"{avg_pb:.2f}" if not np.isnan(avg_pb) else "-")
col6.metric("行业平均ROE", f"{avg_roe*100:.2f}%" if not np.isnan(avg_roe) else "-")

# 行业判断
def compare(val, avg, high_good=True):
    if np.isnan(val) or np.isnan(avg):
        return 0.5
    return 1 if (val > avg if high_good else val < avg) else 0

score_pe = compare(pe, avg_pe, False)
score_pb = compare(pb, avg_pb, False)
score_roe = compare(roe, avg_roe, True)
industry_score = (score_pe + score_pb + score_roe) / 3
industry_judge = "低估" if industry_score >= 0.6 else "高估"
st.markdown(f"### 🧠 行业对比判断：**{industry_judge}**")

# 新闻情绪分析
sentiment_score = fetch_news_sentiment(name_en)
if sentiment_score > 0.2:
    sentiment_judge = "正面"
elif sentiment_score < -0.2:
    sentiment_judge = "负面"
else:
    sentiment_judge = "中性"
st.markdown(f"### 💬 情绪面分析判断：**{sentiment_judge}**")

# 加载模型预测
try:
    model = joblib.load("valuation_model.pkl")
    input_features = pd.DataFrame([{
        "trailingPE": pe, "priceToBook": pb, "returnOnEquity": roe,
        "trailingEps": eps, "revenueGrowth": revenue_growth,
        "grossMargins": gross_margin, "marketCap": market_cap,
        "freeCashflow": free_cashflow, "sentiment": sentiment_score
    }])

    pred_price = model.predict(input_features)[0]
    tech_judge = "低估" if current_price < pred_price else "高估"
except Exception as e:
    pred_price = None
    tech_judge = "-"

# 估值模块展示
st.markdown("### 💲 估值结果")
col7, col8, col9 = st.columns(3)
col7.metric("📉 当前价格", f"${current_price:.2f}" if current_price else "-")
col8.metric("📈 预测价格", f"${pred_price:.2f}" if pred_price else "N/A")
col9.metric("🧠 技术面分析判断", tech_judge)

# 综合估值判断
tech_score = 0 if tech_judge == "低估" else 1
sent_score = {"正面": 0, "中性": 0.5, "负面": 1}.get(sentiment_judge, 0.5)
model_score = tech_score * 0.6 + sent_score * 0.4
model_judge = "低估" if model_score < 0.4 else "高估" if model_score > 0.6 else "合理"

industry_score_final = 0 if industry_judge == "低估" else 1
final_score = model_score * 0.5 + industry_score_final * 0.5
final_judge = "低估" if final_score < 0.4 else "高估" if final_score > 0.6 else "合理"

# 综合结果展示
st.markdown("---")
st.markdown(
    f"<div style='background-color:#fcefdc;padding:10px;border-left:6px solid orange;'>"
    f"<b>🧮 综合估值判断（技术60% + 情绪40%） × 模型50% + 行业50% ：<span style='color:red;'>{final_judge}</span></b>"
    f"</div>",
    unsafe_allow_html=True
)

# 股票走势
st.markdown("### 📈 股票近6个月价格走势")
try:
    hist = stock.history(period="6mo", interval="1d")
    if hist.empty or "Close" not in hist.columns:
        raise ValueError("无有效价格数据")
    close_data = hist["Close"].dropna()
    chart_df = pd.DataFrame({"日期": close_data.index, "收盘价": close_data.values}).set_index("日期")
    st.line_chart(chart_df)
except Exception as e:
    st.warning("⚠️ 无法获取历史价格数据。可能该股票无日度数据或接口异常。")
