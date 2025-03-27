import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")

# 加载公司列表
stock_map = pd.read_csv("stock_map.csv")
stock_map["display"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"

# UI - 搜索栏
st.title("📈 中英文股票估值分析平台")
query = st.text_input("请输入公司名称或股票代码（支持中英文，如 苹果、NVDA、0700.HK）", "")
matched = stock_map[stock_map["display"].str.contains(query, case=False, na=False)] if query else stock_map
selected = st.selectbox("请选择股票：", matched["display"].tolist())

# 选中股票信息
row = stock_map[stock_map["display"] == selected].iloc[0]
code = row["code"]
industry = row["industry"]
stock = yf.Ticker(code)
info = stock.info

# 提取财务指标
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

st.markdown(f"### 📌 股票：{row['name_cn']} ({code})")
st.markdown("### 📊 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "-")
col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "-")
col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "-")

# 行业均值计算
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

st.markdown(f"### 🏭 {industry}行业平均指标")
col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{avg_pe:.2f}" if not np.isnan(avg_pe) else "-")
col5.metric("行业平均PB", f"{avg_pb:.2f}" if not np.isnan(avg_pb) else "-")
col6.metric("行业平均ROE", f"{avg_roe*100:.2f}%" if not np.isnan(avg_roe) else "-")

# 行业估值判断
def tag(val, avg, high_good=True):
    if np.isnan(val) or np.isnan(avg):
        return 0.5
    return 1 if (val > avg if high_good else val < avg) else 0

score_pe = tag(pe, avg_pe, False)
score_pb = tag(pb, avg_pb, False)
score_roe = tag(roe, avg_roe, True)
industry_score = (score_pe + score_pb + score_roe) / 3
industry_judge = "低估" if industry_score >= 0.6 else "高估"
st.markdown(f"### 🧠 行业对比判断：{industry_judge}")

# 获取新闻情绪
analyzer = SentimentIntensityAnalyzer()
def get_sentiment_score(code):
    try:
        news = stock.news[:5]
        headlines = [item["title"] for item in news]
        if not headlines:
            return 0.0
        scores = [analyzer.polarity_scores(title)["compound"] for title in headlines]
        return np.mean(scores)
    except:
        return 0.0

sentiment = get_sentiment_score(code)

# 模型预测与判断（技术 + 情绪）
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

# 情绪面判断
sentiment_judge = "低估" if sentiment > 0 else "高估"

# 展示估值结果
st.markdown("### 💲 估值结果")
col7, col8, col9 = st.columns(3)
col7.metric("📉 当前价格", f"${current_price:.2f}" if current_price else "-")
col8.metric("📈 预测价格", f"${pred_price:.2f}" if pred_price else "N/A")
col9.metric("🧠 技术面分析判断", tech_judge)

st.markdown(f"### 💬 情绪面分析判断：{sentiment_judge}")

# 模型判断 = 技术 60% + 情绪 40%
tech_score = 0 if tech_judge == "低估" else 1
sentiment_score = 0 if sentiment_judge == "低估" else 1
model_score = 0.6 * tech_score + 0.4 * sentiment_score
model_judge = "低估" if model_score < 0.5 else "高估"

# 综合判断 = 模型 50% + 行业 50%
industry_score_final = 0 if industry_judge == "低估" else 1
final_score = 0.5 * model_score + 0.5 * industry_score_final
final_judge = "低估" if final_score < 0.5 else "高估"

st.markdown(f"### 🧮 综合估值判断：{final_judge}")
st.caption("（模型 = 技术60% + 情绪40%，综合 = 模型50% + 行业50%）")

# 股票价格走势图
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
