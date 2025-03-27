import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")

# 全局样式（圆角卡片 + 判断颜色）
st.markdown("""
<style>
.card {
    background-color: white;
    padding: 1.5rem;
    border-radius: 1.25rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 2rem;
}
.judge-green {
    color: green;
    font-weight: bold;
}
.judge-red {
    color: red;
    font-weight: bold;
}
.judge-yellow {
    color: #d99000;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 加载数据
stock_map = pd.read_csv("stock_map.csv")
stock_map["display"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"
query = st.text_input("请输入公司名称或股票代码（支持中英文，如 苹果、NVDA、0700.HK）", "")
matched = stock_map[stock_map["display"].str.contains(query, case=False, na=False)] if query else stock_map
selected = st.selectbox("请选择股票：", matched["display"].tolist())

row = stock_map[stock_map["display"] == selected].iloc[0]
code = row["code"]
industry = row["industry"]

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

st.markdown(f"### 📝 股票：{row['name_cn']} ({code})")

# 股票关键指标卡片
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 股票关键指标")
    col1, col2, col3 = st.columns(3)
    col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "-")
    col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "-")
    col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "-")
    st.markdown('</div>', unsafe_allow_html=True)

# 行业均值判断
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

# 判断行业估值
def tag(val, avg, high_good=True):
    if np.isnan(val) or np.isnan(avg):
        return 0.5
    return 1 if (val > avg if high_good else val < avg) else 0

score_pe = tag(pe, avg_pe, False)
score_pb = tag(pb, avg_pb, False)
score_roe = tag(roe, avg_roe, True)
industry_score = (score_pe + score_pb + score_roe) / 3
industry_judge = "低估" if industry_score >= 0.6 else "高估"
industry_color = "judge-green" if industry_judge == "低估" else "judge-red"

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏭 行业估值判断")
    col4, col5, col6 = st.columns(3)
    col4.metric("行业平均PE", f"{avg_pe:.2f}" if not np.isnan(avg_pe) else "-")
    col5.metric("行业平均PB", f"{avg_pb:.2f}" if not np.isnan(avg_pb) else "-")
    col6.metric("行业平均ROE", f"{avg_roe*100:.2f}%" if not np.isnan(avg_roe) else "-")
    st.markdown(f"<p>行业判断结果：<span class='{industry_color}'>{industry_judge}</span></p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 模型判断
try:
    model = joblib.load("valuation_model.pkl")
    analyzer = SentimentIntensityAnalyzer()
    news = stock.news if hasattr(stock, "news") else []
    headlines = [article["title"] for article in news if "title" in article][:5]
    scores = [analyzer.polarity_scores(title)["compound"] for title in headlines]
    sentiment_score = np.mean(scores) if scores else 0.0
except:
    sentiment_score = 0.0

features = pd.DataFrame([{
    "trailingPE": pe, "priceToBook": pb, "returnOnEquity": roe, "trailingEps": eps,
    "revenueGrowth": revenue_growth, "grossMargins": gross_margin,
    "marketCap": info.get("marketCap", np.nan),
    "freeCashflow": free_cashflow,
    
}])

try:
    pred_price = model.predict(features)[0]
    tech_judge = "低估" if current_price < pred_price else "高估"
    sentiment_judge = "正面" if sentiment_score > 0.1 else "负面" if sentiment_score < -0.1 else "中性"

    tech_score = 0 if tech_judge == "低估" else 1
    senti_score = 0 if sentiment_judge == "负面" else 0.5 if sentiment_judge == "中性" else 1
    model_score = tech_score * 0.6 + senti_score * 0.4
    model_judge = "低估" if model_score < 0.5 else "高估"
except:
    pred_price, tech_judge, sentiment_judge, model_judge = None, "-", "-", "-"

model_color = "judge-green" if model_judge == "低估" else "judge-red"
sentiment_color = {
    "正面": "judge-green", "中性": "judge-yellow", "负面": "judge-red"
}[sentiment_judge]

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🤖 模型估值判断（技术 + 情绪）")
    col7, col8, col9 = st.columns(3)
    col7.metric("📉 当前价格", f"${current_price:.2f}" if current_price else "-")
    col8.metric("📈 预测价格", f"${pred_price:.2f}" if pred_price else "N/A")
    col9.metric("📊 技术面判断", tech_judge)
    st.markdown(f"<p>情绪面判断：<span class='{sentiment_color}'>{sentiment_judge}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p>模型判断结果：<span class='{model_color}'>{model_judge}</span></p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 最终判断
industry_score_final = 0 if industry_judge == "低估" else 1
final_score = model_score * 0.5 + industry_score_final * 0.5
final_judge = "低估" if final_score < 0.4 else "高估" if final_score > 0.6 else "合理"
final_color = {
    "低估": "judge-green", "合理": "judge-yellow", "高估": "judge-red"
}[final_judge]

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧮 最终估值判断（模型 × 行业）")
    st.markdown(f"<h4>最终判断：<span class='{final_color}'>{final_judge}</span></h4>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 股票走势图
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 股票近6个月价格走势")
    try:
        hist = stock.history(period="6mo", interval="1d")
        if hist.empty or "Close" not in hist.columns:
            raise ValueError("无有效价格数据")
        price_data = hist["Close"].dropna()
        price_df = pd.DataFrame({"日期": price_data.index, "收盘价": price_data.values}).set_index("日期")
        st.line_chart(price_df)
    except Exception as e:
        st.warning(f"⚠️ 无法获取历史价格数据。\n错误信息：{e}")
    st.markdown('</div>', unsafe_allow_html=True)
