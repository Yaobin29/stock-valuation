import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from math import pi

# 加载项目模型
model = joblib.load("valuation_model.pkl")

# 读取股票映射列表
stock_map = pd.read_csv("stock_map.csv")
stock_map["label"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"

# 页面配置
st.set_page_config(page_title="股票估值分析平台", layout="wide")
st.title("\U0001F4CA 股票估值分析平台")
st.markdown("### 请搜索公司名称或股票代码 (支持中英文)")

# 搜索框 (使用格式函数显示)
def format_label(label):
    row = stock_map[stock_map["label"] == label].iloc[0]
    return f"{row['name_cn']} ({row['code']})"

selected_label = st.selectbox("", stock_map["label"].tolist(), format_func=format_label)
selected_row = stock_map[stock_map["label"] == selected_label].iloc[0]
ticker = selected_row["code"]
industry = selected_row["industry"]

# 应用 Yahoo Finance API
stock = yf.Ticker(ticker)
try:
    info = stock.info
except:
    st.error("无法获取股票数据")
    st.stop()

# 抽取关键指标
pe = info.get("trailingPE")
pb = info.get("priceToBook")
roe = info.get("returnOnEquity")
eps = info.get("trailingEps")
revenue = info.get("revenueGrowth")
gross = info.get("grossMargins")
cap = info.get("marketCap")
cashflow = info.get("freeCashflow")
price = info.get("currentPrice")

st.markdown(f"### 📄 股票：**{selected_row['name_cn']} ({ticker})**")

col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F4C9 股票关键指标")
    st.metric("PE (市盈率)", f"{pe:.2f}" if pe else "-")
    st.metric("PB (市净率)", f"{pb:.2f}" if pb else "-")
    st.metric("ROE (%)", f"{roe*100:.2f}%" if roe else "-")

with col2:
    st.subheader("\U0001F4CA 行业平均指标")
    industry_df = stock_map[stock_map["industry"] == industry]["code"]
    peer_data = []
    for code in industry_df:
        try:
            peer = yf.Ticker(code).info
            peer_data.append([
                peer.get("trailingPE"),
                peer.get("priceToBook"),
                peer.get("returnOnEquity"),
                peer.get("revenueGrowth"),
                peer.get("grossMargins"),
                peer.get("freeCashflow")
            ])
        except:
            continue
    peer_df = pd.DataFrame(peer_data, columns=["PE", "PB", "ROE", "Revenue", "Gross", "Cashflow"]).dropna()
    st.metric("行业平均PE", f"{peer_df['PE'].mean():.2f}")
    st.metric("行业平均PB", f"{peer_df['PB'].mean():.2f}")
    st.metric("行业平均ROE", f"{peer_df['ROE'].mean()*100:.2f}%")

# 估值预测
model_price = "-"
model_tag = "-"
model_score = None
if all([pe, pb, roe, eps, revenue, gross, cap, cashflow]):
    X_pred = pd.DataFrame([[pe, pb, roe, eps, revenue, gross, cap, cashflow]],
                          columns=["trailingPE", "priceToBook", "returnOnEquity",
                                   "trailingEps", "revenueGrowth", "grossMargins",
                                   "marketCap", "freeCashflow"])
    model_price = model.predict(X_pred)[0]
    model_tag = "高估" if price > model_price else "低估"
    model_score = 1 if price > model_price else 0

# 行业比较判断
industry_judgement = "-"
industry_score = None
score = 0
if pe and pb and roe and not peer_df.empty:
    if pe < peer_df["PE"].mean(): score += 1
    if pb < peer_df["PB"].mean(): score += 1
    if roe > peer_df["ROE"].mean(): score += 1
    if score >= 2:
        industry_judgement = "低估"
        industry_score = 0
    else:
        industry_judgement = "高估"
        industry_score = 1

# 综合判断
final_judgement = "-"
if model_score is not None and industry_score is not None:
    alpha = 0.5
    score_combined = model_score * alpha + industry_score * (1 - alpha)
    final_judgement = "高估" if score_combined >= 0.5 else "低估"

# 显示估值结果
st.subheader("\U0001F4B2 估值结果")
st.write(f"**📅 当前价格：** ${price:.2f}" if price else "")
st.write(f"**🔢 预测价格：** ${model_price:.2f}" if model_price != "-" else "")
st.write(f"**\U0001F4CB 模型判断:** {model_tag}")
st.write(f"**🧠 行业比较判断:** {industry_judgement}")
st.write(f"**🧹 综合估值判断 (50%模型 + 50%行业):** {final_judgement}")

# 加入资产极约化 radar chart
st.subheader("\U0001F4D0 资产极约化指标雷达图")
if not peer_df.empty:
    avg = peer_df.mean()
    target = [pe, pb, roe, revenue, gross, cashflow]
    features = ["PE", "PB", "ROE", "Revenue", "Gross", "Cashflow"]
    normalized = [target[i] / avg[features[i]] if avg[features[i]] else 0 for i in range(len(features))]
    angles = [n / float(len(features)) * 2 * pi for n in range(len(features))]
    normalized += [normalized[0]]
    angles += [angles[0]]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, normalized, linewidth=2)
    ax.fill(angles, normalized, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_yticklabels([])
    st.pyplot(fig)

# 显示6个月价格走势
st.subheader("\U0001F4C6 12 个月股价走势")
hist = stock.history(period="12mo")
if not hist.empty:
    st.line_chart(hist["Close"])
