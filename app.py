import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import matplotlib.pyplot as plt

# 加载项目模型
model = joblib.load("valuation_model.pkl")

# 读取股票映射列表
stock_map = pd.read_csv("stock_map.csv")

# 创建中英文合并表
stock_map["label"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"

# 页面配置
st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")
st.title("\U0001F4C8 中英文股票估值分析平台")

# 搜索框
query = st.selectbox("\n请输入公司名或股票代码（支持中英文）", stock_map["label"].tolist())

# 解析选中股票
selected_row = stock_map[stock_map["label"] == query].iloc[0]
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

# 词汇表示
st.markdown(f"### \U0001F4C4 股票：**{selected_row['name_cn']} ({ticker})**")

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
                peer.get("returnOnEquity")
            ])
        except:
            continue
    peer_df = pd.DataFrame(peer_data, columns=["PE", "PB", "ROE"]).dropna()
    st.metric("行业平均PE", f"{peer_df['PE'].mean():.2f}")
    st.metric("行业平均PB", f"{peer_df['PB'].mean():.2f}")
    st.metric("行业平均ROE", f"{peer_df['ROE'].mean()*100:.2f}%")

# 构造特征进行预测
model_price = "-"
model_tag = "-"
if all([pe, pb, roe, eps, revenue, gross, cap, cashflow]):
    X_pred = pd.DataFrame([[pe, pb, roe, eps, revenue, gross, cap, cashflow]],
                          columns=["trailingPE", "priceToBook", "returnOnEquity",
                                   "trailingEps", "revenueGrowth", "grossMargins",
                                   "marketCap", "freeCashflow"])
    model_price = model.predict(X_pred)[0]
    model_tag = "高估" if price > model_price else "低估"

# 简单的行业平均对比判断
industry_judgement = "-"
score = 0
if pe and pb and roe and not peer_df.empty:
    if pe < peer_df["PE"].mean(): score += 1
    if pb < peer_df["PB"].mean(): score += 1
    if roe > peer_df["ROE"].mean(): score += 1
    if score >= 2:
        industry_judgement = "低估"
    else:
        industry_judgement = "高估"

# 显示估值结果
st.subheader("\U0001F4C8 估值结果")
st.write(f"**📅 当前价格：** ${price:.2f}" if price else "")
st.write(f"**🔢 预测价格：** ${model_price:.2f}" if model_price != "-" else "")
st.write(f"**\U0001F4CB 模型判断 (基于过去数据学习):** {model_tag}")
st.write(f"**🧠 行业比较判断 (基于 PE/PB/ROE):** {industry_judgement}")

# 显示近30天股价
st.subheader("\U0001F4C6 近30天价格走势")
hist = stock.history(period="1mo")
if not hist.empty:
    st.line_chart(hist["Close"])
