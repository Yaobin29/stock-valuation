import streamlit as st
import yfinance as yf
import numpy as np

st.title("股票实时估值分析平台")

stocks = {
    "苹果 (AAPL)": "AAPL",
    "英伟达 (NVDA)": "NVDA",
    "AMD (AMD)": "AMD",
    "诺和诺德 (NVO)": "NVO",
    "礼来 (LLY)": "LLY",
    "Cloudflare (NET)": "NET",
    "百济神州 (BGNE)": "BGNE",
    "药明生物 (2269.HK)": "2269.HK"
}

# 选择股票
stock_name = st.selectbox("请选择股票：", list(stocks.keys()))
stock_code = stocks[stock_name]

stock = yf.Ticker(stock_code)

# 获取实时价格（优先用info，否则用最近收盘价）
try:
    current_price = stock.info["currentPrice"]
except:
    current_price = stock.history(period="1d")["Close"].iloc[-1]

# 获取分析师目标价（避免错误）
try:
    target_price = stock.info["targetMeanPrice"]
except:
    target_price = np.nan

# 显示当前价格和目标价格
col1, col2 = st.columns(2)
col1.metric("当前实时价格", f"${current_price:.2f}")
if not np.isnan(target_price):
    col2.metric("分析师目标价格", f"${target_price:.2f}")
else:
    col2.metric("分析师目标价格", "暂无数据")

# 估值判断
if np.isnan(target_price):
    valuation = "暂无目标价格"
else:
    if current_price < target_price * 0.9:
        valuation = "便宜"
    elif current_price > target_price * 1.1:
        valuation = "贵"
    else:
        valuation = "合理"

st.subheader(f"估值判断：{valuation}")

# 获取最近30天价格趋势
hist = stock.history(period="1mo")
st.subheader("近30天股价走势")
st.line_chart(hist['Close'])
