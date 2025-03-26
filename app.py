import streamlit as st
import yfinance as yf
import numpy as np

st.title("股票实时估值分析平台")

stocks = {
    "苹果 (AAPL)": "AAPL",
    "英伟达 (NVDA)": "NVDA",
    "AMD (AMD)": "AMD",
    "诺和诺德 (NVO)": "NVO",
    "礼来 (LLY)": "LLY"
}

# 选择股票
stock_name = st.selectbox("请选择股票：", list(stocks.keys()))
stock_code = stocks[stock_name]

# 获取实时股票数据
stock = yf.Ticker(stock_code)

# 显示实时价格
current_price = stock.info["currentPrice"]

# 使用目标价作为估值参考 (示例用分析师目标价)
target_price = stock.info.get("targetMeanPrice", np.nan)

# 显示当前价格与目标价格
col1, col2 = st.columns(2)
col1.metric("当前实时价格", f"${current_price:.2f}")
col2.metric("分析师目标价格", f"${target_price:.2f}")

# 判断估值高低
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

# 获取最近30天的真实价格走势
hist = stock.history(period="1mo")
st.subheader("近30天股价走势")
st.line_chart(hist['Close'])
