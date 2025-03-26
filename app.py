import streamlit as st

# Set page title in Chinese
st.title("股票估值分析平台")

# Static data example (当前价格和预测价格仅为示例)
stocks = {
    "苹果 (AAPL)": {"current": 223.75, "predict": 200},
    "英伟达 (NVDA)": {"current": 120, "predict": 130},
    "AMD (AMD)": {"current": 105, "predict": 100},
    "诺和诺德 (NVO)": {"current": 75.33, "predict": 89},
    "礼来 (LLY)": {"current": 852.35, "predict": 620}
}

# Stock selection box
stock_selected = st.selectbox("请选择股票：", list(stocks.keys()))

# Show current price and predict price
col1, col2 = st.columns(2)
col1.metric("当前价格", f"${stocks[stock_selected]['current']}")
col2.metric("预测合理估值", f"${stocks[stock_selected]['predict']}")

# Determine cheap/fair/expensive
current_price = stocks[stock_selected]['current']
predict_price = stocks[stock_selected]['predict']

if current_price < predict_price * 0.9:
    valuation = "便宜"
elif current_price > predict_price * 1.1:
    valuation = "贵"
else:
    valuation = "合理"

st.subheader(f"估值判断：{valuation}")

# Example price trend (static data)
import numpy as np
np.random.seed(0)
price_trend = np.random.normal(current_price, 2, size=30)

st.subheader("近30天股价走势（示例数据）")
st.line_chart(price_trend)

