import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

st.title("机器学习驱动的股票估值平台")

stocks = {
    "苹果 (AAPL)": "AAPL",
    "英伟达 (NVDA)": "NVDA",
    "AMD (AMD)": "AMD",
    "诺和诺德 (NVO)": "NVO",
    "礼来 (LLY)": "LLY",
    "Cloudflare (NET)": "NET",
    "百济神州 (BGNE)": "BGNE"
}

stock_name = st.selectbox("请选择股票：", list(stocks.keys()))
ticker = stocks[stock_name]
stock = yf.Ticker(ticker)
info = stock.info

# 当前价格
try:
    current_price = info.get("currentPrice", stock.history(period="1d")["Close"].iloc[-1])
except:
    st.error("无法获取股票当前价格")
    st.stop()

# 获取特征
features = {
    "pe": info.get("trailingPE"),
    "pb": info.get("priceToBook"),
    "roe": info.get("returnOnEquity"),
    "eps": info.get("trailingEps"),
    "revenue_growth": info.get("revenueGrowth")
}

# 展示基本信息
st.subheader("当前指标")
for k, v in features.items():
    if v is not None:
        st.write(f"📊 {k.upper()}: {round(v, 3)}")
    else:
        st.write(f"📊 {k.upper()}: 暂无数据")

# 检查特征完整性
if None in features.values():
    st.warning("⚠️ 无法进行模型预测，部分财务数据缺失。")
    st.stop()

# 加载模型
try:
    model = joblib.load("valuation_model.pkl")
except:
    st.error("未能加载模型文件，请确认 valuation_model.pkl 已上传至仓库")
    st.stop()

# 构造特征DataFrame并预测
X = pd.DataFrame([features])
predicted_price = model.predict(X)[0]

st.metric("当前价格", f"${current_price:.2f}")
st.metric("模型预测合理价格", f"${predicted_price:.2f}")

# 估值判断
if current_price < predicted_price * 0.9:
    judgment = "💚 低估（便宜）"
elif current_price > predicted_price * 1.1:
    judgment = "❤️ 高估（贵）"
else:
    judgment = "⚖️ 合理"

st.subheader(f"估值判断：{judgment}")

# 股价走势
hist = stock.history(period="1mo")
st.subheader("近30天股价走势")
st.line_chart(hist['Close'])
