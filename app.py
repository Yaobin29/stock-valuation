import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import os
import re

# 自动每日更新 stock_map.csv
@st.cache_data(ttl=86400)
def load_stock_map():
    url = "https://raw.githubusercontent.com/Yaobin29/stock-valuation/main/stock_map.csv"
    return pd.read_csv(url)

stock_map = load_stock_map()

# 模糊匹配搜索支持
st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")
st.title("📊 中英文股票估值分析平台")

query = st.text_input("请输入公司名称或股票代码（支持中英文）")

if query:
    results = stock_map[
        stock_map.apply(lambda row: query.lower() in str(row["code"]).lower() \
                        or query.lower() in str(row["name_cn"]).lower() \
                        or query.lower() in str(row["name_en"]).lower(), axis=1)]

    if not results.empty:
        selected = results.iloc[0]
        stock_name = selected['name_cn']
        stock_code = selected['code']
    else:
        st.warning("未找到匹配的公司")
        st.stop()
else:
    stock_name = "苹果"
    stock_code = "AAPL"

st.markdown(f"### 🗂️ 股票：{stock_name} ({stock_code})")

# 获取财务数据
def get_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "PE": info.get("trailingPE"),
        "PB": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity", 0) * 100
    }

def get_price(ticker):
    stock = yf.Ticker(ticker)
    return stock.info.get("currentPrice")

def get_history(ticker):
    return yf.Ticker(ticker).history(period="6mo")

# 加载模型
def predict_price(metrics):
    try:
        model = joblib.load("valuation_model.pkl")
        features = pd.DataFrame([metrics])
        return model.predict(features)[0]
    except:
        return None

# 显示指标
col1, col2 = st.columns(2)
with col1:
    stock_metrics = get_metrics(stock_code)
    st.subheader("📉 股票关键指标")
    st.metric("PE (市盈率)", f"{stock_metrics['PE']:.2f}")
    st.metric("PB (市净率)", f"{stock_metrics['PB']:.2f}")
    st.metric("ROE (%)", f"{stock_metrics['ROE']:.2f}%")

with col2:
    industry = stock_map[stock_map['code'] == stock_code]['industry'].values[0]
    df_ind = stock_map[stock_map['industry'] == industry]
    st.subheader("📊 行业平均指标")
    st.metric("行业平均PE", f"{df_ind['PE'].mean():.2f}")
    st.metric("行业平均PB", f"{df_ind['PB'].mean():.2f}")
    st.metric("行业平均ROE", f"{df_ind['ROE'].mean():.2f}%")

# 模型估值预测
actual_price = get_price(stock_code)
predicted_price = predict_price(stock_metrics)
if predicted_price:
    model_judgement = "高估" if actual_price > predicted_price else "低估"
    st.subheader("💲 估值结果")
    st.metric("📉 当前价格", f"${actual_price:.2f}")
    st.metric("📈 预测价格", f"${predicted_price:.2f}")
    st.metric("🧠 模型判断", model_judgement)

# 历史走势图
st.subheader("📈 近6个月价格走势")
data = get_history(stock_code)
fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label='收盘价')
ax.set_xlabel("日期")
ax.set_ylabel("价格")
ax.legend()
st.pyplot(fig)
