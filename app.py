import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objs as go
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import isnan

st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")
st.title("📊 中英文股票估值分析平台")
st.markdown("请搜索公司名称或股票代码 (支持中英文)")

# 加载股票映射表
@st.cache_data
def load_stock_map():
    return pd.read_csv("stock_map.csv")

stock_map = load_stock_map()

# 用户输入股票关键词
query = st.text_input("请输入公司中文名、英文名或股票代码：", "苹果")

# 模糊匹配并展示匹配列表
results = stock_map[
    stock_map.apply(lambda row: query.lower() in str(row["name_cn"]).lower()
                    or query.lower() in str(row["name_en"]).lower()
                    or query.lower() in str(row["ticker"]).lower(), axis=1)
]

if results.empty:
    st.warning("未找到匹配的股票。请尝试其他关键词。")
    st.stop()

# 若存在多个匹配项，用户选择其中一个
selected_row = results.iloc[0]
if len(results) > 1:
    selected_label = st.selectbox("请选择股票：", results.apply(lambda row: f"{row['name_cn']} ({row['ticker']})", axis=1))
    selected_row = results[results.apply(lambda row: f"{row['name_cn']} ({row['ticker']})" == selected_label, axis=1)].iloc[0]

name_cn = selected_row['name_cn']
ticker = selected_row['ticker']
industry = selected_row['industry']

st.subheader(f"📄 股票：{name_cn} ({ticker})")

# 获取股票财务指标
def get_stock_metrics(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "roe": info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None,
        "price": info.get("currentPrice")
    }

metrics = get_stock_metrics(ticker)

# 获取行业平均
def get_industry_averages(ind):
    df = stock_map[stock_map["industry"] == ind]
    pe_list = []
    pb_list = []
    roe_list = []
    for t in df['ticker']:
        m = get_stock_metrics(t)
        if m['pe'] and m['pb'] and m['roe']:
            pe_list.append(m['pe'])
            pb_list.append(m['pb'])
            roe_list.append(m['roe'])
    return {
        "pe": np.mean(pe_list) if pe_list else None,
        "pb": np.mean(pb_list) if pb_list else None,
        "roe": np.mean(roe_list) if roe_list else None
    }

industry_avg = get_industry_averages(industry)

# 显示指标
st.subheader("📉 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{metrics['pe']:.2f}" if metrics['pe'] else "N/A")
col2.metric("PB (市净率)", f"{metrics['pb']:.2f}" if metrics['pb'] else "N/A")
col3.metric("ROE (%)", f"{metrics['roe']:.2f}%" if metrics['roe'] else "N/A")

st.subheader("📊 行业平均指标")
col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{industry_avg['pe']:.2f}" if industry_avg['pe'] else "N/A")
col5.metric("行业平均PB", f"{industry_avg['pb']:.2f}" if industry_avg['pb'] else "N/A")
col6.metric("行业平均ROE", f"{industry_avg['roe']:.2f}%" if industry_avg['roe'] else "N/A")

# 模型估值预测
model = joblib.load("valuation_model.pkl")

# 构造特征DataFrame并预测
features = pd.DataFrame([{
    "pe": metrics['pe'],
    "pb": metrics['pb'],
    "roe": metrics['roe'],
    "industry_pe": industry_avg['pe'],
    "industry_pb": industry_avg['pb'],
    "industry_roe": industry_avg['roe']
}])

if features.isnull().any().any():
    st.error("❌ 缺少必要指标，无法进行估值预测。")
    st.stop()

predicted_price = model.predict(features)[0]
model_judgment = "低估" if predicted_price > metrics['price'] else "高估"

# 简单行业比较判断
industry_judgment = "低估" if (metrics['pe'] or 0) < (industry_avg['pe'] or 0) and (metrics['pb'] or 0) < (industry_avg['pb'] or 0) else "高估"

# 综合判断 (加权平均)
model_weight = 0.5
industry_weight = 0.5
combined_score = (1 if model_judgment == "低估" else 0) * model_weight + (1 if industry_judgment == "低估" else 0) * industry_weight
combined_judgment = "低估" if combined_score >= 0.5 else "高估"

st.subheader("💲 估值结果")
st.metric("📉 当前价格", f"${metrics['price']:.2f}")
st.metric("📈 预测价格", f"${predicted_price:.2f}")
st.metric("🤖 模型判断", model_judgment)
st.metric("📊 行业比较判断", industry_judgment)
st.success(f"🧠 综合估值判断（{int(model_weight*100)}%模型 + {int(industry_weight*100)}%行业）: {combined_judgment}")

# 股票价格走势（过去12个月）
st.subheader("📉 近12个月价格走势")
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)
data = yf.download(ticker, start=start_date, end=end_date)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='收盘价'))
fig.update_layout(xaxis_title='日期', yaxis_title='价格', height=400)
st.plotly_chart(fig, use_container_width=True)
