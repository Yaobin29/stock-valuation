import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import plotly.graph_objs as go

# 页面配置
st.set_page_config(page_title="估值分析平台", layout="wide")

# 加载映射表
stock_map = pd.read_csv("stock_map.csv")

# 标题
st.title("📊 中英文股票估值分析平台")
query = st.text_input("请搜索公司名称或股票代码 (支持中英文)", "苹果")

# 精确匹配 ticker
def find_ticker(query):
    match = stock_map[stock_map.apply(
        lambda row: query.lower() in str(row["name_cn"]).lower() or query.lower() in str(row["ticker"]).lower(),
        axis=1
    )]
    if len(match) == 0:
        return None, None
    ticker = match.iloc[0]["ticker"]
    name_cn = match.iloc[0]["name_cn"]
    return ticker, name_cn

ticker, name_cn = find_ticker(query)

if not ticker:
    st.error("未找到该公司，请检查名称或代码是否正确。")
    st.stop()

st.header(f"📄 股票：{name_cn} ({ticker})")

# 获取数据
stock = yf.Ticker(ticker)
try:
    info = stock.info
    pe = info.get("trailingPE")
    pb = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    current_price = info.get("currentPrice")
except:
    st.error("❌ 获取股票信息失败，可能该股票不支持估值数据。")
    st.stop()

# 行业均值
industry = stock_map.loc[stock_map["ticker"] == ticker, "industry"]
if industry.empty:
    st.warning("⚠️ 无法识别行业，部分估值功能不可用")
    df_ind = pd.DataFrame()
else:
    df_ind = stock_map[stock_map["industry"] == industry.values[0]]

# --- 显示指标 ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📉 股票关键指标")
    st.metric("PE (市盈率)", f"{pe:.2f}" if pe else "—")
    st.metric("PB (市净率)", f"{pb:.2f}" if pb else "—")
    st.metric("ROE (%)", f"{roe*100:.2f}%" if roe else "—")

with col2:
    st.subheader("📊 行业平均指标")
    st.metric("行业平均PE", f"{df_ind['PE'].mean():.2f}" if not df_ind.empty else "—")
    st.metric("行业平均PB", f"{df_ind['PB'].mean():.2f}" if not df_ind.empty else "—")
    st.metric("行业平均ROE", f"{df_ind['ROE'].mean()*100:.2f}%" if not df_ind.empty else "—")

# --- 模型估值 ---
st.subheader("💲 估值结果")
model = joblib.load("valuation_model.pkl")
features = {
    "PE": pe,
    "PB": pb,
    "ROE": roe * 100 if roe else None
}
df_feat = pd.DataFrame([features])
predicted_price = model.predict(df_feat)[0]

col3, col4 = st.columns(2)
with col3:
    st.metric("📉 当前价格", f"${current_price:.2f}" if current_price else "—")
    st.metric("🧮 预测价格", f"${predicted_price:.2f}")

# 模型判断
if current_price and predicted_price:
    tag = "低估" if predicted_price > current_price else "高估"
    st.metric("📈 模型判断", tag)

# 行业估值判断
def judge_by_industry(pe, pb, roe, df):
    if df.empty:
        return "—"
    score = 0
    score += pe < df["PE"].mean()
    score += pb < df["PB"].mean()
    score += roe > df["ROE"].mean()
    return "低估" if score >= 2 else "高估"

ind_judge = judge_by_industry(pe, pb, roe, df_ind)
st.metric("📊 行业比较判断", ind_judge)

# 综合判断（可调整权重）
if current_price and predicted_price:
    tag = "低估" if (
        (predicted_price > current_price and ind_judge == "低估")
        or (predicted_price > current_price and ind_judge == "—")
        or (predicted_price > current_price and ind_judge == "高估")
    ) else "高估"
    st.success(f"🧠 综合估值判断：{tag}")

# --- 雷达图 ---
if all([pe, pb, roe]):
    st.subheader("📌 财务指标雷达图")
    radar_df = pd.DataFrame({
        "指标": ["PE", "PB", "ROE"],
        "当前股票": [pe, pb, roe * 100],
        "行业均值": [
            df_ind["PE"].mean() if not df_ind.empty else 0,
            df_ind["PB"].mean() if not df_ind.empty else 0,
            df_ind["ROE"].mean() * 100 if not df_ind.empty else 0
        ]
    })

    fig_radar = go.Figure()
    for col in ["当前股票", "行业均值"]:
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df[col],
            theta=radar_df["指标"],
            fill='toself',
            name=col
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# --- 价格走势 ---
st.subheader("📈 过去12个月价格趋势")
hist = stock.history(period="1y")
if not hist.empty:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="收盘价"))
    fig_price.update_layout(xaxis_title="日期", yaxis_title="价格 ($)")
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.warning("⚠️ 无法获取历史价格数据")
