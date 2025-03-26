import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objs as go

# 页面配置
st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")

# 读取映射表
stock_map = pd.read_csv("stock_map.csv")
stock_dict = {row["name_cn"]: row["code"] for _, row in stock_map.iterrows()}
stock_dict.update({row["name_en"]: row["code"] for _, row in stock_map.iterrows()})
stock_dict.update({row["code"]: row["code"] for _, row in stock_map.iterrows()})

# 侧边栏选择
st.title("📊 中英文股票估值分析平台")
user_input = st.selectbox("请输入公司名称或股票代码（支持中英文）", list(stock_dict.keys()))
symbol = stock_dict[user_input]

# 股票基本信息
stock = yf.Ticker(symbol)
info = stock.info

name_display = f"{user_input} ({symbol})"
st.subheader(f"📄 股票：{name_display}")

# 获取财务指标
pe = info.get("trailingPE", None)
pb = info.get("priceToBook", None)
roe = info.get("returnOnEquity", None)
roe = f"{roe*100:.2f}%" if roe is not None else None

col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if pe else "N/A")
col2.metric("PB (市净率)", f"{pb:.2f}" if pb else "N/A")
col3.metric("ROE (%)", roe if roe else "N/A")

# 行业均值
industry = stock_map[stock_map["code"] == symbol]["industry"].values[0]
industry_df = pd.read_csv("industry_averages.csv")
df_ind = industry_df[industry_df["industry"] == industry]

st.subheader("📊 行业平均指标")
col1, col2, col3 = st.columns(3)
col1.metric("行业平均PE", f"{df_ind['PE'].mean():.2f}")
col2.metric("行业平均PB", f"{df_ind['PB'].mean():.2f}")
col3.metric("行业平均ROE", f"{df_ind['ROE'].mean():.2f}%")

# 模型估值预测
model = joblib.load("valuation_model.pkl")
input_data = pd.DataFrame([{
    "PE": pe, "PB": pb, "ROE": float(roe.strip("%")) if roe else None
}])
prediction = model.predict(input_data)[0]

# 当前价格
current_price = info.get("currentPrice", None)

st.subheader("💲估值结果")
col1, col2, col3 = st.columns(3)
col1.metric("📉 当前价格", f"${current_price:.2f}" if current_price else "N/A")
col2.metric("🔮 预测价格", f"${prediction:.2f}")
if current_price:
    judgment = "高估" if current_price > prediction else "低估"
    col3.metric("📌 模型判断", judgment)

# 综合判断（模型 + 行业）
combined = ""
if current_price:
    industry_pe = df_ind['PE'].mean()
    pe_judgment = "高估" if pe > industry_pe else "低估"
    combined = "高估" if [judgment, pe_judgment].count("高估") >= 1 else "低估"
st.metric("🧠 综合估值判断（50%模型 + 50%行业）", combined)

# 股价走势（6个月）
st.subheader("📉 历史价格走势（6个月）")
hist = stock.history(period="6mo")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="收盘价"))
fig.update_layout(title=f"{symbol} 收盘价走势", xaxis_title="日期", yaxis_title="价格")
st.plotly_chart(fig, use_container_width=True)
