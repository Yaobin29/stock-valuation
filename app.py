import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import joblib

# 设置页面
st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")

# 加载股票映射文件
stock_map = pd.read_csv("stock_map.csv")

# 构建搜索选项（中英文+代码）
stock_map["display"] = stock_map["name_cn"] + " (" + stock_map["code"] + ")"
search_options = stock_map["display"].tolist()

# 搜索栏
st.title("📈 中英文股票估值分析平台")
query = st.text_input("请输入公司名称或股票代码（支持中英文，如 苹果、NVDA、0700.HK）", "")

# 匹配逻辑
matched = stock_map[stock_map["display"].str.contains(query, case=False, na=False)] if query else stock_map
selected = st.selectbox("请选择股票：", matched["display"].tolist())

# 获取选中行
row = stock_map[stock_map["display"] == selected].iloc[0]
code = row["code"]
industry = row["industry"]

# 获取股票数据
stock = yf.Ticker(code)
info = stock.info

# 抓取财务指标
def get_metric(name):
    return info.get(name, np.nan)

pe = get_metric("trailingPE")
pb = get_metric("priceToBook")
roe = get_metric("returnOnEquity")
eps = get_metric("trailingEps")
revenue_growth = get_metric("revenueGrowth")
gross_margin = get_metric("grossMargins")
free_cashflow = get_metric("freeCashflow")
current_price = get_metric("currentPrice")

# 显示标题
st.markdown(f"### 📌 股票：{row['name_cn']} ({code})")

# 主要财务指标展示
st.markdown("### 📊 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "-")
col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "-")
col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "-")

# 获取行业平均
industry_stocks = stock_map[stock_map["industry"] == industry]["code"].tolist()
industry_pe, industry_pb, industry_roe = [], [], []

for ticker in industry_stocks:
    try:
        data = yf.Ticker(ticker).info
        industry_pe.append(data.get("trailingPE", np.nan))
        industry_pb.append(data.get("priceToBook", np.nan))
        industry_roe.append(data.get("returnOnEquity", np.nan))
    except:
        continue

avg_pe = np.nanmean(industry_pe)
avg_pb = np.nanmean(industry_pb)
avg_roe = np.nanmean(industry_roe)

st.markdown(f"### 🏭 {industry}行业平均指标")
col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{avg_pe:.2f}" if not np.isnan(avg_pe) else "-")
col5.metric("行业平均PB", f"{avg_pb:.2f}" if not np.isnan(avg_pb) else "-")
col6.metric("行业平均ROE", f"{avg_roe*100:.2f}%" if not np.isnan(avg_roe) else "-")

# 判断逻辑
def tag(val, avg, high_good=True):
    if np.isnan(val) or np.isnan(avg):
        return 0.5
    return 1 if (val > avg if high_good else val < avg) else 0

score_pe = tag(pe, avg_pe, high_good=False)
score_pb = tag(pb, avg_pb, high_good=False)
score_roe = tag(roe, avg_roe, high_good=True)

industry_score = (score_pe + score_pb + score_roe) / 3
industry_judge = "低估" if industry_score >= 0.6 else "高估"
st.markdown(f"### 🧠 行业对比判断：{industry_judge}")

# 加载模型并预测
try:
    model = joblib.load("valuation_model.pkl")
    features = pd.DataFrame([{
        "trailingPE": pe,
        "priceToBook": pb,
        "returnOnEquity": roe,
        "trailingEps": eps,
        "revenueGrowth": revenue_growth,
        "grossMargins": gross_margin,
        "marketCap": info.get("marketCap", np.nan),
        "freeCashflow": free_cashflow
    }])
    pred_price = model.predict(features)[0]
    model_judge = "低估" if current_price < pred_price else "高估"
except:
    pred_price = None
    model_judge = "-"

st.markdown("### 📉 模型估值结果")
col7, col8, col9 = st.columns(3)
col7.metric("当前价格", f"${current_price:.2f}" if current_price else "-")
col8.metric("预测价格", f"${pred_price:.2f}" if pred_price else "-")
col9.metric("模型判断", model_judge)

# 综合判断
weight = 0.5
model_score = 0 if model_judge == "低估" else 1
industry_score_final = 0 if industry_judge == "低估" else 1
final_score = model_score * weight + industry_score_final * (1 - weight)
final_judge = "低估" if final_score < 0.5 else "高估"
st.markdown(f"### 🧮 综合估值判断（50%模型 + 50%行业）：{final_judge}")

# 📈 股票近60个月价格变化
st.markdown("### 📈 股票近半年（ 6月度）价格走势")

try:
    hist = yf.download(code, period="6mo", interval="1d", progress=False)
   if hist.empty or "Close" not in hist.columns:
    raise ValueError("无有效价格数据")
price_data = hist["Close"].dropna()
price_df = pd.DataFrame({"日期": price_data.index, "收盘价": price_data.values}).set_index("日期")
st.line_chart(price_df))
except:
    st.warning("⚠️ 无法获取历史价格数据。可能该股票无月度数据或接口异常。")

