import streamlit as st
import yfinance as yf
import numpy as np

st.title("多维度股票估值分析平台（PE/PB/ROE）")

# 定义每个股票的行业内对比公司（用于动态计算行业平均值）
industry_stocks = {
    "科技": ["AAPL", "NVDA", "AMD", "MSFT", "GOOGL", "META"],
    "医药": ["NVO", "LLY", "PFE", "JNJ", "MRK", "BGNE", "2269.HK"],
    "云计算": ["NET", "AMZN", "CRM", "NOW"]
}

# 股票及对应行业定义
stocks = {
    "苹果 (AAPL)": {"code": "AAPL", "industry": "科技"},
    "英伟达 (NVDA)": {"code": "NVDA", "industry": "科技"},
    "AMD (AMD)": {"code": "AMD", "industry": "科技"},
    "诺和诺德 (NVO)": {"code": "NVO", "industry": "医药"},
    "礼来 (LLY)": {"code": "LLY", "industry": "医药"},
    "Cloudflare (NET)": {"code": "NET", "industry": "云计算"},
    "百济神州 (BGNE)": {"code": "BGNE", "industry": "医药"},
    "药明生物 (2269.HK)": {"code": "2269.HK", "industry": "医药"}
}

# 用户选择股票
stock_name = st.selectbox("请选择股票：", list(stocks.keys()))
stock_code = stocks[stock_name]["code"]
industry = stocks[stock_name]["industry"]

# 获取目标股票数据
target_stock = yf.Ticker(stock_code)

# 抓取目标股票数据
def get_stock_metrics(ticker):
    info = ticker.info
    pe = info.get("trailingPE", np.nan)
    pb = info.get("priceToBook", np.nan)
    roe = info.get("returnOnEquity", np.nan)
    current_price = info.get("currentPrice", np.nan)
    if np.isnan(current_price):
        current_price = ticker.history(period="1d")["Close"].iloc[-1]
    return current_price, pe, pb, roe

current_price, pe, pb, roe = get_stock_metrics(target_stock)

# 显示目标股票的数据
st.subheader("目标股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "暂无数据")
col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "暂无数据")
col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "暂无数据")

# 动态抓取行业数据计算平均PE、PB、ROE
st.subheader(f"{industry}行业平均指标")
industry_pe, industry_pb, industry_roe = [], [], []

for code in industry_stocks[industry]:
    peer_stock = yf.Ticker(code)
    _, peer_pe, peer_pb, peer_roe = get_stock_metrics(peer_stock)
    if not np.isnan(peer_pe): industry_pe.append(peer_pe)
    if not np.isnan(peer_pb): industry_pb.append(peer_pb)
    if not np.isnan(peer_roe): industry_roe.append(peer_roe)

avg_pe = np.mean(industry_pe)
avg_pb = np.mean(industry_pb)
avg_roe = np.mean(industry_roe)

col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{avg_pe:.2f}")
col5.metric("行业平均PB", f"{avg_pb:.2f}")
col6.metric("行业平均ROE", f"{avg_roe*100:.2f}%")

# 多维估值判断逻辑
valuation_score = 0
if not np.isnan(pe) and pe < avg_pe * 0.9:
    valuation_score += 1
if not np.isnan(pb) and pb < avg_pb * 0.9:
    valuation_score += 1
if not np.isnan(roe) and roe > avg_roe * 1.1:  # ROE越高越好
    valuation_score += 1

# 评分判断
if valuation_score >= 2:
    valuation = "便宜（低估）"
elif valuation_score == 1:
    valuation = "合理"
else:
    valuation = "贵（高估）"

st.subheader(f"综合估值判断：{valuation}")

# 价格走势
st.subheader("近30天价格走势")
hist = target_stock.history(period="1mo")
st.line_chart(hist['Close'])
