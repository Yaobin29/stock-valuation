import streamlit as st
import yfinance as yf
import numpy as np

st.title("股票实时PE估值分析平台")

# 股票和对应的行业平均PE
stocks = {
    "苹果 (AAPL)": {"code": "AAPL", "industry_pe": 28},
    "英伟达 (NVDA)": {"code": "NVDA", "industry_pe": 35},
    "AMD (AMD)": {"code": "AMD", "industry_pe": 30},
    "诺和诺德 (NVO)": {"code": "NVO", "industry_pe": 20},
    "礼来 (LLY)": {"code": "LLY", "industry_pe": 25},
    "Cloudflare (NET)": {"code": "NET", "industry_pe": 45},
    "百济神州 (BGNE)": {"code": "BGNE", "industry_pe": 40},
    "药明生物 (2269.HK)": {"code": "2269.HK", "industry_pe": 35}
}

# 选择股票
stock_name = st.selectbox("请选择股票：", list(stocks.keys()))
stock_code = stocks[stock_name]["code"]
industry_pe = stocks[stock_name]["industry_pe"]

stock = yf.Ticker(stock_code)

# 获取实时股价和PE
try:
    current_price = stock.info["currentPrice"]
except:
    current_price = stock.history(period="1d")["Close"].iloc[-1]

pe_ratio = stock.info.get("trailingPE", np.nan)

# 展示数据
col1, col2 = st.columns(2)
col1.metric("实时价格", f"${current_price:.2f}")

if not np.isnan(pe_ratio):
    col2.metric("实时PE (市盈率)", f"{pe_ratio:.2f}")
else:
    col2.metric("实时PE (市盈率)", "暂无数据")

st.write(f"行业平均PE参考值：{industry_pe}")

# 根据PE进行估值判断
if not np.isnan(pe_ratio):
    if pe_ratio < industry_pe * 0.8:
        valuation = "便宜（低估）"
    elif pe_ratio > industry_pe * 1.2:
        valuation = "贵（高估）"
    else:
        valuation = "合理"
else:
    valuation = "暂无PE数据"

st.subheader(f"估值判断：{valuation}")

# 显示最近一个月的股价走势
hist = stock.history(period="1mo")
st.subheader("近30天股价走势")
st.line_chart(hist['Close'])
