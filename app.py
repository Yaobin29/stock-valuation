import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import difflib

st.set_page_config(layout="wide")
st.title("中英文股票估值分析平台")

# 加载公司映射表
@st.cache_data
def load_stock_map():
    df = pd.read_csv("stock_map.csv")
    return df

stock_df = load_stock_map()

# 输入框：支持中英文或代码
user_input = st.text_input("请输入公司名称或股票代码（支持中英文，如 苹果、NVDA、0700.HK）")

if not user_input:
    st.stop()

# 模糊匹配
matches = []
for _, row in stock_df.iterrows():
    if user_input.lower() in str(row["name_cn"]).lower() or \
       user_input.lower() in str(row["name_en"]).lower() or \
       user_input.lower() in str(row["code"]).lower():
        matches.append(row)

if len(matches) == 0:
    st.error("未找到匹配的股票，请检查输入或扩展 stock_map.csv")
    st.stop()

stock_row = matches[0]
code = stock_row["code"]
industry = stock_row["industry"]
name_cn = stock_row["name_cn"]
name_en = stock_row["name_en"]

st.subheader(f"🎯 股票：{name_cn} ({code})")

# 加载模型
try:
    model = joblib.load("valuation_model.pkl")
except:
    st.error("未找到机器学习模型，请确保 valuation_model.pkl 已上传")
    st.stop()

# 获取目标股票数据
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        data = {
            "pe": info.get("trailingPE", np.nan),
            "pb": info.get("priceToBook", np.nan),
            "roe": info.get("returnOnEquity", np.nan),
            "eps": info.get("trailingEps", np.nan),
            "revenue_growth": info.get("revenueGrowth", np.nan),
            "current_price": info.get("currentPrice", np.nan)
        }
        if np.isnan(data["current_price"]):
            hist = stock.history(period="1d")
            data["current_price"] = hist["Close"].iloc[-1] if not hist.empty else np.nan
        return data, stock
    except:
        return None, None

target_data, stock_obj = get_stock_info(code)

if target_data is None or any(np.isnan(v) for v in list(target_data.values())[:5]):
    st.error("⚠️ 无法获取该股票完整数据，暂时无法估值。")
    st.stop()

# 显示关键财务指标
st.subheader("📊 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{target_data['pe']:.2f}")
col2.metric("PB (市净率)", f"{target_data['pb']:.2f}")
col3.metric("ROE (%)", f"{target_data['roe'] * 100:.2f}")

# 获取行业对比指标
industry_peers = stock_df[(stock_df["industry"] == industry) & (stock_df["code"] != code)]["code"].tolist()

peers_data = []
for peer_code in industry_peers:
    d, _ = get_stock_info(peer_code)
    if d and not any(np.isnan([d["pe"], d["pb"], d["roe"]])):
        peers_data.append(d)

if peers_data:
    peer_df = pd.DataFrame(peers_data)
    avg_pe = peer_df["pe"].mean()
    avg_pb = peer_df["pb"].mean()
    avg_roe = peer_df["roe"].mean()

    st.subheader(f"{industry}行业平均指标")
    col4, col5, col6 = st.columns(3)
    col4.metric("行业平均PE", f"{avg_pe:.2f}")
    col5.metric("行业平均PB", f"{avg_pb:.2f}")
    col6.metric("行业平均ROE", f"{avg_roe * 100:.2f}%")

    # 多维判断
    score = 0
    if target_data["pe"] < avg_pe * 0.9: score += 1
    if target_data["pb"] < avg_pb * 0.9: score += 1
    if target_data["roe"] > avg_roe * 1.1: score += 1

    if score >= 2:
        industry_judgment = "低估"
    elif score == 1:
        industry_judgment = "合理"
    else:
        industry_judgment = "高估"
else:
    st.warning("⚠️ 无法获取行业参考数据")
    industry_judgment = "未知"

# 模型预测价格
X = pd.DataFrame([{
    "pe": target_data["pe"],
    "pb": target_data["pb"],
    "roe": target_data["roe"],
    "eps": target_data["eps"],
    "revenue_growth": target_data["revenue_growth"]
}])
try:
    predicted_price = model.predict(X)[0]
except:
    predicted_price = np.nan

st.subheader("📈 模型估值结果")
col7, col8, col9 = st.columns(3)
col7.metric("当前价格", f"${target_data['current_price']:.2f}")
col8.metric("预测价格", f"${predicted_price:.2f}" if not np.isnan(predicted_price) else "无法预测")
if not np.isnan(predicted_price):
    if target_data["current_price"] < predicted_price * 0.9:
        pred_judgment = "低估"
    elif target_data["current_price"] > predicted_price * 1.1:
        pred_judgment = "高估"
    else:
        pred_judgment = "合理"
    col9.metric("模型判断", pred_judgment)
else:
    col9.write("模型判断不可用")

st.subheader(f"🧠 综合估值判断：{industry_judgment}")

# 走势图
try:
    hist = stock_obj.history(period="1mo")
    st.subheader("📉 近30天价格走势")
    if not hist.empty:
        st.line_chart(hist["Close"])
    else:
        st.write("暂无历史数据")
except:
    st.warning("⚠️ 无法加载价格走势")
