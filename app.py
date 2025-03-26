import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# 页面配置
st.set_page_config(page_title="中英文股票估值分析平台", layout="wide")

# 加载映射表
stock_map = pd.read_csv("stock_map.csv")

# 构建搜索字典
def build_search_dict(df):
    search_dict = {}
    for _, row in df.iterrows():
        key_cn = row["name_cn"]
        key_en = row["name_en"]
        key_code = row["code"]
        label = f"{row['name_cn']} ({row['code']})"
        for key in [key_cn, key_en, key_code]:
            if pd.notna(key):
                search_dict[key.upper()] = {
                    "label": label,
                    "code": row["code"],
                    "industry": row["industry"],
                    "name_cn": row["name_cn"],
                    "name_en": row["name_en"]
                }
    return search_dict

search_dict = build_search_dict(stock_map)

# 输入框搜索（支持中文名/英文名/代码）
st.title("📊 中英文股票估值分析平台")
user_input = st.text_input("请输入公司名称或股票代码（支持中英文）", value="苹果").strip().upper()

if user_input in search_dict:
    info = search_dict[user_input]
    code = info["code"]
    industry = info["industry"]
    name_cn = info["name_cn"]
    name_en = info["name_en"]
else:
    st.warning("⚠️ 未找到对应公司，请检查拼写或更新 stock_map.csv")
    st.stop()

st.subheader(f"📄 股票：{name_cn} ({code})")

# 获取股票数据
stock = yf.Ticker(code)
info = stock.info

try:
    pe = info.get("trailingPE", np.nan)
    pb = info.get("priceToBook", np.nan)
    roe = info.get("returnOnEquity", np.nan)
    price = info.get("currentPrice", np.nan)
    eps = info.get("trailingEps", np.nan)
except:
    st.error("❌ 无法获取股票财务数据，请稍后重试。")
    st.stop()

# 显示股票指标
st.subheader("📉 股票关键指标")
col1, col2, col3 = st.columns(3)
col1.metric("PE (市盈率)", f"{pe:.2f}" if not np.isnan(pe) else "N/A")
col2.metric("PB (市净率)", f"{pb:.2f}" if not np.isnan(pb) else "N/A")
col3.metric("ROE (%)", f"{roe*100:.2f}%" if not np.isnan(roe) else "N/A")

# 加载行业平均指标
industry_df = pd.read_csv("industry_averages.csv")  # 文件应包含 columns: industry, PE, PB, ROE
df_ind = industry_df[industry_df["industry"] == industry]

st.subheader("📊 行业平均指标")
col4, col5, col6 = st.columns(3)
col4.metric("行业平均PE", f"{df_ind['PE'].mean():.2f}")
col5.metric("行业平均PB", f"{df_ind['PB'].mean():.2f}")
col6.metric("行业平均ROE", f"{df_ind['ROE'].mean()*100:.2f}%")

# 综合判断（行业）
pe_diff = pe - df_ind["PE"].mean()
pb_diff = pb - df_ind["PB"].mean()
roe_diff = roe - df_ind["ROE"].mean()

industry_score = 0
industry_score += -1 if pe_diff > 0 else 1
industry_score += -1 if pb_diff > 0 else 1
industry_score += 1 if roe_diff > 0 else -1

industry_judgment = "高估" if industry_score < 0 else "低估"

# 加载模型预测价格
try:
    model = joblib.load("valuation_model.pkl")
    features = pd.DataFrame([{
        "PE": pe, "PB": pb, "ROE": roe, "eps": eps
    }])
    predicted_price = model.predict(features)[0]
    model_judgment = "高估" if price > predicted_price else "低估"
except:
    predicted_price = None
    model_judgment = "N/A"

# 综合估值判断
score = 0
score += 1 if industry_judgment == "低估" else -1
score += 1 if model_judgment == "低估" else -1
final_judgment = "低估" if score >= 0 else "高估"

# 显示估值结果
st.subheader("💲 估值结果")
col7, col8, col9 = st.columns(3)
col7.metric("📉 当前价格", f"${price:.2f}")
col8.metric("📈 预测价格", f"${predicted_price:.2f}" if predicted_price else "N/A")
col9.metric("🧠 模型判断", model_judgment)

st.markdown(f"📐 行业比较判断：**{industry_judgment}**")
st.markdown(f"🧠 综合估值判断（50%模型 + 50%行业）：**{final_judgment}**")

# 财务指标雷达图
st.subheader("📊 财务指标雷达图")
radar_df = pd.DataFrame({
    "指标": ["PE", "PB", "ROE"],
    name_cn: [pe, pb, roe],
    "行业平均": [df_ind["PE"].mean().values[0], df_ind["PB"].mean().values[0], df_ind["ROE"].mean().values[0]]
})
radar_df.set_index("指标", inplace=True)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(radar_df)
scaled_df = pd.DataFrame(scaled, columns=radar_df.columns, index=radar_df.index)

fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
labels = scaled_df.index.tolist()
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
for col in scaled_df.columns:
    values = scaled_df[col].tolist()
    values += values[:1]
    ax.plot(angles, values, label=col)
    ax.fill(angles, values, alpha=0.1)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("归一化财务指标对比", size=14)
ax.legend(loc='upper right')
st.pyplot(fig)

# 历史价格走势（12个月）
st.subheader("📈 近12个月价格走势")
end_date = datetime.today()
start_date = end_date - timedelta(days=365)
hist = stock.history(start=start_date, end=end_date)
st.line_chart(hist["Close"])
