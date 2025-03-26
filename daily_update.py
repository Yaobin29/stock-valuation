import pandas as pd
import yfinance as yf
import numpy as np
from xgboost import XGBRegressor
import joblib
import json

print("📥 Loading company list from stock_map.csv...")
stock_map = pd.read_csv("stock_map.csv")
codes = stock_map["code"].tolist()

data_list = []
print(f"🌐 Fetching data for {len(codes)} companies...")

for code in codes:
    try:
        stock = yf.Ticker(code)
        info = stock.info
        data_list.append({
            "code": code,
            "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"),
            "returnOnEquity": info.get("returnOnEquity"),
            "trailingEps": info.get("trailingEps"),
            "revenueGrowth": info.get("revenueGrowth"),
            "grossMargins": info.get("grossMargins"),
            "marketCap": info.get("marketCap"),
            "freeCashflow": info.get("freeCashflow"),
            "currentPrice": info.get("currentPrice"),
        })
    except Exception as e:
        print(f"⚠️ Failed to fetch {code}: {e}")

df = pd.DataFrame(data_list)
print(f"✅ Raw data shape: {df.shape}")

# 清洗数据
df = df.dropna()
print(f"✅ After dropna: {df.shape}")

# 特征列
features = [
    "trailingPE", "priceToBook", "returnOnEquity", "trailingEps",
    "revenueGrowth", "grossMargins", "marketCap", "freeCashflow"
]
target = "currentPrice"

# 训练模型
X = df[features]
y = df[target]

model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4)
model.fit(X, y)

# 简单评估
from sklearn.metrics import mean_absolute_error
preds = model.predict(X)
mae = mean_absolute_error(y, preds)
print(f"📉 MAE: ${mae:.2f}")

# 保存模型
joblib.dump(model, "valuation_model.pkl")
print("✅ Saved model as valuation_model.pkl")

# 计算行业平均指标
print("📊 Calculating industry averages...")
industry_avg = {}

for industry in stock_map["industry"].unique():
    codes_in_industry = stock_map[stock_map["industry"] == industry]["code"]
    df_ind = df[df["code"].isin(codes_in_industry)]

    if df_ind.empty:
        continue

    industry_avg[industry] = {
        "avg_PE": round(df_ind["trailingPE"].mean(), 2),
        "avg_PB": round(df_ind["priceToBook"].mean(), 2),
        "avg_ROE": round(df_ind["returnOnEquity"].mean() * 100, 2)
    }

# 保存为 JSON 文件
with open("industry_avg.json", "w") as f:
    json.dump(industry_avg, f, indent=2, ensure_ascii=False)

print("✅ Saved industry_avg.json with average indicators.")
