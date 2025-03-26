import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from joblib import dump
from datetime import datetime

# 输出日志
print(f"\U0001F552 Starting model update at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 读取股票列表
stock_df = pd.read_csv("stock_map.csv")
codes = stock_df["code"].tolist()

features = [
    "trailingPE", "priceToBook", "returnOnEquity", "trailingEps",
    "revenueGrowth", "grossMargins", "marketCap", "freeCashflow"
]
data = []

print(f"\U0001F4C8 Fetching data for {len(codes)} companies...")

for code in codes:
    try:
        info = yf.Ticker(code).info
        row = [info.get(feat) for feat in features]
        price = info.get("currentPrice")
        if None not in row and price:
            data.append(row + [price])
    except Exception as e:
        print(f"⚠️ Failed to fetch {code}: {e}")

df = pd.DataFrame(data, columns=features + ["target"])
print(f"✅ Raw data shape: {df.shape}")

# 清洗
X = df[features]
y = df["target"]

# 分组训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 评估效果
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"📉 MAE: ${mae:.2f}")

# 保存模型
dump(model, "valuation_model.pkl")
print("✅ Saved model as valuation_model.pkl")
