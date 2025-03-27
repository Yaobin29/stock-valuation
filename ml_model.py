import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# 股票列表
stock_df = pd.read_csv("stock_map.csv")
tickers = stock_df["code"].unique().tolist()

# 财务特征
FEATURE_KEYS = [
    "trailingPE", "priceToBook", "returnOnEquity", "trailingEps",
    "revenueGrowth", "grossMargins", "marketCap", "freeCashflow"
]

# 获取财务特征
def fetch_features(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        features = {
            "code": ticker,
            "price": info.get("currentPrice", np.nan)
        }
        for key in FEATURE_KEYS:
            features[key] = info.get(key, np.nan)
        return features
    except Exception as e:
        print(f"⚠️ Failed to fetch {ticker}: {e}")
        return None

# 抓取全部数据
all_data = []
print(f"📊 Fetching data for {len(tickers)} companies...")
for code in tickers:
    row = fetch_features(code)
    if row:
        all_data.append(row)

# 构建 DataFrame
df = pd.DataFrame(all_data)
print(f"✅ Raw data shape: {df.shape}")

# 清洗
df = df.dropna()
print(f"✅ After dropna: {df.shape}")

# 模型训练
X = df[FEATURE_KEYS]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 回归预测
df["predicted_price"] = model.predict(X)

# 估值判断标签（±10% 容忍区间）
def judge(row, threshold=0.1):
    diff_ratio = (row["price"] - row["predicted_price"]) / row["predicted_price"]
    if abs(diff_ratio) <= threshold:
        return "合理"
    elif diff_ratio > threshold:
        return "高估"
    else:
        return "低估"

df["valuation_label"] = df.apply(judge, axis=1)

# 模型评估
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📉 MAE: ${mae:.2f}")

# 保存模型与数据
joblib.dump(model, "valuation_model.pkl")
print("✅ Saved model as valuation_model.pkl")

df.to_csv("valuation_labels.csv", index=False)
print("✅ Saved labeled data to valuation_labels.csv")
