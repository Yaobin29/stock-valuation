import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# 加载股票映射文件
stock_df = pd.read_csv("stock_map.csv")
tickers = stock_df["code"].unique().tolist()

# 定义需要提取的特征
FEATURE_KEYS = [
    "trailingPE", "priceToBook", "returnOnEquity", "trailingEps",
    "revenueGrowth", "grossMargins", "marketCap", "freeCashflow"
]

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

# 抓取全部公司数据
all_data = []
print(f"📊 Fetching data for {len(tickers)} companies...")
for code in tickers:
    row = fetch_features(code)
    if row:
        all_data.append(row)

# 构建数据集
df = pd.DataFrame(all_data)
print(f"✅ Raw data shape: {df.shape}")

# 丢弃缺失较多的数据行
df = df.dropna()
print(f"✅ After dropna: {df.shape}")

# 特征列与目标列
features = FEATURE_KEYS
X = df[features]
y = df["price"]

# 训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📉 MAE: ${mae:.2f}")

# 保存模型
joblib.dump(model, "valuation_model.pkl")
print("✅ Saved model as valuation_model.pkl")
