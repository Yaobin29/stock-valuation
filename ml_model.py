import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 股票列表
stock_df = pd.read_csv("stock_map.csv")
tickers = stock_df["code"].unique().tolist()

# 财务特征
FEATURE_KEYS = [
    "trailingPE", "priceToBook", "returnOnEquity", "trailingEps",
    "revenueGrowth", "grossMargins", "marketCap", "freeCashflow"
]

# 初始化情绪分析器
analyzer = SentimentIntensityAnalyzer()

def fetch_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return 0
        scores = [analyzer.polarity_scores(item["title"])["compound"] for item in news if "title" in item]
        return np.mean(scores) if scores else 0
    except Exception as e:
        print(f"⚠️ Sentiment fetch failed for {ticker}: {e}")
        return 0

def fetch_features(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        features = {
            "code": ticker,
            "price": info.get("currentPrice", np.nan),
            "sentiment": fetch_sentiment(ticker)
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

# 丢弃缺失值
df = df.dropna()
print(f"✅ After dropna: {df.shape}")

# 模型训练
features = FEATURE_KEYS + ["sentiment"]
X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 评估指标
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📉 MAE: ${mae:.2f}")

# 保存模型
joblib.dump(model, "valuation_model.pkl")
print("✅ Saved model as valuation_model.pkl")