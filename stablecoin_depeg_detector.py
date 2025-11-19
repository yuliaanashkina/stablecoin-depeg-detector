import requests
import pandas as pd
import matplotlib.pyplot as plt

# 1. Fetch data from CoinGecko
coins = ["usd-coin", "tether", "dai"]
data = {}

for coin in coins:
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=90"
    resp = requests.get(url).json()
    prices = resp["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["coin"] = coin
    data[coin] = df

df = pd.concat(data.values())

# 2. Calculate deviation metrics
df["peg"] = 1.00
df["deviation"] = df["price"] - 1
df["abs_dev"] = df["deviation"].abs()

# 3. Rolling anomaly flags
df["rolling_std"] = df.groupby("coin")["price"].transform(lambda x: x.rolling(7).std())
df["z_score"] = df["deviation"] / df["rolling_std"]
df["anomaly_flag"] = (df["z_score"].abs() > 3)

# 4. Plot
plt.figure(figsize=(10, 6))
for coin in coins:
    subset = df[df["coin"] == coin]
    plt.plot(subset["date"], subset["price"], label=coin)
plt.axhline(1.0, color="black", linestyle="--")
plt.title("Stablecoin Peg Stability (90 Days)")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Simple risk score
risk_scores = df.groupby("coin")["abs_dev"].mean().sort_values(ascending=False)
print("Stablecoin Risk Scores (Mean Deviation):")
print(risk_scores)
