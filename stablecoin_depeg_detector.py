import requests
import pandas as pd
import matplotlib.pyplot as plt

# Fetch price and volume data
coins = ["usd-coin", "tether", "dai"]
data = {}

for coin in coins:
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=90"
    resp = requests.get(url).json()

    prices = resp["prices"]
    volumes = resp["total_volumes"]

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["volume"] = [v[1] for v in volumes]
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["coin"] = coin
    data[coin] = df

df = pd.concat(data.values())

# Peg deviation and volatility
df["peg"] = 1.0
df["deviation"] = df["price"] - 1
df["abs_dev"] = df["deviation"].abs()

df["rolling_std"] = df.groupby("coin")["price"].transform(lambda x: x.rolling(7).std())
df["rolling_vol"] = df.groupby("coin")["volume"].transform(lambda x: x.rolling(7).mean())

df["z_score"] = df["deviation"] / df["rolling_std"]

# Depeg event detection
DEPEG_THRESHOLD = -0.003
events = []

for coin in coins:
    coin_df = df[df["coin"] == coin].sort_values("date")
    coin_df["is_depeg"] = coin_df["deviation"] < DEPEG_THRESHOLD

    in_event = False
    start_date = None

    for idx, row in coin_df.iterrows():
        if row["is_depeg"] and not in_event:
            in_event = True
            start_date = row["date"]

        elif not row["is_depeg"] and in_event:
            end_date = row["date"]
            duration = (end_date - start_date).days

            period = coin_df[(coin_df["date"] >= start_date) & (coin_df["date"] <= end_date)]
            min_price = period["price"].min()
            max_dev = period["abs_dev"].max()

            events.append([coin, start_date.date(), end_date.date(), duration, min_price, max_dev])
            in_event = False

events_df = pd.DataFrame(events, columns=[
    "coin", "start_date", "end_date", "duration_days", "min_price", "max_deviation"
])

print("\nDEPEG EVENTS DETECTED\n")
print(events_df if not events_df.empty else "No depeg events in the last 90 days.")

# Liquidity-adjusted risk score
df["liquidity_risk"] = 1 / (df["rolling_vol"] + 1e-6)
df["combined_risk"] = df["abs_dev"] * df["liquidity_risk"]

risk_scores = df.groupby("coin")[["abs_dev", "combined_risk"]].mean()
risk_scores.columns = ["peg_risk", "liquidity_adjusted_risk"]

print("\nSTABILITY AND LIQUIDITY RISK SCORES\n")
print(risk_scores.sort_values("liquidity_adjusted_risk", ascending=False))

# Plot price data
fig, ax1 = plt.subplots(figsize=(12, 6))

for coin in coins:
    subset = df[df["coin"] == coin]
    ax1.plot(subset["date"], subset["price"], label=f"{coin} price")

ax1.axhline(1.0, color="black", linestyle="--")
ax1.set_ylabel("Price (USD)")
ax1.set_title("Stablecoin Peg Stability (90 Days)")

plt.legend()
plt.tight_layout()
plt.show()
