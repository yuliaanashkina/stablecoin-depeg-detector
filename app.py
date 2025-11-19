import streamlit as st
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("Stablecoin Risk Dashboard")

coins = ["usd-coin", "tether", "dai"]

with open("anomaly_classifier.pkl", "rb") as f:
    model = pickle.load(f)

coin = st.sidebar.selectbox("Select stablecoin", coins)
days = st.sidebar.slider("Days of history", min_value=30, max_value=180, value=90)

url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
resp = requests.get(url).json()

prices = resp["prices"]
volumes = resp["total_volumes"]

df = pd.DataFrame(prices, columns=["timestamp", "price"])
df["volume"] = [v[1] for v in volumes]
df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
df["deviation"] = df["price"] - 1
df["abs_dev"] = df["deviation"].abs()
df["rolling_std"] = df["price"].rolling(7).std()
df["rolling_vol"] = df["volume"].rolling(7).mean()
df["liquidity_risk"] = 1 / (df["rolling_vol"] + 1e-6)
df["combined_risk"] = df["abs_dev"] * df["liquidity_risk"]

# Price plot
st.subheader("Price History")
fig, ax = plt.subplots()
ax.plot(df["date"], df["price"], label=coin)
ax.axhline(1.0, linestyle="--", color="black")
st.pyplot(fig)

# Anomaly prediction
st.subheader("Anomaly Prediction (Latest Data Point)")
latest = df.iloc[-1]
feat = latest[["price","deviation","abs_dev","rolling_std","rolling_vol","liquidity_risk","combined_risk"]].values.reshape(1,-1)

pred = model.predict(feat)[0]
prob = model.predict_proba(feat)[0][1]

st.write("Probability of anomaly:", f"{prob:.2%}")
st.write("Anomaly detected:", "✔️" if pred==1 else "❌")

# Forecasting
st.subheader("7-Day Forecast (Prophet)")
fc = df[["date","price"]].rename(columns={"date":"ds","price":"y"})
m = Prophet()
m.fit(fc)
future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
fig2 = m.plot(forecast)
st.pyplot(fig2)
