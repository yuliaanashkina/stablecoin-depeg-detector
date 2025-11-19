import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import shap
import pickle

coins = ["usd-coin", "tether", "dai"]
data = {}

# Fetch data
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

# Basic features
df["peg"] = 1.0
df["deviation"] = df["price"] - 1
df["abs_dev"] = df["deviation"].abs()
df["rolling_std"] = df.groupby("coin")["price"].transform(lambda x: x.rolling(7).std())
df["rolling_vol"] = df.groupby("coin")["volume"].transform(lambda x: x.rolling(7).mean())
df["z_score"] = df["deviation"] / df["rolling_std"]

# Depeg events
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

print("\nDEPEG EVENTS\n")
print(events_df if not events_df.empty else "No depeg events detected.")

# Liquidity-adjusted stability
df["liquidity_risk"] = 1 / (df["rolling_vol"] + 1e-6)
df["combined_risk"] = df["abs_dev"] * df["liquidity_risk"]

risk_scores = df.groupby("coin")[["abs_dev", "combined_risk"]].mean()
risk_scores.columns = ["peg_risk", "liquidity_adjusted_risk"]

print("\nRISK SCORES\n")
print(risk_scores.sort_values("liquidity_adjusted_risk", ascending=False))

# Classifier
df["anomaly"] = (df["z_score"].abs() > 3).astype(int)
df = df.dropna()

features = df[[
    "price",
    "deviation",
    "abs_dev",
    "rolling_std",
    "rolling_vol",
    "liquidity_risk",
    "combined_risk"
]]

labels = df["anomaly"]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

print("\nCLASSIFIER PERFORMANCE\n")
print("Accuracy:", round(acc, 4))
print("\nConfusion Matrix:\n", cm)

# ROC
fpr, tpr, thresholds = roc_curve(y_test, prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label='AUC=%0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Anomaly Classifier")
plt.legend()
plt.tight_layout()
plt.show()

# SHAP explainability
explainer = shap.TreeExplainer(model)
raw_shap = explainer.shap_values(X_test)

# Handle binary classifier output shape
if isinstance(raw_shap, list):
    shap_values = raw_shap[1]
else:
    shap_values = raw_shap

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300)
plt.show()

print("Saved SHAP feature importance plot.")


# Save model
with open("anomaly_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model to anomaly_classifier.pkl")

# Plot price lines
fig, ax = plt.subplots(figsize=(12,6))
for coin in coins:
    subset = df[df["coin"] == coin]
    ax.plot(subset["date"], subset["price"], label=coin)
ax.axhline(1.0, color="black", linestyle="--")
ax.set_title("Stablecoin Prices")
ax.set_ylabel("Price (USD)")
plt.legend()
plt.show()
