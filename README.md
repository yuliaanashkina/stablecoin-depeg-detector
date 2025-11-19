# Stablecoin Stability & Anomaly Detection

This project analyzes the stability of major stablecoins (USDC, USDT, DAI) using real market data, risk metrics, and machine learning. It includes peg-tracking, depeg detection, liquidity analysis, anomaly classification, forecasting, and an interactive dashboard.

What This Project Does
1. Peg Stability & Depeg Detection

Pulls 90 days of price + volume data from CoinGecko

Computes deviation from the $1 peg, rolling volatility, and liquidity indicators

Automatically detects continuous depeg events with start/end dates and severity

2. Liquidity-Adjusted Risk Scoring

Calculates a risk metric combining deviation and low-liquidity conditions

Ranks stablecoins by relative stability

3. Machine Learning Anomaly Classifier

Trains a Random Forest to predict anomalies from volatility, volume, and deviation features

Includes accuracy, confusion matrix, ROC/AUC, and SHAP interpretability

Saves the trained model (anomaly_classifier.pkl) for downstream use

4. Streamlit Dashboard

Run:

streamlit run app.py


Dashboard includes:

Price chart

Real-time anomaly probability

7-day forecasting with Prophet

📂 Project Structure
stablecoin-depeg-detector/
│
├── stablecoin_depeg_detector.py   # Full analysis + ML + SHAP + ROC
├── app.py                          # Interactive dashboard
├── anomaly_classifier.pkl          # Saved model for inference
├── shap_feature_importance.png     # SHAP interpretation
└── requirements.txt

▶️ How to Run

Install dependencies:

pip install -r requirements.txt


Run analysis:

python stablecoin_depeg_detector.py


Launch dashboard:

streamlit run app.py

