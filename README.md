# Stablecoin Stability & Anomaly Detection

A data science and machine learning project analyzing the stability of major stablecoins (**USDC**, **USDT**, **DAI**) using real price and volume data.  
Includes peg tracking, depeg detection, liquidity-adjusted risk scoring, anomaly classification, forecasting, and an interactive dashboard.

---

## Run the Dashboard

streamlit run app.py

### Dashboard includes:
- Price chart  
- Anomaly probability (real-time)  
- 7-day forecasting (Prophet)  

---

## How to Run

### Install dependencies:
pip install -r requirements.txt

### Run analysis:
python stablecoin_depeg_detector.py

### Launch dashboard:
streamlit run app.py

---

## Project Summary

This project demonstrates:

- Peg deviation analysis  
- Depeg event detection  
- Liquidity-adjusted risk scoring  
- Random Forest anomaly classification  
- ROC curve + AUC evaluation  
- SHAP model explainability  
- Prophet forecasting  
- Streamlit dashboard for real-time monitoring  


