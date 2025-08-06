# 📈 S&P 500 Price Predictor using Random Forest & yFinance

A machine learning project that predicts the **next-day S&P 500 closing price** using historical stock market data retrieved via the `yfinance` API. The model is built and trained using a **Random Forest Regressor**.

---

## 🔎 Overview

This notebook fetches real-time historical data of the S&P 500 index (`^GSPC`) using `yfinance`, performs feature engineering on it, and trains a Random Forest model to forecast future closing prices.

---

## 📚 Workflow Summary

1. **Data Collection**
   - Uses [`yfinance`](https://github.com/ranaroussi/yfinance) to download historical OHLCV data:
     ```python
     import yfinance as yf
     data = yf.download("^GSPC", start="2010-01-01", end="2024-01-01")
     ```

2. **Feature Engineering**
   - Creates lag features (previous 5 days' prices)
   - Computes rolling averages (MA7, MA21)
   - Adds percentage daily returns

3. **Target Definition**
   - Predicts `Close` price of the next day using:
     ```python
     df['target'] = df['Close'].shift(-1)
     ```

4. **Modeling**
   - Splits into train/test sets (no shuffle)
   - Trains a `RandomForestRegressor`
   - Evaluates using **MSE** and **visual comparison**

---

## 🧰 Tech Stack

- **Language:** Python
- **Data:** `yfinance`
- **ML Library:** `scikit-learn`
- **Notebook Environment:** Google Colab
- **Visualization:** Matplotlib

---

## 📊 Sample Output

- Predicted vs actual closing prices
- MSE score
- Feature importance bar chart

---

## 📈 Code Snippets

### Load Data:
```python
import yfinance as yf
df = yf.download("^GSPC", start="2010-01-01", end="2024-01-01")
