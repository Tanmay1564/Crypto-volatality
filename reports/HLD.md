# High Level Design (HLD)

Components:
- Data ingestion: CSV loader (daily OHLC, volume, market cap)
- Preprocessing: cleaning & type conversion
- Feature engineering: returns, rolling volatility, moving averages, liquidity ratio
- Model training: baseline RandomForest; optional XGBoost/LSTM
- Evaluation: RMSE, MAE, R2
- Deployment: Streamlit demo for local testing
