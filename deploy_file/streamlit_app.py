import streamlit as st
import pandas as pd
import joblib
from src.data_preprocessing import load_and_clean
from src.features import add_features

st.title("Crypto Volatility Prediction - Demo")

uploaded = st.file_uploader("Upload CSV with columns: date,symbol,open,high,low,close,volume,market_cap", type=["csv"])
model = None
if st.button("Load baseline model"):
    try:
        model = joblib.load("models/rf_baseline_model.pkl")
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['date'] if 'date' in pd.read_csv(uploaded, nrows=0).columns else None)
    st.write("Preview:")
    st.dataframe(df.head())

    if st.button("Compute features & predict (demo)"):
        df_clean = load_and_clean(uploaded)
        df_feat = add_features(df_clean)
        df_feat = df_feat.dropna(subset=['vol_7d'])
        if model is None:
            st.warning("Model not loaded — please click 'Load baseline model' first.")
        else:
            features = ['vol_7d','vol_30d','ma_7','ma_30','liq_ratio','price_range','return']
            X = df_feat[features].fillna(0)
            preds = model.predict(X)
            df_feat['pred_vol_next_1d'] = preds
            st.write(df_feat[['date','symbol','close','vol_7d','pred_vol_next_1d']].head(50))
