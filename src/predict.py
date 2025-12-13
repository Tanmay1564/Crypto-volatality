import joblib
import pandas as pd

def load_model(model_path):
    return joblib.load(model_path)

def prepare_features(df):
    # assumes features already present (or you may call add_features to create them)
    features = ['vol_7d','vol_30d','ma_7','ma_30','liq_ratio','price_range','return']
    X = df[features].fillna(0)
    return X

def predict_from_df(df, model_path):
    model = load_model(model_path)
    X = prepare_features(df)
    preds = model.predict(X)
    return preds
