import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_preprocessing import load_and_clean
from features import add_features

def train_pipeline(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    df = load_and_clean(data_path)
    df = add_features(df)

    # Prepare model dataset
    df_model = df.dropna(subset=['volatility_next_1d'])
    features = ['vol_7d','vol_30d','ma_7','ma_30','liq_ratio','price_range','return']
    X = df_model[features].fillna(0)
    y = df_model['volatility_next_1d'].fillna(0)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # baseline model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        'rmse': mean_squared_error(y_test, preds, squared=False),
        'mae': mean_absolute_error(y_test, preds),
        'r2': r2_score(y_test, preds)
    }

    # save model
    model_path = os.path.join(models_dir, 'rf_baseline_model.pkl')
    joblib.dump(model, model_path)

    # save sample cleaned file
    sample_path = os.path.join(out_dir, 'data', 'sample_cleaned.csv')
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    df_model.head(200).to_csv(sample_path, index=False)

    print("Training complete. Metrics:", metrics)
    print("Model saved to:", model_path)
    return metrics, model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args()
    train_pipeline(args.data_path, args.out_dir)
