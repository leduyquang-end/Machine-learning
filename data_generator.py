"""
data_generator.py - Tai du lieu Kaggle House Prices
Nguon: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""

import pandas as pd
import os


def load_data(path="house_price.csv"):
    """Tai du lieu tu CSV (da download tu Kaggle)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Khong tim thay {path}")
    df = pd.read_csv(path)
    print(f"[OK] {df.shape[0]} mau, {df.shape[1]} cot")
    return df


def get_X_y(df):
    """Tach features (X) va target (y)"""
    X = df[['sqft_living']].values  # Dien tich song
    y = df['price'].values          # Gia nha (USD)
    return X, y
