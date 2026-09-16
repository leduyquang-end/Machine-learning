"""
data_generator.py
=================
Tai du lieu tu Kaggle: House Prices (Seattle, WA)
Nguon: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""

import pandas as pd
import numpy as np
import os


def load_kaggle_data(csv_path="house_price.csv"):
    """Tai du lieu tu file CSV da download tu Kaggle"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Khong tim thay {csv_path}.\n"
            "Hay download tu: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction"
        )

    df = pd.read_csv(csv_path)
    print(f"[OK] Tai du lieu: {df.shape[0]} mau, {df.shape[1]} cot")
    return df


def prepare_features(df, feature_cols=None, target_col='price'):
    """
    Chuan bi features va target
    Mac dinh chi su dung cot 'sqft_living' (dien tich song)
    """
    if feature_cols is None:
        feature_cols = ['sqft_living']

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"[OK] Features: {feature_cols} | Target: {target_col}")
    return X, y


if __name__ == "__main__":
    df = load_kaggle_data()
    print(f"\n5 dong dau tien:")
    print(df[['sqft_living', 'price']].head())
    X, y = prepare_features(df)
