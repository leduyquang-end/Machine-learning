"""
data_generator.py
==================
Tai du lieu tu Kaggle dataset: House Prices (Seattle, WA)
Nguon: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
       Hoac: https://github.com/SarahShafqat/Kaggle-Datasets

Du lieu gom 4601 mau nha o Seattle, WA
Cac features: bedrooms, bathrooms, sqft_living, sqft_lot, floors, etc.
Target: price (gia nha USD)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os


def load_kaggle_data(csv_path="house_price.csv"):
    """
    Tai du lieu tu file CSV da download tu Kaggle

    Parameters:
    -----------
    csv_path : str
        Duong dan den file CSV

    Returns:
    --------
    df : pd.DataFrame - Du lieu day du
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Khong tim thay {csv_path}.\n"
            "Hay download tu Kaggle:\n"
            "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction"
        )

    df = pd.read_csv(csv_path)
    print(f"Tai du lieu thanh cong: {df.shape[0]} mau, {df.shape[1]} cot")
    return df


def prepare_features(df, feature_cols=None, target_col='price'):
    """
    Chuan bi features va target tu DataFrame

    Parameters:
    -----------
    df : pd.DataFrame
        Du lieu day du
    feature_cols : list
        Danh sach cot features (mac dinh: sqft_living)
    target_col : str
        Cot target (mac dinh: price)

    Returns:
    --------
    X : np.array - Features
    y : np.array - Target
    """
    if feature_cols is None:
        # Mac dinh chi su dung sqft_living (dien tich song)
        feature_cols = ['sqft_living']

    X = df[feature_cols].values
    y = df[target_col].values

    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Kich thuoc: X={X.shape}, y={y.shape}")

    return X, y


def visualize_data(X, y, feature_name='sqft_living', target_name='price'):
    """Visualize du lieu duoi dang scatter plot"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.3, c='steelblue', edgecolors='white', s=40)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel(f'{target_name} (USD)', fontsize=12)
    plt.title(f'Du lieu Kaggle: {feature_name} vs {target_name}\n'
              f'Nguon: Kaggle House Sales Prediction', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data_preview.png', dpi=100)
    print("Da luu: data_preview.png")


def get_data_summary(df):
    """In thong tin tong quan ve du lieu"""
    print("\n" + "=" * 60)
    print("THONG TIN DU LIEU KAGGLE")
    print("=" * 60)
    print(f"Nguon: Kaggle - House Sales Prediction (Seattle, WA)")
    print(f"So mau: {df.shape[0]}")
    print(f"So features: {df.shape[1]}")
    print(f"\nCac cot: {list(df.columns)}")
    print(f"\nThong ke mo ta:")
    print(df.describe().round(2))


if __name__ == "__main__":
    # Tai du lieu
    df = load_kaggle_data()

    # Hien thi thong tin
    get_data_summary(df)

    # Chuan bi features (chi su dung sqft_living)
    X, y = prepare_features(df, feature_cols=['sqft_living'])

    # Visualize
    visualize_data(X, y)
