"""
data_generator.py
=================
TAI DU LIEU TU KAGGLE

Nguon: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
Bo du lieu: House Prices (Seattle, WA)
- So mau: 4600 can nha
- Features: dien tich, so phong ngu, so phong tam, tang, nam xay dung, ...
- Target: gia nha (USD)

Muc dich: Du doan gia nha dua tren dien tich song (sqft_living)
"""

import pandas as pd
import os


def load_data(path="house_price.csv"):
    """
    Tai du lieu tu file CSV da download tu Kaggle

    Parameters:
    -----------
    path : str
        Duong dan den file CSV (mac dinh: house_price.csv)

    Returns:
    --------
    df : pd.DataFrame
        Du lieu day du
    """
    # Kiem tra file co ton tai khong
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay file {path}\n"
            "Hay download tu Kaggle:\n"
            "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction"
        )

    # Doc file CSV
    df = pd.read_csv(path)
    print(f"[OK] Tai du lieu thanh cong: {df.shape[0]} mau, {df.shape[1]} cot")
    return df


def get_X_y(df):
    """
    Tach du lieu thanh features (X) va target (y)

    Parameters:
    -----------
    df : pd.DataFrame
        Du lieu day du

    Returns:
    --------
    X : np.array
        Ma tran features (chi lay cot sqft_living - dien tich song)
    y : np.array
        Mang target (cot price - gia nha USD)
    """
    # Chi su dung 1 feature: dien tich song (sqft_living)
    X = df[['sqft_living']].values  # Shape: (4600, 1)
    y = df['price'].values          # Shape: (4600,)
    return X, y


# ============================================================
# CHAY THU
# ============================================================
if __name__ == "__main__":
    # Tai du lieu
    df = load_data()

    # Hien thi 5 dong dau tien
    print("\n5 dong dau tien (sqft_living vs price):")
    print(df[['sqft_living', 'price']].head())

    # Tach X, y
    X, y = get_X_y(df)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
