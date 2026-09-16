"""
overfit_example.py
==================
PHAN 1: TAO MO HINH BI OVERFITTING

OVERFITTING LA GI?
- Mo hinh hoc QUA TOT tren du lieu training
- Nhung hoat dong RAT TE tren du lieu moi (test)
- Giong nhu hoc sinh "hoc vet" bai tap cu, khong hieu noi dung

DAU HIEU NHAN BIET:
- Train R2 CAO (gan 1.0)
- Test R2 THAP (co the am)
- Chenh lech giua Train va Test R2 RAT LON

TAI SAO OVERFIT XAY RA?
- Mo hinh qua PHUC TAP so voi du lieu thuc
- Vi du: Duong gia nha la DUONG THANG (degree=1)
          nhung ta dung polynomial degree=15 (15 o so)
          -> Mo hinh "cong" vao moi diem du lieu
          -> Khong duoc tren du lieu moi

NGUON DU LIEU:
- Kaggle: House Prices (Seattle, WA)
- https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend khong interactive (khong mo cua so)
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# HAM TAI DU LIEU
# ============================================================

def load_data():
    """Tai du lieu tu file CSV da download tu Kaggle"""
    path = "house_price.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay {path}\n"
            "Hay download: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction"
        )
    df = pd.read_csv(path)
    print(f"[OK] {df.shape[0]} mau, {df.shape[1]} cot")
    return df


def get_X_y(df):
    """Tach features (X) va target (y)"""
    X = df[['sqft_living']].values  # Dien tich song
    y = df['price'].values          # Gia nha (USD)
    return X, y


# ============================================================
# CHAY CHINH
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PHAN 1: TAO MO HINH OVERFITTING")
    print("=" * 60)

    # Tai du lieu
    df = load_data()
    X, y = get_X_y(df)

    # Chia train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Tao mo hinh voi degree cu the
    def fit(degree):
        m = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )
        m.fit(X_train, y_train)
        return m

    # Mo hinh 1: OVERFIT (degree=15 - qua phuc tap)
    # Mo hinh 2: DON GIAN (degree=1 - hop ly)
    model_of, model_ok = fit(15), fit(1)

    # Ve so sanh
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    for ax, name, model, color in [
        (axes[0], "OVERFIT (degree=15)", model_of, 'red'),
        (axes[1], "DON GIAN (degree=1)", model_ok, 'green')
    ]:
        tr = r2_score(y_train, model.predict(X_train))
        te = r2_score(y_test, model.predict(X_test))
        ax.scatter(X_train, y_train, alpha=0.3, s=20, c='steelblue', label='Train')
        ax.scatter(X_test, y_test, alpha=0.3, s=20, c='orange', label='Test')
        ax.plot(X_range, model.predict(X_range), color=color, linewidth=2)
        ax.set_title(f'{name}\nTrain={tr:.4f} | Test={te:.4f}\nChenh lech={abs(tr-te):.4f}',
                     color=color)
        ax.set_xlabel('sqft_living'); ax.set_ylabel('price (USD)')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hinh_anh/overfit_comparison.png', dpi=100)
    print("Da luu: hinh_anh/overfit_comparison.png")

    # In ket qua
    tr_of = r2_score(y_train, model_of.predict(X_train))
    te_of = r2_score(y_test, model_of.predict(X_test))
    tr_ok = r2_score(y_train, model_ok.predict(X_train))
    te_ok = r2_score(y_test, model_ok.predict(X_test))

    print(f"\n{'':20} {'Overfit':>12} {'Simple':>12}")
    print(f"{'Train R2':20} {tr_of:>12.4f} {tr_ok:>12.4f}")
    print(f"{'Test R2':20} {te_of:>12.4f} {te_ok:>12.4f}")
    print(f"\n[!] OVERFIT: Train tot, Test TE -> Hoc vet!")
    print(f"[OK] SIMPLE: Train va Test gan nhau -> Tot!")
