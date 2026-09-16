"""
overfit_example.py
==================
Vi du OVERFITTING: Mo hinh hoc qua tot tren train nhung kem tren test
Du lieu: Kaggle House Prices (Seattle, WA)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from data_generator import load_kaggle_data, prepare_features


# ============================================================
# OVERFITTING LA GI?
# ============================================================
# Mo hinh "hoc vet" - nho het du lieu training nhung khong hieu
# ban chat cua du lieu. Dau hieu: Train R2 cao, Test R2 thap.
#
# Vi du: Duong gia nha la DUONG THANG don gian,
# nhung mo hinh degree=15 co 15 o so -> no "cong" vao moi diem
# -> qua phuc tap, khong duoc tren du lieu moi.
# ============================================================


def create_model(X_train, y_train, degree=1):
    """
    Tao mo hinh Polynomial Regression

    Parameters:
    -----------
    degree : int
        - degree=1: Duong thang (don gian, it overfit)
        - degree=15: Duong cong phuc tap (de overfit)
    """
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    # 1. Tai du lieu Kaggle
    print("=== OVERFITTING EXAMPLE ===")
    df = load_kaggle_data()
    X, y = prepare_features(df, feature_cols=['sqft_living'])

    # 2. Chia train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)} mau | Test: {len(X_test)} mau\n")

    # 3. Tao 2 mo hinh:
    #    - degree=15: Qua phuc tap -> OVERFIT
    #    - degree=1:  Don gian -> Tot
    model_overfit = create_model(X_train, y_train, degree=15)
    model_simple = create_model(X_train, y_train, degree=1)

    # 4. Danh gia
    X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    train_r2_of = r2_score(y_train, model_overfit.predict(X_train))
    test_r2_of = r2_score(y_test, model_overfit.predict(X_test))

    train_r2_s = r2_score(y_train, model_simple.predict(X_train))
    test_r2_s = r2_score(y_test, model_simple.predict(X_test))

    # 5. Ve bieu do so sanh
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, name, model, tr, te, color in [
        (axes[0], "OVERFIT (degree=15)", model_overfit, train_r2_of, test_r2_of, 'red'),
        (axes[1], "DON GIAN (degree=1)", model_simple, train_r2_s, test_r2_s, 'green')
    ]:
        ax.scatter(X_train, y_train, alpha=0.3, s=30, c='steelblue', label='Train')
        ax.scatter(X_test, y_test, alpha=0.3, s=30, c='orange', label='Test')
        ax.plot(X_range, model.predict(X_range), color=color, linewidth=2, label=name)
        ax.set_title(f'{name}\nTrain R2={tr:.4f} | Test R2={te:.4f}\nChenh lech={abs(tr-te):.4f}',
                     fontsize=11, color=color)
        ax.set_xlabel('Dien tich (sqft)')
        ax.set_ylabel('Gia nha (USD)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('overfit_comparison.png', dpi=100)
    print("Da luu: overfit_comparison.png")

    # 6. In ket qua
    print(f"\n{'='*50}")
    print(f"{'':20} {'Overfit':>12} {'Simple':>12}")
    print(f"{'Train R2':20} {train_r2_of:>12.4f} {train_r2_s:>12.4f}")
    print(f"{'Test R2':20} {test_r2_of:>12.4f} {test_r2_s:>12.4f}")
    print(f"{'Chenh lech':20} {abs(train_r2_of-test_r2_of):>12.4f} {abs(train_r2_s-test_r2_s):>12.4f}")
    print(f"\n[!] Overfit: Train tot nhung Test kem -> Hoc vet!")
    print(f"[OK] Simple: Train va Test gan nhau -> Hieu du lieu!")
