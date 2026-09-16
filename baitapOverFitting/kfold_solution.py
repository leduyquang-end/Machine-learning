"""
kfold_solution.py
=================
PHAN 2: KHAC PHUC OVERFITTING BANG K-FOLD CROSS VALIDATION

TẠI SAO K-FOLD GIUP KHAC PHUC OVERFITTING?
- K-Fold chia du lieu thanh K phan (fold)
- Moi lan: K-1 phan de TRAIN, 1 phan de VALIDATE
- Lap lai K lan, ket qua = trung binh

LOI ICH:
+ Phat hien overfitting SOM (truoc khi dung test)
+ Danh gia mo hinh ON DINH hon
+ Chon hyperparameter tot nhat (degree, alpha, ...)

NGUON DU LIEU:
- Kaggle: House Prices (Seattle, WA)
- https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    KFold, GridSearchCV, learning_curve, train_test_split
)

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
    print("PHAN 2: K-FOLD CROSS VALIDATION")
    print("=" * 60)

    df = load_data()
    X, y = get_X_y(df)

    # --- PHAN A: Ve K-Fold de hieu ---
    # 30 mau nho, chia 5 fold, moi fold ~6 mau
    # Xanh = Train, Do = Validate
    print("\n[PHAN A] Ve K-Fold...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fig, axes = plt.subplots(5, 1, figsize=(12, 5))

    for i, (train_idx, val_idx) in enumerate(kfold.split(np.arange(30))):
        colors = ['lightgray'] * 30
        for j in train_idx: colors[j] = 'steelblue'  # Train = xanh
        for j in val_idx:   colors[j] = 'red'         # Validate = do
        axes[i].scatter(range(30), [0]*30, c=colors, s=200, edgecolors='black')
        axes[i].set_yticks([])
        axes[i].set_xlabel(f'Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}')

    plt.suptitle('K-Fold: Xanh=Train | Do=Validation', fontweight='bold')
    plt.tight_layout()
    plt.savefig('hinh_anh/kfold_visualization.png', dpi=100)
    print("[OK] Da luu: hinh_anh/kfold_visualization.png")

    # --- PHAN B: Tim degree tot nhat voi GridSearchCV ---
    # Thu degree 1-10, moi degree danh gia bang K-Fold (K=5)
    print("\n[PHAN B] Tim degree tot nhat...")
    model = make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
    grid = GridSearchCV(
        model,
        {'polynomialfeatures__degree': list(range(1, 11))},
        cv=5, scoring='r2', return_train_score=True
    )
    grid.fit(X, y)
    res = grid.cv_results_

    print(f"\n{'Deg':>4} {'Train':>8} {'Val':>8} {'Lech':>8}")
    for d, tr, v in zip(range(1, 11), res['mean_train_score'], res['mean_test_score']):
        star = " <-- BEST" if v == max(res['mean_test_score']) else ""
        print(f"{d:>4} {tr:>8.4f} {v:>8.4f} {tr-v:>8.4f}{star}")

    best_deg = grid.best_params_['polynomialfeatures__degree']

    # Ve bieu do
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), res['mean_train_score'], 'o-', label='Train')
    plt.plot(range(1, 11), res['mean_test_score'], 'o-', label='Val', color='red')
    plt.axvspan(5, 10, alpha=0.1, color='red', label='Overfitting')
    plt.xlabel('Degree'); plt.ylabel('R2')
    plt.title(f'GridSearchCV + K-Fold: Best Degree={best_deg}')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hinh_anh/degree_comparison.png', dpi=100)
    print(f"\n[OK] Da luu: hinh_anh/degree_comparison.png")

    # --- PHAN C: Learning Curve ---
    # Xem mo hinh tot hon khi co nhieu du lieu hon hay khong
    print("\n[PHAN C] Learning Curve...")
    best_model = grid.best_estimator_
    sizes, tr_sc, val_sc = learning_curve(
        best_model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10), cv=5
    )

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, tr_sc.mean(axis=1), 'o-', label='Train')
    plt.plot(sizes, val_sc.mean(axis=1), 'o-', label='Val', color='red')
    plt.xlabel('Kich thuoc du lieu'); plt.ylabel('R2')
    plt.title(f'Learning Curve (Degree={best_deg})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hinh_anh/learning_curve.png', dpi=100)
    print("[OK] Da luu: hinh_anh/learning_curve.png")

    # --- PHAN D: So sanh 3 mo hinh ---
    # 1. Overfit (deg=15): qua phuc tap -> TE
    # 2. CV (deg=best_deg): chon boi GridSearchCV -> TOT
    # 3. Ridge (deg=10): regularization -> giam overfit
    print("\n[PHAN D] So sanh 3 mo hinh...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    models = {
        'Overfit (deg=15)':    make_pipeline(PolynomialFeatures(15), LinearRegression()),
        f'CV (deg={best_deg})': make_pipeline(PolynomialFeatures(best_deg), LinearRegression()),
        'Ridge (deg=10)':      make_pipeline(PolynomialFeatures(10), Ridge(alpha=1.0))
    }

    print(f"\n{'Mo hinh':22} {'Train':>10} {'Test':>10}")
    print("-" * 45)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train, y_train)
        tr = r2_score(y_train, model.predict(X_train))
        te = r2_score(y_test, model.predict(X_test))
        ax.scatter(X_train, y_train, alpha=0.3, s=20, c='steelblue', label='Train')
        ax.scatter(X_test, y_test, alpha=0.3, s=20, c='orange', label='Test')
        ax.plot(X_range, model.predict(X_range), 'r-', linewidth=2)
        ax.set_title(f'{name}\nTest={te:.4f}', fontsize=10)
        ax.legend(); ax.grid(True, alpha=0.3)
        print(f"{name:22} {tr:>10.4f} {te:>10.4f}")

    plt.tight_layout()
    plt.savefig('hinh_anh/final_comparison.png', dpi=100)
    print("\n[OK] Da luu: hinh_anh/final_comparison.png")
    print(f"\n>> CV (degree={best_deg}) tot nhat -> Khong overfit!")
