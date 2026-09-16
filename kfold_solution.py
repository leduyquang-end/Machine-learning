"""
kfold_solution.py
=================
GIAI PHAP: K-Fold Cross Validation de khac phuc OVERFITTING
Du lieu: Kaggle House Prices (Seattle, WA)

K-Fold CV la gi?
- Chia du lieu thanh K phan (fold)
- Moi lan: K-1 phan de train, 1 phan de validate
- Lap lai K lan, ket qua = trung binh
- Loi ich: Phat hien overfitting som, danh gia on dinh hon
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    GridSearchCV,
    learning_curve,
    train_test_split
)
from data_generator import load_kaggle_data, prepare_features


# ============================================================
# PHAN 1: GIAI THICH K-FOLD DON GIAN
# ============================================================

def visualize_kfold():
    """Truc quan hoa cach chia du lieu K-Fold"""
    print("=== GIAI THICH K-FOLD ===")

    n = 30  # Mau nho de de visualize
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    fig, axes = plt.subplots(5, 1, figsize=(12, 6))

    for i, (train_idx, val_idx) in enumerate(kfold.split(np.arange(n))):
        colors = ['lightgray'] * n
        for j in train_idx: colors[j] = 'steelblue'  # Train = xanh
        for j in val_idx:   colors[j] = 'red'         # Validate = do

        axes[i].scatter(range(n), [0]*n, c=colors, s=200, edgecolors='black')
        axes[i].set_yticks([])
        axes[i].set_xlabel(f'Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}')

    plt.suptitle('K-Fold: Xanh=Train | Do=Validation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kfold_visualization.png', dpi=100)
    print("Da luu: kfold_visualization.png\n")


# ============================================================
# PHAN 2: TIM DEGREE TOT NHAT VOI GRIDSEARCHCV
# ============================================================

def find_best_degree(X, y):
    """
    Su dung GridSearchCV de thu tat ca degree tu 1-10
    Moi danh gia bang K-Fold CV (K=5)

    Ket qua: Degree nao co Validation R2 cao nhat -> chon
    """
    print("=== TIM DEGREE TOT NHAT ===")

    degrees = list(range(1, 11))
    model = make_pipeline(
        PolynomialFeatures(include_bias=False),
        LinearRegression()
    )

    # GridSearchCV: Thu het cac degree, danh gia bang K-Fold
    grid = GridSearchCV(
        model,
        {'polynomialfeatures__degree': degrees},
        cv=5,              # K-Fold voi K=5
        scoring='r2',
        return_train_score=True
    )
    grid.fit(X, y)
    results = grid.cv_results_

    # In bang ket qua
    print(f"\n{'Degree':>8} {'Train R2':>10} {'Val R2':>10} {'Chenh lech':>12}")
    print("-" * 45)

    best_score = -np.inf
    best_deg = 1

    for deg, tr, val in zip(degrees, results['mean_train_score'], results['mean_test_score']):
        gap = tr - val
        star = " <-- BEST" if val > best_score else ""
        print(f"{deg:>8} {tr:>10.4f} {val:>10.4f} {gap:>12.4f}{star}")
        if val > best_score:
            best_score = val
            best_deg = deg

    print(f"\n>> Degree tot nhat: {best_deg} (Val R2={best_score:.4f})")

    # Ve bieu do
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, results['mean_train_score'], 'o-', color='steelblue', label='Train')
    ax.plot(degrees, results['mean_test_score'], 'o-', color='red', label='Validation')
    ax.axvspan(5, 10, alpha=0.1, color='red', label='Overfitting')
    ax.set_xlabel('Degree')
    ax.set_ylabel('R2 Score')
    ax.set_title('GridSearchCV + K-Fold: Chon Degree tot nhat')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('degree_comparison.png', dpi=100)
    print("Da luu: degree_comparison.png\n")

    return grid.best_estimator_, best_deg


# ============================================================
# PHAN 3: LEARNING CURVE
# ============================================================

def show_learning_curve(X, y, degree=1):
    """
    Learning Curve: Xem mo hinh hoat dong tot voi bao nhieu du lieu
    - Train va Val gan nhau -> Mo hinh tot
    - Train >> Val -> Overfitting
    """
    print("=== LEARNING CURVE ===")

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    sizes, train_sc, val_sc = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='r2'
    )

    train_mean = np.mean(train_sc, axis=1)
    val_mean = np.mean(val_sc, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, 'o-', color='steelblue', label='Train')
    ax.plot(sizes, val_mean, 'o-', color='red', label='Validation')
    ax.set_xlabel('Kich thuoc du lieu')
    ax.set_ylabel('R2 Score')
    ax.set_title(f'Learning Curve (Degree={degree})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=100)
    print("Da luu: learning_curve.png\n")


# ============================================================
# PHAN 4: SO SANH CUOI CUNG
# ============================================================

def final_comparison(X, y):
    """So sanh 3 phuong phap: Overfit vs CV vs Ridge"""
    print("=== SO SANH 3 PHUONG PHAP ===")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3 mo hinh
    models = {
        'Overfit (deg=15)': make_pipeline(PolynomialFeatures(15, include_bias=False), LinearRegression()),
        'CV Selected (deg=1)': make_pipeline(PolynomialFeatures(1, include_bias=False), LinearRegression()),
        'Ridge (deg=10)': make_pipeline(PolynomialFeatures(10, include_bias=False), Ridge(alpha=1.0))
    }

    print(f"\n{'Mo hinh':22} {'Train R2':>10} {'Test R2':>10}")
    print("-" * 45)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train, y_train)
        tr = r2_score(y_train, model.predict(X_train))
        te = r2_score(y_test, model.predict(X_test))

        ax.scatter(X_train, y_train, alpha=0.3, s=20, c='steelblue', label='Train')
        ax.scatter(X_test, y_test, alpha=0.3, s=20, c='orange', label='Test')
        ax.plot(X_range, model.predict(X_range), 'r-', linewidth=2)
        ax.set_title(f'{name}\nTest R2={te:.4f}', fontsize=10)
        ax.set_xlabel('sqft_living')
        ax.set_ylabel('price (USD)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"{name:22} {tr:>10.4f} {te:>10.4f}")

    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=100)
    print("\nDa luu: final_comparison.png")
    print("\n>> CV va Ridge tot hon Overfit!")
    print(">> Bai hoc: Luon dung Cross Validation de danh gia mo hinh!")


# ============================================================
# CHAY CHINH
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("K-FOLD CROSS VALIDATION - Kaggle House Prices")
    print("=" * 50)

    # Tai du lieu
    df = load_kaggle_data()
    X, y = prepare_features(df, feature_cols=['sqft_living'])

    # 1. Giai thich K-Fold
    visualize_kfold()

    # 2. Tim degree tot nhat
    best_model, best_deg = find_best_degree(X, y)

    # 3. Learning curve
    show_learning_curve(X, y, best_deg)

    # 4. So sanh cuoi cung
    final_comparison(X, y)
