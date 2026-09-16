"""
overfit_example.py
==================
Vi du ve OVERFITTING trong Machine Learning
Du lieu: Kaggle House Prices Dataset (Seattle, WA)
Nguon: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

Khi mo hinh hoc qua tot tren du lieu nhung hoat dong te tren du lieu moi.
Dau hieu: Train R2 cao, Test R2 thap, chenh lech lon.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from data_generator import load_kaggle_data, prepare_features


def create_overfitted_model(X_train, y_train, degree=15):
    """
    Tao mo hinh OVERFITTING bang polynomial degree cao

    degree=15: Mo hinh se co 15 features (x^1, x^2, ..., x^15)
    Qua phuc tap so voi du lieu thuc (quan he tuyen tinh don gian)

    Parameters:
    -----------
    X_train : array - Du lieu hoc
    y_train : array - Nhan
    degree : int - Bac polynomial (cang cao = overfit)

    Returns:
    --------
    model : fitted model
    """
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    return model


def plot_overfitting_comparison(X_train, y_train, X_test, y_test,
                                 model_overfit, model_simple):
    """Ve bieu do so sanh overfit vs simple model"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    X_range = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)

    # ---- SUBPLOT 1: Overfit Model ----
    ax1 = axes[0]
    ax1.scatter(X_train, y_train, alpha=0.3, c='steelblue', s=40,
                label=f'Train ({len(X_train)} mau)')
    ax1.scatter(X_test, y_test, alpha=0.3, c='orange', s=40,
                label=f'Test ({len(X_test)} mau)')

    y_pred_overfit = model_overfit.predict(X_range)
    ax1.plot(X_range, y_pred_overfit, 'r-', linewidth=2,
             label='Overfit (degree=15)')

    train_r2 = r2_score(y_train, model_overfit.predict(X_train))
    test_r2 = r2_score(y_test, model_overfit.predict(X_test))

    ax1.set_xlabel('Dien tich (sqft)', fontsize=12)
    ax1.set_ylabel('Gia nha (USD)', fontsize=12)
    ax1.set_title(f'OVERFITTING - Mo hinh qua phuc tap\n'
                  f'Train R2={train_r2:.4f} | Test R2={test_r2:.4f}\n'
                  f'Chenh lech={train_r2-test_r2:.4f} (LON!)',
                  fontsize=12, color='red')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- SUBPLOT 2: Simple Model ----
    ax2 = axes[1]
    ax2.scatter(X_train, y_train, alpha=0.3, c='steelblue', s=40,
                label=f'Train ({len(X_train)} mau)')
    ax2.scatter(X_test, y_test, alpha=0.3, c='orange', s=40,
                label=f'Test ({len(X_test)} mau)')

    y_pred_simple = model_simple.predict(X_range)
    ax2.plot(X_range, y_pred_simple, 'g-', linewidth=2,
             label='Simple (degree=1)')

    train_r2_s = r2_score(y_train, model_simple.predict(X_train))
    test_r2_s = r2_score(y_test, model_simple.predict(X_test))

    ax2.set_xlabel('Dien tich (sqft)', fontsize=12)
    ax2.set_ylabel('Gia nha (USD)', fontsize=12)
    ax2.set_title(f'GOOD MODEL - Mo hinh don gian\n'
                  f'Train R2={train_r2_s:.4f} | Test R2={test_r2_s:.4f}\n'
                  f'Chenh lech={train_r2_s-test_r2_s:.4f} (NHO)',
                  fontsize=12, color='green')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('overfit_comparison.png', dpi=100)
    print("Da luu: overfit_comparison.png")

    # In ket qua chi tiet
    print("\n" + "=" * 60)
    print("SO SANH MO HINH")
    print("=" * 60)
    print(f"{'':20} {'Overfit':>12} {'Simple':>12}")
    print(f"{'Train R2':20} {train_r2:>12.4f} {train_r2_s:>12.4f}")
    print(f"{'Test R2':20} {test_r2:>12.4f} {test_r2_s:>12.4f}")
    print(f"{'Chenh lech':20} {train_r2-test_r2:>12.4f} {train_r2_s-test_r2_s:>12.4f}")
    print(f"\n[!] Overfit: Train tot nhung Test kem -> Mo hinh hoc vet!")
    print(f"[OK] Simple: Train va Test gan nhau -> Mo hinh hieu du lieu!")


if __name__ == "__main__":
    # 1. Tai du lieu tu Kaggle
    print("=" * 60)
    print("OVERFITTING EXAMPLE - Kaggle House Prices Dataset")
    print("=" * 60)
    print("\n1. Tai du lieu tu Kaggle...")
    df = load_kaggle_data()

    # 2. Chuan bi features (chi su dung sqft_living)
    X, y = prepare_features(df, feature_cols=['sqft_living'])

    # 3. Chia train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n2. Chia du lieu: Train={len(X_train)} | Test={len(X_test)}")

    # 4. Tao mo hinh Overfitting (degree=15 - qua phuc tap!)
    print("\n3. Tao mo hinh OVERFITTING (degree=15)...")
    model_overfit = create_overfitted_model(X_train, y_train, degree=15)

    # 5. Tao mo hinh don gian (degree=1 - tuyen tinh)
    print("   Tao mo hinh DON GIAN (degree=1)...")
    model_simple = create_overfitted_model(X_train, y_train, degree=1)

    # 6. Visualize so sanh
    print("\n4. Ve bieu do so sanh...")
    plot_overfitting_comparison(X_train, y_train, X_test, y_test,
                                model_overfit, model_simple)

    print("\n=> Ket qua luu vao: overfit_comparison.png")
