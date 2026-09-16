"""
kfold_solution.py
=================
GIAI PHAP K-FOLD CROSS VALIDATION de khac phuc OVERFITTING
Du lieu: Kaggle House Prices Dataset (Seattle, WA)
Nguon: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

K-Fold CV:
- Chia du lieu thanh K fold
- Moi lan lay K-1 fold de train, 1 fold de validate
- Ket qua = trung binh K lan

Tai sao K-Fold giup giam overfitting?
- Mo hinh phai hoat dong tot tren nhieu cach chia du lieu
- Phat hien overfitting som bang validation
- Danh gia on dinh hon train/test chi chia 1 lan
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    GridSearchCV,
    learning_curve,
    train_test_split
)
from data_generator import load_kaggle_data, prepare_features


def simple_cross_val_example():
    """Giai thich don gian ve K-Fold Cross Validation"""
    print("=" * 70)
    print("K-FOLD CROSS VALIDATION - GIAI THICH DON GIAN")
    print("=" * 70)

    # Tao du lieu mau nho de visualize
    n_samples = 30
    X = np.arange(n_samples).reshape(-1, 1)
    y = np.random.randn(n_samples)

    # Tao K-Fold voi K=5
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\nTong so mau: {n_samples}")
    print(f"So fold (K): 5")
    print(f"Moi fold: ~{n_samples//5} mau")

    # Visualize cach chia fold
    fig, axes = plt.subplots(5, 1, figsize=(12, 8))

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        ax = axes[fold_idx]
        colors = ['lightgray'] * n_samples
        for i in train_idx:
            colors[i] = 'steelblue'
        for i in val_idx:
            colors[i] = 'red'

        ax.scatter(range(n_samples), [0]*n_samples, c=colors,
                   s=200, edgecolors='black', linewidth=1)
        ax.set_yticks([])
        ax.set_xlabel(f'Fold {fold_idx+1}: '
                      f'Train={len(train_idx)} mau, '
                      f'Validation={len(val_idx)} mau',
                      fontsize=10)

    plt.suptitle('Cach chia du lieu trong K-Fold Cross Validation\n'
                 'Xanh duong = Train | Do = Validation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kfold_visualization.png', dpi=100)
    pass  # plt.show() disabled for non-interactive


def find_best_degree_with_cv(X, y):
    """
    Su dung GridSearchCV de tim polynomial degree tot nhat

    GridSearchCV:
    1. Thu tat ca degree tu 1 den 10
    2. Moi degree danh gia bang K-Fold CV (K=5)
    3. Chon degree co score tot nhat
    """
    print("\n" + "=" * 70)
    print("TIM POLYNOMIAL DEGREE TOT NHAT VOI K-FOLD CV")
    print("=" * 70)

    degrees = list(range(1, 11))

    model = make_pipeline(
        PolynomialFeatures(include_bias=False),
        LinearRegression()
    )

    param_grid = {
        'polynomialfeatures__degree': degrees
    }

    # GridSearchCV voi K=5
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='r2',
        return_train_score=True,
        n_jobs=-1
    )

    grid_search.fit(X, y)
    results = grid_search.cv_results_

    # In ket qua chi tiet
    print(f"\n{'Degree':>8} {'Train R2':>12} {'Val R2':>12} {'Chenh lech':>12}")
    print("-" * 50)

    best_val_score = -np.inf
    best_deg = 1

    for degree, train_score, val_score in zip(
        degrees,
        results['mean_train_score'],
        results['mean_test_score']
    ):
        gap = train_score - val_score
        marker = " <-- BEST" if val_score > best_val_score else ""
        print(f"{degree:>8} {train_score:>12.4f} {val_score:>12.4f} {gap:>12.4f}{marker}")
        if val_score > best_val_score:
            best_val_score = val_score
            best_deg = degree

    print(f"\n[KET QUA] Degree tot nhat: {best_deg}")
    print(f"[KET QUA] Validation R2: {best_val_score:.4f}")

    return grid_search.best_estimator_, best_deg, results


def visualize_degree_comparison(results, degrees):
    """Ve bieu do so sanh train/val score theo degree"""
    fig, ax = plt.subplots(figsize=(10, 6))

    train_scores = results['mean_train_score']
    val_scores = results['mean_test_score']

    ax.plot(degrees, train_scores, 'o-', color='steelblue',
            linewidth=2, markersize=8, label='Train R2 Score')
    ax.plot(degrees, val_scores, 'o-', color='red',
            linewidth=2, markersize=8, label='Validation R2 Score')

    # Danh dau vung overfitting
    ax.axvspan(5, 10, alpha=0.1, color='red', label='Overfitting zone')
    ax.axvspan(1, 2, alpha=0.1, color='green', label='Good zone')

    ax.annotate('Overfitting!\nTrain >> Validation',
                xy=(7, train_scores[6]),
                xytext=(7.5, 0.95),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    ax.annotate('Good fit!\nTrain ~ Validation',
                xy=(1, val_scores[0]),
                xytext=(2.5, 0.7),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', fontweight='bold')

    ax.set_xlabel('Polynomial Degree', fontsize=12)
    ax.set_ylabel('R2 Score', fontsize=12)
    ax.set_title('Su dung K-Fold CV de chon Degree tot nhat\n'
                 'Kaggle House Prices Dataset', fontsize=13)
    ax.set_xticks(degrees)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    plt.savefig('degree_comparison.png', dpi=100)
    pass  # plt.show() disabled for non-interactive


def learning_curve_analysis(X, y, best_degree=1):
    """Phan tich learning curve de hieu mo hinh"""
    print("\n" + "=" * 70)
    print("PHAN TICH LEARNING CURVE")
    print("=" * 70)

    model = make_pipeline(
        PolynomialFeatures(degree=best_degree, include_bias=False),
        LinearRegression()
    )

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color='steelblue')
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.1, color='red')

    ax.plot(train_sizes, train_mean, 'o-', color='steelblue',
            linewidth=2, label='Train Score')
    ax.plot(train_sizes, val_mean, 'o-', color='red',
            linewidth=2, label='Validation Score')

    ax.set_xlabel('Kich thuoc du lieu hoc', fontsize=12)
    ax.set_ylabel('R2 Score', fontsize=12)
    ax.set_title(f'Learning Curve (Degree = {best_degree})\n'
                 f'Kaggle House Prices Dataset', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=100)
    pass  # plt.show() disabled for non-interactive

    print(f"Kich thuoc du lieu cuoi: {train_sizes[-1]:.0f} mau")
    print(f"Train Score: {train_mean[-1]:.4f} (+/- {train_std[-1]:.4f})")
    print(f"Val Score:   {val_mean[-1]:.4f} (+/- {val_std[-1]:.4f})")
    print(f"Chenh lech:  {train_mean[-1] - val_mean[-1]:.4f}")


def final_comparison(X, y):
    """So sanh cuoi cùng: Overfit vs CV vs Ridge"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n" + "=" * 70)
    print("SO SANH CUOI CUNG: 3 PHUONG PHAP")
    print("=" * 70)

    # 1. Overfit model (degree=15)
    model_overfit = make_pipeline(
        PolynomialFeatures(degree=15, include_bias=False),
        LinearRegression()
    )
    model_overfit.fit(X_train, y_train)

    # 2. CV-selected model (degree=1)
    model_cv = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        LinearRegression()
    )
    model_cv.fit(X_train, y_train)

    # 3. Ridge model (degree=10, alpha=1.0)
    model_ridge = make_pipeline(
        PolynomialFeatures(degree=10, include_bias=False),
        Ridge(alpha=1.0)
    )
    model_ridge.fit(X_train, y_train)

    models = {
        'Overfit (degree=15)': model_overfit,
        'CV Selected (degree=1)': model_cv,
        'Ridge (degree=10, a=1)': model_ridge
    }

    print(f"\n{'Phuong phap':25} {'Train R2':>10} {'Test R2':>10} {'MAE':>12}")
    print("-" * 60)

    for name, model in models.items():
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        print(f"{name:25} {train_r2:>10.4f} {test_r2:>10.4f} {test_mae:>12.2f}")

    # Ve so sanh
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

    for ax, (name, model) in zip(axes, models.items()):
        ax.scatter(X_train, y_train, alpha=0.3, s=40, c='steelblue',
                   label='Train')
        ax.scatter(X_test, y_test, alpha=0.3, s=40, c='orange',
                   label='Test')

        y_range_pred = model.predict(X_range)
        ax.plot(X_range, y_range_pred, 'r-', linewidth=2)

        test_r2 = r2_score(y_test, model.predict(X_test))
        ax.set_title(f'{name}\nTest R2 = {test_r2:.4f}', fontsize=11)
        ax.set_xlabel('Dien tich (sqft)')
        ax.set_ylabel('Gia nha (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('SO SANH: Overfit vs Cross-Validated vs Regularized\n'
                 'Kaggle House Prices Dataset',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=100)
    pass  # plt.show() disabled for non-interactive

    print("\n=> Ket qua: CV va Ridge tot hon Overfit!")
    print("=> Bai hoc: Luon dung Cross Validation de danh gia mo hinh!")


if __name__ == "__main__":
    # 1. Tai du lieu tu Kaggle
    print("=" * 70)
    print("K-FOLD CROSS VALIDATION - Kaggle House Prices Dataset")
    print("=" * 70)
    print("\n1. Tai du lieu tu Kaggle...")
    df = load_kaggle_data()

    # 2. Chuan bi features
    X, y = prepare_features(df, feature_cols=['sqft_living'])

    # 3. Giai thich K-Fold don gian
    print("\n2. Giai thich K-Fold don gian...")
    simple_cross_val_example()

    # 4. Tim best degree voi GridSearchCV
    print("\n3. Tim degree tot nhat voi GridSearchCV...")
    best_model, best_degree, results = find_best_degree_with_cv(X, y)

    # 5. Visualize degree comparison
    print("\n4. Ve bieu do so sanh degree...")
    degrees = list(range(1, 11))
    visualize_degree_comparison(results, degrees)

    # 6. Learning curve analysis
    print("\n5. Phan tich learning curve...")
    learning_curve_analysis(X, y, best_degree=best_degree)

    # 7. Final comparison
    print("\n6. So sanh cuoi cung...")
    final_comparison(X, y)

    print("\n=> Tat ca ket qua da luu vao cac file anh!")
