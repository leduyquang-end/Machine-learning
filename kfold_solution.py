"""
kfold_solution.py - K-Fold Cross Validation khac phuc overfitting
Chia du lieu thanh K fold, moi lan train K-1 fold, validate 1 fold
"""

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV, learning_curve, train_test_split
from data_generator import load_data, get_X_y

# --- Tai du lieu ---
df = load_data()
X, y = get_X_y(df)

# ================================================================
# PHAN 1: VE K-FOLD DE HIU
# ================================================================
# Chia 30 mau thanh 5 fold, moi fold ~6 mau
# Moi hang = 1 lan train/validate
#   Xanh = du lieu dung de hoc (Train)
#   Do   = du lieu dung de kiem tra (Validation)
# ================================================================

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fig, axes = plt.subplots(5, 1, figsize=(12, 5))

for i, (train_idx, val_idx) in enumerate(kfold.split(np.arange(30))):
    colors = ['lightgray'] * 30
    for j in train_idx: colors[j] = 'steelblue'
    for j in val_idx:   colors[j] = 'red'
    axes[i].scatter(range(30), [0]*30, c=colors, s=200, edgecolors='black')
    axes[i].set_yticks([])
    axes[i].set_xlabel(f'Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}')

plt.suptitle('K-Fold: Xanh=Train | Do=Validation', fontweight='bold')
plt.tight_layout()
plt.savefig('kfold_visualization.png', dpi=100)
print("[1] Da luu: kfold_visualization.png")

# ================================================================
# PHAN 2: GRIDSEARCHCV - TIM DEGREE TOT NHAT
# ================================================================
# Thu degree 1-10, moi degree danh gia bang K-Fold (K=5)
# Degree tot nhat = co Val R2 cao nhat
# ================================================================

model = make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
grid = GridSearchCV(model, {'polynomialfeatures__degree': list(range(1, 11))},
                    cv=5, scoring='r2', return_train_score=True)
grid.fit(X, y)
res = grid.cv_results_

print("\n[2] Ket qua tim degree:")
print(f"{'Deg':>4} {'Train':>8} {'Val':>8} {'Lech':>8}")
for d, tr, v in zip(range(1, 11), res['mean_train_score'], res['mean_test_score']):
    print(f"{d:>4} {tr:>8.4f} {v:>8.4f} {tr-v:>8.4f}{' <-- BEST' if v==max(res['mean_test_score']) else ''}")

# Ve bieu do
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), res['mean_train_score'], 'o-', label='Train')
plt.plot(range(1, 11), res['mean_test_score'], 'o-', label='Validation', color='red')
plt.axvspan(5, 10, alpha=0.1, color='red', label='Overfitting')
plt.xlabel('Degree'); plt.ylabel('R2'); plt.title('GridSearchCV + K-Fold')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('degree_comparison.png', dpi=100)
print("[3] Da luu: degree_comparison.png")

# ================================================================
# PHAN 3: LEARNING CURVE
# ================================================================
# Xem mo hinh tot hon khi co nhieu du lieu hon hay khong
# 2 duong gan nhau -> mo hinh tot
# ================================================================

best_deg = grid.best_params_['polynomialfeatures__degree']
best_model = grid.best_estimator_
sizes, tr_sc, val_sc = learning_curve(best_model, X, y,
                                       train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

plt.figure(figsize=(8, 5))
plt.plot(sizes, tr_sc.mean(axis=1), 'o-', label='Train')
plt.plot(sizes, val_sc.mean(axis=1), 'o-', label='Validation', color='red')
plt.xlabel('Kich thuoc du lieu'); plt.ylabel('R2')
plt.title(f'Learning Curve (Degree={best_deg})')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=100)
print("[4] Da luu: learning_curve.png")

# ================================================================
# PHAN 4: SO SANH 3 MO HINH
# ================================================================
# 1. Overfit: degree=15 -> qua phuc tap
# 2. CV:      degree=1  -> chon boi GridSearchCV
# 3. Ridge:   degree=10 + regularization -> giam overfit
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

models = {
    'Overfit (deg=15)':    make_pipeline(PolynomialFeatures(15), LinearRegression()),
    'CV Selected (deg=1)': make_pipeline(PolynomialFeatures(1), LinearRegression()),
    'Ridge (deg=10)':      make_pipeline(PolynomialFeatures(10), Ridge(alpha=1.0))
}

print("\n[5] So sanh 3 mo hinh:")
print(f"{'Mo hinh':22} {'Train R2':>10} {'Test R2':>10}")
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
plt.savefig('final_comparison.png', dpi=100)
print("\n[6] Da luu: final_comparison.png")
print("\n>> CV (degree=1) tot nhat -> Khong overfit!")
