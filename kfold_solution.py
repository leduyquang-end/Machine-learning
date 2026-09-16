"""
kfold_solution.py
=================

PHAN 2: KHAC PHUC OVERFITTING BANG K-FOLD CROSS VALIDATION

TẠI SAO K-FOLD GIUP KHAC PHUC OVERFITTING?
-----------------------------------------
- K-Fold chia du lieu thanh K phan (fold)
- Moi lan: K-1 phan de TRAIN, 1 phan de VALIDATE
- Lap lai K lan, moi lan validate phan khac nhau
- Ket qua cuoi = trung binh K lan

Loi ich:
  + Phat hien overfitting SOM (truoc khi dung test)
  + Danh gia mo hinh ON DINH hon (khong phu thuoc vao 1 cach chia)
  + Chon hyperparameter tot nhat (degree, alpha, ...)

NGUON DU LIEU:
- Kaggle: House Prices (Seattle, WA)
- https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
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
    KFold,
    GridSearchCV,
    learning_curve,
    train_test_split
)
from data_generator import load_data, get_X_y


# ============================================================
# TAI DU LIEU
# ============================================================
print("=" * 60)
print("PHAN 2: KHAC PHUC OVERFITTING BANG K-FOLD CV")
print("=" * 60)

df = load_data()
X, y = get_X_y(df)


# ============================================================
# PHAN A: TRUC QUAN HOA K-FOLD
# ============================================================
# VD: 30 mau, chia thanh 5 fold, moi fold ~6 mau
#   - Fold 1: Train=24 mau, Validate=6 mau
#   - Fold 2: Train=24 mau, Validate=6 mau
#   - ...
#   - Fold 5: Train=24 mau, Validate=6 mau
#
#   Ket qua = (R2_fold1 + R2_fold2 + ... + R2_fold5) / 5
# ============================================================

print("\n[PHAN A] Truc quan hoa K-Fold...")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fig, axes = plt.subplots(5, 1, figsize=(12, 6))

for i, (train_idx, val_idx) in enumerate(kfold.split(np.arange(30))):
    # Tao mau sac: xanh = train, do = validate, xam = khong dung
    colors = ['lightgray'] * 30
    for j in train_idx:
        colors[j] = 'steelblue'  # Train = xanh duong
    for j in val_idx:
        colors[j] = 'red'        # Validate = do

    axes[i].scatter(range(30), [0]*30, c=colors, s=200, edgecolors='black')
    axes[i].set_yticks([])
    axes[i].set_xlabel(
        f'Fold {i+1}: Train={len(train_idx)} mau, Validate={len(val_idx)} mau',
        fontsize=10
    )

plt.suptitle(
    'K-Fold Cross Validation (K=5)\n'
    'Xanh duong = Train | Do = Validate | Xam = Khong dung',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('kfold_visualization.png', dpi=100)
print("[OK] Da luu: kfold_visualization.png")


# ============================================================
# PHAN B: TIM DEGREE TOT NHAT VOI GRIDSEARCHCV
# ============================================================
# GridSearchCV thu tat ca degree tu 1 den 10
# Moi degree danh gia bang K-Fold CV (K=5)
# Chon degree co Validation R2 CAO NHAT
#
# Ket qua mong doi:
#   degree=1: Val R2 CAO (tot) -> CHON
#   degree=5+: Val R2 THAP (te) -> BO (overfit)
# ============================================================

print("\n[PHAN B] Tim degree tot nhat voi GridSearchCV...")

# Tao mo hinh co PolynomialFeatures (chua chon degree)
model = make_pipeline(
    PolynomialFeatures(include_bias=False),
    LinearRegression()
)

# GridSearchCV: Thu het cac degree, danh gia bang K-Fold
grid = GridSearchCV(
    model,
    {'polynomialfeatures__degree': list(range(1, 11))},  # Thu degree 1-10
    cv=5,              # K-Fold voi K=5
    scoring='r2',      # Danh gia bang R2 score
    return_train_score=True  # Luu ca train score
)
grid.fit(X, y)  # Chay GridSearch
res = grid.cv_results_  # Lay ket qua

# In bang ket qua chi tiet
print(f"\n{'Degree':>8} {'Train R2':>10} {'Val R2':>10} {'Chenh lech':>12}")
print("-" * 50)

for deg, tr, val in zip(
    range(1, 11),
    res['mean_train_score'],
    res['mean_test_score']
):
    gap = tr - val
    star = " <-- BEST" if val == max(res['mean_test_score']) else ""
    print(f"{deg:>8} {tr:>10.4f} {val:>10.4f} {gap:>12.4f}{star}")

# Lay ket qua tot nhat
best_deg = grid.best_params_['polynomialfeatures__degree']
best_val_r2 = grid.best_score_
print(f"\n>> KET QUA: Degree tot nhat = {best_deg} (Val R2={best_val_r2:.4f})")


# ============================================================
# PHAN C: VE BIEU DO SO SANH DEGREE
# ============================================================
# Bieu do cho thay:
#   - Train R2 (xanh): Luon cao, it thay doi
#   - Val R2 (do): Giam dan khi degree tang -> OVERFITTING
#   - Vung do (degree 5-10): Overfitting zone
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), res['mean_train_score'], 'o-',
         color='steelblue', linewidth=2, markersize=8, label='Train R2')
plt.plot(range(1, 11), res['mean_test_score'], 'o-',
         color='red', linewidth=2, markersize=8, label='Val R2')

# Danh dau vung overfitting
plt.axvspan(5, 10, alpha=0.1, color='red', label='Overfitting zone')
plt.axvspan(1, 2, alpha=0.1, color='green', label='Good zone')

# Them chu thich
plt.annotate('Overfitting!\nTrain >> Val',
             xy=(7, res['mean_train_score'][6]),
             xytext=(7.5, 0.95),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red', fontweight='bold')

plt.annotate('Good fit!\nTrain ~ Val',
             xy=(1, res['mean_test_score'][0]),
             xytext=(2.5, 0.6),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, color='green', fontweight='bold')

plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('R2 Score', fontsize=12)
plt.title('GridSearchCV + K-Fold: Chon Degree tot nhat\n'
          'Kaggle House Prices Dataset', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('degree_comparison.png', dpi=100)
print("\n[OK] Da luu: degree_comparison.png")


# ============================================================
# PHAN D: LEARNING CURVE
# ============================================================
# Learning Curve xem mo hinh hoat dong tot voi bao nhieu du lieu
#   - 2 duong gan nhau -> mo hinh TOT
#   - Train >> Val -> OVERFITTING
#   - Ca hai deu thap -> UNDERFITTING
# ============================================================

print("\n[PHAN D] Phan tich Learning Curve...")

best_model = grid.best_estimator_
sizes, train_sc, val_sc = learning_curve(
    best_model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),  # Tu 10% den 100% du lieu
    cv=5,                                      # K-Fold
    scoring='r2'
)

# Ve learning curve
plt.figure(figsize=(8, 5))
plt.plot(sizes, train_sc.mean(axis=1), 'o-',
         color='steelblue', linewidth=2, label='Train Score')
plt.plot(sizes, val_sc.mean(axis=1), 'o-',
         color='red', linewidth=2, label='Val Score')

plt.xlabel('Kich thuoc du lieu huấn luyen', fontsize=12)
plt.ylabel('R2 Score', fontsize=12)
plt.title(f'Learning Curve (Degree={best_deg})\n'
          f'Kaggle House Prices Dataset', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=100)
print("[OK] Da luu: learning_curve.png")


# ============================================================
# PHAN E: SO SANH 3 MO HINH CUOI CUNG
# ============================================================
# 1. Overfit (degree=15): Qua phuc tap -> TE
# 2. CV Selected (degree=1): Chon boi GridSearchCV -> TOT
# 3. Ridge (degree=10 + alpha=1): Regularization -> Giam overfit
# ============================================================

print("\n[PHAN E] So sanh 3 mo hinh cuoi cung...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tao 3 mo hinh
models = {
    'Overfit (degree=15)': make_pipeline(
        PolynomialFeatures(15, include_bias=False),
        LinearRegression()
    ),
    f'CV Selected (degree={best_deg})': make_pipeline(
        PolynomialFeatures(best_deg, include_bias=False),
        LinearRegression()
    ),
    'Ridge (degree=10, alpha=1)': make_pipeline(
        PolynomialFeatures(10, include_bias=False),
        Ridge(alpha=1.0)  # Ridge = LinearRegression + regularization
    )
}

# In bang ket qua
print(f"\n{'Mo hinh':30} {'Train R2':>10} {'Test R2':>10}")
print("-" * 55)

# Ve bieu do so sanh
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

for ax, (name, model) in zip(axes, models.items()):
    # Hoc va danh gia
    model.fit(X_train, y_train)
    tr = r2_score(y_train, model.predict(X_train))
    te = r2_score(y_test, model.predict(X_test))

    # Ve du lieu
    ax.scatter(X_train, y_train, alpha=0.3, s=20, c='steelblue', label='Train')
    ax.scatter(X_test, y_test, alpha=0.3, s=20, c='orange', label='Test')

    # Ve duong du doan
    ax.plot(X_range, model.predict(X_range), 'r-', linewidth=2)

    # Tieu de va luoi
    ax.set_title(f'{name}\nTest R2 = {te:.4f}', fontsize=11)
    ax.set_xlabel('Dien tich (sqft_living)')
    ax.set_ylabel('Gia nha (USD)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    print(f"{name:30} {tr:>10.4f} {te:>10.4f}")

plt.tight_layout()
plt.savefig('final_comparison.png', dpi=100)
print("\n[OK] Da luu: final_comparison.png")


# ============================================================
# KET LUAN
# ============================================================
print("\n" + "=" * 60)
print("KET LUAN")
print("=" * 60)
print(f"1. Overfit (degree=15):  Train tot nhung Test TE -> Hoc vet!")
print(f"2. CV Selected (degree={best_deg}):  Train va Test gan nhau -> TOT!")
print(f"3. Ridge (degree=10):    Giam overfit nhung van chua tot bang CV")
print(f"\n=> BAI HOC: Luon dung Cross Validation de danh gia mo hinh!")
print("=> K-Fold giup phat hien overfitting som va chon mo hinh tot nhat!")
