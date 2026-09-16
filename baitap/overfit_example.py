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

TẠI SAO OVERFIT XAY RA?
- Mo hinh qua PHUC TAP so voi du lieu thuc
- Vi du: Duong gia nha la DUONG THANG (degree=1)
          nhung ta dung polynomial degree=15 (15 o so)
          -> Mo hinh "cong" vao moi diem du lieu
          -> Khong duoc tren du lieu moi

NGUON DU LIEU:
- Kaggle: House Prices (Seattle, WA)
- https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend khong interactive (khong mo cua so)
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from data_generator import load_data, get_X_y


# ============================================================
# BUOC 1: TAI DU LIEU TU KAGGLE
# ============================================================
print("=" * 60)
print("PHAN 1: TAO MO HINH OVERFITTING")
print("=" * 60)

df = load_data()
X, y = get_X_y(df)

# Chia du lieu thanh 2 phan:
#   Train (80%): Dung de hoc
#   Test (20%):  Dung de kiem tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% cho test
    random_state=42      # Fix seed de ket qua co the tai tao
)
print(f"Train: {len(X_train)} mau | Test: {len(X_test)} mau\n")


# ============================================================
# BUOC 2: TAO 2 MO HINH
# ============================================================

def fit_model(degree):
    """
    Tao mo hinh Polynomial Regression voi degree cu the

    Parameters:
    -----------
    degree : int
        - degree=1:  Duong thang (don gian, it overfit)
        - degree=15: Duong cong phuc tap (de overfit)
    """
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X_train, y_train)  # Hoc tu du lieu train
    return model


# Mo hinh 1: OVERFITTING (degree=15 - qua phuc tap!)
print("Tao mo hinh OVERFIT (degree=15)...")
model_overfit = fit_model(15)

# Mo hinh 2: DON GIAN (degree=1 - hop ly)
print("Tao mo hinh DON GIAN (degree=1)...\n")
model_simple = fit_model(1)


# ============================================================
# BUOC 3: DANH GIA MO HINH
# ============================================================

# Tinh R2 score cho moi mo hinh
# R2 gan 1.0 = tot, R2 am = rat te
train_r2_of = r2_score(y_train, model_overfit.predict(X_train))
test_r2_of = r2_score(y_test, model_overfit.predict(X_test))

train_r2_s = r2_score(y_train, model_simple.predict(X_train))
test_r2_s = r2_score(y_test, model_simple.predict(X_test))

# In ket qua
print("=" * 60)
print("KET QUA DANH GIA")
print("=" * 60)
print(f"{'':20} {'Overfit':>12} {'Simple':>12}")
print(f"{'Train R2':20} {train_r2_of:>12.4f} {train_r2_s:>12.4f}")
print(f"{'Test R2':20} {test_r2_of:>12.4f} {test_r2_s:>12.4f}")
print(f"{'Chenh lech':20} {abs(train_r2_of-test_r2_of):>12.4f} {abs(train_r2_s-test_r2_s):>12.4f}")

print(f"\n[!] OVERFIT: Train R2 tot nhung Test R2 RAT TE -> Hoc vet!")
print(f"[OK] SIMPLE: Train va Test R2 gan nhau -> Hieu du lieu!")


# ============================================================
# BUOC 4: VE BIEU DO SO SANH
# ============================================================
X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, name, model, color in [
    (axes[0], "OVERFIT (degree=15)", model_overfit, 'red'),
    (axes[1], "DON GIAN (degree=1)", model_simple, 'green')
]:
    # Ve du lieu
    ax.scatter(X_train, y_train, alpha=0.3, s=20, c='steelblue', label='Train')
    ax.scatter(X_test, y_test, alpha=0.3, s=20, c='orange', label='Test')

    # Ve duong du doan
    y_pred = model.predict(X_range)
    ax.plot(X_range, y_pred, color=color, linewidth=2, label=f'Pred ({name})')

    # Tinh va hien thi R2
    tr = r2_score(y_train, model.predict(X_train))
    te = r2_score(y_test, model.predict(X_test))
    gap = abs(tr - te)

    ax.set_title(f'{name}\nTrain R2={tr:.4f} | Test R2={te:.4f}\n'
                 f'Chenh lech={gap:.4f}', fontsize=11, color=color)
    ax.set_xlabel('Dien tich (sqft_living)', fontsize=10)
    ax.set_ylabel('Gia nha (USD)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfit_comparison.png', dpi=100)
print(f"\n[OK] Da luu: overfit_comparison.png")
print("\n=> Xem anh de thay su khac biet giua 2 mo hinh!")
print("=> Mo hinh Overfit 'cong' vao moi diem du lieu, khong phai thuc te!")
