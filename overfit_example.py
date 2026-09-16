"""
overfit_example.py - Vi du OVERFITTING
Mo hinh degree=15 qua phuc tap -> "hoc vet" du lieu training
"""

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from data_generator import load_data, get_X_y

# Tai du lieu Kaggle
df = load_data()
X, y = get_X_y(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tao 2 mo hinh:
#   degree=15 -> qua phuc tap (OVERFIT)
#   degree=1  -> don gian (OK)
def fit(degree):
    m = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
    m.fit(X_train, y_train)
    return m

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
    ax.set_title(f'{name}\nTrain={tr:.4f} | Test={te:.4f} | Lech={abs(tr-te):.4f}', color=color)
    ax.set_xlabel('sqft_living'); ax.set_ylabel('price (USD)')
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfit_comparison.png', dpi=100)
print("Da luu: overfit_comparison.png")

# In ket qua
tr_of, te_of = r2_score(y_train, model_of.predict(X_train)), r2_score(y_test, model_of.predict(X_test))
tr_ok, te_ok = r2_score(y_train, model_ok.predict(X_train)), r2_score(y_test, model_ok.predict(X_test))
print(f"\n{'':20} {'Overfit':>12} {'Simple':>12}")
print(f"{'Train R2':20} {tr_of:>12.4f} {tr_ok:>12.4f}")
print(f"{'Test R2':20} {te_of:>12.4f} {te_ok:>12.4f}")
print(f"\n[!] Overfit: Train tot, Test kem -> Hoc vet!")
print(f"[OK] Simple: Train va Test gan nhau -> Tot!")
