"""
bai1_gradient_descent.py
========================
BÀI 1: Tìm giá trị cực tiểu của f(x) = x² - 2 bằng Gradient Descent

Phân tích (như trong sổ):
  f(x)  = x² - 2
  f'(x) = 2x = 0  ->  x = 0  (cực tiểu, f(0) = -2)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Ham so f(x) = x^2 - 2
def cost(x):
    return x * x - 2


# Dao ham f'(x) = 2x
def grad(x):
    return 2 * x


# Gradient Descent
# x0: diem khoi tao, eta: learning rate
def myGD(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])   # Cap nhat: x = x - eta * f'(x)
        if abs(grad(x_new)) < 1e-3:          # Dung khi f'(x) gan 0
            break
        x.append(x_new)
    return x, it


# ============================================================
# CHAY CHINH
# ============================================================
if __name__ == "__main__":
    # Chay Gradient Descent
    x, it = myGD(x0=5.0, eta=0.1)

    # In ket qua tung buoc
    print("Buoc |       x |     f(x) |    f'(x)")
    print("-" * 45)
    for i, xi in enumerate(x):
        print(f"{i:>4} | {xi:>7.4f} | {cost(xi):>8.4f} | {grad(xi):>8.4f}")

    # Ket qua
    print(f"\nKET QUA:")
    print(f"  So buoc lap: {it}")
    print(f"  x* = {x[-1]:.6f}  (ky vong: 0)")
    print(f"  f(x*) = {cost(x[-1]):.6f}  (ky vong: -2)")

    # Ve hinh
    xs = np.linspace(-6, 6, 200)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, cost(xs), 'b-', linewidth=2, label='f(x) = x² - 2')
    plt.plot(x, [cost(xi) for xi in x], 'ro-', markersize=5, label='Cac buoc GD')
    plt.plot(0, -2, 'g*', markersize=18, label='Cuc tieu (0, -2)')
    plt.xlabel('x'); plt.ylabel('f(x)')
    plt.title('Bai 1: Gradient Descent - f(x) = x² - 2')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hinh_anh/bai1_gradient_descent.png', dpi=100)
    print("\n[OK] Da luu: hinh_anh/bai1_gradient_descent.png")
