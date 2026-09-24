"""
bai2_gradient_descent.py
========================
BÀI 2: Tìm giá trị cực tiểu của g(x) = (1/3)x³ - x bằng Gradient Descent

Phân tích (như trong sổ):
  g(x)  = (1/3)x³ - x
  g'(x) = 3·(1/3)·x² - 1 = x² - 1 = 0  ->  x = ±1
  x = 1  -> g(1) = -2/3  (cuc tieu)
  x = -1 -> g(-1) = 2/3  (cuc dai)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Ham so g(x) = (1/3)x^3 - x
def cost(x):
    return (1/3) * x**3 - x


# Dao ham g'(x) = x^2 - 1
def grad(x):
    return x * x - 1


# Gradient Descent
# x0: diem khoi tao, eta: learning rate
def myGD(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])   # Cap nhat: x = x - eta * g'(x)
        if abs(grad(x_new)) < 1e-3:          # Dung khi g'(x) gan 0
            break
        x.append(x_new)
    return x, it


# ============================================================
# CHAY CHINH
# ============================================================
if __name__ == "__main__":
    # Chay Gradient Descdescent (x0 = 3 > 1 de tim cuc tieu tai x = 1)
    x, it = myGD(x0=3.0, eta=0.05)

    # In ket qua tung buoc
    print("Buoc |       x |     g(x) |    g'(x)")
    print("-" * 45)
    for i, xi in enumerate(x):
        print(f"{i:>4} | {xi:>7.4f} | {cost(xi):>8.4f} | {grad(xi):>8.4f}")

    # Ket qua
    print(f"\nKET QUA:")
    print(f"  So buoc lap: {it}")
    print(f"  x* = {x[-1]:.6f}  (ky vong: 1)")
    print(f"  g(x*) = {cost(x[-1]):.6f}  (ky vong: -0.6667 = -2/3)")

    # Ve hinh
    xs = np.linspace(-3, 3, 300)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, cost(xs), 'b-', linewidth=2, label='g(x) = (1/3)x³ - x')
    plt.plot(x, [cost(xi) for xi in x], 'ro-', markersize=5, label='Cac buoc GD')
    plt.plot(1, cost(1), 'g*', markersize=18, label='Cuc tieu (1, -2/3)')
    plt.plot(-1, cost(-1), 'm*', markersize=18, label='Cuc dai (-1, 2/3)')
    plt.xlabel('x'); plt.ylabel('g(x)')
    plt.title('Bai 2: Gradient Descent - g(x) = (1/3)x³ - x')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hinh_anh/bai2_gradient_descent.png', dpi=100)
    print("\n[OK] Da luu: hinh_anh/bai2_gradient_descent.png")
