"""
bai2_gradient_descent.py
========================
BÀI 2: Tìm giá trị cực tiểu của hàm số g(x) = (1/3)x³ - x
bằng thuật toán Gradient Descent.

Công thức đạo hàm: (a·xⁿ)' = n·(a·xⁿ⁻¹)
  g'(x) = 3·(1/3)·x² - 1·x⁰ = x² - 1

Lưu ý: g(x) là hàm bậc 3, có 2 điểm critical (g'(x)=0):
  x = 1  -> g(1)  = -2/3  (cực tiểu)
  x = -1 -> g(-1) =  2/3  (cực đại)
Gradient Descent sẽ tìm CỰC TIỂU (x = 1) nếu chọn điểm khởi tạo phù hợp.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# HÀM SỐ VÀ ĐẠO HÀM
# ============================================================
def g(x):
    """Hàm số g(x) = (1/3)x³ - x"""
    return (1/3) * x**3 - x


def g_prime(x):
    """Đạo hàm g'(x) = x² - 1"""
    return x**2 - 1


# ============================================================
# GRADIENT DESCENT
# ============================================================
def gradient_descent(lr=0.05, x0=3.0, epochs=50, tol=1e-6):
    """
    Gradient Descent tìm cực tiểu g(x).

    Parameters:
    -----------
    lr     : learning_rate
    x0     : điểm khởi tạo (chon > 1 de tim cuc tieu tai x=1)
    epochs : số lần lặp tối đa
    tol    : ngưỡng dừng

    Returns:
    --------
    history : danh sách (x, g(x)) qua từng bước
    """
    x = x0
    history = [(x, g(x))]
    print(f"{'Buoc':>5} {'x':>12} {'g(x)':>12} {'g_der':>12}")
    print("-" * 45)

    for i in range(epochs):
        grad = g_prime(x)          # Tính đạo hàm
        x = x - lr * grad          # Cập nhật x = x - lr·g'(x)
        history.append((x, g(x)))

        print(f"{i+1:>5} {x:>12.6f} {g(x):>12.6f} {g_prime(x):>12.6f}")

        if abs(g_prime(x)) < tol:
            print(f"\n>> Dừng ở buoc {i+1}: |g'(x)| < {tol}")
            break

    return history


# ============================================================
# VẼ HÌNH
# ============================================================
def plot_result(history):
    """Vẽ hàm số và các bước Gradient Descent"""
    xs = np.linspace(-3, 3, 300)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, g(xs), 'b-', linewidth=2, label="g(x) = (1/3)x³ - x")

    # Các bước GD
    hx = [h[0] for h in history]
    hy = [h[1] for h in history]
    plt.plot(hx, hy, 'ro-', markersize=6, label='Cac buoc Gradient Descent')

    # Đánh dấu các điểm critical
    plt.plot(1, g(1), 'g*', markersize=20, label='Cuc tieu (1, -2/3)')
    plt.plot(-1, g(-1), 'm*', markersize=20, label='Cuc dai (-1, 2/3)')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('g(x)', fontsize=12)
    plt.title("Bai 2: Gradient Descent tim cuc tieu g(x) = (1/3)x³ - x", fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hinh_anh/bai2_gradient_descent.png', dpi=100)
    print("\n[OK] Da luu: hinh_anh/bai2_gradient_descent.png")


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("BÀI 2: g(x) = (1/3)x³ - x  |  g'(x) = x² - 1")
    print("=" * 50)

    history = gradient_descent(lr=0.05, x0=3.0, epochs=50)

    # Kết quả cuối
    x_final, gx_final = history[-1]
    print(f"\nKET QUA:")
    print(f"  Gia tri cuc toi x* = {x_final:.6f}  (ky vong: 1)")
    print(f"  g(x*) = {gx_final:.6f}           (ky vong: -0.666667 = -2/3)")

    plot_result(history)
