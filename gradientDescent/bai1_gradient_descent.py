"""
bai1_gradient_descent.py
========================
BÀI 1: Tìm giá trị cực tiểu của hàm số f(x) = x² - 2
bằng thuật toán Gradient Descent.

Công thức đạo hàm: (a·xⁿ)' = n·(a·xⁿ⁻¹)
=> f'(x) = 2x

Gradient Descent:
  x_mới = x_cũ - learning_rate · f'(x_cũ)
  Lặp đến khi |f'(x)| nhỏ hơn ngưỡng cho trước.

Kết quả mong đợi: x* ≈ 0 (cực tiểu), f(x*) ≈ -2
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# HÀM SỐ VÀ ĐẠO HÀM
# ============================================================
def f(x):
    """Hàm số f(x) = x² - 2"""
    return x**2 - 2


def f_prime(x):
    """Đạo hàm f'(x) = 2x"""
    return 2 * x


# ============================================================
# GRADIENT DESCENT
# ============================================================
def gradient_descent(lr=0.1, x0=5.0, epochs=50, tol=1e-6):
    """
    Gradient Descent tìm cực tiểu f(x).

    Parameters:
    -----------
    lr     : learning_rate (tốc độ học)
    x0     : điểm khởi tạo
    epochs : số lần lặp tối đa
    tol    : ngưỡng dừng (khi |f'(x)| < tol)

    Returns:
    --------
    history : danh sách các cặp (x, f(x)) qua từng bước
    """
    x = x0
    history = [(x, f(x))]
    print(f"{'Buoc':>5} {'x':>12} {'f(x)':>12} {'f_der':>12}")
    print("-" * 45)

    for i in range(epochs):
        grad = f_prime(x)          # Tính đạo hàm (gradient)
        x = x - lr * grad          # Cập nhật x: x = x - lr·f'(x)
        history.append((x, f(x)))

        print(f"{i+1:>5} {x:>12.6f} {f(x):>12.6f} {f_prime(x):>12.6f}")

        # Dừng khi đạo hàm gần bằng 0 (đã tới cực tiểu)
        if abs(f_prime(x)) < tol:
            print(f"\n>> Dừng ở buoc {i+1}: |f'(x)| < {tol}")
            break

    return history


# ============================================================
# VẼ HÌNH
# ============================================================
def plot_result(history):
    """Vẽ hàm số và các bước Gradient Descent"""
    xs = np.linspace(-6, 6, 200)

    plt.figure(figsize=(9, 5))
    plt.plot(xs, f(xs), 'b-', linewidth=2, label='f(x) = x² - 2')

    # Vẽ các bước GD
    hx = [h[0] for h in history]
    hy = [h[1] for h in history]
    plt.plot(hx, hy, 'ro-', markersize=6, label='Cac buoc Gradient Descent')

    # Đánh dấu cực tiểu
    plt.plot(0, -2, 'g*', markersize=20, label='Cuc tieu (0, -2)')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Bai 1: Gradient Descent tim cuc tieu f(x) = x² - 2', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hinh_anh/bai1_gradient_descent.png', dpi=100)
    print("\n[OK] Da luu: hinh_anh/bai1_gradient_descent.png")


# ============================================================
# CHẠY CHÍNH
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("BÀI 1: f(x) = x² - 2  |  f'(x) = 2x")
    print("=" * 50)

    history = gradient_descent(lr=0.1, x0=5.0, epochs=50)

    # Kết quả cuối
    x_final, fx_final = history[-1]
    print(f"\nKET QUA:")
    print(f"  Gia tri cuc toi x* = {x_final:.6f}  (ky vong: 0)")
    print(f"  f(x*) = {fx_final:.6f}            (ky vong: -2)")

    plot_result(history)
