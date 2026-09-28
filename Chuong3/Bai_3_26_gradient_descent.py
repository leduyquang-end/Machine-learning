def f(x):
    return x**2 - 4 * x + 5


def df(x):
    return 2 * x - 4


x = 5.0
eta = 0.2
n_steps = 4

print("=== Bài 3.26: Gradient Descent ===")
print("f'(x) = 2x - 4")
print("{:<6}{:<12}{:<12}{:<12}".format("Bước", "x", "f(x)", "f'(x)"))
print("{:<6}{:<12.6f}{:<12.6f}{:<12.6f}".format(0, x, f(x), df(x)))

history = [(0, x, f(x))]
for i in range(1, n_steps + 1):
    grad = df(x)
    x = x - eta * grad
    history.append((i, x, f(x)))
    print("{:<6}{:<12.6f}{:<12.6f}{:<12.6f}".format(i, x, f(x), df(x)))

x_star = 2.0
print("\nCực tiểu lý thuyết tại x* = 2, f(x*) = {}".format(f(x_star)))
print("Sau {} bước, x = {:.6f}, f(x) = {:.6f}".format(n_steps, x, f(x)))
print("=> Thuật toán hội tụ về điểm cực tiểu (do η nhỏ hơn 1/|f''| = 1/2).")