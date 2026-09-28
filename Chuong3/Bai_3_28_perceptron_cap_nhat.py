import numpy as np


print("=== Bài 3.28: Perceptron cập nhật trọng số ===")
w = np.array([-2, 1, 0], dtype=float)
x = np.array([2, 3, 1], dtype=float)
y = 1
eta = 1.0

z = np.dot(w, x)
y_pred = 1 if z >= 0 else -1
print("w^T x = {}, dự đoán = {}, thực tế = {}".format(z, y_pred, y))

if y_pred != y:
    print("=> Phân lớp SAI, thực hiện cập nhật:")
    w_new = w + eta * y * x
    z_new = np.dot(w_new, x)
    print("w_mới = w + y*x = {}".format(w_new))
    print("w_mới^T x = {}".format(z_new))
else:
    print("=> Phân lớp ĐÚNG, không cần cập nhật.")