import numpy as np


print("=== Bài 3.27: Perceptron kiểm tra 1 điểm ===")
w = np.array([1, 2, -10], dtype=float)
x = np.array([3, 4, 1], dtype=float)
y_true = -1

z = np.dot(w, x)
y_pred = 1 if z >= 0 else -1

print("w = {}".format(w))
print("x = {}".format(x))
print("w^T x = {}".format(z))
print("Nhãn dự đoán: {}".format(y_pred))
if y_pred != y_true:
    print("Nhãn thực tế y = {} => Điểm bị PHÂN LỚP SAI.".format(y_true))
else:
    print("Nhãn thực tế y = {} => Điểm phân lớp ĐÚNG.".format(y_true))