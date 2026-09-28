import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000, random_state=42):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None
        self.b = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape
        self.w = rng.normal(size=n_features) * 0.01
        self.b = 0.0

        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            errors = 0
            for xi, yi in zip(X, y_):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1
            if errors == 0:
                break

    def predict(self, X):
        linear = np.dot(X, self.w) + self.b
        return np.where(linear >= 0, 1, -1)


print("=== Bài 3.30: Perceptron trên Breast Cancer Dataset ===")
data = load_breast_cancer()
X, y = data.data, np.where(data.target == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = Perceptron(learning_rate=0.01, n_iters=1000, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("Accuracy : {:.4f}".format(acc))
print("Precision: {:.4f}".format(pre))
print("Recall   : {:.4f}".format(rec))
print("F1-score : {:.4f}".format(f1))
print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Malignant (-1)", "Benign (+1)"]
    )
)