import numpy as np


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


if __name__ == "__main__":
    print("=== Bài 3.29: Demo lớp Perceptron ===")
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = iris.data[:100, :2]
    y = np.where(iris.target[:100] == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = Perceptron(learning_rate=0.01, n_iters=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy trên tập test: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("w = {}, b = {}".format(clf.w, clf.b))