from collections import Counter
from math import log2

# Bảng dữ liệu Play Tennis chuẩn của bài ID3
DATA = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play": "No"},
]

FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]
TARGET = "Play"


def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0
    result = 0.0
    for count in counts.values():
        p = count / total
        result -= p * log2(p)
    return result


def majority_value(rows):
    labels = [row[TARGET] for row in rows]
    return Counter(labels).most_common(1)[0][0]


def info_gain(rows, attribute):
    total_entropy = entropy([row[TARGET] for row in rows])
    values = {row[attribute] for row in rows}
    weighted_entropy = 0.0

    for value in values:
        subset = [row for row in rows if row[attribute] == value]
        weighted_entropy += (len(subset) / len(rows)) * entropy([row[TARGET] for row in subset])

    return total_entropy - weighted_entropy


def choose_best_feature(rows, features):
    gains = {feature: info_gain(rows, feature) for feature in features}
    return max(gains, key=gains.get)


def build_tree(rows, features):
    labels = [row[TARGET] for row in rows]

    if len(set(labels)) == 1:
        return labels[0]

    if not features:
        return majority_value(rows)

    best = choose_best_feature(rows, features)
    tree = {best: {}}
    remaining = [f for f in features if f != best]

    for value in sorted({row[best] for row in rows}):
        subset = [row for row in rows if row[best] == value]
        if not subset:
            tree[best][value] = majority_value(rows)
        else:
            tree[best][value] = build_tree(subset, remaining)

    return tree


def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    for key, subtree in tree.items():
        value = sample[key]
        if value in subtree:
            return predict(subtree[value], sample)
        return majority_value([sample])

    return None


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f"{indent}{tree}")
        return

    for key, subtree in tree.items():
        print(f"{indent}{key}:")
        for value, child in subtree.items():
            print(f"{indent}  {value} -> ", end="")
            if isinstance(child, dict):
                print()
                print_tree(child, indent + "    ")
            else:
                print(child)


if __name__ == "__main__":
    tree = build_tree(DATA, FEATURES)
    print("Cây quyết định ID3 cho bài Play Tennis:\n")
    print_tree(tree)

    print("\nDự đoán mẫu mới:")
    sample = {
        "Outlook": "Sunny",
        "Temperature": "Cool",
        "Humidity": "Normal",
        "Wind": "Strong"
    }
    print(sample)
    print("Kết quả dự đoán:", predict(tree, sample))
