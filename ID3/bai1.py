from collections import Counter
from math import log2


# Dữ liệu mẫu theo kiểu bài toán ID3 (phân lớp Yes/No)
# Có 15 mẫu, 4 thuộc tính: Thời tiết, Nhiệt độ, Độ ẩm, Gió
# Thuộc tính nhãn: KetQua
DATA = [
    {"ThoiTiet": "Nang", "NhietDo": "Nhiet", "DoAm": "Cao", "Gio": "Yeu", "KetQua": "No"},
    {"ThoiTiet": "Nang", "NhietDo": "Nhiet", "DoAm": "Cao", "Gio": "Manh", "KetQua": "No"},
    {"ThoiTiet": "Am", "NhietDo": "Nhiet", "DoAm": "Cao", "Gio": "Yeu", "KetQua": "Yes"},
    {"ThoiTiet": "Mua", "NhietDo": "TrungBinh", "DoAm": "Cao", "Gio": "Yeu", "KetQua": "Yes"},
    {"ThoiTiet": "Mua", "NhietDo": "Lat", "DoAm": "Thuong", "Gio": "Yeu", "KetQua": "Yes"},
    {"ThoiTiet": "Mua", "NhietDo": "Lat", "DoAm": "Thuong", "Gio": "Manh", "KetQua": "No"},
    {"ThoiTiet": "Am", "NhietDo": "Lat", "DoAm": "Thuong", "Gio": "Manh", "KetQua": "Yes"},
    {"ThoiTiet": "Nang", "NhietDo": "TrungBinh", "DoAm": "Cao", "Gio": "Yeu", "KetQua": "No"},
    {"ThoiTiet": "Nang", "NhietDo": "Lat", "DoAm": "Thuong", "Gio": "Yeu", "KetQua": "Yes"},
    {"ThoiTiet": "Mua", "NhietDo": "TrungBinh", "DoAm": "Thuong", "Gio": "Yeu", "KetQua": "Yes"},
    {"ThoiTiet": "Nang", "NhietDo": "TrungBinh", "DoAm": "Thuong", "Gio": "Manh", "KetQua": "Yes"},
    {"ThoiTiet": "Am", "NhietDo": "TrungBinh", "DoAm": "Cao", "Gio": "Manh", "KetQua": "Yes"},
    {"ThoiTiet": "Am", "NhietDo": "Nhiet", "DoAm": "Thuong", "Gio": "Yeu", "KetQua": "Yes"},
    {"ThoiTiet": "Mua", "NhietDo": "TrungBinh", "DoAm": "Cao", "Gio": "Manh", "KetQua": "No"},
    {"ThoiTiet": "Mua", "NhietDo": "TrungBinh", "DoAm": "Thuong", "Gio": "Manh", "KetQua": "Yes"},
]


FEATURES = ["ThoiTiet", "NhietDo", "DoAm", "Gio"]
TARGET = "KetQua"


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
        sub_rows = [row for row in rows if row[attribute] == value]
        weighted_entropy += (len(sub_rows) / len(rows)) * entropy([row[TARGET] for row in sub_rows])

    return total_entropy - weighted_entropy


def choose_best_feature(rows, features):
    gains = {feature: info_gain(rows, feature) for feature in features}
    return max(gains, key=gains.get)


def build_tree(rows, features):
    labels = [row[TARGET] for row in rows]

    # Nếu tất cả cùng nhãn thì dừng
    if len(set(labels)) == 1:
        return labels[0]

    # Nếu không còn thuộc tính để chia thì trả về đa số
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

    for key, value in tree.items():
        if isinstance(value, dict):
            attr_value = sample[key]
            if attr_value in value:
                return predict(value[attr_value], sample)
            return majority_value([sample])
        return value

    return None


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f"{indent}{tree}")
        return

    for key, subtree in tree.items():
        print(f"{indent}{key}:")
        for val, next_node in subtree.items():
            print(f"{indent}  {val} -> ", end="")
            if isinstance(next_node, dict):
                print()
                print_tree(next_node, indent + "    ")
            else:
                print(next_node)


if __name__ == "__main__":
    tree = build_tree(DATA, FEATURES)
    print("Cây quyết định ID3:\n")
    print_tree(tree)

    print("\nDự đoán mẫu mới:")
    sample = {
        "ThoiTiet": "Nang",
        "NhietDo": "Lat",
        "DoAm": "Thuong",
        "Gio": "Yeu"
    }
    result = predict(tree, sample)
    print(sample)
    print("Ket qua du doan:", result)
