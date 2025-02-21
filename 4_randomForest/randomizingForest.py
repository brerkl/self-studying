from treeDecisioning import DecisionTree
from collections import Counter
import numpy as np

class RandomForest:
    def __init__(self, n_trees = 10, min_samples_split = 2, max_depth = 10, n_feature = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feature = n_feature
        self.forest = []
    
    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Sắp xếp thứ tự các mẫu/điểm dữ liệu theo một cách ngẫu nhiên (bootstrapping samples) tạo ra tập dữ liệu con
            randomzed_indices = np.random.choice(X.shape[0], X.shape[0], replace = False) 
            X_sample, y_sample = X[randomzed_indices], y[randomzed_indices]

            # Dùng thuật toán cây quyết định lên tập dữ liệu con 
            tree = DecisionTree(max_depth = self.max_depth, 
                                min_samples_split = self.min_samples_split,
                                n_features = self.n_feature)
            tree.fit(X_sample, y_sample)
            self.forest.append(tree) # thêm cây vào khu rừng

    def predict(self, X):
        # sử dụng từng cây quyết định để dự đoán đầu vào
        predictions = np.array([tree.predict(X) for tree in self.forest])

        # với dự đoán của từng cây quyết định, mình lấy nhãn dự đoán xuất hiện nhiều nhất 
        # trong số các cây tồn tại cho từng đầu vào
        final_preds = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(len(X))]
        return final_preds