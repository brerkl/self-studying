from collections import Counter
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,value=None):
        self.feature = feature # index đặc trưng để phân chia cây
        self.threshold = threshold # giá trị ngưỡng để phân chia cây
        self.left = left # nhánh cây trái 
        self.right = right # nhánh cây phải
        self.value = value # nút lá
        
    def is_leaf_node(self):
        # kiểm tra xem nút lá có rỗng hay không
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split # số lượng mẫu/điểm dữ liệu cần thiết để phân chia nhánh cây
        self.max_depth=max_depth # giới hạn độ dài của cây để tránh overfitting
        self.n_features=n_features # số lượng đặc trưng
        self.root=None # nút nguồn bằng rỗng (depth = 0)

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features) # lưu trữ số lượng đặc trưng
        self.root = self.building_tree(X, y) # xây dựng cây quyết định (bắt đầu với nút nguồn)

    def building_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape # số lượng mẫu/điểm dữ liệu (dòng), số lượng đặc trưng (cột)
        n_labels = len(set(y)) # số lượng nhãn duy nhất trong biến mục tiêu

        # kiểm tra điều kiện dừng khi: 
        # - độ sâu của cây đã đạt
        # - số lượng nhãn trong biến mục tiêu chỉ cò 1 nhãn
        # - hoặc khi số lượng mẫu/điểm dữ liệu trong tập nhỏ hơn 2
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self.label_from_majority(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # tìm đặc trưng và ngưỡng phù hợp để phân chia dữ liệu
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # đặc trưng và ngưỡng tìm được tạo thành 2 nhánh cây bên trái và bên phải
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self.building_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self.building_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        # vòng lặp đi qua từng đặc trưng và các giá trị ngưỡng duy nhất của chúng
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # tình thông tin nhận vào
                gain = self.information_gain(y, X_column, thr)

                # chọn đặc trưng và ngưỡng tốt nhất với thông tin nhận vào cao nhất
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def information_gain(self, y, X_column, threshold):
        # tính entropy trước và sau khi phân chia nhánh cây
        parent_entropy = self.entropy(y)

        # tạo ra nhánh cây trái và phải
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # tính trọng số trung bình của các entropy con theo số lượng lớp trong đặc trưng
        n = len(y)
        n_lef, n_rig = len(left_idxs), len(right_idxs)
        e_lef, e_rig = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_lef/n) * e_lef + (n_rig/n) * e_rig

        # tính thông tin nhận vào
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        # chia tập dữ liệu thành các nhánh cây trái và phải dựa trên ngưỡng tốt nhất
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def entropy(self, y):
        # tính độ vẩn đục trong 1 đặc trưng cụ thể
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def label_from_majority(self, y):
        # gắn nhãn xuất hiện thường xuyên nhất trong tập vào nút lá
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        # đi qua từng vòng mẫu/điểm dữ liệu và gọi hàm chức năng "_traverse_tree" để phân loại
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # 2. khi đạt đền nút lá, trả về kết quả phân loại
        if node.is_leaf_node():
            return node.value

        # 1. đi qua từng nút dựa trên đầu vào X 
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)