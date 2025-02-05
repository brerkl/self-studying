import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class LosgisticRegression: 
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate # Cho mình biết được hướng mình đi nhanh hay chậm trong quá trình cập nhật w và b giảm thiểu sai số
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0

    def predict(self, X): 
        prediction = sigmoid(np.dot(X, self.weights) + self.bias)
        return np.array([[1 if i > 0.5 else 0 for i in prediction]]).reshape(40, 1)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape # Lấy số lượng mẫu và số lượng đặc trưng của tập dữ liệu 
        self.weights = np.zeros(n_features) # Đặt kích thước trọng số (w) theo số lượng đặc trưng từ tập dữ liệu
        self.bias = 0 # bias
        
        # Vòng lặp tìm w và bias để tối thiểu sai số lớn nhất
        for _ in range(self.n_iterations):
            
            # Tìm sai số bằng cách làm dự đoán
            predictions = sigmoid(np.dot(X, self.weights) + self.bias)

            # Để cập nhật trọng số và bias mới, mình tính đạo hàm, hoặc độ dốc của sai số bình phương trung bình (MSE)
            dw = (-2/n_samples) * np.dot(X.T, (y - predictions))
            db = (-2/n_samples) * np.sum(y - predictions)

            # Sau khi tính đạo hàm của hàm mất mát, mình cập nhất trọng số (w) và bias (b) mới
            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db
        
        return self.weights, self.bias