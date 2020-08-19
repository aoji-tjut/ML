from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 参数设置
param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]

knn_clf = KNeighborsClassifier()
# 寻找最佳超参数 n_jobs使用cpu核数 verbose时刻输出信息，值越大越详细
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)  # 拟合
print("score =", grid_search.score(X_test, y_test))  # 先对X_test预测y_predict 再结合y_test进行评估 得到模型准确率
print("best_estimator_ =", grid_search.best_estimator_)  # 最佳参数
knn_clf = grid_search.best_estimator_  # 保存最佳分类器
