import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_score, best_k, best_p = 0, 0, 0
for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p, n_jobs=-1)
        scores = cross_val_score(knn_clf, X_train, y_train)  # 交叉验证 cv=分组数 极端情况：留一法
        score = np.mean(scores)
        if score > best_score:
            best_score, best_k, best_p = score, k, p
print("best_score =", best_score)
print("best_k =", best_k)
print("best_p =", best_p)

#得到最佳超参数后 重新创建模型拟合预测
knn_clf_best = KNeighborsClassifier(weights="distance", n_neighbors=best_k, p=best_p)
knn_clf_best.fit(X_train, y_train)
print("knn_clf_best.score =", knn_clf_best.score(X_test, y_test))
