import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

digits = datasets.load_digits()  # [1797, 64]
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 二维可视化处理
pca_n2 = PCA(n_components=2)
pca_n2.fit(X)
X_pca_n2 = pca_n2.transform(X)
plt.figure("2D digits")
for i in range(10):
    plt.scatter(X_pca_n2[y == i, 0], X_pca_n2[y == i, 1])

# 三维可视化处理
pca_n3 = PCA(n_components=3)
pca_n3.fit(X)
X_pca_n3 = pca_n3.transform(X)
fig = plt.figure("3D digits")
ax = Axes3D(fig)
for i in range(10):
    ax.scatter(X_pca_n3[y == i, 0], X_pca_n3[y == i, 1], X_pca_n3[y == i, 2])

# 选择最佳n_components
pca_n64 = PCA(64)
pca_n64.fit(X_train)
plt.figure("pca_n64.explained_variance_ratio_ ")
# 横坐标为1-64个特征 纵坐标为前i个特征的方差比例总和
plt.plot([i for i in range(X_train.shape[1])],
         [np.sum(pca_n64.explained_variance_ratio_[:i + 1]) for i in range(X_train.shape[1])])
for i in range(X_train.shape[1]):
    print("前%d个维度方差比例 = %.2f" % (i + 1, float(np.sum(pca_n64.explained_variance_ratio_[:i + 1]) * 100)))

# 普通knn
knn_clf = KNeighborsClassifier()
t = time.time()
knn_clf.fit(X_train, y_train)
print("normal knn\ttime = %.2fms, score = %f" % ((time.time() - t) * 1000, knn_clf.score(X_test, y_test)))

# 40维knn
pca_n40 = PCA(n_components=40)  # 降到40维
pca_n40.fit(X_train)
print("方差比例总和:", np.sum(pca_n40.explained_variance_ratio_))
X_train_pca_n40 = pca_n40.transform(X_train)
X_test_pca_n40 = pca_n40.transform(X_test)
knn_clf = KNeighborsClassifier()
t = time.time()
knn_clf.fit(X_train_pca_n40, y_train)
print("pca_n40 knn\ttime = %.2fms, score = %f" % ((time.time() - t) * 1000, knn_clf.score(X_test_pca_n40, y_test)))

# 保留99%方差knn
pca_r99 = PCA(n_components=0.99)  # 保留99%的方差
pca_r99.fit(X_train)
print("特征总数:", pca_r99.n_components_)
X_train_pca_r99 = pca_r99.transform(X_train)
X_test_pca_r99 = pca_r99.transform(X_test)
knn_clf = KNeighborsClassifier()
t = time.time()
knn_clf.fit(X_train_pca_r99, y_train)
print("pca_r99 knn\ttime = %.2fms, score = %f" % ((time.time() - t) * 1000, knn_clf.score(X_test_pca_r99, y_test)))

plt.show()

# X = np.zeros((100, 2))
# noise = np.random.uniform(-5, 5, size=100)
# X[:, 0] = np.random.uniform(0, 100, size=100)
# X[:, 1] = 0.5 * X[:, 0] + 2 + noise
#
# pca = PCA(n_components=1)  # 主成分个数
# pca.fit(X)
# print("pca.components_ =", pca.components_)  # 方向
# X_reduction = pca.transform(X)  # 降维X
# x_restore = pca.inverse_transform(X_reduction)  # 恢复X_reduction
#
# plt.figure("PCA")
# plt.scatter(X[:, 0], X[:, 1], color='b', alpha=0.5)
# plt.plot(x_restore[:, 0], x_restore[:, 1], color='g')
# plt.scatter(x_restore[:, 0], x_restore[:, 1], color='r')
# plt.show()
