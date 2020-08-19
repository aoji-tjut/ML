import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[:, 1:3]
plt.figure("KMeans", (10, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

km = KMeans(n_clusters=3)  # 聚成2类
km.fit(X)
y_predict = km.predict(X)
center = km.cluster_centers_
print("silhouette_score =", silhouette_score(X, y_predict))

plt.sca(ax1)
plt.title("y")
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])

plt.sca(ax2)
plt.title("y_predict")
plt.scatter(X[y_predict == 0, 0], X[y_predict == 0, 1])
plt.scatter(X[y_predict == 1, 0], X[y_predict == 1, 1])
plt.scatter(X[y_predict == 2, 0], X[y_predict == 2, 1])
plt.scatter(center[0][0], center[0][1], c='r', marker='*')
plt.scatter(center[1][0], center[1][1], c='r', marker='*')
plt.scatter(center[2][0], center[2][1], c='r', marker='*')

plt.show()
