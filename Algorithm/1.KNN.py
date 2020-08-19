import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 分类------------------------------------------------------------------------------------------------------------------
X = np.array([[3, 2], [3, 1], [1, 3], [3, 4], [2, 2], [7, 4], [5, 3], [9, 2], [7, 3], [7, 0]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

knn_clf = KNeighborsClassifier()  # 默认k=5
knn_clf.fit(X, y)
print("[3,3]类别：", knn_clf.predict([[3, 3]]))
print("[8,3]类别：", knn_clf.predict([[8, 3]]))

plt.figure("KNN Classifier")
plt.scatter(X[:, 0], X[:, 1], c='r')
plt.scatter(3, 3, c='b')
plt.scatter(8, 3, c='b')
plt.show()

# 回归------------------------------------------------------------------------------------------------------------------
boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50.0]
y = y[y < 50.0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_test, y_test)
print("score =", knn_reg.score(X_test, y_test))

'''
KNeighborsClassifier

参数:
1.n_neighbors: 一个整数，指定k值。
2.weights: 一字符串或者可调用对象，指定投票权重类型。即这些邻居投票权可以为相同或者不同。 
   uniform: 本节点的所有邻居节点的投票权重都相等。
   distance: 本节点的所有邻居节点的投票权重与距离成反比，即越近节点，其投票权重越大。
   [callable]: 一个可调用对象，它传入距离的数组，返回同样形状的权重数组。
3.algorithm: 一个字符串，指定最近邻的算法，可以为下： 
   ball_tree: 使用BallTree算法。
   kd_tree: 使用KDTree算法。
   brute: 使用暴力搜索算法。
   auto: 自动决定最合适算法。
4.leaf_size: 一个整数，指定BallTree或者KDTree叶节点的规模。它影响树的构建和查询速度。
5.metric: 一个字符串，指定距离度量。默认为‘minkowski’(闵可夫斯基)距离。
6.p: 整数值。 
   p=1： 对应曼哈顿距离。
   p=2: 对应欧氏距离。
7.n_jobs: 并行性。默认为-1表示派发任务到所有计算机的CPU上。

方法: 
1.fit(X,y): 训练模型。
2.predict(X): 预测模型。
3.score(X,y): 返回在(X,y)上预测的准确率(accuracy)。
4.predict_proba(X): 返回样本为每种标记的概率。
5.kneighbors([X,n_neighbors,return_distace]): 返回样本点的k邻近点。
  如果return_distance=True，同时还返回到这些近邻点的距离。
6.kneighbors_graph([X,n_neighbors,mode]): 返回样本点的连接图。
'''

'''
KNeighborsRegressor

参数: 
1.n_neighbors: 一个整数，指定k值。
2.weights: 一字符串或者可调用对象，指定投票权重类型。即这些邻居投票权可以为相同或者不同。 
   uniform: 本节点的所有邻居节点的投票权重都相等。
   distance: 本节点的所有邻居节点的投票权重与距离成反比，即越近节点，其投票权重越大。
   [callable]: 一个可调用对象，它传入距离的数组，返回同样形状的权重数组。
3.algorithm: 一个字符串，指定最近邻的算法，可以为下： 
   ball_tree: 使用BallTree算法。
   kd_tree: 使用KDTree算法。
   brute: 使用暴力搜索算法。
   auto: 自动决定最合适算法。
4.leaf_size: 一个整数，指定BallTree或者KDTree叶节点的规模。它影响树的构建和查询速度。
5.metric: 一个字符串，指定距离度量。默认为‘minkowski’(闵可夫斯基)距离。
6.p: 整数值。 
   p=1： 对应曼哈顿距离。
   p=2: 对应欧氏距离。
7.n_jobs: 并行性。默认为-1表示派发任务到所有计算机的CPU上。

方法: 
1.fit(X,y): 训练模型。
2.predict(X): 预测模型。
3.score(X,y): 返回在(X,y)上预测的准确率(accuracy)。
4.predict_proba(X): 返回样本为每种标记的概率。
5.kneighbors([X,n_neighbors,return_distace]): 返回样本点的k邻近点。
  如果return_distance=True，同时还返回到这些近邻点的距离。
6.kneighbors_graph([X,n_neighbors,mode]): 返回样本点的连接图。
'''
