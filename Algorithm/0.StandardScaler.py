import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train = np.arange(1, 121)  # 1-120个训练集
test = np.arange(1, 31)  # 1-30个测试集

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()  # 实例化
scaler.fit(X_train)  # 拟合X_train
print("mean_ =", scaler.mean_)  # 均值
print("scale_ =", scaler.scale_)  # 标准差
X_train_scaler = scaler.transform(X_train)  # 返回归一化结果 X_train本身不改变
X_test_scaler = scaler.transform(X_test)  # 测试集也用训练集拟合的模型进行归一化

plt.figure("X_train_scaler")
plt.scatter(train, X_train_scaler[:, 0], alpha=0.4)
plt.scatter(train, X_train_scaler[:, 1], alpha=0.4)
plt.scatter(train, X_train_scaler[:, 2], alpha=0.4)
plt.scatter(train, X_train_scaler[:, 3], alpha=0.4)
plt.figure("X_test_scaler")
plt.scatter(test, X_test_scaler[:, 0], alpha=0.4)
plt.scatter(test, X_test_scaler[:, 1], alpha=0.4)
plt.scatter(test, X_test_scaler[:, 2], alpha=0.4)
plt.scatter(test, X_test_scaler[:, 3], alpha=0.4)
plt.show()
