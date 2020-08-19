import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDRegressor  # 只能解决线性模型
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50.0]
y = y[y < 50.0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaler = scaler.transform(X_train)
X_test_scaler = scaler.transform(X_test)

sgd_reg = SGDRegressor()  # 默认max_iter=100 迭代100次
sgd_reg.fit(X_train_scaler, y_train)
print("k =", sgd_reg.coef_)
print("b =", sgd_reg.intercept_)
print("score =", sgd_reg.score(X_test_scaler, y_test))

# # 导数
# def dJ(theta):
#     return 2 * (theta - 1)
#
#
# # 损失函数
# def J(theta):
#     return (theta - 1) ** 2 + 3
#
#
# x = np.linspace(-1, 3, 1000)
# y = (x - 1) ** 2 + 3
#
# plt.figure("SGD")
#
# eta = 0.99  # 学习率
# x_init = theta = -1.0  # x位置
# yy = np.zeros([0])
# xx = np.zeros([0])
# iter = 0
#
# while True:
#     iter = iter + 1
#     yy = np.append(yy, J(theta))
#     xx = np.append(xx, theta)
#     plt.cla()
#     plt.plot(x, y, c='b')
#     plt.plot(xx, yy, c='g')
#     plt.scatter(xx, yy, c='r')
#     plt.ylim(2, 8)
#     plt.pause(0.001)
#
#     gradient = dJ(theta)
#     last_theta = theta
#     theta = theta - eta * gradient
#
#     if (abs(J(theta) - J(last_theta)) < 1e-8):
#         print("iter =", iter)
#         break
#
# print("theta =", theta)
# print("J(theta) =", J(theta))
# plt.show()
