import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x = np.random.uniform(-3, 3, 100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + 2 * x + 3 + np.random.normal(0, 1, 100)
plt.figure("Polynomial Features", (10, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# poly+line_reg---------------------------------------------------------------------------------------------------------
poly = PolynomialFeatures(degree=2)  # 多项式次数
poly.fit(X)
X_poly = poly.transform(X)  # 增维转换为多项式特征 X_poly[:,0]为x的0次幂,X_poly[:,1]为x的1次幂,X_poly[:,2]为x的2次幂
line_reg = LinearRegression()
line_reg.fit(X_poly, y)
y_predict = line_reg.predict(X_poly)
print("x^2系数 = %f\nx^1系数 = %f\nx^0系数 = %f\n常数 = %f" % (
    line_reg.coef_[2], line_reg.coef_[1], line_reg.coef_[0], line_reg.intercept_))

plt.sca(ax1)
plt.title("poly+line_reg")
plt.scatter(x, y, c='b', alpha=0.5, label="data")
plt.plot(np.sort(x), y_predict[np.argsort(x)], c='r', alpha=0.5, label="y_predict", lw=3)
plt.legend()

# 管道-------------------------------------------------------------------------------------------------------------------
pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("line_reg", LinearRegression())
])
pipe.fit(X, y)
y_predict = pipe.predict(X)

plt.sca(ax2)
plt.title("pipe")
plt.scatter(x, y, c='b', alpha=0.5, label="data")
plt.plot(np.sort(x), y_predict[np.argsort(x)], c='r', alpha=0.5, label="y_predict", lw=3)
plt.legend()

plt.show()
