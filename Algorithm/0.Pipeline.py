import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

x = np.random.uniform(-3, 3, 100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + 2 * x + 3 + np.random.normal(0, 1, 100)

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("scaler", StandardScaler()),
    ("line_reg", LinearRegression())
])
pipe.fit(X, y)
y_predict = pipe.predict(X)

plt.figure("pipe")
plt.scatter(x, y, c='b', alpha=0.5, label="data")
plt.plot(np.sort(x), y_predict[np.argsort(x)], c='r', alpha=0.5, label="y_predict", lw=3)
plt.legend()
plt.show()
