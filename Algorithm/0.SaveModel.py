from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.externals import joblib

boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50.0]
y = y[y < 50.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
line_reg = LinearRegression()
line_reg.fit(X_train, y_train)

joblib.dump(line_reg, "model")  # 保存
model = joblib.load("model")  # 加载

y_predict = model.predict(X_test)
print(r2_score(y_test, y_predict))
