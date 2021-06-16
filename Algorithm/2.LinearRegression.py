from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

boston = datasets.load_boston()  # [506, 13]
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
line_reg = LinearRegression()
line_reg.fit(X_train, y_train)
y_predict = line_reg.predict(X_test)

print("k =", line_reg.coef_)  # 正数正相关 负数负相关 绝对值越大相关程度越大
print("b =", line_reg.intercept_)
print("score =", line_reg.score(X_test, y_test))
print("mse =", mean_squared_error(y_test, y_predict))
print("rmse =", sqrt(mean_squared_error(y_test, y_predict)))
print("mae =", mean_absolute_error(y_test, y_predict))
print("r2_score =", r2_score(y_test, y_predict))

'''
参数含义：
1.fit_intercept:布尔值，指定是否需要计算线性回归中的截距，即b值。如果为False，那么不计算b值。
2.normalize:布尔值。如果为False，那么训练样本会进行归一化处理。
3.copy_X：布尔值。如果为True，会复制一份训练数据。
4.n_jobs:一个整数。任务并行时指定的CPU数量。如果取值为-1则使用所有可用的CPU。

属性
1.coef_:权重向量
2.intercept_:截距b值

方法：
1.fit(X,y)：训练模型。
2.predict(X)：用训练好的模型进行预测，并返回预测值。
3.score(X,y)：返回预测性能的得分。计算公式为：score=(1 - u/v)
其中u=((y_true - y_pred) ** 2).sum()，v=((y_true - y_true.mean()) ** 2).sum()
score最大值是1，但有可能是负值(预测效果太差)。score越大，预测性能越好。
'''
