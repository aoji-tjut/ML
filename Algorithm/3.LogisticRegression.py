import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


def boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)


def pipe(degree, C, penalty, multi_class='auto', solver="lbfgs"):
    return Pipeline([
        ("ploy", PolynomialFeatures(degree=degree)),
        ("scaler", StandardScaler()),
        ("log_reg", LogisticRegression(C=C, penalty=penalty, multi_class=multi_class, solver=solver))
        # C正则化系数 penalty正则化方式 multi_class分类方式 solver损失函数优化方法
    ])


iris = datasets.load_iris()
plt.figure("Logistic Regression", (7.5, 7.5))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

# 线性二分类-------------------------------------------------------------------------------------------------------------
X = iris.data
y = iris.target
X = X[y < 2, :2]
y = y[y < 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)
print("score1 =", log_reg.score(X_test, y_test))
print("precision_score1 =", precision_score(y_test, y_predict))  # 精准率
print("recall_score1 =", recall_score(y_test, y_predict))  # 召回率
print("f1_score1 =", f1_score(y_test, y_predict))  # f1-score
print()

plt.sca(ax1)
plt.title("Linear Two Classification")
boundary(log_reg, [4, 7.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.sca(ax2)

# 非线性二分类------------------------------------------------------------------------------------------------------------
X = iris.data
y = iris.target
X = X[y < 2, :2]
y = y[y < 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe1 = pipe(10, 100, "l2")
pipe1.fit(X_train, y_train)
y_predict = pipe1.predict(X_test)
print("score2 =", pipe1.score(X_test, y_test))
print("precision_score2 =", precision_score(y_test, y_predict))
print("recall_score2 =", recall_score(y_test, y_predict))
print("f1_score2 =", f1_score(y_test, y_predict))
print()

plt.sca(ax2)
plt.title("Non-linear Two Classification")
boundary(pipe1, [4, 7.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])

# 线性OvO多分类----------------------------------------------------------------------------------------------------------
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_reg_ovo = LogisticRegression(multi_class="multinomial", solver="newton-cg")  # OvO不能使用solver="liblinear"
log_reg_ovo.fit(X_train, y_train)
y_predict = log_reg_ovo.predict(X_test)
print("score3 =", log_reg_ovo.score(X_test, y_test))
print("precision_score3 =", precision_score(y_test, y_predict, average="micro"))
print("recall_score3 =", recall_score(y_test, y_predict, average="micro"))
print("f1_score3 =", f1_score(y_test, y_predict, average="micro"))
print()

plt.sca(ax3)
plt.title("Linear OvO Classification")
boundary(log_reg_ovo, [4, 8.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])

# 非线性OvR多分类---------------------------------------------------------------------------------------------------------
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe2 = pipe(10, 100, "l2", "ovr", "liblinear")
pipe2.fit(X_train, y_train)
y_predict = pipe2.predict(X_test)
print("score4 =", pipe2.score(X_test, y_test))
print("precision_score4 =", precision_score(y_test, y_predict, average="micro"))
print("recall_score4 =", recall_score(y_test, y_predict, average="micro"))
print("f1_score4 =", f1_score(y_test, y_predict, average="micro"))
print()

plt.sca(ax4)
plt.title("Non-linear OvR Classification")
boundary(pipe2, [4, 8.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])

# 任意模型多分类器--------------------------------------------------------------------------------------------------------
X = iris.data
y = iris.target
# OneVsRestClassifier
ovr = OneVsRestClassifier(log_reg)
ovr.fit(X_train, y_train)
print("ovr =", ovr.score(X_test, y_test))
# OneVsOneClassifier
ovo = OneVsOneClassifier(log_reg)
ovo.fit(X_train, y_train)
print("ovo =", ovo.score(X_test, y_test))

plt.show()

'''
参数含义：
1.penalty:字符串，指定了正则化策略。默认为"l2"
    (1)如果为"l2",则优化的目标函数为：0.5*||w||^2_2+C*L(w),C>0,
        L(w)为极大似然函数。
    (2)如果为"l1",则优化的目标函数为||w||_1+C*L(w),C>0,
        L(w)为极大似然函数。
2.dual:布尔值。默认为False。如果等于True，则求解其对偶形式。
  只有在penalty="l2"并且solver="liblinear"时才有对偶形式。如果为False，则求解原始形式。
  当n_samples > n_features，偏向于dual=False。
3.tol:阈值。判断迭代是否收敛或者是否满足精度的要求。
4.C:float,默认为1.0.指定了正则化项系数的倒数。必须是一个正的浮点数。他的值越小，正则化项就越大。
5.fit_intercept:bool值。默认为True。如果为False,就不会计算b值。
6.intercept_scaling：float, default 1。
  只有当solver="liblinear"并且  fit_intercept=True时，才有意义。
  在这种情况下，相当于在训练数据最后一列增加一个特征，该特征恒为1。其对应的权重为b。
7.class_weight：dict or 'balanced', default: None。
    (1)如果是字典，则给出每个分类的权重。按照{class_label: weight}这种形式。
    (2)如果是"balanced"：则每个分类的权重与该分类在样本集中出现的频率成反比。
       n_samples / (n_classes * np.bincount(y))
    (3)如果未指定，则每个分类的权重都为1。
8.random_state: int, RandomState instance or None, default: None
    (1):如果为整数，则它指定了随机数生成器的种子。
    (2):如果为RandomState实例，则它指定了随机数生成器。
    (3):如果为None，则使用默认的随机数生成器。
9.solver: 字符串，指定求解最优化问题的算法。
{'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},default: 'liblinear'
   (1)solver='liblinear',对于小数据集，'liblinear'是很好的选择。
       对于大规模数据集，'sag'和'saga'处理起来速度更快。
   (2)solver='newton-cg',采用牛顿法
   (3)solver='lbfgs',采用L-BFGS拟牛顿法。
   (4)solver='sag',采用Stochastic Average Gradient descent算法。
   (5)对于多分类问题，只有'newton-cg'，'sag'，'saga'和'lbfgs'处理多项损失;
      'liblinear'仅限于'ovr'方案。
   (6)newton-cg', 'lbfgs' and 'sag' 只能处理 L2 penalty,
      'liblinear' and 'saga' 能处理 L1 penalty。
10.max_iter: 指定最大迭代次数。default: 100。只对'newton-cg', 'sag' and 'lbfgs'适用。
11.multi_class：{'ovr', 'multinomial'}, default: 'ovr'。指定对分类问题的策略。
    (1)multi_class='ovr',采用'one_vs_rest'策略。
    (2)multi_class='multinomal',直接采用多分类逻辑回归策略。
12.verbose: 用于开启或者关闭迭代中间输出日志功能。
13.warm_start: 布尔值。如果为True，那么使用前一次训练结果继续训练。否则从头开始训练。
14.n_jobs: int, default: 1。指定任务并行时的CPU数量。如果为-1，则使用所有可用的CPU。

属性：
1.coef_：权重向量。
2.intercept_：截距b值。
3.n_iter_：实际迭代次数。

方法：
1.fit(X,y): 训练模型。
2.predict(X): 用训练好的模型进行预测，并返回预测值。
3.predict_log_proba(X): 返回一个数组，数组元素依次是X预测为各个类别的概率的对数值。
4.predict_proba(X): 返回一个数组，数组元素依次是X预测为各个类别的概率值。
5.score(X,y): 返回预测的准确率。
'''
