import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


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


plt.figure("Ensemble Learning", (10, 7.5))
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

# Voting----------------------------------------------------------------------------------------------------------------
X, y = datasets.make_moons(n_samples=300, noise=0.3, random_state=666)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 非加权
vot_clf_hard = VotingClassifier(estimators=[
    ("log_reg", LogisticRegression()),
    ("svc", SVC()),
    ("dt_clf", DecisionTreeClassifier())
], voting="hard")  # hard非加权投票
vot_clf_hard.fit(X_train, y_train)
print("vot_clf_hard =", vot_clf_hard.score(X_test, y_test))

plt.sca(ax1)
plt.title("vot_clf_hard")
boundary(vot_clf_hard, [-2, 3, -1.5, 2])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([-2, 3, -1.5, 2])

# 加权
vot_clf_soft = VotingClassifier(estimators=[
    ("log_reg", LogisticRegression()),
    ("svc", SVC(probability=True)),
    ("dt_clf", DecisionTreeClassifier())
], voting="soft")  # soft加权投票
vot_clf_soft.fit(X_train, y_train)
print("vot_clf_soft =", vot_clf_soft.score(X_test, y_test))

plt.sca(ax4)
plt.title("vot_clf_soft")
boundary(vot_clf_soft, [-2, 3, -1.5, 2])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([-2, 3, -1.5, 2])

# Bagging---------------------------------------------------------------------------------------------------------------
# 不放回取样
X, y = datasets.make_moons(n_samples=300, noise=0.3, random_state=666)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

bag_clf_false = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                  bootstrap=False)  # 决策树模型 集成500个子模型 每个子模型处理100个数据 不放回取样
bag_clf_false.fit(X_train, y_train)
print("bag_clf_false =", bag_clf_false.score(X_test, y_test))

plt.sca(ax2)
plt.title("bag_clf_false")
boundary(bag_clf_false, [-2, 3, -1.5, 2])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([-2, 3, -1.5, 2])

# 放回取样 不区分训练集测试集
X, y = datasets.make_moons(n_samples=300, noise=0.3, random_state=666)

bag_clf_true = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=500, max_samples=100,
                                 bootstrap=True, oob_score=True)  # 决策树模型 集成500个子模型 每个子模型处理100个数据 放回取样 oob
bag_clf_true.fit(X, y)
print("bag_clf_true =", bag_clf_true.oob_score_)

plt.sca(ax5)
plt.title("bag_clf_true")
boundary(bag_clf_true, [-2, 3, -1.5, 2])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([-2, 3, -1.5, 2])

# Boosting--------------------------------------------------------------------------------------------------------------
X, y = datasets.make_moons(n_samples=300, noise=0.3, random_state=666)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Ada
ab_clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=500)
ab_clf.fit(X_train, y_train)
print("ab_clf =", ab_clf.score(X_test, y_test))

plt.sca(ax3)
plt.title("ab_clf")
boundary(bag_clf_true, [-2, 3, -1.5, 2])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([-2, 3, -1.5, 2])

# Gradient
gb_clf = GradientBoostingClassifier()  # 默认决策树
gb_clf.fit(X_train, y_train)
print("gb_clf =", gb_clf.score(X_test, y_test))

plt.sca(ax6)
plt.title("gb_clf")
boundary(bag_clf_true, [-2, 3, -1.5, 2])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([-2, 3, -1.5, 2])

plt.show()


'''
GradientBoostingClassifier

参数
1.loss: 损失函数，'deviance'：对数损失函数，'exponential'：指数损失函数，只能用于二分类。
2.learning_rate：学习率
3.n_ estimators: 基学习器的个数，这里是树的颗数
4.subsample: 取值在(0, 1)之间，取原始训练集中的一个子集用于训练基础决策树
5.criterion: 'friedman_mse'改进型的均方误差;'mse'标准的均方误差; 'mae'平均绝对误差。
6.min_samples_split:一个整数，指定了每个基础决策树模型分裂所需最小样本数。
7.min_samples_leaf:一个整数，指定了每个基础决策树模型叶节点所包含的最小样本数。
8.min_weight_fraction_leaf:一个浮点数。叶节点的最小加权权重。当不提供sample_weight时，
  样本的权重是相等的。
9.max_depth:一个整数或者None，指定每一个基础决策树模型的最大深度。如果max_leaf_noeds不是None，则忽略此参数。
10.max_features:一个整数，浮点数或者None。代表节点分裂是参与判断的最大特征数。整数为个数，
  浮点数为所占比重。
11.max_leaf_nodes:为整数或者None，指定了每个基础决策树模型的最大叶节点数量。
12.min_impurity_split:一个浮点数，指定在树生长的过程中，节点分裂的阈值，默认为1e-7。
13.init：一个基础分类器对象或者None
14.verbose：如果为0则不输出日志，如果为1，则每隔一段时间输出日志
15.warm_start：热启动，当你训练GBM到一定程度停止时，如果你想在这个基础上接着训练，就需要用到该参数减少重复训练

属性 
1.n_estimators_ : 基学习器的个数，这里是树的颗数
2.feature_importance_:一个数组，给出了每个特征的重要性（值越高重要性越大）。
3.oob_improvement_:一个数组，给出了每增加一棵基础决策树，在包外估计（即测试集）的损失函数的减少值。
4.train_score_:一个数组，给出每增加一棵基础决策树，在训练集上的损失函数的值。
5.loss:具体损失函数对象。
6.init:初始预测使用的分类器。
7.estimators_:一个数组，给出了每个基础决策树。

方法 
1.apply(X)	Apply trees in the ensemble to X, return leaf indices.
2.decision_function(X)	Compute the decision function of X.
3.fit(X,y):训练模型。
4.get_params([deep])	Get parameters for this estimator.
5.predict(X):用模型进行预测，返回预测值。
6.predict_log_proba(X):返回一个数组，数组的元素依次是X预测为各个类别的概率的对数 
  值。
7.predict_proba(X):返回一个数组，数组的元素依次是X预测为各个类别的概率值。
8.score(X,y):返回在（X,y）上预测的准确率。
9.set_params(**params)	Set the parameters of this estimator.
10.staged_predict(X):返回一个数组，数组元素依次是每一轮迭代结束时尚未完成的集成分类器的预测值。
11.staged_predict_proba(X):返回一个二维数组，
  数组元素依次是每一轮迭代结束时尚未完成的集成分类器预测X为各个类别的概率值。
'''



'''
GradientBoostingRegressor

参数
1.loss: 损失函数，‘ls’：此时损失函数为平方损失函数。 
- ‘lad’：此时使用指数绝对值损失函数。 
- ‘quantile’：分位数回归（分位数指的是百分之几），采用绝对值损失。 
- ‘huber’：此时损失函数为上述两者的综合，即误差较小时，采用平方损失，在误差较大时，采用绝对值损失。
2.learning_rate：学习率
3.n_ estimators: 基学习器的个数，这里是树的颗数
4.subsample: 取值在(0, 1)之间，取原始训练集中的一个子集用于训练基础决策树
5.criterion: 'friedman_mse'改进型的均方误差;'mse'标准的均方误差; 'mae'平均绝对误差。
6.min_samples_split:一个整数，指定了每个基础决策树模型分裂所需最小样本数。
7.min_samples_leaf:一个整数，指定了每个基础决策树模型叶节点所包含的最小样本数。
8.min_weight_fraction_leaf:一个浮点数。叶节点的最小加权权重。当不提供sample_weight时，样本的权重是相等的。
9.max_depth:一个整数或者None，指定每一个基础决策树模型的最大深度。如果max_leaf_noeds不是None，则忽略此参数。
10.max_features:一个整数，浮点数或者None。代表节点分裂是参与判断的最大特征数。整数为个数，浮点数为所占比重。
11.max_leaf_nodes:为整数或者None，指定了每个基础决策树模型的最大叶节点数量。
12.min_impurity_split:一个浮点数，指定在树生长的过程中，节点分裂的阈值，默认为1e-7。
13.init：一个基础分类器对象或者None
14.alpha:一个浮点数，只有当loss=‘huber’或者loss=‘quantile’时才有效。
15.verbose：如果为0则不输出日志，如果为1，则每隔一段时间输出日志
16.warm_start：热启动，当你训练GBM到一定程度停止时，如果你想在这个基础上接着训练，就需要用到该参数减少重复训练；

属性 
1.feature_importance_:一个数组，给出了每个特征的重要性（值越高重要性越大）。
2.oob_improvement_:一个数组，给出了每增加一棵基础决策树，在包外估计（即测试集）的损失函数的减少值。
3.train_score_:一个数组，给出每增加一棵基础决策树，在训练集上的损失函数的值。
4.loss:具体损失函数对象。
5.init:初始预测使用的分类器。
6.estimators_:一个数组，给出了每个基础决策树。

方法 
1.apply(X)	Apply trees in the ensemble to X, return leaf indices.
2.fit(X,y):训练模型。
3.get_params([deep])	Get parameters for this estimator.
4.predict(X):用模型进行预测，返回预测值。
5.score(X,y):返回在（X,y）上预测的准确率。
6.set_params(**params)	Set the parameters of this estimator.
7.staged_predict(X):返回一个数组，数组元素依次是每一轮迭代结束时尚未完成的集成分类器的预测值。
'''