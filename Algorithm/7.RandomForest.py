import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


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


X, y = datasets.make_moons(n_samples=300, noise=0.3)
plt.figure("Random Forest", (10, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

# 随机森林
rf_clf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf_clf.fit(X, y)
print("rf_clf =", rf_clf.oob_score_)

plt.sca(ax1)
plt.title("rf_clf")
boundary(rf_clf, [-2, 3, -1.5, 2])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([-2, 3, -1.5, 2])

# 极其随机森林 抑制过拟合
ef_clf = ExtraTreesClassifier(n_estimators=500, bootstrap=True, oob_score=True)
ef_clf.fit(X, y)
print("ef_clf =", ef_clf.oob_score_)

plt.sca(ax2)
plt.title("ef_clf")
boundary(ef_clf, [-2, 3, -1.5, 2])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([-2, 3, -1.5, 2])

plt.show()


'''
RandomForestClassifier

参数:
1.n_estimators :一个整数，指定基础决策树的数量（默认为10）.
2.criterion:字符串。指定分裂的标准，可以为‘entory’或者‘gini’。
3.max_depth:一个整数或者None，指定每一个基础决策树模型的最大深度。如果max_leaf_nodes不是None，则忽略此参数。
4.min_samples_split:一个整数，指定了每个基础决策树模型分裂所需最小样本数。
5.min_samples_leaf:一个整数，指定了每个基础决策树模型叶节点所包含的最小样本数。
6.min_weight_fraction_leaf:一个浮点数。叶节点的最小加权权重。当不提供sample_weight时，样本的权重是相等的。
7.max_features:一个整数，浮点数或者None。代表节点分裂是参与判断的最大特征数。整数为   个数，浮点数为所占比重。
8.max_leaf_nodes:为整数或者None，指定了每个基础决策树模型的最大叶节点数量。
9.bootstrap:为布尔值。如果为True，则使用采样法bootstrap sampling来产生决策树的训练数据。
10.oob_score：为布尔值。如果为True，则使用包外样本来计算泛化误差。
11.n_jobs：一个整数，指定并行性。如果为-1，则表示将训练和预测任务派发到所有CPU上。
12.verbose:一个整数，如果为0则不输出日志，如果为1，则每隔一段时间输出日志，大于1输出日志会更频繁。
13.warm_start:布尔值。当为True是，则继续使用上一次训练结果。否则重新开始训练。
14.random_state:一个整数或者一个RandomState实例，或者None。 
  如果为整数，指定随机数生成器的种子。
  如果为RandomState，指定随机数生成器。
  如果为None，指定使用默认的随机数生成器。
15.class_weight:一个字典，或者字典的列表，或者字符串‘balanced’，或者字符串
  ‘balanced_subsample’，或者为None。 
  如果为字典，则字典给出每个分类的权重，如{class_label：weight}
  如果为字符串‘balanced’，则每个分类的权重与该分类在样本集合中出现的频率成反比。
  如果为字符串‘balanced_subsample’，则样本为采样法bootstrap sampling产生的决策树的训练数据，
  每个分类的权重与该分类在样本集合中出现的频率成反比。
  如果为None，则每个分类的权重都为1。

属性 
1.estimators_:一个数组，存放所有训练过的决策树。
2.classes_:一个数组，形状为[n_classes]，为类别标签。
3.n_classes_:一个整数，为类别数量。
4.n_features_:一个整数，在训练时使用的特征数量。
5.n_outputs_:一个整数，在训练时输出的数量。
6.feature_importances_:一个数组，形状为[n_features]。如果base_estimator支持，
  则他给出每个特征的重要性。
7.oob_score_:一个浮点数，训练数据使用包外估计时的得分。

方法 
1.fit(X,y):训练模型。 
2.predict(X):用模型进行预测，返回预测值。
3.predict_log_proba(X):返回一个数组，数组的元素依次是X预测为各个类别的概率的对数
  值。
4.predict_proba(X):返回一个数组，数组的元素依次是X预测为各个类别的概率值。
5.score(X,y):返回在（X,y）上预测的准确度。
'''


'''
RandomForestRegressor

参数:
1.n_estimators :一个整数，指定基础决策树的数量（默认为10）.
2.criterion:字符串。指定分裂的标准，默认为sse
3.max_depth:一个整数或者None，指定每一个基础决策树模型的最大深度。如果      
  max_leaf_nodes不是None，则忽略此参数。
4.min_samples_split:一个整数，指定了每个基础决策树模型分裂所需最小样本数。
5.min_samples_leaf:一个整数，指定了每个基础决策树模型叶节点所包含的最小样本数。
6.min_weight_fraction_leaf:一个浮点数。叶节点的最小加权权重。当不提供
  sample_weight时，样本的权重是相等的。
7.max_features:一个整数，浮点数或者None。代表节点分裂是参与判断的最大特征数。
8.max_leaf_nodes:为整数或者None，指定了每个基础决策树模型的最大叶节点数量。
9.bootstrap:为布尔值。如果为True，则使用采样法bootstrap sampling来产生决策树的训练数据。
10.oob_score：为布尔值。如果为True，则使用包外样本来计算泛化误差。
11.n_jobs：一个整数，指定并行性。如果为-1，则表示将训练和预测任务派发到所有CPU上。
12.verbose:一个整数，如果为0则不输出日志，如果为1，则每隔一段时间输出日志，大于1输出日志会更频繁。
13.warm_start:布尔值。当为True是，则继续使用上一次训练结果。否则重新开始训练。
14.random_state:一个整数或者一个RandomState实例，或者None。 
  如果为整数，指定随机数生成器的种子。
  如果为RandomState，指定随机数生成器。
  如果为None，指定使用默认的随机数生成器。

属性 
1.estimators_:一个数组，存放所有训练过的决策树。
2.oob_prediction_:一个数组，训练数据使用包外估计时的预测值。
3.n_features_:一个整数，在训练时使用的特征数量。
4.n_outputs_:一个整数，在训练时输出的数量。
5.feature_importances_:一个数组，形状为[n_features]。如果base_estimator支持，
  则他给出每个特征的重要性。
6.oob_score_:一个浮点数，训练数据使用包外估计时的得分。

方法 
1.fit(X,y):训练模型。 
2.predict(X):用模型进行预测，返回预测值。
3.score(X,y):返回在（X,y）上预测的准确度。
'''