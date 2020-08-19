import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


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


# 分类-------------------------------------------------------------------------------------------------------------------
iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target[:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
plt.figure("Decision Tree", (10, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

dt_clf_entropy = DecisionTreeClassifier(max_depth=3, criterion="entropy")  # 信息熵(慢) 熵越大不确定性越高
dt_clf_entropy.fit(X_train, y_train)
print("dt_clf_entropy =", dt_clf_entropy.score(X_test, y_test))

plt.sca(ax1)
plt.title("entropy")
boundary(dt_clf_entropy, [0.5, 7.5, 0, 3])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])

dt_clf_gini = DecisionTreeClassifier(max_depth=3, criterion="gini")  # 基尼系数(快) 系数越大不确定越高
dt_clf_gini.fit(X_train, y_train)
print("dt_clf_gini =", dt_clf_gini.score(X_test, y_test))

plt.sca(ax2)
plt.title("gini")
boundary(dt_clf_gini, [0.5, 7.5, 0, 3])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])

# 回归-------------------------------------------------------------------------------------------------------------------
boston = datasets.load_boston()
X = boston.data
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
print("dt_reg =", dt_reg.score(X_test, y_test))

plt.show()

'''
DecisionTreeClassifier
参数:
1.criterion : 一个字符串，指定切分质量的评价标准。可以为：
   ‘gini’ ：表示切分标准是Gini系数。切分时选取基尼系数小的属性切分。
   ‘entropy’ ： 表示切分标准是熵。
2.splitter : 一个字符串，指定切分原则，可以为：
   best : 表示选择最优的切分。
   random ： 表示随机切分。
   默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random"。
3.max_features : 可以为整数、浮点、字符或者None，指定寻找best split时考虑的特征数量。 
   如果是整数，则每次切分只考虑max_features个特征。
   如果是浮点数，则每次切分只考虑max_features*n_features个特征(max_features指定了百分比)。
   如果是字符串‘auto’，则max_features等于n_features。
   如果是字符串‘sqrt’，则max_features等于sqrt(n_features)。
   如果是字符串‘log2’，则max_features等于log2(n_features)。
   如果是字符串None，则max_features等于n_features。
4.max_depth : 可以为整数或者None，指定树的最大深度，防止过拟合
   如果为None，表示树的深度不限(知道每个叶子都是纯的，即叶子结点中的所有样本点都属于一个类，
   或者叶子中包含小于min_sanples_split个样本点)。
   如果max_leaf_nodes参数非None，则忽略此项。
5.min_samples_split : 为整数，指定每个内部节点(非叶子节点)包含的最少的样本数。
6.min_samples_leaf : 为整数，指定每个叶子结点包含的最少的样本数。
7.min_weight_fraction_leaf : 为浮点数，叶子节点中样本的最小权重系数。
8.max_leaf_nodes : 为整数或None，指定叶子结点的最大数量。 
   如果为None，此时叶子节点数不限。
   如果非None，则max_depth被忽略。
9.min_impurity_decrease=0.0 如果该分裂导致不纯度的减少大于或等于该值，则将分裂节点。
10.min_impurity_split=1e-07, 限制决策树的增长，
11.class_weight : 一个字典、字典的列表、字符串‘balanced’或者None，他指定了分类的权重。
   权重形式为：{class_label:weight} 如果为None，则每个分类权重都为1.
   字符串‘balanced’表示每个分类的权重是各分类在样本出现的频率的反比。
12.random_state : 一个整数或者一个RandomState实例，或者None。 
   如果为整数，则它指定了随机数生成器的种子。
   如果为RandomState实例，则指定了随机数生成器。
   如果为None，则使用默认的随机数生成器。
13.presort : 一个布尔值，指定了是否要提前排序数据从而加速寻找最优切分的过程。
  设置为True时，对于大数据集会减慢总体训练过程，但对于小数据集或者设定了最大深度的情况下，则会加速训练过程。

属性:
1.classes_ : 分类的标签值。
2.feature_importances_ : 给出了特征的重要程度。该值越高，则特征越重要(也称为Gini 
  importance)。
3.max_features_ : max_feature的推断值。
4.n_classes_ : 给出了分类的数量。
5.n_features_ : 当执行fit后，特征的数量。
6.n_outputs_ : 当执行fit后，输出的数量。
7.tree_ : 一个Tree对象，即底层的决策树。

方法: 
1.fit(X,y) : 训练模型。
2.predict(X) : 用模型预测，返回预测值。
3.predict_log_proba(X) : 返回一个数组，数组元素依次为X预测为各个类别的概率值的对数 
  值。
4.predict_proba(X) : 返回一个数组，数组元素依次为X预测为各个类别的概率值。
5.score(X,y) : 返回在(X,y)上预测的准确率(accuracy)。
'''

'''
DecisionTreeRegressor
参数： 
1.criterion : 一个字符串，指定切分质量的评价标准。默认为‘mse’，且只支持该字符串，表示均方误差。
2.splitter : 一个字符串，指定切分原则，可以为： 
   best : 表示选择最优的切分。
   random ： 表示随机切分。
3.max_features : 可以为整数、浮点、字符或者None，指定寻找best split时考虑的特征数量。 
   如果是整数，则每次切分只考虑max_features个特征。
   如果是浮点数，则每次切分只考虑max_features*n_features个特征(max_features指定了百分比)。
   如果是字符串‘auto’，则max_features等于n_features。
   如果是字符串‘sqrt’，则max_features等于sqrt(n_features)。
   如果是字符串‘log2’，则max_features等于log2(n_features)。
   如果是字符串None，则max_features等于n_features。
4.max_depth : 可以为整数或者None，指定树的最大深度。 
  如果为None，表示树的深度不限(知道每个叶子都是纯的，即叶子结点中的所有样本点都属于一个类，
  或者叶子中包含小于min_sanples_split个样本点)。
如果max_leaf_nodes参数非None，则忽略此项。
5.min_samples_split : 为整数，指定每个内部节点(非叶子节点)包含的最少的样本数。
6.min_samples_leaf : 为整数，指定每个叶子结点包含的最少的样本数。
7.min_weight_fraction_leaf : 为浮点数，叶子节点中样本的最小权重系数。
8.max_leaf_nodes : 为整数或None，指定叶子结点的最大数量。 
   如果为None，此时叶子节点数不限。
   如果非None，则max_depth被忽略。
9.class_weight : 一个字典、字典的列表、字符串‘balanced’或者None，它指定了分类的权重。
  权重形式为：{class_label:weight} 如果为None，则每个分类权重都为1.
  字符串‘balanced’表示每个分类的权重是各分类在样本出现的频率的反比。
10.random_state : 一个整数或者一个RandomState实例，或者None。 
   如果为整数，则它指定了随机数生成器的种子。
   如果为RandomState实例，则指定了随机数生成器。
   如果为None，则使用默认的随机数生成器。
11.presort : 一个布尔值，指定了是否要提前排序数据从而加速寻找最优切分的过程。
  设置为True时，对于大数据集会减慢总体训练过程，但对于小数据集或者设定了最大深度的情况下，则会加速训练过程。

属性: 
1.feature_importances_ : 给出了特征的重要程度。该值越高，则特征越重要(也称为Gini importance)。
2.max_features_ : max_feature的推断值。
3.n_features_ : 当执行fit后，特征的数量。
4.n_outputs_ : 当执行fit后，输出的数量。
5.tree_ : 一个Tree对象，即底层的决策树。

方法: 
1.fit(X,y) : 训练模型。
2.predict(X) : 用模型预测，返回预测值。
3.score(X,y) : 返回性能得分
'''
