import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.metrics import r2_score, f1_score


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


plt.figure("Support Vector Machines", (12, 7.5))
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)
ax6 = plt.subplot(2, 3, 6)

# 线性分类---------------------------------------------------------------------------------------------------------------
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y < 2, :2]
y = y[y < 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def ScalerLinearSvcPipe(C):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("log_reg", LinearSVC(C=C, max_iter=1e5))  # C 大-容错小-过拟合 小-容错大-欠拟合
    ])


# hard
svc_hard = ScalerLinearSvcPipe(C=1e10)
svc_hard.fit(X_train, y_train)
y_predict = svc_hard.predict(X_test)
print("svc_hard f1_score =", f1_score(y_test, y_predict))

plt.sca(ax1)
plt.title("Hard LinearSVC")
boundary(svc_hard, [4, 7.5, 1.75, 4.75])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([4, 7.5, 1.75, 4.75])

# soft
svc_soft = ScalerLinearSvcPipe(C=1e-10)
svc_soft.fit(X_train, y_train)
y_predict = svc_soft.predict(X_test)
print("svc_soft f1_score =", f1_score(y_test, y_predict))

plt.sca(ax4)
plt.title("Soft LinearSVC")
boundary(svc_soft, [4, 7.5, 1.75, 4.75])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.axis([4, 7.5, 1.75, 4.75])

# 非线性分类-------------------------------------------------------------------------------------------------------------
X = iris.data
y = iris.target
X = X[:, :2]
y = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# poly+svc
def PolyLinearSvcPipe(degree, C):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(C=C, max_iter=1e5))
    ])


svc_poly = PolyLinearSvcPipe(5, 1)
svc_poly.fit(X_train, y_train)
y_predict = svc_poly.predict(X_test)
print("svc_poly f1_score =", f1_score(y_test, y_predict, average="micro"))

plt.sca(ax2)
plt.title("Poly + LinearSVC")
boundary(svc_poly, [4, 8.5, 1.75, 4.75])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.axis([4, 8.5, 1.75, 4.75])


# svc_rbf
def RbfKernelSvcPipe(gamma):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kernel_svc", SVC(kernel="rbf", gamma=gamma))  # gamma 小-欠拟合 大-过拟合
    ])


svc_rbf = RbfKernelSvcPipe(0.2)
svc_rbf.fit(X_train, y_train)
y_predict = svc_rbf.predict(X_test)
print("svc_rbf f1_score =", f1_score(y_test, y_predict, average="micro"))

plt.sca(ax5)
plt.title("SVC_RBF")
boundary(svc_rbf, [4, 8.5, 1.75, 4.75])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.axis([4, 8.5, 1.75, 4.75])

# 回归------------------------------------------------------------------------------------------------------------------
# 线性回归
x = np.linspace(0, 100, 100)
X = x.reshape(-1, 1)
y = 2 * x + 5 + np.random.uniform(-10, 10, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svr_line = LinearSVR(max_iter=1e5)
svr_line.fit(X_train, y_train)
y_predict = svr_line.predict(X_test)
print("svr_line r2_score =", r2_score(y_test, y_predict))

plt.sca(ax3)
plt.title("LinearSVR")
plt.scatter(X_train, y_train, c='b')
plt.plot(X_test, y_predict, c='r')

# 非线性回归
x = np.linspace(-2, 2, 100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + 2 * x + 3 + np.random.normal(0, 0.5, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svr_rbf = SVR(kernel="rbf", gamma=0.5)
svr_rbf.fit(X_train, y_train)
y_predict = svr_rbf.predict(X_test)
print("svr_rbf r2_score =", r2_score(y_test, y_predict))

plt.sca(ax6)
plt.title("SVR")
plt.scatter(X_train, y_train, c='b')
plt.plot(np.sort(X_test.flatten()), y_predict[np.argsort(X_test.flatten())], c='r')

plt.show()

'''
LinearSVC

参数：
1.C: 一个浮点数，罚项系数。C值越大对误分类的惩罚越大。
2.loss: 字符串，表示损失函数，可以为如下值： 
   ‘hinge’：此时为合页损失(标准的SVM损失函数)，
   ‘squared_hinge’：合页损失函数的平方。
3.penalty: 字符串，指定‘l1’或者‘l2’，罚项范数。默认为‘l2’(他是标准的SVM范数)。
4.dual: 布尔值，如果为True，则解决对偶问题，如果是False，则解决原始问题。
  当n_samples>n_features是，倾向于采用False。
5.tol: 浮点数，指定终止迭代的阈值。
6.multi_class: 字符串，指定多分类的分类策略。 
   ‘ovr’：采用one-vs-rest策略。
   ‘crammer_singer’: 多类联合分类，很少用，因为他计算量大，而且精度不会更佳，此时忽略loss,penalty,dual等参数。
7.fit_intecept: 布尔值，如果为True，则计算截距，即决策树中的常数项，否则忽略截距。
8.intercept_scaling: 浮点值，若提供了，则实例X变成向量[X,intercept_scaling]。
  此时相当于添加一个人工特征，该特征对所有实例都是常数值。
9.class_weight: 可以是个字典或者字符串‘balanced’。指定个各类的权重，若未提供，则认为类的权重为1。 
  如果是字典，则指定每个类标签的权重。如果是‘balanced’，则每个累的权重是它出现频数的倒数。
10.verbose: 一个整数，表示是否开启verbose输出。
11.random_state: 一个整数或者一个RandomState实例，或者None。 
    如果为整数，指定随机数生成器的种子。
    如果为RandomState，指定随机数生成器。
    如果为None，指定使用默认的随机数生成器。
12.max_iter: 一个整数，指定最大迭代数。

属性：
1.coef_: 一个数组，它给出了各个特征的权重。
2.intercept_: 一个数组，它给出了截距。

方法 
1.fit(X,y): 训练模型。
2.predict(X): 用模型进行预测，返回预测值。
3.score(X,y): 返回在(X,y)上的预测准确率。
'''

'''
SVC

参数：
1.C：惩罚参数，默认值是1.0
  C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
  这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
2.kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
‘precomputed’(表示使用自定义的核函数矩阵)
3.degree ：多项式poly函数的阶数，默认是3，选择其他核函数时会被忽略。
4.gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
5.coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
6.probability ：是否采用概率估计.这必须在调用fit()之前启用，并且会fit()方法速度变慢。默认为False
7.shrinking ：是否采用启发式收缩方式方法，默认为True
8.tol ：停止训练的误差值大小，默认为1e-3
9.cache_size ：指定训练所需要的内存，以MB为单位，默认为200MB。
10.class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(SVC中的C)
11.verbose ：用于开启或者关闭迭代中间输出日志功能。
12.max_iter ：最大迭代次数。-1为无限制。
13.decision_function_shape ：‘ovo’, ‘ovr’ or None, default=‘ovr’
14.random_state ：数据洗牌时的种子值，int值

属性：
1.support_  支持向量的索引
2.support_vectors_   支持向量
3.n_support_   每个类别的支持向量的数目
4.dual_coef_ :  一个数组，形状为[n_class-1,n_SV]。对偶问题中，在分类决策函数中每一个支持向量的系数。
5.coef_ : 一个数组，形状为[n_class-1,n_features]。原始问题中，每个特征的系数。只有在linear kernel中有效。
6.intercept_ : 一个数组，形状为[n_class*(n_class-1)/2]。决策函数中的常数项。
7.fit_status_ : 整型，表示拟合的效果，如果正确拟合为0，否则为1
8.probA_ : array, shape = [n_class * (n_class-1) / 2]
9.probB_ : array, shape = [n_class * (n_class-1) / 2]
10.probability: If probability=True, the parameters learned in Platt scaling to produce probability
estimates from decision values. 
If probability=False, an empty array. Platt scaling uses the logistic function 
1 / (1 + exp(decision_value * probA_ + probB_)) where probA_ and probB_ are learned from the dataset. 
For more information on the multiclass case and training procedure see section 8 of LIBSVM: 
A Library for Support Vector Machines (in References) for more.

用法：
1.decision_function(X)：	样本X到决策平面的距离
2.fit(X, y[, sample_weight])：	训练模型
3.get_params([deep])：	获取参数
4.predict(X)：	预测
5.score(X, y[, sample_weight])：	返回预测的平均准确率
6.set_params(**params)：	设定参数
'''

'''
LinearSVR

参数：
1.C: 一个浮点值，罚项系数。
2.loss:字符串，表示损失函数，可以为： 
　　‘epsilon_insensitive’:此时损失函数为L_ϵ(标准的SVR)
　　‘squared_epsilon_insensitive’:此时损失函数为LϵLϵ
3.epsilon: 浮点数，用于lose中的ϵϵ参数。
4.dual: 布尔值。如果为True，则解决对偶问题，如果是False则解决原始问题。
5.tol: 浮点数，指定终止迭代的阈值。
6.fit_intercept: 布尔值。如果为True，则计算截距，否则忽略截距。
7.intercept_scaling: 浮点值。如果提供了，则实例X变成向量[X,intercept_scaling]。此时相当于添加了一个人工特征，
  该特征对所有实例都是常数值。
8.verbose: 是否输出中间的迭代信息。
9.random_state: 指定随机数生成器的种子。
10.max_iter: 一个整数，指定最大迭代次数。

属性：
1.coef_: 一个数组，他给出了各个特征的权重。
2.intercept_: 一个数组，他给出了截距，及决策函数中的常数项。

方法：
1.fit(X,y): 训练模型。
2.predict(X): 用模型进行预测，返回预测值。
3.score(X,y): 返回性能得分。
'''

'''
SVR

参数： 
1.C: 一个浮点值，罚项系数。
2.epsilon: 浮点数，用于lose中的ϵ参数。
3.kernel: 一个字符串，指定核函数。 
   ’linear’ : 线性核
   ‘poly’: 多项式核
   ‘rbf’: 默认值，高斯核函数
   ‘sigmoid’: Sigmoid核函数
   ‘precomputed’: 表示支持自定义核函数
4.degree: 一个整数，指定当核函数是多项式核函数时，多项式的系数。对于其它核函数该参数无效。
5.gamma: 一个浮点数。当核函数是’rbf’,’poly’,’sigmoid’时，核函数的系数。如果为‘auto’，则表示系数为1/n_features。
6.coef0: 浮点数，用于指定核函数中的自由项。只有当核函数是‘poly’和‘sigmoid’时有效。
7.shrinking: 布尔值。如果为True，则使用启发式收缩。
8.tol: 浮点数，指定终止迭代的阈值。
9.cache_size: 浮点值，指定了kernel cache的大小，单位为MB。
10.verbose: 指定是否开启verbose输出。
11.max_iter: 一个整数，指定最大迭代步数。

属性: 
1.support_: 一个数组，形状为[n_SV]，支持向量的下标。
2.support_vectors_: 一个数组，形状为[n_SV,n_features]，支持向量。
3.n_support_: 一个数组，形状为[n_class]，每一个分类的支持向量个数。
4.dual_coef_: 一个数组，形状为[n_class-1,n_SV]。对偶问题中，在分类决策函数中每一个支持向量的系数。
5.coef_: 一个数组，形状为[n_class-1,n_features]。原始问题中，每个特征的系数。只有在linear kernel中有效。
5.intercept_: 一个数组，形状为[n_class*(n_class-1)/2]。决策函数中的常数项。

方法: 
1.fit(X,y): 训练模型。
2.predict(X): 用模型进行预测，返回预测值。
3.score(X,y): 返回性能得分。
'''
