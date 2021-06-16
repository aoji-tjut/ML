import numpy as np  # 矩阵
import matplotlib.pyplot as plt  # 画图
from sklearn import datasets  # 数据集
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split  # 分割数据集
from sklearn.model_selection import GridSearchCV  # 网格搜索最佳超参数
from sklearn.pipeline import Pipeline  # 管道
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # KNN分类 KNN回归
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.metrics import r2_score  # 回归模型评估
from sklearn.linear_model import SGDRegressor  # 梯度下降
from sklearn.decomposition import PCA  # PCA
from sklearn.preprocessing import PolynomialFeatures  # 多项式回归
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier  # 逻辑回归OvR 逻辑回归OvO
from sklearn.metrics import precision_score, recall_score, f1_score  # 分类模型评估(精准率 召回率 f1-score)
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score  # PR曲线 ROC曲线
from sklearn.svm import LinearSVC, SVC  # SVM线性分类 SVM非线性分类
from sklearn.svm import LinearSVR, SVR  # SVM线性回归 SVM非线性回归
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 决策树分类 决策树回归
from sklearn.ensemble import VotingClassifier, VotingRegressor  # 集成学习 投票
from sklearn.ensemble import BaggingClassifier, BaggingRegressor  # 集成学习 随机取样
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor  # 集成学习 提升树
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # 集成学习 GBDT
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # 集成学习 随机森林
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor  # 集成学习 极其随机森林
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯
from sklearn.cluster import KMeans  # 聚类
from sklearn.metrics import silhouette_score  # 聚类模型评估
from sklearn.externals import joblib  # 保存模型


def TrainTestSplit(x, y, test_ratio):
    index_shuffle = np.random.permutation(len(x))  # 对x大小的0～x-1索引乱序
    train_size = int(len(x) * (1.0 - test_ratio))  # 训练数据集大小
    train_index = index_shuffle[:train_size]  # 训练数据集索引
    test_index = index_shuffle[train_size:]  # 测试数据集索引
    x_train = x[train_index]
    y_train = y[train_index]
    x_test = x[test_index]
    y_test = y[test_index]
    return x_train, y_train, x_test, y_test


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
