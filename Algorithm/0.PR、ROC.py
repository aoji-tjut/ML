import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, f1_score, auc

digits = datasets.load_digits()
X = digits.data
y = digits.target.copy()
y[digits.target == 9] = 1
y[digits.target != 9] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_predict = log_reg.predict(X_test)
print("old f1_score =", f1_score(y_test, y_predict))
decision_score = log_reg.decision_function(X_test)  # 输入sigmod中的分数值
precisions, recalls, thresholds = precision_recall_curve(y_test, decision_score)
for i in range(thresholds.shape[0]):
    if precisions[i] == recalls[i]:
        y_predict = np.array(decision_score > thresholds[i], dtype=int)
        print("thresholds =", thresholds[i])
        print("new f1_score =", f1_score(y_test, y_predict))

plt.figure("Threshold Precision Recall")
plt.plot(thresholds, precisions[:-1], label="precision")  # last thresholds, precisions=1, recalls=0
plt.plot(thresholds, recalls[:-1], label="recall")
plt.legend()

plt.figure("PR")
plt.plot(recalls, precisions)

fprs, tprs, thresholds = roc_curve(y_test, decision_score)
print("roc_auc_score =", roc_auc_score(y_test, decision_score))
print("auc =", auc(fprs, tprs))
plt.figure("ROC")
plt.plot(fprs, tprs)

plt.show()
