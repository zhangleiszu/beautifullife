# -*- coding:utf-8

#Author : 石头
#Date : 20181206
#Content ：第三节《模型泛化能力探讨》
import matplotlib.pyplot as plt

from sklearn.ensemble import  AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import  make_gaussian_quantiles
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
from plot_learn_curve import plot_learning_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.metrics import zero_one_loss
# ##########################
n_estimators = 200
# 生成2维正态分布，生成的数据按分位数分为两类，50个样本特征，5000个样本数据
X,y = make_gaussian_quantiles(cov=2.0,n_samples=5000,n_features=50,n_classes=2,random_state=1)
# 数据划分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
# 根据上一节的参数择优，选择最优参数来构建模型
estimatorCart = DecisionTreeClassifier(max_depth=1)
dt_stump1 = AdaBoostClassifier(base_estimator=estimatorCart,
                                    n_estimators=n_estimators,learning_rate=0.8)
dt_stump2 = AdaBoostClassifier(base_estimator=estimatorCart,
                                    n_estimators=n_estimators,learning_rate=0.1)
dt_stump1.fit(X_train,y_train)
dt_stump_err1 = 1.0 - dt_stump1.score(X_test,y_test)
#
dt_stump2.fit(X_train,y_train)
dt_stump_err2 = 1.0 - dt_stump2.score(X_test,y_test)
# ###########
test_errors1 = []
# 每迭代一次，得到一个测试结果
ada_discrete_err1 = np.zeros((n_estimators,))
ada_discrete_err2 = np.zeros((n_estimators,))
for i,ypred in enumerate(dt_stump1.staged_predict(X_test)):
    ada_discrete_err1[i] = zero_one_loss(ypred,y_test)

for i,ypred in enumerate(dt_stump2.staged_predict(X_test)):
    ada_discrete_err2[i] = zero_one_loss(ypred,y_test)

# 画出迭代次数与准确率的关系
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(np.arange(n_estimators) + 1, ada_discrete_err1,
        label='learning rate = 0.8',
        color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err2,
        label='learning rate = 0.1',
        color='green')
ax.set_ylim((0.0, 1))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
plt.show()