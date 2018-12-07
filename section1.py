# -*- coding:utf-8

#Author : 石头
#Date : 20181206
#Content ：第一节《决策树与AdaBoost算法比较》
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
# ##########################
# 生成2维正态分布，生成的数据按分位数分为两类，50个样本特征，5000个样本数据
X,y = make_gaussian_quantiles(cov=2.0,n_samples=5000,n_features=50,n_classes=2,random_state=1)
# 设置一百折交叉验证参数，数据集分层越多，交叉最优模型越接近原模型
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=1)
# 分别画出CART分类决策树和AdaBoost分类决策树的学习曲线
estimatorCart = DecisionTreeClassifier(max_depth=1)
estimatorBoost = AdaBoostClassifier(base_estimator=estimatorCart,
                                    n_estimators=270)
# 画CART决策树和AdaBoost的学习曲线
estimatorTuple = (estimatorCart,estimatorBoost)
titleTuple =("decision learning curve","adaBoost learning curve")
title = "decision learning curve"
for i in range(2):
    estimator = estimatorTuple[i]
    title = titleTuple[i]
    plot_learning_curve(estimator,title, X, y, cv=cv)
    plt.show()

