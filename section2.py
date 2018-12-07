# -*- coding:utf-8

#Author : 石头
#Date : 20181206
#Content ：第二节《AdaBoost算法参数择优》
# -*- coding:utf-8

import matplotlib.pyplot as plt

from sklearn.ensemble import  AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import  make_gaussian_quantiles
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
from plot_learn_curve import plot_learning_curve

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
# ##########################
# 生成2维正态分布，生成的数据按分位数分为两类，50个样本特征，5000个样本数据
X,y = make_gaussian_quantiles(cov=2.0,n_samples=5000,n_features=50,n_classes=2,random_state=1)
# 分成训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
# 设置弱学习器决策树的最大深度为1
estimatorCart = DecisionTreeClassifier(max_depth=1)
# 对框架参数 弱学习器个数进行择优
param_test1 = {"n_estimators":range(150,300,50)}
# 框架参数择优
gsearch1 = GridSearchCV(estimator = AdaBoostClassifier(estimatorCart)
                                           ,param_grid=param_test1,scoring="roc_auc",cv=5)
gsearch1.fit(X,y)
print gsearch1.best_params_,gsearch1.best_score_
n_estimators1 = gsearch1.best_params_["n_estimators"]
# 继续优化弱学习器个数，在最优化学习器个数的范围内再次搜寻
param_test2 = {"n_estimators":range(n_estimators1-30,n_estimators1+30,10)}
gsearch2 = GridSearchCV(estimator = AdaBoostClassifier(estimatorCart)
                                           ,param_grid=param_test2,scoring="roc_auc",cv=5)
gsearch2.fit(X,y)
print gsearch2.best_params_,gsearch2.best_score_
n_estimators2 = gsearch2.best_params_["n_estimators"]
# ###########
# 对弱学习器参数择优,弱学习器择优的重要参数：决策树的最大深度（max_depth）
# 最小分裂节点样本数（min_samples_split），决策树分类的最大特征个数（max_features）
# AdaBoost的基学习器是若分类器，因此决策树的最大深度考虑1~3，最小分裂节点样本数考虑10~30
# 优化这两个参数
tree_depth = 0
samples_split = 0
score = 0
for i in range(1,3):      # 决策树最大深度循环
    print i
    for j in range(20,21):   # 最小可分节点数在该项目中没起作用，所以为了运行速度，设置20
        print j
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i,min_samples_split=j),
                                n_estimators=n_estimators2)
        cv_result = cross_validate(bdt,X,y,return_train_score=False,cv=5)
        cv_value_vec = cv_result["test_score"]
        cv_mean = np.mean(cv_value_vec)
        if cv_mean >= score:
            score = cv_mean
            tree_depth = i
            samples_split = j
print tree_depth,samples_split,score
# ########################
# 用该最优参数重新构建模型，并输出测试结果
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_depth),
                         n_estimators=n_estimators2)
bdt.fit(X_train,y_train)
print bdt.score(X_test,y_test)

