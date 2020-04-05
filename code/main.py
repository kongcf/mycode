# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:19:51 2020

@author: 王均坐
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

data = pd.read_csv(r'collection.txt',header=None,sep='\t')
label = pd.read_csv(r'label.txt',header=None,sep='\t')


x,y = data.iloc[:,:].values,label.iloc[:,:].values
train_data,test_data,train_label,test_label = train_test_split(x,y,test_size=0.3,random_state=0)

model = RandomForestClassifier(oob_score=True,random_state=10)
model.fit(train_data,train_label)
print(model.oob_score_)
print("accuracy:%f"%model.oob_score_)

"n_estimators:对原始数据集进行有放回抽样生成的子数据集个数，即决策树的个数"
param_test1 = {"n_estimators":range(1,101,10)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_test1,cv=10)
gsearch1.fit(train_data,train_label)

print(gsearch1.grid_scores_)
print(gsearch1.best_params_)
print("best accuracy:%f" % gsearch1.best_score_)

"max_features:构建决策树最优模型时考虑的最大特征数。"
param_test2 = {"max_features":range(1,10,1)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=81,
                        random_state=10),
                        param_grid = param_test2,cv=10)
gsearch1.fit(train_data,train_label)
print(gsearch1.grid_scores_)
print(gsearch1.best_params_)
print('best accuracy:%f' % gsearch1.best_score_)

"max_depth"
param_test2 = {"max_depth":range(10,50,1)}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=81,max_features=5,
                        random_state=10),
                        param_grid = param_test2,cv=10)
gsearch2.fit(train_data,train_label)
print(gsearch2.grid_scores_)
print(gsearch2.best_params_)
print('best accuracy:%f' % gsearch1.best_score_)


rf0 = RandomForestClassifier(n_estimators=81,max_features=5,max_depth=20,
                             oob_score=True,random_state=10)
rf0.fit(train_data,train_label)
print(rf0.oob_score_)
print("accuracy: %f" % rf0.oob_score_)


modelpredict = model.predict(test_data); "预测标签"
modelpra = model.predict_proba(test_data); "预测得分"
print(accuracy_score(test_label, modelpredict))

rf0predict = rf0.predict(test_data)
rf0pra = rf0.predict_proba(test_data)
print(accuracy_score(test_label, rf0predict))


"保存数据"
dt = pd.DataFrame(train_data);
dt.to_excel("训练集.xlsx",index=0,header=None);
dt = pd.DataFrame(train_label);
dt.to_excel("训练集标签.xlsx",index=0,header=None);
dt = pd.DataFrame(test_data);
dt.to_excel("测试集.xlsx",index=0,header=None);
dt = pd.DataFrame(test_label);
dt.to_excel("测试集标签.xlsx",index=0,header=None);

dt = pd.DataFrame(modelpra)
dt.to_excel("随机森林测试集预测得分.xlsx",index=0,header=None)
dt = pd.DataFrame(rf0pra)
dt.to_excel("优化随机森林测试集预测得分.xlsx",index=0,header=None)

alldata = pd.read_csv(r'data.txt',header=None,sep='\t')
alllabel = rf0.predict(alldata)
dg = pd.DataFrame(alldata)
dg.to_csv("结果.csv",index = 0,header=None)
