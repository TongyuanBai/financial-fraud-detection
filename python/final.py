# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 20:17:36 2022

@author: Tongyuan Bai
"""

###导入包
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score

###数据导入及处理
#统计频数
data = pd.read_csv("E:/essay/data/data1.csv")
num = data.groupby(["category","FLAG"]).size()

#判断数据是否为空
mis_val = data.drop(columns=['TICKER_SYMBOL', 'category','FLAG']).isnull().sum()
mis_val_percent =100 * data.drop(columns=['TICKER_SYMBOL', 'category','FLAG']).isnull().sum() / len(data)
print(mis_val_percent)
mis_val1 = mis_val_percent.tolist()

#缺失值可视化
plt.figure(figsize=(15,8))
x = list(range(352))
plt.bar(x,mis_val1,color="dodgerblue",label=False)
plt.xlabel("特征指标",fontsize=23)
plt.ylabel("缺失度",fontsize=23)
plt.show()

#对列名重命名
clear_col = mis_val_percent.to_frame()
clear_col.columns = ['value']

#缺失值填充
names1 = clear_col[clear_col['value']>=70]
drop_col_name1 = names1.index.tolist()
drop_col_name1
for i in drop_col_name1:
    data = data.drop(columns=[i])

names2 = clear_col[(clear_col['value'] < 70)&(clear_col['value'] >= 40)]
drop_col_name2 = names2.index.tolist()
drop_col_name2
for i in drop_col_name2:
    data[i] = data[i].fillna(0)

names3 = clear_col[(clear_col['value'] < 40)]
drop_col_name3 = names3.index.tolist()
drop_col_name3
for i in drop_col_name3:
    data[i] = data[i].fillna(data[i].mean())

#检查数据缺失情况
data.isnull().sum()

#处理后缺失值情况
mis_val_percent_process =100 * data.isnull().sum() / len(data)
print(mis_val_percent_process)

#缺失值可视化
mis_val2 = mis_val_percent_process.tolist()
plt.figure(figsize=(15,8))
x = list(range(240))
plt.bar(x,mis_val2,color="dodgerblue",label=False)
plt.xlabel("特征指标",fontsize=23)
plt.ylabel("缺失度",fontsize=23)
plt.show()

#导出文件
data.to_csv('F:/essay/data/data_clean.csv',index=False,header=True,encoding="utf_8_sig") 
nodeDataList = list(data.groupby(['category']))
for nodeData in nodeDataList:
   nodeData[1].to_excel('E:/essay/data/' + str(nodeData[0])+ '.xlsx', sheet_name=nodeData[0], index=False)

###特征选择
from sklearn import preprocessing
data_new = pd.read_excel("E:/essay/data/制造业.xlsx")
FLAG_industury = data_new['FLAG']

data_new.drop(['category','FLAG'], axis=1, inplace=True)
zscore = preprocessing.StandardScaler()
zscore = zscore.fit_transform(data_new)
data_new_zscore = pd.DataFrame(zscore,index=data_new.index,columns=data_new.columns)
data_new_zscore_new = pd.concat([data_new_zscore,FLAG_industury], axis=1)

X = data_new_zscore_new.iloc[:,0:data_new_zscore_new.shape[1]-1]
Y = data_new_zscore_new["FLAG"]
print((Counter(Y)))

# 逻辑斯蒂回归
Logistic = LogisticRegression()
Logistic.fit(X, Y)
m = 60
features_coefs = dict(zip(X.columns, Logistic.coef_.tolist()[0]))
features_coefs = sorted(features_coefs.items(), key=lambda i: abs(i[1]), reverse=True)[:m]
features_Logistic = list(map(lambda i: i[0], features_coefs))
    
# 随机森林
RandomForest = RandomForestClassifier()
RandomForest.fit(X, Y)
features_coefs = dict(zip(X.columns, RandomForest.feature_importances_))
features_coefs = sorted(features_coefs.items(), key=lambda i: abs(i[1]), reverse=True)[:m]
features_RandomForest = list(map(lambda i: i[0], features_coefs))
    
# GBDT
GBDT = GradientBoostingClassifier()
GBDT.fit(X, Y)
features_coefs = dict(zip(X.columns, GBDT.feature_importances_))
features_coefs = sorted(features_coefs.items(), key=lambda i: abs(i[1]), reverse=True)[:m]
features_gbdt = list(map(lambda i: i[0], features_coefs))
    
# 决策树   
clf = DecisionTreeClassifier()
clf.fit(X,Y)
features_coefs = dict(zip(X.columns, clf.feature_importances_))
features_coefs = sorted(features_coefs.items(), key=lambda i: abs(i[1]), reverse=True)[:m]
features_clf = list(map(lambda i: i[0], features_coefs))
    
#XGBOOST
xgb = XGBClassifier()              
xgb.fit(X,Y)           
features_coefs = dict(zip(X.columns, xgb.feature_importances_))
features_coefs = sorted(features_coefs.items(), key=lambda i: abs(i[1]), reverse=True)[:m]
features_xgb = list(map(lambda i: i[0], features_coefs))
    
features = features_Logistic
features.extend(features_RandomForest)
features.extend(features_gbdt)
features.extend(features_clf)
features.extend(features_xgb)
    
features_times = dict(Counter(features))
features_times = list(filter(lambda i: i[1] == 4, features_times.items()))
features = list(map(lambda i: i[0], features_times))

X = data_new_zscore_new[features]

#数据集划分及模型搭建
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y,test_size=0.3)
print((Counter(Ytrain)))

from imblearn.over_sampling import RandomOverSampler
X_resampled, Y_resampled = RandomOverSampler(random_state=0).fit_resample(Xtrain, Ytrain)
print((Counter(Y_resampled)))

def acu_curve(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

#LR
LR = LogisticRegression()
LR_para = [{'penalty':['l1','l2'],'C':[0.01,0.05,0.1,0.5,10,50,100]}]
grid_LR = GridSearchCV(Logistic,param_grid = LR_para)
grid_LR.fit(X_resampled, Y_resampled)
best_LR = grid_LR.best_estimator_
best_LR.fit(Xtrain,Ytrain)
y_pre_LR = best_LR.predict(Xtest)       # 调用模型对测试样本进行预测
print(accuracy_score(Ytest, y_pre_LR))
print(recall_score(Ytest, y_pre_LR))
print(f1_score(Ytest, y_pre_LR))
best_LR.score(Xtest, y_pre_LR)         # 模型在测试集商的精度
confusion_matrix(Ytest, y_pre_LR)   # 混淆矩阵
print(classification_report(Ytest, y_pre_LR))    # 打印分类报告（包含了各模型性能评价指标）
y_score_LR = best_LR.fit(Xtrain,Ytrain).decision_function(Xtest)
acu_curve(Ytest, y_score_LR)

#SVM
SVM = SVC()
param = [{'C':[0.01,0.05,0.1,0.5,10,50,100],'gamma':[0.01,0.05,0.1,0.5,10,50,100]}]
grid_svm = GridSearchCV(estimator=SVM, param_grid=param)
grid_svm.fit(X_resampled, Y_resampled)
best_SVM = grid_svm.best_estimator_
best_SVM.fit(Xtrain,Ytrain)
y_pre_SVM = best_SVM.predict(Xtest)       # 调用模型对测试样本进行预测
print(accuracy_score(Ytest, y_pre_SVM))
print(precision_score(Ytest, y_pre_SVM))
print(recall_score(Ytest, y_pre_SVM))
print(f1_score(Ytest, y_pre_SVM))
best_SVM.score(Xtest, Ytest)         # 模型在测试集商的精度
confusion_matrix(Ytest, y_pre_SVM)   # 混淆矩阵
print(classification_report(Ytest, y_pre_SVM))    # 打印分类报告（包含了各模型性能评价指标）
y_score_SVM = best_SVM.fit(Xtrain, Ytrain).decision_function(Xtest)
acu_curve(Ytest, y_score_SVM)

#RandomForestset
RandomForestset = RandomForestClassifier()
param = [{'n_estimators': [1,10,20,50,100],"max_features":[1,5,10,15,20,22]}]
RandomForest = GridSearchCV(estimator=RandomForestset, param_grid=param)
RandomForest.fit(X_resampled, Y_resampled)
best_RandomForestset = RandomForest.best_estimator_
best_RandomForestset.fit(Xtrain, Ytrain)
y_pre_RF = best_RandomForestset.predict(Xtest)       # 调用模型对测试样本进行预测
print(accuracy_score(Ytest, y_pre_RF))
print(precision_score(Ytest, y_pre_RF))
print(recall_score(Ytest, y_pre_RF))
print(f1_score(Ytest, y_pre_RF))
best_RandomForestset.score(Xtest, Ytest)         # 模型在测试集商的精度
confusion_matrix(Ytest, y_pre_RF)   # 混淆矩阵
print(classification_report(Ytest, y_pre_RF))    # 打印分类报告（包含了各模型性能评价指标）
y_score_RF = best_RandomForestset.fit(Xtrain, Ytrain).predict_proba(Xtest)[:, 1]
acu_curve(Ytest, y_score_RF)

#XGBoost
XGBoost = XGBClassifier()
param = [{"max_depth":range(1,16),'learning_rate': [0.01,0.1,0.5,1]}]
XGBoost = GridSearchCV(estimator=XGBoost, param_grid=param)
XGBoost.fit(X_resampled, Y_resampled)
best_XGBoost = XGBoost.best_estimator_
best_XGBoost.fit(Xtrain, Ytrain)
y_pre_XG = best_XGBoost.predict(Xtest)       # 调用模型对测试样本进行预测
print(accuracy_score(Ytest, y_pre_XG))
print(precision_score(Ytest, y_pre_XG))
print(recall_score(Ytest, y_pre_XG))
print(f1_score(Ytest, y_pre_XG))
best_XGBoost.score(Xtest, Ytest)         # 模型在测试集商的精度
confusion_matrix(Ytest, y_pre_XG)   # 混淆矩阵
print(classification_report(Ytest, y_pre_XG))    # 打印分类报告（包含了各模型性能评价指标）
y_score_XG = best_XGBoost.fit(Xtrain, Ytrain).predict_proba(Xtest)[:, 1]
acu_curve(Ytest, y_score_XG)

#Voting classifier
from sklearn.ensemble import VotingClassifier 
#两种分类器的比较
best_XGBoost = XGBoost.best_estimator_
best_LR = grid_LR.best_estimator_
best_SVM = grid_svm.best_estimator_
best_SVM = SVC(C=10, gamma=100,probability=True)
best_RandomForestset = RandomForest.best_estimator_

lr_xgb=VotingClassifier(estimators=[('LR',best_LR),('XGB',best_XGBoost)],voting='soft')
lr_xgb=lr_xgb.fit(Xtrain, Ytrain)
y_pred_lr_xgb = lr_xgb.predict(Xtest)
confusion_matrix(Ytest, y_pred_lr_xgb) 
print(accuracy_score(Ytest, y_pred_lr_xgb))
print(precision_score(Ytest, y_pred_lr_xgb))
print(recall_score(Ytest, y_pred_lr_xgb))
print(f1_score(Ytest, y_pred_lr_xgb))
y_score = lr_xgb.fit(Xtrain, Ytrain).predict_proba(Xtest)[:, 1]
acu_curve(Ytest, y_score)

SVM_xgb=VotingClassifier(estimators=[('SVM',best_SVM),('XGB',best_XGBoost)],voting='soft')
SVM_xgb=SVM_xgb.fit(Xtrain, Ytrain)
y_pred_SVM_xgb = SVM_xgb.predict(Xtest)
confusion_matrix(Ytest, y_pred_SVM_xgb) 
print(accuracy_score(Ytest, y_pred_SVM_xgb))
print(precision_score(Ytest, y_pred_SVM_xgb))
print(recall_score(Ytest, y_pred_SVM_xgb))
print(f1_score(Ytest, y_pred_SVM_xgb))
y_score = SVM_xgb.fit(Xtrain, Ytrain).predict_proba(Xtest)[:, 1]
acu_curve(Ytest, y_score)

RandomForestse_xgb=VotingClassifier(estimators=[('RF',best_RandomForestset),('XGB',best_XGBoost)],voting='soft')
RandomForestse_xgb=RandomForestse_xgb.fit(Xtrain, Ytrain)
y_pred_RandomForestse_xgb = RandomForestse_xgb.predict(Xtest)
confusion_matrix(Ytest, y_pred_RandomForestse_xgb) 
print(accuracy_score(Ytest, y_pred_RandomForestse_xgb))
print(precision_score(Ytest, y_pred_RandomForestse_xgb))
print(recall_score(Ytest, y_pred_RandomForestse_xgb))
print(f1_score(Ytest, y_pred_RandomForestse_xgb))
y_score = RandomForestse_xgb.fit(Xtrain, Ytrain).predict_proba(Xtest)[:, 1]
acu_curve(Ytest, y_score)

###
from sklearn import metrics

Font={'size':18, 'family':'Times New Roman'}
fpr1,tpr1,thres1 = metrics.roc_curve(Ytest, y_pred_lr_xgb)
fpr2,tpr2,thres2 = metrics.roc_curve(Ytest, y_pred_SVM_xgb)
fpr3,tpr3,thres3 = metrics.roc_curve(Ytest, y_pred_RandomForestse_xgb)
roc_auc1 = metrics.auc(fpr1, tpr1)
roc_auc2 = metrics.auc(fpr2, tpr2)
roc_auc3 = metrics.auc(fpr3, tpr3)
  
plt.figure(figsize=(6,6))
plt.plot(fpr1, tpr1, 'b', label = 'LR-XGB = %0.3f' % roc_auc1, color='Red')
plt.plot(fpr2, tpr2, 'b', label = 'SVM-XGB = %0.3f' % roc_auc2, color='k')
plt.plot(fpr3, tpr3, 'b', label = 'RF-XGB = %0.3f' % roc_auc3, color='RoyalBlue')
plt.legend(loc = 'lower right', prop=Font)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate', Font)
plt.xlabel('False Positive Rate', Font)
plt.tick_params(labelsize=15)
plt.show()