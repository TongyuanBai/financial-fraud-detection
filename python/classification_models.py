
# 导入所需模块

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import precision_score

db = pd.read_csv("F:/essay/data/data_make.csv")
db = db.drop(columns=['Unnamed: 0'])
X = db[['CIP',
 'RETAINED_EARNINGS',
 'GOING_CONCERN_NI',
 'MINORITY_GAIN',
 'FCFF',
 'PROC_SELL_INVEST',
 'N_CE_BEG_BAL',
 'BASIC_EPS',
 'OTH_PAYABLE',
 'T_CA',
 'NOPERATE_EXP',
 'REVENUE']]
Y = db["FLAG"]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.3)
X_resampled, Y_resampled = RandomOverSampler(random_state=0).fit_resample(Xtrain, Ytrain)

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

def Acc(y, pred):
    N = y.shape[0]
    return (y == pred).sum() / N

def classify(X_resampled, Xtest, Y_resampled, Ytest):    
    acc = {}
    precision = [] #存放准确率
    recall = []
    f1 = []
  
    ## 逻辑斯谛回归模型
    Logistic = LogisticRegression()
    lr_para = [{'penalty':['l1','l2'],'C':[0.01,0.05,0.1,0.5,10,50,100],'solver':['liblinear'],'multi_class':['ovr']}]
    grid_lr = GridSearchCV(Logistic,param_grid = lr_para)
    print(grid_lr.fit(X_resampled, Y_resampled))
    print(grid_lr.best_params_)
    print(grid_lr.best_estimator_)
    print(grid_lr.best_score_)
    lr1 = LogisticRegression(C=0.5, multi_class='ovr', solver='liblinear')
    lr1.fit(X_resampled, Y_resampled)
    pred = lr1.predict(Xtest)
    acc['逻辑斯谛回归模型'] = [pred, Acc(Ytest, pred)]
    precision.append(precision_score(Ytest,pred))
    recall.append(recall_score(Ytest,pred))
    f1.append(f1_score(Ytest,pred))
    y_score = lr1.fit(X_resampled, Y_resampled).decision_function(Xtest)
    fpr,tpr,threshold = roc_curve(Ytest, y_score) ###计算真正率和假正率
    acu_curve(Ytest, y_score)
    con_lr = confusion_matrix(Ytest,pred)
    print(con_lr)
    
    ## SVM
    SVM = SVC()
    param = [{'C':[0.01,0.05,0.1,0.5,10,50,100],'gamma':[0.01,0.05,0.1,0.5,10,50,100]}]
    svm = GridSearchCV(estimator=SVM, param_grid=param)
    svm.fit(Xtrain, Ytrain)
    print(svm.best_params_)
    print(svm.best_estimator_)
    print(svm.best_score_)
    SVM1 = SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    SVM1.fit(Xtrain, Ytrain)
    pred = SVM1.predict(Xtest)
    acc['SVM'] = [pred, Acc(Ytest, pred)]
    precision.append(precision_score(Ytest,pred))
    recall.append(recall_score(Ytest,pred))
    f1.append(f1_score(Ytest,pred))
    y_score = SVM.fit(Xtrain, Ytrain).decision_function(Xtest)
    fpr,tpr,threshold = roc_curve(Ytest, y_score) ###计算真正率和假正率
    acu_curve(Ytest, y_score)
    con_lr = confusion_matrix(Ytest,pred)
    print(con_lr)
    
    ## 随机森林
    RandomForestset = RandomForestClassifier()
    param = [{'n_estimators': [1,10,20,50,100],"max_features":[1,5,10,15,20,22]}]
    RandomForest = GridSearchCV(estimator=RandomForestset, param_grid=param)
    RandomForest.fit(Xtrain, Ytrain)
    print(RandomForest.best_params_)
    print(RandomForest.best_estimator_)
    print(RandomForest.best_score_)
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=1, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    rf.fit(Xtrain, Ytrain)
    pred = rf.predict(Xtest)
    acc['随机森林'] = [pred, Acc(Ytest, pred)]
    precision.append(precision_score(Ytest,pred))
    recall.append(recall_score(Ytest,pred))
    f1.append(f1_score(Ytest,pred))
    y_score = rf.fit(Xtrain, Ytrain).predict_proba(Xtest)[:, 1]
    fpr,tpr,threshold = roc_curve(Ytest, y_score) ###计算真正率和假正率
    acu_curve(Ytest, y_score)
    con_lr = confusion_matrix(Ytest,pred)
    print(con_lr)
    
    ## XGBoost
    XGBoost = XGBClassifier()
    param = [{"max_features":[1,5,10,15],'learning_rate': [0.01,0.1,0.5,1]}]
    XGBoost = GridSearchCV(estimator=XGBoost, param_grid=param)
    XGBoost.fit(Xtrain, Ytrain)
    print(XGBoost.best_params_)
    print(XGBoost.best_estimator_)
    print(XGBoost.best_score_)
    XGBoost.fit(Xtrain, Ytrain)
    pred = XGBoost.predict(Xtest)
    acc['XGBoost'] = [pred, Acc(Ytest, pred)]
    precision.append(precision_score(Ytest,pred))
    recall.append(recall_score(Ytest,pred))
    f1.append(f1_score(Ytest,pred))
    y_score = XGBoost.predict_proba(np.array(Xtest))[:,1]
    fpr,tpr,threshold = roc_curve(Ytest, y_score) ###计算真正率和假正率
    acu_curve(Ytest, y_score)
    con_lr = confusion_matrix(Ytest,pred)
    print(con_lr)

    ## 决策树
    DecisionTree = DecisionTreeClassifier()
    DecisionTree.fit(Xtrain, Ytrain)
    pred = DecisionTree.predict(Xtest)
    acc['决策树'] = [pred, Acc(Ytest, pred)]
    precision.append(precision_score(Ytest,pred))
    recall.append(recall_score(Ytest,pred))
    f1.append(f1_score(Ytest,pred))
    y_score = DecisionTree.fit(Xtrain, Ytrain).predict_proba(Xtest)[:, 1]
    fpr,tpr,threshold = roc_curve(Ytest, y_score) ###计算真正率和假正率
    acu_curve(Ytest, y_score)
    con_lr = confusion_matrix(Ytest,pred)
    print(con_lr)
    feature_name = ['PAYROLL_PAYABLE','DEFER_TAX_ASSETS','C_FR_CAP_CONTR','BTAX_SURCHG_TR',
         'C_PAID_FOR_OTH_OP_A','FCFF','ST_BORR','T_FIXED_A_TA','ASSETS_IMPAIR_LOSS',
         'T_SH_EQUITY','TAXES_PAYABLE','OTH_COMPR_INCOME','N_CF_OPERATE_A','T_CA','GOODWILL','C_FR_OTH_FINAN_A']
    dot_data = export_graphviz(DecisionTree, out_file=None,
                                    feature_names=feature_name,  
                                    class_names=["no","yes"],
                                    filled=True, rounded=True)  
    graph = graphviz.Source(dot_data)  
    graph


###
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
m = 70
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
Y = data_new_zscore_new["FLAG"]
df1 = pd.concat([X,Y], axis=1)
df_corr = df1.corr()
corr = df_corr['FLAG'].drop('FLAG').abs()

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


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y,test_size=0.3)
print((Counter(Ytrain)))

from imblearn.over_sampling import RandomOverSampler
X_resampled, Y_resampled = RandomOverSampler(random_state=0).fit_resample(Xtrain, Ytrain)
print((Counter(Y_resampled)))

LR = LogisticRegression()
LR_para = [{'penalty':['l1','l2'],'C':[0.01,0.05,0.1,0.5,10,50,100]}]
grid_LR = GridSearchCV(Logistic,param_grid = LR_para)
grid_LR.fit(X_resampled, Y_resampled)

best_LR = grid_LR.best_estimator_
best_LR.fit(Xtrain,Ytrain)
y_pre = best_LR.predict(Xtest)       # 调用模型对测试样本进行预测
best_LR.score(Xtest, Ytest)         # 模型在测试集商的精度
confusion_matrix(Ytest, y_pre)   # 混淆矩阵
print(classification_report(Ytest, y_pre))    # 打印分类报告（包含了各模型性能评价指标）

y_score = best_LR.fit(Xtrain,Ytrain).decision_function(Xtest)
acu_curve(Ytest, y_score)

SVM = SVC()
param = [{'C':[0.01,0.05,0.1,0.5,10,50,100],'gamma':[0.01,0.05,0.1,0.5,10,50,100]}]
grid_svm = GridSearchCV(estimator=SVM, param_grid=param)
grid_svm.fit(X_resampled, Y_resampled)
best_SVM = grid_svm.best_estimator_
best_SVM.fit(Xtrain,Ytrain)
y_pre = best_SVM.predict(Xtest)       # 调用模型对测试样本进行预测
best_SVM.score(Xtest, Ytest)         # 模型在测试集商的精度
confusion_matrix(Ytest, y_pre)   # 混淆矩阵
print(classification_report(Ytest, y_pre))    # 打印分类报告（包含了各模型性能评价指标）

y_score = best_SVM.fit(Xtrain, Ytrain).decision_function(Xtest)
acu_curve(Ytest, y_score)

RandomForestset = RandomForestClassifier()
param = [{'n_estimators': [1,10,20,50,100],"max_features":[1,5,10,15,20,22]}]
RandomForest = GridSearchCV(estimator=RandomForestset, param_grid=param)
RandomForest.fit(X_resampled, Y_resampled)
best_RandomForestset = RandomForest.best_estimator_
best_RandomForestset.fit(Xtrain, Ytrain)
y_pre = best_RandomForestset.predict(Xtest)       # 调用模型对测试样本进行预测
best_RandomForestset.score(Xtest, Ytest)         # 模型在测试集商的精度
confusion_matrix(Ytest, y_pre)   # 混淆矩阵
print(classification_report(Ytest, y_pre))    # 打印分类报告（包含了各模型性能评价指标）

y_score = best_RandomForestset.fit(Xtrain, Ytrain).predict_proba(Xtest)[:, 1]
acu_curve(Ytest, y_score)
