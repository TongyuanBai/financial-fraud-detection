
# 导入所需模块
from collections import Counter
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import pandas as pd

db = pd.read_csv("E:/essay/data/data_make.csv")
db = db.drop(columns=['Unnamed: 0'])
X = db.iloc[:,0:db.shape[1]-1]
Y = db["FLAG"]

def select_features(X, Y):
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
    clf = tree.DecisionTreeClassifier()
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
    
    return X[features]

select_features(X,Y)
